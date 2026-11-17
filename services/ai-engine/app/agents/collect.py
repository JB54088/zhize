"""Collect-agent actions: natural-language intent and real search dispatch."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from ..stages.collect import SearchAdapter, collect_records, parse_query_intent
from ._runtime import invoke_json, json_message, make_log, now_iso
from .models import AgentExecution, IntentAnalysis


_INTENT_SYSTEM = """你是知择的信息检索意图分析器。只理解用户需求并生成可用于真实搜索的查询式，不得宣称搜索已经发生，不得创造公司、项目、来源或资格条件。硬性条件由程序从原始需求确定；你只能总结它们，不得删改、弱化或新增硬性条件。输出符合 schema 的 JSON。搜索式最多 5 条，保留原始需求含义，不要引入用户未提到的专有实体。"""


class IntentAgent:
    """LLM-assisted search intent; deterministic hard requirements stay final."""

    async def run(self, query: str, llm_client: Any = None) -> AgentExecution:
        started_iso, started_clock = now_iso(), time.perf_counter()
        deterministic = parse_query_intent(query)
        output: dict[str, Any] = {
            "query": query,
            "domain": deterministic["domain"],
            "semantic_domain": deterministic["domain"],
            "hard_conditions": deterministic["hard_conditions"],
            "soft_conditions": deterministic["soft_conditions"],
            "intent_summary": _deterministic_summary(deterministic),
            "search_queries": [query],
        }
        if not _enabled(llm_client):
            log = make_log(
                agent="collect",
                action="intent_understanding",
                status="skipped_unconfigured",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary="已解析 1 条用户需求；硬性条件由确定性规则提取。",
                output_summary=f"保留 {len(output['hard_conditions'])} 个规则硬条件；LLM 意图理解未配置。",
                skip_reason="llm_not_configured",
            )
            return AgentExecution(output=output, run_log=log)

        messages = [
            {"role": "system", "content": _INTENT_SYSTEM},
            {
                "role": "user",
                "content": json_message({
                    "query": query,
                    "deterministic_domain": deterministic["domain"],
                    "immutable_hard_conditions": deterministic["hard_conditions"],
                    "known_soft_conditions": deterministic["soft_conditions"],
                }),
            },
        ]
        data, metadata, error_code = await invoke_json(
            llm_client,
            agent_name="collect",
            messages=messages,
            response_model=IntentAnalysis,
        )
        if error_code:
            log = make_log(
                agent="collect",
                action="intent_understanding",
                status="failed",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary="已解析 1 条用户需求；硬性条件由确定性规则提取。",
                output_summary="LLM 意图理解调用失败；未将规则结果标记为模型成功。",
                metadata=metadata,
                error_code=error_code,
            )
            return AgentExecution(output=output, run_log=log)

        try:
            parsed = IntentAnalysis.model_validate(data)
        except ValidationError:
            log = make_log(
                agent="collect",
                action="intent_understanding",
                status="failed",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary="已解析 1 条用户需求；硬性条件由确定性规则提取。",
                output_summary="模型响应未通过意图结构校验。",
                metadata=metadata,
                error_code="response_schema_invalid",
            )
            return AgentExecution(output=output, run_log=log)

        queries = _unique_nonempty([query, *parsed.search_queries], maximum=6)
        output.update({
            # Semantic classification is informative only. The rule domain
            # remains deterministic so an LLM cannot omit domain hard gates.
            "semantic_domain": parsed.domain,
            "intent_summary": parsed.intent_summary or output["intent_summary"],
            "search_queries": queries,
            "soft_preferences": parsed.soft_preferences,
        })
        log = make_log(
            agent="collect",
            action="intent_understanding",
            status="completed",
            started_iso=started_iso,
            started_clock=started_clock,
            input_summary=f"已解析需求及 {len(deterministic['hard_conditions'])} 个规则硬条件。",
            output_summary=f"模型生成 {len(queries)} 条去重搜索式；规则硬条件保持不变。",
            metadata=metadata,
        )
        return AgentExecution(output=output, run_log=log)


class CollectAgent:
    """Dispatch actual queries to configured search adapters concurrently."""

    async def run(
        self,
        query: str,
        search_queries: Sequence[str],
        adapters: Sequence[SearchAdapter],
        limit: int,
        llm_client: Any = None,
    ) -> AgentExecution:
        del query, llm_client  # Query variants come from IntentAgent; search is a tool action.
        started_iso, started_clock = now_iso(), time.perf_counter()
        queries = _unique_nonempty(list(search_queries), maximum=6)
        if not queries:
            queries = [""]
        # Provider result caps vary; keep requests within the shared 20-item limit.
        # candidate_limit remains a post-normalization upper bound, while each
        # provider request stays inside that documented request limit.
        per_query_limit = min(20, max(1, int(limit)))
        records: list[dict[str, Any]] = []
        errors: list[str] = []

        async def one_search(index: int, search_query: str):
            return index, search_query, await collect_records(search_query, adapters, per_query_limit)

        responses = await asyncio.gather(
            *(one_search(index, item) for index, item in enumerate(queries)),
            return_exceptions=True,
        )
        for response in responses:
            if isinstance(response, BaseException):
                errors.append(type(response).__name__)
                continue
            index, search_query, (found, query_errors) = response
            records.extend(found)
            errors.extend(query_errors)
            for record in found:
                provenance = record.get("_provenance")
                if isinstance(provenance, dict):
                    provenance["search_query_index"] = index
                    provenance["search_query"] = search_query

        if not records and errors:
            output: dict[str, Any] = {"records": [], "errors": list(dict.fromkeys(errors)), "search_queries": queries}
            log = make_log(
                agent="collect",
                action="search",
                status="failed",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary=f"向 {len(adapters)} 个已配置搜索源分发 {len(queries)} 条查询。",
                output_summary="所有搜索请求均未返回记录。",
                error_code="search_sources_failed",
            )
            return AgentExecution(output=output, run_log=log)

        output = {"records": records, "errors": list(dict.fromkeys(errors)), "search_queries": queries}
        log = make_log(
            agent="collect",
            action="search",
            status="partial" if errors else "completed",
            started_iso=started_iso,
            started_clock=started_clock,
            input_summary=f"向 {len(adapters)} 个已配置搜索源分发 {len(queries)} 条查询。",
            output_summary=f"搜索源返回 {len(records)} 条带来源追溯信息的原始记录；源错误数 {len(errors)}。",
        )
        return AgentExecution(output=output, run_log=log)


def _enabled(client: Any) -> bool:
    from ._runtime import llm_is_enabled

    return llm_is_enabled(client)


def _unique_nonempty(values: Sequence[Any], maximum: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.split())
        normalized = cleaned.casefold()
        if cleaned and normalized not in seen:
            seen.add(normalized)
            result.append(cleaned)
        if len(result) >= maximum:
            break
    return result


def _deterministic_summary(intent: dict[str, Any]) -> str:
    domain = intent.get("domain", "general")
    descriptions = {
        "employment": "求职与招聘信息检索",
        "procurement": "政府采购与招标信息检索",
        "general": "通用信息检索",
    }
    return descriptions.get(str(domain), descriptions["general"])
