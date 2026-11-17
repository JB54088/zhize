"""Judge-agent semantic assistance and evidence-grounded answer generation."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from ..stages.judge import judge_records
from ._runtime import invoke_json, json_message, make_log, now_iso
from .models import AgentExecution, AnswerResponse, MatchResponse


_MATCH_SYSTEM = """你是知择的小判语义匹配助手。只依据输入中的可核验事实评估语义相关程度。不要修改规则条件，不得把未知条件视为满足，不得生成新的岗位、公司、资格、截止时间或来源。返回的 evidence_quotes 必须逐字复制 verified facts 中的 quote。reason 只解释这些证据如何对应用户偏好，不得声称硬条件已通过；硬条件的最终分类由程序规则决定。输出 JSON。"""
_ANSWER_SYSTEM = """你是知择的最终结论生成器。只能使用输入中的规则判断和已核验事实，不可添加来源、公司、岗位、资格、日期或外部知识。每个事实性 claim 必须引用同一结果中的 evidence_quotes，且 quote 必须从输入的 verified_facts 原样复制。明确区分推荐、待核实和不匹配；未知条件不能写成满足。未找到足够证据时应简洁说明证据不足。不得输出输入以外的 URL。输出 JSON。"""


class JudgeAgent:
    """Run deterministic hard gates first; model analysis is only additive."""

    async def run(
        self,
        query: str,
        intent: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        llm_client: Any = None,
    ) -> AgentExecution:
        started_iso, started_clock = now_iso(), time.perf_counter()
        judged = judge_records(query, intent, records)
        output = {"records": judged}
        candidates = []
        allowed_by_id: dict[str, set[str]] = {}
        employment_mode = intent.get("domain") == "employment"
        skipped_scope = 0
        for record in judged:
            record_id = str(record.get("id", ""))
            if employment_mode and record.get("employment_source_scope") == "corporate_campus_page":
                record["llm_semantic_match"] = {"status": "not_analyzed"}
                skipped_scope += 1
                continue
            quotes = _verified_quotes(record)
            if not record_id or not quotes:
                record["llm_semantic_match"] = {"status": "not_analyzed"}
                continue
            allowed_by_id[record_id] = quotes
            candidates.append({
                "id": record_id,
                "title": record.get("title", ""),
                "rule_decision": record.get("decision"),
                "rule_match_status": record.get("match_status"),
                "rule_match_score": record.get("match_score"),
                "hard_conditions": record.get("key_conditions", []),
                "verified_facts": _verified_facts(record),
            })
        if not candidates:
            log = make_log(
                agent="judge",
                action="semantic_match",
                status="skipped_no_evidence",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary=f"规则裁决 {len(records)} 条候选结果；没有岗位级已核验事实可供语义解释。",
                output_summary=(
                    f"硬性条件按规则分类；{skipped_scope} 条企业校招主页不进入岗位语义匹配。"
                    if skipped_scope
                    else "硬性条件按规则分类；模型语义匹配未运行。"
                ),
                skip_reason="verified_fact_missing",
            )
            return AgentExecution(output=output, run_log=log)
        if not _enabled(llm_client):
            for item in judged:
                item.setdefault("llm_semantic_match", {"status": "not_analyzed"})
            log = make_log(
                agent="judge",
                action="semantic_match",
                status="skipped_unconfigured",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary=f"规则裁决 {len(records)} 条结果；{len(candidates)} 条有已核验事实。",
                output_summary="规则分类已完成；LLM 语义解释未配置。",
                skip_reason="llm_not_configured",
            )
            return AgentExecution(output=output, run_log=log)

        data, metadata, error_code = await invoke_json(
            llm_client,
            agent_name="judge",
            messages=[
                {"role": "system", "content": _MATCH_SYSTEM},
                {"role": "user", "content": json_message({"query": query, "intent": dict(intent), "results": candidates})},
            ],
            response_model=MatchResponse,
        )
        if error_code:
            log = make_log(
                agent="judge",
                action="semantic_match",
                status="failed",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary=f"模型待解释 {len(candidates)} 条带已核验证据的结果。",
                output_summary="LLM 语义匹配调用失败；硬条件规则结果仍单独保留。",
                metadata=metadata,
                error_code=error_code,
            )
            return AgentExecution(output=output, run_log=log)
        try:
            response = MatchResponse.model_validate(data)
        except ValidationError:
            log = make_log(
                agent="judge",
                action="semantic_match",
                status="failed",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary=f"模型待解释 {len(candidates)} 条带已核验证据的结果。",
                output_summary="模型响应未通过语义匹配结构校验。",
                metadata=metadata,
                error_code="response_schema_invalid",
            )
            return AgentExecution(output=output, run_log=log)

        by_id = {str(item.get("id", "")): item for item in judged}
        accepted = rejected = 0
        for item in response.items:
            record = by_id.get(item.id)
            allowed = allowed_by_id.get(item.id)
            refs = list(dict.fromkeys(item.evidence_quotes))
            if record is None or not allowed or not refs or any(ref not in allowed for ref in refs):
                rejected += 1
                continue
            record["llm_semantic_match"] = {
                "status": "analyzed",
                "fit": item.semantic_fit,
                "score": item.score,
                "reason": item.reason,
                "evidence_quotes": refs,
            }
            accepted += 1
        for record in judged:
            record.setdefault("llm_semantic_match", {"status": "not_analyzed"})

        log = make_log(
            agent="judge",
            action="semantic_match",
            status="completed",
            started_iso=started_iso,
            started_clock=started_clock,
            input_summary=f"模型评估 {len(candidates)} 条带已核验证据的结果。",
            output_summary=f"接受 {accepted} 条带精确证据引用的语义分析，拒绝 {rejected} 条；规则决策和分数不变。",
            metadata=metadata,
        )
        return AgentExecution(output=output, run_log=log)


class AnswerAgent:
    """Generate a concise response whose every claim cites existing verified text."""

    async def run(
        self,
        query: str,
        intent: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        llm_client: Any = None,
    ) -> AgentExecution:
        started_iso, started_clock = now_iso(), time.perf_counter()
        eligible: list[dict[str, Any]] = []
        allowed_by_id: dict[str, set[str]] = {}
        for record in records:
            record_id = str(record.get("id", ""))
            quotes = _verified_quotes(record)
            if not record_id or not quotes:
                continue
            allowed_by_id[record_id] = quotes
            eligible.append({
                "id": record_id,
                "title": record.get("title", ""),
                "decision": record.get("decision", "needs_verification"),
                "summary": record.get("summary", ""),
                "hard_conditions": record.get("key_conditions", []),
                "verified_facts": _verified_facts(record),
                "llm_semantic_match": record.get("llm_semantic_match", {}),
            })
        output: dict[str, Any] = {"answer": None, "claims": []}
        if not eligible:
            log = make_log(
                agent="judge",
                action="final_answer",
                status="skipped_no_evidence",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary=f"读取 {len(records)} 条规则判断结果；没有已核验事实可引用。",
                output_summary="未生成事实性结论，等待可核验证据。",
                skip_reason="verified_fact_missing",
            )
            return AgentExecution(output=output, run_log=log)
        if not _enabled(llm_client):
            log = make_log(
                agent="judge",
                action="final_answer",
                status="skipped_unconfigured",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary=f"读取 {len(eligible)} 条含已核验事实的规则结果。",
                output_summary="LLM 最终结论生成未配置。",
                skip_reason="llm_not_configured",
            )
            return AgentExecution(output=output, run_log=log)

        data, metadata, error_code = await invoke_json(
            llm_client,
            agent_name="judge",
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user", "content": json_message({"query": query, "intent": dict(intent), "results": eligible})},
            ],
            response_model=AnswerResponse,
        )
        if error_code:
            log = make_log(
                agent="judge",
                action="final_answer",
                status="failed",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary=f"模型待总结 {len(eligible)} 条含已核验事实的规则结果。",
                output_summary="LLM 最终结论调用失败；没有静默生成替代结论。",
                metadata=metadata,
                error_code=error_code,
            )
            return AgentExecution(output=output, run_log=log)
        try:
            response = AnswerResponse.model_validate(data)
        except ValidationError:
            log = make_log(
                agent="judge",
                action="final_answer",
                status="failed",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary=f"模型待总结 {len(eligible)} 条含已核验事实的规则结果。",
                output_summary="模型响应未通过最终结论结构校验。",
                metadata=metadata,
                error_code="response_schema_invalid",
            )
            return AgentExecution(output=output, run_log=log)

        validated_claims: list[dict[str, Any]] = []
        invalid = False
        for claim in response.claims:
            allowed = allowed_by_id.get(claim.result_id)
            refs = list(dict.fromkeys(claim.evidence_quotes))
            if not allowed or not refs or any(ref not in allowed for ref in refs):
                invalid = True
                break
            if not _specific_tokens_are_grounded(claim.text, refs):
                invalid = True
                break
            validated_claims.append({
                "result_id": claim.result_id,
                "text": claim.text,
                "evidence_quotes": refs,
            })
        # A non-empty result set must yield at least one source-linked claim;
        # otherwise the free-form summary would be ungrounded.
        cited_quotes = [
            quote for claim in validated_claims for quote in claim["evidence_quotes"]
        ]
        if (
            invalid
            or not validated_claims
            or not _specific_tokens_are_grounded(response.summary, cited_quotes)
        ):
            log = make_log(
                agent="judge",
                action="final_answer",
                status="failed",
                started_iso=started_iso,
                started_clock=started_clock,
                input_summary=f"模型待总结 {len(eligible)} 条含已核验事实的规则结果。",
                output_summary="最终回答引用校验失败，未发布模型文本。",
                metadata=metadata,
                error_code="citation_validation_failed",
            )
            return AgentExecution(output=output, run_log=log)

        output = {"answer": response.summary, "claims": validated_claims}
        log = make_log(
            agent="judge",
            action="final_answer",
            status="completed",
            started_iso=started_iso,
            started_clock=started_clock,
            input_summary=f"模型总结 {len(eligible)} 条含已核验事实的规则结果。",
            output_summary=f"最终结论包含 {len(validated_claims)} 条可追溯引证；逐字引用校验通过。",
            metadata=metadata,
        )
        return AgentExecution(output=output, run_log=log)


def _verified_quotes(record: Mapping[str, Any]) -> set[str]:
    quotes: set[str] = set()
    official_url = record.get("official_source_url")
    if record.get("official_source_status") != "verified" or not isinstance(official_url, str) or not official_url:
        return quotes
    checks = record.get("field_verification")
    if isinstance(checks, Mapping):
        for check in checks.values():
            if (
                isinstance(check, Mapping)
                and check.get("status") == "verified"
                and check.get("evidence_url") == official_url
            ):
                quote = check.get("quote")
                if isinstance(quote, str) and quote.strip():
                    quotes.add(quote)
    conditions = record.get("key_conditions")
    if isinstance(conditions, list):
        for check in conditions:
            if (
                isinstance(check, Mapping)
                and check.get("status") == "verified"
                and check.get("evidence_url") == official_url
            ):
                quote = check.get("quote")
                if isinstance(quote, str) and quote.strip():
                    quotes.add(quote)
    # The already-verified original notice title can support a project/title
    # claim; an arbitrary search-result title cannot.
    title = record.get("official_source_title")
    if record.get("official_source_status") == "verified" and isinstance(title, str) and title.strip():
        quotes.add(title)
    return quotes


def _verified_facts(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    official_url = record.get("official_source_url")
    if record.get("official_source_status") != "verified" or not isinstance(official_url, str) or not official_url:
        return facts
    checks = record.get("field_verification")
    if isinstance(checks, Mapping):
        for field, check in checks.items():
            if (
                not isinstance(check, Mapping)
                or check.get("status") != "verified"
                or check.get("evidence_url") != official_url
            ):
                continue
            quote = check.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                continue
            facts.append({
                "field": str(field),
                "value": check.get("value"),
                "evidence_url": check.get("evidence_url"),
                "quote": quote,
            })
    if record.get("official_source_status") == "verified":
        title = record.get("official_source_title")
        if isinstance(title, str) and title.strip():
            facts.append({"field": "official_source_title", "value": title, "evidence_url": official_url, "quote": title})
    return facts


def _specific_tokens_are_grounded(claim: str, quotes: Sequence[str]) -> bool:
    """Reject model-added dates, URLs, amounts, and number claims."""
    grounded = "\n".join(quotes)
    tokens = re.findall(
        r"(?:https?://[^\s，。；;）)]+|(?:20\d{2}(?:\s*届)?|\d+(?:\.\d+)?\s*(?:元|万元|亿元|人|名|个|%)))",
        claim,
        flags=re.IGNORECASE,
    )
    return all(token in grounded for token in tokens)


def _enabled(client: Any) -> bool:
    from ._runtime import llm_is_enabled

    return llm_is_enabled(client)
