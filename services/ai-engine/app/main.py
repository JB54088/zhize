from pathlib import Path
import asyncio
import json
import re
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .agents import AnswerAgent, CleanAgent, CollectAgent, IntentAgent, JudgeAgent, VerifyAgent
from .llm import LLMClient, LLMConfig
from .official_verifier import verify_official_source
from .provider import ProviderError, create_search_provider, create_search_providers
from .research.contracts import ResearchOutcome
from .research.config import research_budget_from_env
from .research.loop import ResearchLoop
from .result_cards import build_result_cards
from .search_tools import SearchToolbox
from .semantic_matching import rank_and_explain_results
from .source_records import build_source_records
from .schemas import AgentRunSnapshot, AnswerClaim, SearchRequest, SearchResponse, SearchResult, StageSnapshot
from .settings import Settings
from .stages.collect import ProviderAdapter
from .stages.clean import canonicalize_url
from .stages.verify import (
    find_official_employment_evidence,
    verification_counts,
    verify_records,
)


load_dotenv(Path(__file__).resolve().parents[3] / ".env")

app = FastAPI(title="知择 AI 信息决策引擎", version="0.3.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "zhize-ai-engine"}


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must contain non-whitespace text")
    try:
        return await _execute_search(payload)
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="AI search pipeline failed.") from exc


@app.post("/api/v1/search/stream")
async def search_stream(payload: SearchRequest) -> StreamingResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must contain non-whitespace text")

    async def events():
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        stages = ("collect", "clean", "verify", "judge")
        for key in stages:
            pending = StageSnapshot(key=key, status="pending").model_dump(mode="json")
            yield _ndjson({"type": "stage", "stage": pending})

        errors: list[str] = []
        task = asyncio.create_task(_execute_search(payload, progress=queue.put_nowait, errors_out=errors))
        failure_stage = "pipeline"
        try:
            while not task.done() or not queue.empty():
                if not queue.empty():
                    event = queue.get_nowait()
                    if event.get("type") == "stage":
                        stage_event = event.get("stage", {})
                        if stage_event.get("status") == "failed":
                            failure_stage = stage_event.get("key", failure_stage)
                    yield _ndjson(event)
                    continue
                await asyncio.sleep(0.02)
            response = await task
            yield _ndjson({
                "type": "complete",
                "query": response.query,
                "results": [item.model_dump(mode="json") for item in response.results],
                "cards": [item.model_dump(mode="json") for item in response.cards],
                "source_records": [item.model_dump(mode="json") for item in response.source_records],
                "answer": response.answer,
                "answer_claims": [item.model_dump(mode="json") for item in response.answer_claims],
                "agent_runs": [item.model_dump(mode="json") for item in response.agent_runs],
                "research": response.research.model_dump(mode="json") if response.research else None,
                "search_metrics": [item.model_dump(mode="json") for item in response.search_metrics],
                "errors": errors,
            })
        except ProviderError as exc:
            yield _ndjson({"type": "error", "stage": "collect", "message": str(exc)})
            yield _ndjson({"type": "complete", "query": query, "results": [], "errors": [str(exc)]})
        except Exception:
            yield _ndjson({"type": "error", "stage": failure_stage, "message": "AI search pipeline failed."})
            yield _ndjson({"type": "complete", "query": query, "results": [], "errors": ["AI search pipeline failed."]})

    return StreamingResponse(events(), media_type="application/x-ndjson")


async def _execute_search(payload: SearchRequest, progress=None, errors_out=None) -> SearchResponse:
    query = payload.query.strip()
    settings = Settings.from_env()
    research_budget = research_budget_from_env()
    candidate_limit = min(payload.candidate_limit, research_budget.max_candidates)
    llm_client = LLMClient(LLMConfig(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        thinking_mode=settings.llm_thinking_mode,
        timeout_seconds=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
        input_usd_per_million=settings.llm_input_usd_per_million,
        output_usd_per_million=settings.llm_output_usd_per_million,
    ))
    errors: list[str] = errors_out if errors_out is not None else []
    agent_errors: list[str] = []
    agent_runs: list[AgentRunSnapshot] = []
    research_outcome: ResearchOutcome | None = None
    search_metrics = []
    stage_started: dict[str, tuple[str, float]] = {}

    def emit(key: str, status: str, **values: Any) -> None:
        if progress is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        if status == "running":
            stage_started[key] = (now, time.perf_counter())
            data = {
                "key": key, "status": status, "started_at": now,
                "completed_at": None, "duration_ms": None,
                "input_count": None, "output_count": None,
                "verified_count": None, "unverified_count": None,
                "conflict_count": None, "error": None,
            }
        else:
            started_at, started_clock = stage_started.pop(key, (now, time.perf_counter()))
            data = {
                "key": key, "status": status, "started_at": started_at,
                "completed_at": now,
                "duration_ms": max(0, int((time.perf_counter() - started_clock) * 1000)),
                "input_count": values.pop("input_count", None),
                "output_count": values.pop("output_count", None),
                "verified_count": values.pop("verified_count", None),
                "unverified_count": values.pop("unverified_count", None),
                "conflict_count": values.pop("conflict_count", None),
                "error": values.pop("error", None),
            }
        data.update(values)
        progress({"type": "stage", "stage": StageSnapshot.model_validate(data).model_dump(mode="json")})

    def emit_agent(log: Mapping[str, Any]) -> AgentRunSnapshot:
        data = dict(log)
        data["model"] = data.pop("actual_model", data.get("model"))
        data.pop("retry_count", None)
        data.pop("skip_reason", None)
        snapshot = AgentRunSnapshot.model_validate(data)
        identity = (snapshot.agent, snapshot.action)
        for index, previous in enumerate(agent_runs):
            if (previous.agent, previous.action) == identity:
                agent_runs[index] = snapshot
                break
        else:
            agent_runs.append(snapshot)
        if progress is not None:
            progress({"type": "agent", "run": snapshot.model_dump(mode="json")})
        return snapshot

    async def run_agent(agent: str, action: str, input_summary: str, operation):
        started_at = datetime.now(timezone.utc).isoformat()
        started_clock = time.perf_counter()
        emit_agent({
            "agent": agent,
            "action": action,
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "duration_ms": None,
            "model": None,
            "request_id": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "estimated_cost_usd": None,
            "input_summary": input_summary,
            "output_summary": "",
            "error_code": None,
        })
        try:
            execution = await operation
        except Exception:
            emit_agent({
                "agent": agent,
                "action": action,
                "status": "failed",
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": max(0, int((time.perf_counter() - started_clock) * 1000)),
                "input_summary": input_summary,
                "output_summary": "Agent action failed before producing a validated output.",
                "error_code": "agent_runtime_error",
            })
            raise
        log = dict(execution.run_log)
        snapshot = emit_agent(log)
        if snapshot.status == "failed":
            safe_code = snapshot.error_code or "agent_action_failed"
            message = f"{agent}/{action}: {safe_code}"
            agent_errors.append(message)
            errors.append(message)
        return execution.output

    emit("collect", "running")
    collect_error_count = len(agent_errors)
    try:
        intent = (await run_agent(
            "collect", "intent_understanding", "解析 1 条用户需求；硬性条件由规则提取。",
            IntentAgent().run(query, llm_client),
        ))
        search_providers = (
            create_search_providers(settings)
            if settings.search_providers
            else [(settings.search_provider, create_search_provider(settings))]
        )
        collection_started = time.perf_counter()
        collection = await run_agent(
            "collect", "search",
            f"准备通过 {len(search_providers)} 个已配置搜索源执行真实搜索。",
            CollectAgent().run(
            query,
            intent.get("search_queries", [query]),
            [ProviderAdapter(provider, name) for name, provider in search_providers],
            candidate_limit,
            llm_client,
        ))
        raw_results = collection.get("records", [])
        initial_collection_duration_ms = max(
            0, int((time.perf_counter() - collection_started) * 1000)
        )
        collection_errors = collection.get("errors", [])
        errors.extend(collection_errors)
    except Exception as exc:
        emit("collect", "failed", input_count=None, output_count=0, error=type(exc).__name__)
        raise
    emit(
        "collect", "partial" if len(agent_errors) > collect_error_count or collection_errors else "completed",
        input_count=len(search_providers) * max(1, len(intent.get("search_queries", [query]))),
        output_count=len(raw_results), error="; ".join(errors) or None,
    )

    emit("clean", "running")
    clean_error_count = len(agent_errors)
    try:
        cleaned = await run_agent(
            "clean", "extraction",
            f"准备规范化并去重 {len(raw_results)} 条搜索记录。",
            CleanAgent().run(
                query, raw_results, candidate_limit,
                official_source_domains=settings.official_source_domains,
                llm_client=llm_client,
            ),
        )
        # Candidate limits apply only after all source records have been
        # normalized and de-duplicated; UI page size never truncates cleaning.
        records = list(cleaned.get("records", []))[:candidate_limit]
        results = [SearchResult.model_validate(record) for record in records]
    except Exception as exc:
        emit("clean", "failed", input_count=len(raw_results), output_count=0, error=type(exc).__name__)
        raise
    initial_search_queries = intent.get("search_queries", [query])
    search_toolbox = SearchToolbox(
        search_providers=search_providers,
        max_candidates=candidate_limit,
    )
    raw_counts_by_provider: dict[str, int] = {}
    for raw in raw_results:
        provenance = raw.get("_provenance") if isinstance(raw, Mapping) else None
        name = provenance.get("provider") if isinstance(provenance, Mapping) else None
        if isinstance(name, str):
            raw_counts_by_provider[name] = raw_counts_by_provider.get(name, 0) + 1
    error_provider_names = {
        item.split(":", 1)[0].strip()
        for item in collection_errors
        if isinstance(item, str) and ":" in item
    }
    for provider_name, _provider in search_providers:
        search_toolbox.record_external_search(
            provider_name,
            requests=len(initial_search_queries),
            returned_sources=raw_counts_by_provider.get(provider_name, 0),
            failures=1 if provider_name in error_provider_names else 0,
            latency_ms=initial_collection_duration_ms,
        )
    search_metrics = search_toolbox.provider_metrics()
    search_toolbox.set_candidates(records)
    context_state: dict[str, Any] = {"records": records}
    research_reader_records: list[dict[str, Any]] = []

    if progress is not None:
        progress({
            "type": "partial_results",
            "results": [
                SearchResult.model_validate(record).model_dump(mode="json")
                for record in records
            ],
        })

    def event_from_research(event: dict[str, Any]) -> None:
        if progress is not None:
            progress(event)

    async def refresh_research_context(new_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Add real search records, deterministically normalize and dedupe them."""
        if new_records:
            captured_at = datetime.now(timezone.utc).isoformat()
            for item in new_records:
                if not isinstance(item, Mapping):
                    continue
                raw_item = dict(item)
                # The safe reader returns text; the existing cleaning pipeline
                # consumes raw_content/content. Preserve the same fetched bytes.
                page_text = raw_item.get("text")
                if isinstance(page_text, str) and page_text.strip():
                    raw_item.setdefault("raw_content", page_text)
                    raw_item.setdefault("content", page_text)
                    raw_item.setdefault("source_provider", "public_web_reader")
                    raw_item.setdefault("_provenance", {
                        "provider": "public_web_reader",
                        "captured_at": captured_at,
                    })
                    research_reader_records.append(raw_item)
                raw_results.append(raw_item)
        from .stages.clean import clean_records

        refreshed = clean_records(
            query,
            raw_results,
            candidate_limit,
            settings.official_source_domains,
        )
        old_by_url = {
            str(item.get("url", "")): item
            for item in context_state["records"]
            if isinstance(item, Mapping) and item.get("url")
        }
        merged_records: list[dict[str, Any]] = []
        for item in refreshed:
            previous = old_by_url.get(str(item.get("url", "")))
            if previous:
                combined = {**item, **previous}
                # Keep the longest freshly retrieved source text/raw payload,
                # while retaining previously extracted LLM fields.
                for key in ("raw_record", "raw_text", "source_provenance"):
                    if len(str(item.get(key, ""))) > len(str(previous.get(key, ""))):
                        combined[key] = item.get(key)
                combined["extracted_fields"] = {
                    **dict(item.get("extracted_fields", {})),
                    **dict(previous.get("extracted_fields", {})),
                }
                merged_records.append(combined)
            else:
                merged_records.append(item)
        context_state["records"] = merged_records[:candidate_limit]
        search_toolbox.set_candidates(context_state["records"])
        if progress is not None:
            progress({
                "type": "partial_results",
                "results": [
                    SearchResult.model_validate(record).model_dump(mode="json")
                    for record in context_state["records"]
                ],
            })
        gaps = _research_evidence_gaps(
            context_state["records"], intent.get("hard_conditions", [])
        )
        return {
            "candidates": _research_candidate_summaries(context_state["records"]),
            "evidence_gaps": gaps,
        }

    try:
        initial_request_count = len(search_providers) * max(1, len(initial_search_queries))
        evidence_gaps = _research_evidence_gaps(records, intent.get("hard_conditions", []))
        if not settings.llm_api_key or not settings.llm_model:
            research_outcome = ResearchOutcome(
                status="stopped", stop_reason="llm_not_configured", partial_results=bool(records)
            )
        elif initial_request_count >= research_budget.max_search_requests:
            research_outcome = ResearchOutcome(
                status="stopped",
                stop_reason="initial_search_exhausted_search_request_budget",
                partial_results=bool(records),
            )
        else:
            research_outcome, _ = await ResearchLoop(llm_client).run(
                query=query,
                initial_search_queries=list(initial_search_queries),
                candidates=_research_candidate_summaries(records),
                evidence_gaps=evidence_gaps,
                tool_dispatcher=search_toolbox.dispatch,
                # ResearchLoop reads the toolbox's cumulative provider metrics,
                # which already include the initial requests recorded above.
                # Pass the total ceiling so those requests are counted once.
                budget=research_budget,
                on_event=event_from_research,
                refresh_context=refresh_research_context,
            )
            records = context_state["records"]
            search_metrics = search_toolbox.provider_metrics()
    except Exception as exc:
        # Keep initial real search results usable, but make autonomous research
        # failure visible and sanitized rather than silently claiming success.
        safe_code = getattr(exc, "code", "research_loop_failed")
        if not isinstance(safe_code, str) or not re.fullmatch(r"[a-z0-9_]{1,64}", safe_code):
            safe_code = "research_loop_failed"
        errors.append(f"research: {safe_code}")
        research_outcome = ResearchOutcome(
            status="partial" if raw_results else "failed",
            stop_reason=safe_code,
            partial_results=bool(raw_results),
        )
        search_metrics = search_toolbox.provider_metrics()

    if research_reader_records:
        previous_clean_run = next(
            (item for item in agent_runs if item.agent == "clean" and item.action == "extraction"),
            None,
        )
        enriched_output = await run_agent(
            "clean", "extraction",
            f"对自主研究中新增的 {len(research_reader_records)} 条公开网页正文补充提取字段；每条只处理一个缺失字段。",
            CleanAgent().run(
                query,
                research_reader_records,
                research_budget.max_candidates,
                official_source_domains=settings.official_source_domains,
                llm_client=llm_client,
            ),
        )
        enriched_records = enriched_output.get("records", [])
        records = _merge_research_extractions(context_state["records"], enriched_records)
        context_state["records"] = records
        search_toolbox.set_candidates(records)

        # Keep one honest cumulative小洗 execution summary while preserving the
        # initial extraction and bounded post-research extraction usage.
        latest_clean_run = next(
            (item for item in agent_runs if item.agent == "clean" and item.action == "extraction"),
            None,
        )
        if previous_clean_run is not None and latest_clean_run is not None:
            aggregate = latest_clean_run.model_copy(update={
                "started_at": previous_clean_run.started_at,
                "duration_ms": previous_clean_run.duration_ms + latest_clean_run.duration_ms,
                "latency_ms": _sum_optional(previous_clean_run.latency_ms, latest_clean_run.latency_ms),
                "prompt_tokens": _sum_optional(previous_clean_run.prompt_tokens, latest_clean_run.prompt_tokens),
                "completion_tokens": _sum_optional(previous_clean_run.completion_tokens, latest_clean_run.completion_tokens),
                "total_tokens": _sum_optional(previous_clean_run.total_tokens, latest_clean_run.total_tokens),
                "estimated_cost_usd": _sum_optional(previous_clean_run.estimated_cost_usd, latest_clean_run.estimated_cost_usd),
                "status": (
                    "completed"
                    if previous_clean_run.status == latest_clean_run.status == "completed"
                    else "partial"
                ),
                "request_id": None,
                "input_summary": f"初始搜索来源与自主研究网页正文共处理；{len(research_reader_records)} 条正文进入补充提取。",
                "output_summary": f"初始搜索和自主研究网页正文分别经过小洗；补充尝试 {len(research_reader_records)} 条正文，字段仍须通过逐字证据校验。",
            })
            emit_agent(aggregate.model_dump(mode="json"))

    emit(
        "clean", "partial" if len(agent_errors) > clean_error_count or (research_outcome and research_outcome.status == "partial") else "completed",
        input_count=len(raw_results), output_count=len(records),
    )

    emit("verify", "running")
    verify_error_count = len(agent_errors)
    verify_error_offset = len(errors)
    try:
        results = _rank_results(query, results, raw_results, domain=intent["domain"])
        candidates = [
            result for result in results
            if result.procurement_category != "unclear"
            and result.match_status in {"matched", "partial", "keyword_only"}
        ][:4]
        procurement_verifications: dict[str, dict[str, Any]] = {}

        async def search_all_sources(search_query, limit, include_domains=()):
            async def call(name, configured_provider):
                try:
                    return await configured_provider.search(
                        search_query, limit, include_domains=include_domains
                    )
                except TypeError:
                    return await configured_provider.search(search_query, limit)

            source_results = await asyncio.gather(
                *(call(name, configured_provider) for name, configured_provider in search_providers),
                return_exceptions=True,
            )
            combined: list[Mapping[str, Any]] = []
            failures = []
            for (name, _), source_result in zip(search_providers, source_results):
                if isinstance(source_result, BaseException):
                    failures.append(f"{name}: {type(source_result).__name__}")
                else:
                    combined.extend(item for item in source_result if isinstance(item, Mapping))
            if failures:
                errors.extend(failures)
            if not combined and failures:
                raise ProviderError("All configured search adapters failed.")
            return combined

        async def official_search(
            search_query: str, include_domains: Sequence[str], limit: int
        ) -> Sequence[Mapping[str, Any]]:
            match = re.match(r'"([^\"]+)"', search_query)
            project_name = match.group(1) if match else ""
            normalized_project_name = re.sub(r"\s+", "", project_name).casefold()
            if normalized_project_name:
                for record in raw_results:
                    url = record.get("url")
                    if not isinstance(url, str):
                        continue
                    host = (urlparse(url).hostname or "").lower().rstrip(".")
                    if not any(host == domain or host.endswith("." + domain) for domain in include_domains):
                        continue
                    source_text = " ".join(
                        value for value in (record.get("title"), record.get("raw_content"), record.get("content"))
                        if isinstance(value, str)
                    )
                    if normalized_project_name in re.sub(r"\s+", "", source_text).casefold():
                        return [record]
            return await search_all_sources(search_query, limit, include_domains)

        async def verify_procurement_candidate(result):
            source_record = next((record for record in raw_results if record.get("url") == result.url), {})
            source_text = source_record.get("raw_content") or source_record.get("content")
            claim = _supporting_phrase(
                result.procurement_category, result.title, result.evidence, result.snippet,
                source_text if isinstance(source_text, str) else "",
            )
            try:
                verification = await verify_official_source(
                    project_name=_project_name_for_verification(result.title),
                    support_claim=claim,
                    search=official_search,
                    official_domains=settings.official_source_domains,
                    limit=min(10, candidate_limit),
                )
            except Exception as exc:
                errors.append(f"official verification {result.id}: {type(exc).__name__}")
                verification = {
                    "official_source_status": "insufficient_evidence",
                    "official_source_url": None, "verified_fields": [],
                    "unverified_fields": ["project_name", "purchasing_unit", "published_at", "budget", "deadline", "evidence"],
                    "purchasing_unit": None, "budget": None, "deadline_at": None,
                    "published_at": None, "deadline_status": "unverified", "evidence": "",
                    "field_evidence": {},
                }
            procurement_verifications[result.url] = verification
            return _apply_verification(result, verification)

        if candidates:
            verified_candidates = await asyncio.gather(*(verify_procurement_candidate(item) for item in candidates))
            by_id = {result.id: result for result in verified_candidates}
            results = [by_id.get(result.id, result) for result in results]

        official_by_url = dict(procurement_verifications)
        if intent["domain"] == "employment":
            # Search only candidates likely to represent a distinct role and
            # verify employer ownership plus each relevant field independently.
            employment_candidates = [
                record for record in records
                if record.get("extracted_fields", {}).get("graduate_cohort")
                or record.get("extracted_fields", {}).get("majors")
            ][:4]

            async def employment_search(search_query, _domains, limit):
                try:
                    return await search_all_sources(search_query, limit)
                except Exception as exc:
                    errors.append(f"employment official lookup: {type(exc).__name__}")
                    return []

            official_by_url.update(await find_official_employment_evidence(
                employment_candidates, employment_search, limit=min(5, candidate_limit),
                required_conditions=intent["hard_conditions"],
            ))

        verified_records = verify_records(
            records, official_by_url, required_conditions=intent["hard_conditions"]
        )
        result_by_id = {result.id: result for result in results}
        for record in verified_records:
            result = result_by_id.get(record["id"])
            if result is not None:
                record.update(result.model_dump(mode="python"))
                # Preserve field-level checks created above.
                record["field_verification"] = next(
                    item["field_verification"] for item in verified_records if item["id"] == record["id"]
                )
                official = official_by_url.get(str(record.get("url", "")), {})
                if official.get("official_source_url"):
                    record["official_source_url"] = official["official_source_url"]
                    record["official_source_status"] = "verified"
                field_checks = record.get("field_verification", {})
                if isinstance(field_checks, Mapping):
                    fields = {
                        field for field, check in field_checks.items()
                        if isinstance(check, Mapping) and check.get("status") == "verified"
                    }
                    conflicts = {
                        field for field, check in field_checks.items()
                        if isinstance(check, Mapping) and check.get("status") == "conflict"
                    }
                    record["verified_fields"] = list(fields)
                    record["unverified_fields"] = list(
                        (set(record.get("unverified_fields", [])) | conflicts) - fields
                    )

        verification = await run_agent(
            "verify", "source_verification",
            f"准备逐项检查 {len(verified_records)} 条记录的官方来源和字段证据。",
            VerifyAgent().run(
                verified_records,
                required_conditions=intent["hard_conditions"],
                llm_client=llm_client,
            ),
        )
        verified_records = verification.get("records", verified_records)

        verification_stats = verification_counts(verified_records)
    except Exception as exc:
        emit("verify", "failed", input_count=len(records), output_count=0, error=type(exc).__name__)
        raise
    emit(
        "verify", "partial" if len(agent_errors) > verify_error_count or len(errors) > verify_error_offset else "completed",
        input_count=len(records), output_count=len(verified_records),
        verified_count=verification_stats["verified"],
        unverified_count=verification_stats["unverified"],
        conflict_count=verification_stats["conflict"],
        error="; ".join(errors[verify_error_offset:]) or None,
    )

    emit("judge", "running")
    judge_error_count = len(agent_errors)
    try:
        judgement = await run_agent(
            "judge", "semantic_match",
            f"准备对 {len(verified_records)} 条已核验记录进行规则裁决和语义分析。",
            JudgeAgent().run(query, intent, verified_records, llm_client),
        )
        judged = judgement.get("records", [])
        final_results = [
            _with_result_kind(SearchResult.model_validate(item)) for item in judged
        ]
        answer_output = await run_agent(
            "judge", "final_answer",
            f"准备基于 {len(final_results)} 条最终规则结果生成有证据引用的结论。",
            AnswerAgent().run(query, intent, judged, llm_client),
        )
        answer = answer_output.get("answer")
        answer = answer if isinstance(answer, str) else ""
        answer_claims = [
            AnswerClaim.model_validate(claim)
            for claim in answer_output.get("claims", [])
            if isinstance(claim, Mapping)
        ]
        cards = build_result_cards(query, judged)
        source_records = build_source_records(raw_results)
    except Exception as exc:
        emit("judge", "failed", input_count=len(verified_records), output_count=0, error=type(exc).__name__)
        raise
    emit(
        "judge", "partial" if len(agent_errors) > judge_error_count else "completed",
        input_count=len(verified_records), output_count=len(final_results),
    )
    return SearchResponse(
        query=query,
        results=final_results,
        cards=cards,
        source_records=source_records,
        answer=answer,
        answer_claims=answer_claims,
        agent_runs=agent_runs,
        errors=errors,
        research=research_outcome,
        search_metrics=search_metrics,
    )


def _ndjson(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"

def _rank_results(query: str, results, raw_results, *, domain: str = "procurement"):
    # The legacy category ranker is intentionally procurement-specific. For
    # other domains, leave facts untouched; the generic rule judge assigns the
    # match status after field-level verification below.
    if domain != "procurement":
        return results
    source_by_url = {
        record.get("url"): record
        for record in raw_results
        if isinstance(record, Mapping) and isinstance(record.get("url"), str)
    }
    candidates = []
    for result in results:
        data = result.model_dump(mode="python")
        source = source_by_url.get(result.url, {})
        source_text = source.get("raw_content") or source.get("content")
        if isinstance(source_text, str) and source_text.strip():
            data["content"] = source_text
        candidates.append(data)
    ranked = rank_and_explain_results(query, candidates)
    return [type(results[0]).model_validate(item) for item in ranked] if results else []


def _supporting_phrase(
    category: str,
    title: str,
    evidence: str,
    snippet: str,
    source_text: str = "",
) -> str:
    """Use a literal candidate phrase as the claim to check against the notice."""
    text = f"{title} {evidence} {snippet} {source_text}"
    cues = {
        "ai_training": (
            "人工智能编程培训", "人工智能拔尖创新人才培养", "人工智能培训",
            "教师素养提升服务", "AI培训", "大模型培训",
        ),
        "ai_system": (
            "人工智能应用系统", "人工智能系统", "AI系统", "人工智能平台",
            "大模型平台", "AI平台",
        ),
        "digital_transformation": (
            "数字化改造", "数字化转型", "信息化改造", "数字化建设",
        ),
        "compute_hardware": (
            "GPU服务器", "GPU集群", "算力设备", "计算集群", "高性能计算",
        ),
    }.get(category, ())
    # A result title is not evidence for the user's requested conclusion.
    # Keep the claim empty when no literal category phrase exists in the
    # retrieved text; official notice metadata can still be checked separately.
    return next((cue for cue in cues if cue.casefold() in text.casefold()), "")


def _project_name_for_verification(title: str) -> str:
    """Strip common publisher/category suffixes from a search result title."""
    project_name = title.split(" _ ", 1)[0].strip()
    for suffix in (
        "单一来源采购公告",
        "单一来源公告",
        "竞争性磋商公告",
        "竞争性谈判公告",
        "公开招标公告",
        "询价公告",
        "招标公告",
        "采购公告",
        "采购结果公告",
        "中标公告",
        "公告",
    ):
        if project_name.endswith(suffix) and len(project_name) > len(suffix):
            project_name = project_name[: -len(suffix)].rstrip(" -_｜|：:")
            break
    return re.sub(r"\s+", " ", project_name).strip()


def _merge_research_extractions(
    records: Sequence[Mapping[str, Any]],
    extractions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge newly extracted fields without overwriting existing facts or evidence."""
    by_url: dict[str, Mapping[str, Any]] = {}
    for extraction in extractions:
        url = extraction.get("url")
        if isinstance(url, str):
            canonical = canonicalize_url(url)
            if canonical:
                by_url[canonical] = extraction

    merged: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        url = item.get("url")
        canonical = canonicalize_url(url) if isinstance(url, str) else ""
        extraction = by_url.get(canonical)
        if extraction is None:
            merged.append(item)
            continue

        existing_fields = item.get("extracted_fields")
        existing_fields = dict(existing_fields) if isinstance(existing_fields, Mapping) else {}
        added_fields = extraction.get("extracted_fields")
        if isinstance(added_fields, Mapping):
            for field, value in added_fields.items():
                existing_fields.setdefault(str(field), value)
        item["extracted_fields"] = existing_fields

        existing_sources = item.get("field_sources")
        existing_sources = dict(existing_sources) if isinstance(existing_sources, Mapping) else {}
        added_sources = extraction.get("field_sources")
        if isinstance(added_sources, Mapping):
            for field, value in added_sources.items():
                existing_sources.setdefault(str(field), value)
        item["field_sources"] = existing_sources

        extracted_text = extraction.get("raw_text")
        if isinstance(extracted_text, str) and len(extracted_text) > len(str(item.get("raw_text", ""))):
            item["raw_text"] = extracted_text
        merged.append(item)
    return merged


def _sum_optional(left: int | float | None, right: int | float | None):
    if left is None or right is None:
        return None
    return left + right


def _apply_verification(result, verification: dict[str, Any]):
    data = result.model_dump(mode="python")
    status = verification.get("official_source_status")
    if status not in {"verified", "third_party", "insufficient_evidence"}:
        status = "insufficient_evidence"
    data["official_source_status"] = status
    data["official_source_url"] = (
        verification.get("official_source_url") if status == "verified" else None
    )
    aliases = {"deadline_at": "deadline", "support_claim": "evidence"}

    def normalize_fields(values: Any) -> list[str]:
        if not isinstance(values, (list, tuple, set)):
            return []
        allowed = {
            "project_name", "purchasing_unit", "published_at", "budget", "deadline", "evidence", "region"
        }
        normalized = []
        for value in values:
            field = aliases.get(value, value) if isinstance(value, str) else None
            if field in allowed and field not in normalized:
                normalized.append(field)
        return normalized

    data["verified_fields"] = normalize_fields(verification.get("verified_fields"))
    data["unverified_fields"] = normalize_fields(verification.get("unverified_fields"))

    for field in ("purchasing_unit", "budget", "deadline_at", "published_at"):
        data[field] = verification.get(field) if status == "verified" else None
    if status == "verified":
        data["deadline_status"] = verification.get("deadline_status", "unverified")
    else:
        data["deadline_status"] = "unverified"

    quote_value = verification.get("evidence")
    if isinstance(quote_value, str):
        quote = quote_value.strip()
    elif isinstance(quote_value, list):
        quotes = [
            item.get("quote", "").strip()
            for item in quote_value
            if isinstance(item, dict)
            and item.get("field") in {"evidence", "support_claim", "project_name"}
            and isinstance(item.get("quote"), str)
            and item.get("quote", "").strip()
        ]
        quote = "\n".join(dict.fromkeys(quotes))
    else:
        quote = ""
    if quote:
        data["evidence"] = quote
        data["evidence_status"] = "available"
    elif status == "verified":
        data["evidence"] = ""
        data["evidence_status"] = "missing"
    data["match_score"] = data.get("match_score", 0.0)
    return type(result).model_validate(data)


def _with_result_kind(result: SearchResult) -> SearchResult:
    """Classify only when a source-backed role/project or official lead exists."""
    verified = set(result.verified_fields)
    source_text = " ".join((result.title, result.url, result.official_source_url or "")).casefold()
    if result.official_source_status == "verified" and (
        ("role" in verified and result.application_url)
        or ("project_name" in verified and result.official_source_url)
    ):
        kind = "specific_opportunity"
    elif result.official_source_url and any(
        marker in source_text
        for marker in ("招聘", "校园招聘", "校招", "careers", "career", "招聘门户")
    ):
        kind = "organization_lead"
    else:
        kind = "unknown"
    return result.model_copy(update={"result_kind": kind})


def _research_candidate_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep model context compact while retaining URLs and literal field quotes."""
    summaries: list[dict[str, Any]] = []
    for record in records[:30]:
        fields = record.get("extracted_fields", {})
        clean_fields: dict[str, dict[str, str]] = {}
        if isinstance(fields, Mapping):
            for name, value in list(fields.items())[:12]:
                if not isinstance(name, str) or not isinstance(value, Mapping):
                    continue
                field_value, quote = value.get("value"), value.get("quote")
                clean_fields[name] = {
                    "value": str(field_value or "")[:240],
                    "quote": str(quote or "")[:360],
                }
        raw_text = record.get("raw_text", "")
        summaries.append({
            "id": str(record.get("id", ""))[:80],
            "title": str(record.get("title", ""))[:300],
            "url": str(record.get("url", ""))[:2048],
            "official_source_url": str(record.get("official_source_url") or "")[:2048],
            "official_source_status": str(record.get("official_source_status", "unverified")),
            "extracted_fields": clean_fields,
            "evidence_excerpt": str(raw_text or record.get("evidence", ""))[:1200],
        })
    return summaries


def _research_evidence_gaps(
    records: Sequence[Mapping[str, Any]], hard_conditions: Sequence[Mapping[str, Any]]
) -> list[str]:
    gaps: list[str] = []
    requested_fields = [
        item.get("field") for item in hard_conditions
        if isinstance(item, Mapping) and isinstance(item.get("field"), str)
    ]
    aliases = {"major": "majors", "organization": "organization"}
    for record in records[:20]:
        title = str(record.get("title", "候选记录"))[:100]
        extracted = record.get("extracted_fields", {})
        extracted = extracted if isinstance(extracted, Mapping) else {}
        if not record.get("official_source_url"):
            gaps.append(f"{title}: official original source not located")
        for field in requested_fields[:5]:
            key = aliases.get(field, field)
            if key not in extracted:
                gaps.append(f"{title}: missing evidence for {field}")
        if len(gaps) >= 16:
            break
    if not records:
        gaps.append("No candidate records have been collected yet")
    return list(dict.fromkeys(gaps))[:16]
