from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

import httpx
from pydantic import ValidationError

from .schemas import (
    AgentRun,
    EngineSearchResponse,
    ResearchOutcome,
    ResearchRound,
    ResultCard,
    SearchProviderMetric,
    SearchResult,
    StageKey,
    StageProgress,
    ToolCallRecord,
)


_MAX_PARTIAL_RESULTS = 100


class EngineClientError(Exception):
    """A sanitized upstream AI-engine error safe to store on a search task."""


class EngineStreamError(EngineClientError):
    def __init__(
        self,
        message: str,
        stage: StageKey | None = None,
        agent: StageKey | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.agent = agent


StageCallback = Callable[[StageProgress], Awaitable[None]]
AgentCallback = Callable[[AgentRun], Awaitable[None]]
ResearchRoundCallback = Callable[[ResearchRound], Awaitable[None]]
ToolCallCallback = Callable[[ToolCallRecord], Awaitable[None]]
PartialResultsCallback = Callable[[list[SearchResult | ResultCard]], Awaitable[None]]

_SECRET_VALUE = re.compile(
    r"(?i)(api[_-]?key|authorization|access[_-]?token|secret|password)\s*[:=]\s*[^\s,;]+"
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_API_KEY_SHAPE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_SAFE_ERROR_CODE = re.compile(r"^[a-z0-9_]{1,80}$")


def _safe_summary(value: str | None) -> str | None:
    if value is None:
        return None
    safe = _SECRET_VALUE.sub(r"\1=[redacted]", value)
    safe = _BEARER_VALUE.sub("Bearer [redacted]", safe)
    safe = _API_KEY_SHAPE.sub("[redacted]", safe)
    return safe[:500]


def _safe_agent_run(run: AgentRun) -> AgentRun:
    """Bound and redact streamed summaries before storing or returning them."""
    return run.model_copy(
        update={
            "model": _safe_summary(run.model)[:120] if run.model else None,
            "request_id": _safe_summary(run.request_id)[:180] if run.request_id else None,
            "input_summary": _safe_summary(run.input_summary),
            "output_summary": _safe_summary(run.output_summary),
            "error_code": (
                run.error_code
                if run.error_code is None or _SAFE_ERROR_CODE.fullmatch(run.error_code)
                else "agent_failed"
            ),
        }
    )


def _safe_code(value: str | None) -> str | None:
    if value is None or _SAFE_ERROR_CODE.fullmatch(value):
        return value
    return "tool_failed"


def _safe_tool_call(call: ToolCallRecord) -> ToolCallRecord:
    return call.model_copy(
        update={
            "input_summary": _safe_summary(call.input_summary) or "",
            "output_summary": _safe_summary(call.output_summary) or "",
            "provider": _safe_summary(call.provider)[:80] if call.provider else None,
            "model": _safe_summary(call.model)[:120] if call.model else None,
            "request_id": _safe_summary(call.request_id)[:180] if call.request_id else None,
            "error_code": _safe_code(call.error_code),
        }
    )


def _safe_research_round(item: ResearchRound) -> ResearchRound:
    return item.model_copy(
        update={
            "reason": _safe_summary(item.reason) or "",
            "planned_queries": [_safe_summary(query) or "" for query in item.planned_queries],
            "evidence_gaps": [_safe_summary(gap) or "" for gap in item.evidence_gaps],
        }
    )


def _safe_research_outcome(value: ResearchOutcome | None) -> ResearchOutcome | None:
    if value is None:
        return None
    return value.model_copy(
        update={
            "rounds": [_safe_research_round(item) for item in value.rounds],
            "tool_calls": [_safe_tool_call(item) for item in value.tool_calls],
            "providers": [_safe_search_metric(item) for item in value.providers],
            "stop_reason": _safe_summary(value.stop_reason) or "",
        }
    )


def _safe_search_metric(value: SearchProviderMetric) -> SearchProviderMetric:
    return value.model_copy(
        update={
            "provider": (_safe_summary(value.provider) or "unknown")[:80],
            "error_code": _safe_code(value.error_code),
        }
    )


class HttpEngineClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 90.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def search_stream(
        self,
        query: str,
        candidate_limit: int,
        on_stage: StageCallback,
        on_agent: AgentCallback | None = None,
        on_research_round: ResearchRoundCallback | None = None,
        on_tool_call: ToolCallCallback | None = None,
        on_partial_results: PartialResultsCallback | None = None,
    ) -> EngineSearchResponse:
        """Consume the AI-engine NDJSON stream and persist each progress event."""
        latest_stage: StageKey | None = None
        latest_agent: StageKey | None = None
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/v1/search/stream",
                    json={"query": query, "candidate_limit": candidate_limit},
                ) as response:
                    if response.is_error:
                        raise EngineClientError("AI engine could not complete the search.")
                    complete: EngineSearchResponse | None = None
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except (ValueError, TypeError) as exc:
                            raise EngineClientError(
                                "AI engine returned an invalid progress event."
                            ) from exc
                        if not isinstance(event, dict):
                            raise EngineClientError(
                                "AI engine returned an invalid progress event."
                            )
                        event_type = event.get("type")
                        if event_type == "stage":
                            try:
                                stage = StageProgress.model_validate(event.get("stage"))
                            except (ValidationError, TypeError) as exc:
                                raise EngineClientError(
                                    "AI engine returned an invalid stage update."
                                ) from exc
                            if stage.error:
                                stage = stage.model_copy(update={"error": "Stage reported an error."})
                            latest_stage = stage.key
                            await on_stage(stage)
                        elif event_type == "complete":
                            try:
                                complete = EngineSearchResponse.model_validate(event)
                            except (ValidationError, TypeError) as exc:
                                raise EngineClientError(
                                    "AI engine returned an invalid search response."
                                ) from exc
                            complete.agent_runs = [
                                _safe_agent_run(run) for run in complete.agent_runs
                            ]
                            complete.research = _safe_research_outcome(complete.research)
                            complete.search_metrics = [
                                _safe_search_metric(item) for item in complete.search_metrics
                            ]
                        elif event_type == "agent":
                            try:
                                run = _safe_agent_run(
                                    AgentRun.model_validate(event.get("run"))
                                )
                            except (ValidationError, TypeError) as exc:
                                raise EngineClientError(
                                    "AI engine returned an invalid agent update."
                                ) from exc
                            latest_agent = run.agent
                            if on_agent is not None:
                                await on_agent(run)
                        elif event_type == "research_round":
                            try:
                                research_round = _safe_research_round(
                                    ResearchRound.model_validate(event.get("round"))
                                )
                            except (ValidationError, TypeError) as exc:
                                raise EngineClientError(
                                    "AI engine returned an invalid research-round update."
                                ) from exc
                            if on_research_round is not None:
                                await on_research_round(research_round)
                        elif event_type == "tool_call":
                            try:
                                tool_call = _safe_tool_call(
                                    ToolCallRecord.model_validate(event.get("call"))
                                )
                            except (ValidationError, TypeError) as exc:
                                raise EngineClientError(
                                    "AI engine returned an invalid tool-call update."
                                ) from exc
                            if on_tool_call is not None:
                                await on_tool_call(tool_call)
                        elif event_type == "partial_results":
                            raw_results = event.get("results")
                            raw_cards = event.get("cards")
                            if isinstance(raw_cards, list):
                                if len(raw_cards) > _MAX_PARTIAL_RESULTS:
                                    raise EngineClientError(
                                        "AI engine returned an invalid partial result update."
                                    )
                                try:
                                    cards = [
                                        ResultCard.model_validate(item).model_copy(
                                            update={"matching_status": "needs_verification"}
                                        )
                                        for item in raw_cards
                                    ]
                                except (ValidationError, TypeError) as exc:
                                    raise EngineClientError(
                                        "AI engine returned an invalid partial result update."
                                    ) from exc
                                if on_partial_results is not None:
                                    await on_partial_results(cards)
                                continue
                            if (
                                not isinstance(raw_results, list)
                                or len(raw_results) > _MAX_PARTIAL_RESULTS
                            ):
                                raise EngineClientError(
                                    "AI engine returned an invalid partial result update."
                                )
                            try:
                                # Early candidates are useful for progressive display, but
                                # they have not passed the final evidence/match decision.
                                candidates = [
                                    SearchResult.model_validate(item).model_copy(
                                        update={"decision": "needs_verification"}
                                    )
                                    for item in raw_results
                                ]
                            except (ValidationError, TypeError) as exc:
                                raise EngineClientError(
                                    "AI engine returned an invalid partial result update."
                                ) from exc
                            if on_partial_results is not None:
                                await on_partial_results(candidates)
                        elif event_type == "error":
                            stage_key = event.get("stage")
                            stage = stage_key if stage_key in {
                                "collect", "clean", "verify", "judge"
                            } else None
                            agent_key = event.get("agent")
                            agent = agent_key if agent_key in {
                                "collect", "clean", "verify", "judge"
                            } else latest_agent
                            raise EngineStreamError(
                                "AI engine stage failed.", stage=stage, agent=agent
                            )
                        else:
                            raise EngineClientError(
                                "AI engine returned an unknown progress event."
                            )
                    if complete is None:
                        raise EngineClientError(
                            "AI engine stream ended before search results were returned."
                        )
                    return complete
        except EngineClientError:
            raise
        except httpx.RequestError as exc:
            if isinstance(exc, httpx.TimeoutException):
                raise EngineStreamError(
                    "AI engine search timed out.", latest_stage, latest_agent
                ) from exc
            raise EngineStreamError(
                "AI engine is unavailable.", latest_stage, latest_agent
            ) from exc

    async def search(self, query: str, candidate_limit: int) -> EngineSearchResponse:
        """Legacy non-streaming endpoint kept for existing internal callers."""
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/search",
                    json={"query": query, "candidate_limit": candidate_limit},
                )
        except httpx.TimeoutException as exc:
            raise EngineClientError("AI engine search timed out.") from exc
        except httpx.RequestError as exc:
            raise EngineClientError("AI engine is unavailable.") from exc

        if response.is_error:
            raise EngineClientError("AI engine could not complete the search.")

        try:
            payload = EngineSearchResponse.model_validate(response.json())
        except (ValueError, TypeError, ValidationError) as exc:
            raise EngineClientError("AI engine returned an invalid search response.") from exc
        return payload
