from __future__ import annotations

import logging
import os
import inspect
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .engine_client import EngineClientError, EngineStreamError, HttpEngineClient
from .repository import SQLAlchemyTaskRepository, TaskRepository, task_to_values
from .result_cards import legacy_results_to_cards, rank_cards
from .schemas import (
    AgentRun,
    EngineSearchResponse,
    EvidenceSource,
    ResultCard,
    ResearchRound,
    SearchAccepted,
    SearchRequest,
    SearchResult,
    SearchTaskResponse,
    StageProgress,
    ToolCallRecord,
)

logger = logging.getLogger(__name__)

_SAFE_ENGINE_ERRORS = {
    "AI engine is unavailable.",
    "AI engine search timed out.",
    "AI engine could not complete the search.",
    "AI engine returned an invalid progress event.",
    "AI engine returned an invalid stage update.",
    "AI engine returned an invalid agent update.",
    "AI engine returned an invalid search response.",
    "AI engine returned an unknown progress event.",
    "AI engine stream ended before search results were returned.",
}


def _safe_engine_error(error: EngineClientError) -> str:
    if isinstance(error, EngineStreamError):
        return "AI engine stage failed."
    message = str(error)
    return message if message in _SAFE_ENGINE_ERRORS else "AI engine could not complete the search."


def _mark_failed_stages(
    task_repository: TaskRepository, task_id: str, failed_key: str | None
) -> None:
    keys = ("collect", "clean", "verify", "judge")
    row = task_repository.get(task_id)
    recorded = {
        item.get("key"): item
        for item in (row.stages or [])
        if row and isinstance(item, dict)
    } if row else {}
    if failed_key not in keys:
        failed_key = next(
            (
                key
                for key in keys
                if recorded.get(key, {}).get("status") == "running"
            ),
            "collect",
        )
    failed_index = keys.index(failed_key)
    for index, key in enumerate(keys):
        previous = recorded.get(key)
        previous_stage = StageProgress.model_validate(previous) if previous else None
        if index == failed_index:
            stage = (previous_stage or StageProgress(key=key)).model_copy(
                update={"status": "failed", "error": "AI engine stage failed."}
            )
        elif index > failed_index and (
            previous_stage is None or previous_stage.status == "pending"
        ):
            stage = StageProgress(key=key, status="skipped")
        elif previous_stage is not None:
            continue
        else:
            continue
        task_repository.update_stage(task_id, stage)


def _verified_card_sources(card: ResultCard) -> list[EvidenceSource]:
    """Expose only sources whose source record or field evidence is verified."""
    sources: list[EvidenceSource] = []
    seen: set[str] = set()
    if card.source_url:
        seen.add(card.source_url)
        sources.append(
            EvidenceSource(
                title=card.source_name or card.source_url,
                url=card.source_url,
                source_type="official",
                status="verified",
            )
        )
    for source in card.additional_sources:
        if source.status != "verified" or source.url in seen:
            continue
        seen.add(source.url)
        sources.append(
            EvidenceSource(
                title=source.name,
                url=source.url,
                source_type=source.type,
                status="verified",
            )
        )
    for attribute in card.attributes:
        if (
            attribute.status != "verified"
            or not attribute.evidence_url
            or attribute.evidence_url in seen
        ):
            continue
        seen.add(attribute.evidence_url)
        sources.append(
            EvidenceSource(
                title=attribute.evidence_url,
                url=attribute.evidence_url,
                source_type="unknown",
                status="verified",
            )
        )
    return sources


def create_app(
    *,
    repository: TaskRepository | None = None,
    engine_client: HttpEngineClient | None = None,
    database_url: str | None = None,
) -> FastAPI:
    """Build the API app with injectable persistence and engine boundaries."""
    task_repository = repository or SQLAlchemyTaskRepository(
        database_url
        or os.getenv("DATABASE_URL", "sqlite:///./zhize-api-local.db")
    )
    search_engine = engine_client or HttpEngineClient(
        os.getenv("AI_ENGINE_URL", "http://127.0.0.1:8001"),
        timeout_seconds=float(os.getenv("AI_ENGINE_TIMEOUT_SECONDS", "210")),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task_repository.ensure_schema()
        yield

    app = FastAPI(title="知择 API", version="0.9.0", lifespan=lifespan)
    origins = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "zhize-api"}

    async def execute_search(task_id: str, query: str, candidate_limit: int) -> None:
        task_repository.set_status(task_id, "running")
        try:
            if hasattr(search_engine, "search_stream"):
                async def persist_stage(stage: StageProgress) -> None:
                    if stage.error:
                        stage = stage.model_copy(
                            update={"error": "Stage reported an error."}
                        )
                    task_repository.update_stage(task_id, stage)

                async def persist_agent(run: AgentRun) -> None:
                    task_repository.update_agent_run(task_id, run)

                async def persist_research_round(research_round: ResearchRound) -> None:
                    task_repository.update_research_round(task_id, research_round)

                async def persist_tool_call(tool_call: ToolCallRecord) -> None:
                    task_repository.update_tool_call(task_id, tool_call)

                async def persist_partial_results(
                    results: list[SearchResult | ResultCard]
                ) -> None:
                    task_repository.update_partial_results(task_id, results)

                stream_method = search_engine.search_stream
                # Keep older injected clients working while the streaming
                # boundary gains agent-run snapshots.
                stream_parameters = inspect.signature(stream_method).parameters
                if (
                    "on_research_round" in stream_parameters
                    or "on_tool_call" in stream_parameters
                    or "on_partial_results" in stream_parameters
                ):
                    stream_callbacks = {}
                    if "on_agent" in stream_parameters:
                        stream_callbacks["on_agent"] = persist_agent
                    if "on_research_round" in stream_parameters:
                        stream_callbacks["on_research_round"] = persist_research_round
                    if "on_tool_call" in stream_parameters:
                        stream_callbacks["on_tool_call"] = persist_tool_call
                    if "on_partial_results" in stream_parameters:
                        stream_callbacks["on_partial_results"] = persist_partial_results
                    response = await stream_method(
                        query, candidate_limit, persist_stage, **stream_callbacks
                    )
                elif "on_agent" in stream_parameters:
                    response = await stream_method(
                        query, candidate_limit, persist_stage, on_agent=persist_agent
                    )
                else:
                    response = await stream_method(query, candidate_limit, persist_stage)
                # Upstream error text is intentionally not echoed or persisted:
                # it could include credentials, URLs with secrets, or user data.
                safe_errors = (
                    ["AI engine returned partial results."] if response.errors else []
                )
                cards = response.cards or legacy_results_to_cards(
                    response.results, query=query
                )
                task_repository.complete(
                    task_id,
                    rank_cards(cards),
                    safe_errors,
                    response.agent_runs,
                    response.answer,
                    response.answer_claims,
                    response.research,
                    response.search_metrics
                    or (response.research.providers if response.research else []),
                    source_records=response.source_records
                    or response.results
                    or response.cards,
                )
            else:
                # Compatibility for existing integrations that implement the
                # original client boundary while callers move to NDJSON.
                search_response = await search_engine.search(query, candidate_limit)
                if isinstance(search_response, EngineSearchResponse):
                    results = search_response.cards or legacy_results_to_cards(
                        search_response.results, query=query
                    )
                    raw_source_records = (
                        search_response.source_records
                        or search_response.results
                        or search_response.cards
                    )
                else:
                    results = [
                        result for result in search_response
                        if isinstance(result, ResultCard)
                    ]
                    legacy_results = [
                        result for result in search_response
                        if isinstance(result, SearchResult)
                    ]
                    if legacy_results:
                        results.extend(
                            legacy_results_to_cards(legacy_results, query=query)
                        )
                    raw_source_records = search_response
                # A legacy one-shot response did not run the four-stage
                # pipeline, so expose those stages as skipped instead of
                # presenting stale pending states as real progress.
                for key in ("collect", "clean", "verify", "judge"):
                    task_repository.update_stage(
                        task_id, StageProgress(key=key, status="skipped")
                    )
                task_repository.complete(
                    task_id,
                    rank_cards(results),
                    source_records=raw_source_records,
                )
        except EngineStreamError as exc:
            _mark_failed_stages(task_repository, task_id, exc.stage)
            task_repository.fail(task_id, _safe_engine_error(exc))
        except EngineClientError as exc:
            _mark_failed_stages(task_repository, task_id, None)
            task_repository.fail(task_id, _safe_engine_error(exc))
        except Exception:
            logger.error(
                "Unexpected search task error (task_id=%s)", task_id
            )
            task_repository.fail(task_id, "Search processing failed.")

    @app.post(
        "/api/v1/search",
        status_code=202,
        response_model=SearchAccepted,
    )
    async def create_search(
        payload: SearchRequest, background_tasks: BackgroundTasks
    ) -> SearchAccepted:
        task_id = str(uuid4())
        task_repository.create(task_id, payload.query)
        background_tasks.add_task(
            execute_search, task_id, payload.query, payload.candidate_limit
        )
        return SearchAccepted(task_id=task_id, status="queued")

    @app.get(
        "/api/v1/search/{task_id}",
        response_model=SearchTaskResponse,
    )
    async def get_search(
        task_id: str,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=10, ge=1, le=100),
    ) -> SearchTaskResponse:
        row = task_repository.get(task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Search task not found.")
        values = task_to_values(row)
        all_results = values["results"]
        assert isinstance(all_results, list)
        verified_sources_by_id = {
            result.id: _verified_card_sources(result)
            for result in all_results
        }
        answer_claims = values["answer_claims"]
        assert isinstance(answer_claims, list)
        values["answer_claims"] = [
            claim.model_copy(
                update={
                    "sources": verified_sources_by_id.get(claim.result_id, [])
                }
            )
            for claim in answer_claims
        ]
        start = (page - 1) * page_size
        values["total_results"] = len(all_results)
        values["page"] = page
        values["page_size"] = page_size
        values["results"] = all_results[start : start + page_size]
        values.pop("source_records", None)
        return SearchTaskResponse.model_validate(values)

    return app


app = create_app()
