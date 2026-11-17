import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from services.api.app.engine_client import HttpEngineClient
from services.api.app.main import create_app
from services.api.app.repository import SQLAlchemyTaskRepository
from services.api.app.schemas import (
    EngineSearchResponse,
    ResearchOutcome,
    ResearchRound,
    SearchProviderMetric,
    SearchResult,
    ToolCallRecord,
)


def research_fixture() -> ResearchOutcome:
    return ResearchOutcome(
        status="completed",
        rounds=[
            ResearchRound(
                round_number=1,
                status="completed",
                reason="Still missing an official role page.",
                planned_queries=["site:careers.example.org graduate hiring"],
                search_requests=2,
                sources_found=5,
                new_sources=3,
                candidates_before=0,
                candidates_after=2,
                evidence_gaps=["application deadline"],
                duration_ms=1200,
            )
        ],
        tool_calls=[
            ToolCallRecord(
                round_number=1,
                name="search_official",
                status="completed",
                input_summary="Search the employer career site.",
                output_summary="Found two role pages.",
                provider="tavily",
                source_count=2,
                duration_ms=800,
            )
        ],
        providers=[
            SearchProviderMetric(
                provider="tavily",
                configured=True,
                requests=2,
                returned_sources=5,
                failures=0,
                latency_ms=800,
            )
        ],
        model_calls=2,
        total_tokens=700,
        elapsed_ms=1200,
        stop_reason="The evidence gap remains unresolved within the budget.",
        partial_results=True,
    )


class ResearchStreamingEngine:
    def __init__(self, research_status: str = "completed") -> None:
        self.research_status = research_status

    async def search_stream(
        self,
        query,
        limit,
        on_stage,
        on_agent=None,
        on_research_round=None,
        on_tool_call=None,
    ):
        round_record = research_fixture().rounds[0]
        running_tool = research_fixture().tool_calls[0].model_copy(
            update={"status": "running", "output_summary": ""}
        )
        complete_tool = research_fixture().tool_calls[0]
        await on_research_round(round_record)
        await on_tool_call(running_tool)
        await on_tool_call(complete_tool)
        research = research_fixture().model_copy(update={"status": self.research_status})
        return EngineSearchResponse(
            query=query,
            research=research,
            search_metrics=research.providers,
        )


def test_research_events_and_terminal_metrics_are_persisted_and_pollable() -> None:
    repository = SQLAlchemyTaskRepository("sqlite:///:memory:")
    with TestClient(create_app(repository=repository, engine_client=ResearchStreamingEngine())) as client:
        accepted = client.post("/api/v1/search", json={"query": "Find an open graduate role"})
        task_id = accepted.json()["task_id"]
        body = client.get(f"/api/v1/search/{task_id}").json()

    assert body["status"] == "completed"
    assert body["research"]["status"] == "completed"
    assert body["research_rounds"][0]["evidence_gaps"] == ["application deadline"]
    assert len(body["tool_calls"]) == 1
    assert body["tool_calls"][0]["status"] == "completed"
    assert body["search_metrics"][0]["returned_sources"] == 5
    assert body["research"]["model_calls"] == 2


def test_partial_research_marks_task_partial_without_an_error() -> None:
    repository = SQLAlchemyTaskRepository("sqlite:///:memory:")
    with TestClient(
        create_app(
            repository=repository,
            engine_client=ResearchStreamingEngine(research_status="partial"),
        )
    ) as client:
        accepted = client.post("/api/v1/search", json={"query": "Find an open graduate role"})
        body = client.get(f"/api/v1/search/{accepted.json()['task_id']}").json()

    assert body["status"] == "partial"
    assert body["errors"] == []
    assert body["research"]["status"] == "partial"


def test_engine_stream_accepts_research_events_and_redacts_tool_summaries() -> None:
    research = research_fixture().model_dump(mode="json")
    round_payload = research["rounds"][0]
    call_payload = research["tool_calls"][0]
    call_payload["input_summary"] = "api_key=must-not-be-stored search query"
    partial_result = {
        "id": "candidate-1",
        "title": "Example Graduate Role",
        "url": "https://careers.example.org/role/1",
        "source_name": "Example Careers",
        "snippet": "A current role listing.",
        "evidence": "",
        "match_reason": "Awaiting official verification.",
        "verdict": "unverified",
        "confidence": 0.0,
        "decision": "recommended",
        "result_kind": "specific_opportunity",
    }
    terminal = {
        "type": "complete",
        "query": "question",
        "results": [],
        "errors": [],
        "agent_runs": [],
        "answer": None,
        "answer_claims": [],
        "research": research,
        "search_metrics": research["providers"],
    }
    lines = [
        {"type": "partial_results", "results": [partial_result]},
        {"type": "research_round", "round": round_payload},
        {"type": "tool_call", "call": call_payload},
        terminal,
    ]
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content="\n".join(json.dumps(line) for line in lines).encode(),
            headers={"content-type": "application/x-ndjson"},
        )
    )
    client = HttpEngineClient("http://engine.test", transport=transport)
    seen_rounds: list[ResearchRound] = []
    seen_calls: list[ToolCallRecord] = []
    seen_candidates: list[list[SearchResult]] = []

    async def collect() -> EngineSearchResponse:
        async def noop(_):
            return None

        return await client.search_stream(
            "question",
            10,
            noop,
            on_research_round=lambda item: _append(seen_rounds, item),
            on_tool_call=lambda item: _append(seen_calls, item),
            on_partial_results=lambda items: _append(seen_candidates, items),
        )

    response = asyncio.run(collect())

    assert seen_rounds[0].round_number == 1
    assert "must-not-be-stored" not in seen_calls[0].input_summary
    assert "[redacted]" in seen_calls[0].input_summary
    assert seen_candidates[0][0].decision == "needs_verification"
    assert seen_candidates[0][0].result_kind == "specific_opportunity"
    assert response.research is not None
    assert "must-not-be-stored" not in response.research.tool_calls[0].input_summary
    assert response.search_metrics[0].provider == "tavily"


def test_engine_stream_accepts_configured_research_candidate_budget() -> None:
    partial_results = [
        {
            "id": f"candidate-{index}",
            "title": f"Opportunity {index}",
            "url": f"https://careers.example.org/role/{index}",
            "source_name": "Example Careers",
            "snippet": "A source-backed opportunity clue.",
            "evidence": "",
            "match_reason": "Awaiting official verification.",
            "verdict": "unverified",
            "confidence": 0.0,
            "decision": "needs_verification",
            "result_kind": "specific_opportunity",
        }
        for index in range(60)
    ]
    lines = [
        {"type": "partial_results", "results": partial_results},
        {"type": "complete", "query": "question", "results": [], "errors": []},
    ]
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content="\n".join(json.dumps(line) for line in lines).encode(),
            headers={"content-type": "application/x-ndjson"},
        )
    )
    client = HttpEngineClient("http://engine.test", transport=transport)
    seen_candidates: list[SearchResult] = []

    async def collect() -> EngineSearchResponse:
        async def noop(_):
            return None

        async def capture(items):
            seen_candidates.extend(items)

        return await client.search_stream(
            "question", 10, noop, on_partial_results=capture
        )

    response = asyncio.run(collect())

    assert response.query == "question"
    assert len(seen_candidates) == 60
    assert all(item.decision == "needs_verification" for item in seen_candidates)


def test_partial_candidates_stay_running_and_terminal_results_replace_them() -> None:
    repository = SQLAlchemyTaskRepository("sqlite:///:memory:")
    repository.ensure_schema()
    task_id = "00000000-0000-0000-0000-000000000001"
    repository.create(task_id, "query")
    repository.set_status(task_id, "running")
    candidate = SearchResult(
        id="candidate",
        title="Candidate opportunity",
        url="https://example.org/candidate",
        source_name="Example",
        snippet="A clue.",
        evidence="",
        match_reason="Not yet verified.",
        verdict="unverified",
        confidence=0,
        decision="recommended",
    )

    repository.update_partial_results(task_id, [candidate])
    running = repository.get(task_id)
    assert running is not None and running.status == "running"
    assert running.results[0]["matching_status"] == "needs_verification"

    final = candidate.model_copy(update={"decision": "recommended"})
    repository.complete(task_id, [final])
    completed = repository.get(task_id)
    assert completed is not None and completed.status == "completed"
    assert completed.results[0]["matching_status"] == "recommended"


async def _append(items, item):
    items.append(item)
