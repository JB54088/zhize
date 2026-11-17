from __future__ import annotations

import json
import sqlite3

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from services.api.app.engine_client import HttpEngineClient
from services.api.app.main import create_app
from services.api.app.repository import SQLAlchemyTaskRepository
from services.api.app.schemas import (
    AgentRun,
    EngineSearchResponse,
    SearchResult,
    StageProgress,
)


def make_run(status: str = "completed", **updates: object) -> AgentRun:
    values: dict[str, object] = {
        "agent": "collect",
        "action": "intent_understanding",
        "status": status,
        "started_at": "2026-10-08T00:00:00Z",
        "completed_at": "2026-10-08T00:00:01Z",
        "duration_ms": 1000,
        "model": "deepseek-chat",
        "request_id": "req-123",
        "prompt_tokens": 14,
        "completion_tokens": 9,
        "total_tokens": 23,
        "estimated_cost_usd": None,
        "input_summary": "User asks for graduate hiring roles.",
        "output_summary": "Intent extracted.",
        "error_code": None,
    }
    values.update(updates)
    return AgentRun.model_validate(values)


def make_result() -> SearchResult:
    return SearchResult(
        id="role-1",
        title="校招岗位",
        url="https://example.org/role",
        source_name="招聘官网",
        snippet="电子信息相关岗位",
        evidence="2027届毕业生可投递。",
        match_reason="专业和届别均有来源证据。",
        verdict="supports",
        confidence=0.9,
        decision="recommended",
        evidence_sources=[
            {
                "title": "官方招聘公告",
                "url": "https://example.org/role",
                "source_type": "official",
                "status": "verified",
            },
            {
                "title": "第三方转载",
                "url": "https://news.example.net/copy",
                "source_type": "third_party",
                "status": "unverified",
            },
        ],
        llm_verification={
            "status": "analyzed",
            "facts": [
                {
                    "field": "graduate_cohort",
                    "value": "招收2027届",
                    "assessment": "supports",
                    "evidence_url": "https://example.org/role",
                    "quote": "2027届毕业生可投递。",
                }
            ],
        },
        llm_semantic_match={
            "status": "analyzed",
            "fit": "strong",
            "score": 0.92,
            "reason": "岗位专业要求与用户专业相关。",
            "evidence_quotes": ["电子信息相关岗位"],
        },
    )


class AgentStreamEngine:
    async def search_stream(self, query, limit, on_stage, on_agent=None):
        await on_stage(StageProgress(key="collect", status="completed", duration_ms=12))
        run = make_run()
        if on_agent:
            await on_agent(run)
        from services.api.app.schemas import AnswerClaim

        return EngineSearchResponse(
            query=query,
            results=[make_result()],
            agent_runs=[run],
            answer="该岗位与给定专业和届别相符。",
            answer_claims=[
                AnswerClaim(
                    result_id="role-1",
                    text="岗位接受2027届毕业生。",
                    evidence_quotes=["2027届毕业生可投递。"],
                )
            ],
        )


def test_task_persists_real_agent_run_usage_and_enriches_answer_citations() -> None:
    repository = SQLAlchemyTaskRepository("sqlite:///:memory:")
    with TestClient(create_app(repository=repository, engine_client=AgentStreamEngine())) as client:
        accepted = client.post("/api/v1/search", json={"query": "找校招"})
        payload = client.get(f"/api/v1/search/{accepted.json()['task_id']}").json()

    assert payload["status"] == "completed"
    assert payload["agent_runs"][0]["model"] == "deepseek-chat"
    assert payload["agent_runs"][0]["total_tokens"] == 23
    assert payload["agent_runs"][0]["duration_ms"] == 1000
    assert payload["answer"] == "该岗位与给定专业和届别相符。"
    assert payload["answer_claims"][0]["result_id"] == "role-1"
    assert payload["answer_claims"][0]["sources"] == [
        {
            "title": "官方招聘公告",
            "url": "https://example.org/role",
            "source_type": "official",
            "status": "verified",
        }
    ]
    card = payload["results"][0]
    assert card["category"] == "employment"
    assert card["matching_status"] == "recommended"
    assert card["source_url"] == "https://example.org/role"
    # The public card stays compact; original model evidence is retained in the
    # task source records for traceability.
    task_id = payload["task_id"]
    row = repository.get(task_id)
    assert row is not None
    assert row.source_records[0]["llm_verification"]["facts"][0]["assessment"] == "supports"
    assert row.source_records[0]["llm_semantic_match"]["score"] == 0.92


def test_agent_event_is_redacted_before_it_reaches_storage() -> None:
    lines = [
        {
            "type": "agent",
            "run": make_run(input_summary="api_key=super-secret-token").model_dump(mode="json"),
        },
        {"type": "complete", "query": "q", "results": [], "errors": []},
    ]

    async def handler(_: httpx.Request) -> httpx.Response:
        content = "\n".join(json.dumps(line) for line in lines) + "\n"
        return httpx.Response(200, content=content.encode())

    seen: list[AgentRun] = []

    async def stage(_: StageProgress) -> None:
        return None

    async def agent(run: AgentRun) -> None:
        seen.append(run)

    import asyncio

    client = HttpEngineClient(
        "http://engine.local", transport=httpx.MockTransport(handler)
    )
    asyncio.run(client.search_stream("q", 3, stage, agent))

    assert len(seen) == 1
    assert "super-secret-token" not in (seen[0].input_summary or "")
    assert "[redacted]" in (seen[0].input_summary or "")


def test_additive_migration_adds_agent_and_answer_fields_to_legacy_tasks(tmp_path) -> None:
    database_path = tmp_path / "before-v03.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """CREATE TABLE search_tasks (
            task_id VARCHAR(36) PRIMARY KEY,
            query VARCHAR(4000) NOT NULL,
            status VARCHAR(16) NOT NULL,
            results JSON NOT NULL,
            error TEXT,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )"""
    )
    connection.execute(
        "INSERT INTO search_tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "old-task",
            "old query",
            "completed",
            "[]",
            None,
            "2026-10-01 00:00:00",
            "2026-10-01 00:00:00",
        ),
    )
    connection.commit()
    connection.close()

    repository = SQLAlchemyTaskRepository(f"sqlite:///{database_path}")
    repository.ensure_schema()
    row = repository.get("old-task")
    columns = {column["name"] for column in inspect(repository.engine).get_columns("search_tasks")}

    assert row is not None
    assert row.agent_runs is None
    assert row.answer is None
    assert row.answer_claims is None
    assert {"agent_runs", "answer", "answer_claims"}.issubset(columns)


def test_unconfigured_and_not_analyzed_snapshots_remain_explicit() -> None:
    skipped = make_run(
        "skipped_unconfigured",
        model=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        duration_ms=None,
        error_code="llm_unconfigured",
    )
    result_data = make_result().model_dump(mode="json")
    result_data["llm_semantic_match"] = {"status": "not_analyzed"}
    result = SearchResult.model_validate(result_data)

    assert skipped.status == "skipped_unconfigured"
    assert skipped.model is None
    assert result.llm_semantic_match is not None
    assert result.llm_semantic_match.fit is None
    assert result.llm_semantic_match.score is None
