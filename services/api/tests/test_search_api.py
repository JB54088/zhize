from fastapi.testclient import TestClient

from services.api.app.engine_client import EngineClientError
from services.api.app.main import create_app
from services.api.app.repository import SQLAlchemyTaskRepository
from services.api.app.schemas import SearchResult


class MockEngine:
    def __init__(self, results: list[SearchResult] | None = None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        return self.results


def sample_result() -> SearchResult:
    return SearchResult(
        id="source-1",
        title="Primary source",
        url="https://example.org/source",
        source_name="Example Institute",
        snippet="A source-provided summary.",
        evidence="The source states the requested fact.",
        match_reason="Directly addresses the query.",
        verdict="supports",
        confidence=0.91,
        published_at="2026-10-01T12:00:00Z",
    )


def make_client(engine: MockEngine) -> TestClient:
    repository = SQLAlchemyTaskRepository("sqlite:///:memory:")
    return TestClient(create_app(repository=repository, engine_client=engine))


def test_search_persists_engine_results_and_returns_task_status() -> None:
    engine = MockEngine([sample_result()])
    with make_client(engine) as client:
        accepted = client.post(
            "/api/v1/search", json={"query": "source query", "limit": 4}
        )
        assert accepted.status_code == 202
        task_id = accepted.json()["task_id"]
        assert accepted.json()["status"] == "queued"
        assert engine.calls == [("source query", 4)]

        task = client.get(f"/api/v1/search/{task_id}")
        assert task.status_code == 200
        body = task.json()
        assert body["query"] == "source query"
        assert body["status"] == "completed"
        assert body["results"] == [sample_result().model_dump(mode="json")]
        assert "error" not in body


def test_search_failure_is_stored_without_fabricating_results() -> None:
    engine = MockEngine(error=EngineClientError("AI engine is unavailable."))
    with make_client(engine) as client:
        accepted = client.post("/api/v1/search", json={"query": "question"})
        task = client.get(f"/api/v1/search/{accepted.json()['task_id']}")

        assert task.status_code == 200
        assert task.json()["status"] == "failed"
        assert task.json()["results"] == []
        assert task.json()["error"] == "AI engine is unavailable."


def test_search_rejects_blank_queries_and_unknown_task_ids() -> None:
    with make_client(MockEngine()) as client:
        invalid = client.post("/api/v1/search", json={"query": "  "})
        missing = client.get("/api/v1/search/not-a-task")

        assert invalid.status_code == 422
        assert missing.status_code == 404
