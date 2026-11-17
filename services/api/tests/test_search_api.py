from fastapi.testclient import TestClient

from services.api.app.engine_client import EngineClientError, EngineStreamError
from services.api.app.main import create_app
from services.api.app.repository import SQLAlchemyTaskRepository
from services.api.app.schemas import EngineSearchResponse, SearchResult, SourceRecord, StageProgress


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


class StreamingMockEngine:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        errors: list[str] | None = None,
        failure: Exception | None = None,
        stage_error: str | None = None,
        partial_stage: str | None = None,
    ):
        self.results = results or []
        self.errors = errors or []
        self.failure = failure
        self.stage_error = stage_error
        self.partial_stage = partial_stage

    async def search_stream(self, query: str, limit: int, on_stage):
        for index, key in enumerate(("collect", "clean", "verify", "judge")):
            await on_stage(
                StageProgress(
                    key=key,
                    status="partial" if key == self.partial_stage else "completed",
                    input_count=index + 1,
                    output_count=index + 1,
                    duration_ms=17 + index,
                    error=self.stage_error if index == 0 else None,
                )
            )
            if self.failure and key == "verify":
                raise self.failure
        from services.api.app.schemas import EngineSearchResponse

        return EngineSearchResponse(query=query, results=self.results, errors=self.errors)


class SourceRecordEngine:
    async def search_stream(self, query: str, candidate_limit: int, on_stage):
        return EngineSearchResponse(
            query=query,
            results=[sample_result()],
            source_records=[
                SourceRecord(
                    id="captured-source-1",
                    title="Official source page",
                    url="https://example.org/source",
                    source_name="Example authority",
                    provider="tavily",
                    captured_at="2026-10-09T00:00:00Z",
                    search_query_index=0,
                    search_query=query,
                    raw_text="Full captured public source body for evidence review.",
                    metadata={"content_type": "text/html"},
                )
            ],
        )


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
        decision="recommended",
        summary="A concise verified summary.",
        key_conditions=[
            {
                "label": "截止时间",
                "value": "2026-10-31",
                "status": "verified",
                "evidence_url": "https://example.org/source",
                "quote": "The deadline is 2026-10-31.",
            }
        ],
        evidence_sources=[
            {
                "title": "Official notice",
                "url": "https://example.org/source",
                "source_type": "official",
                "status": "verified",
            }
        ],
        evidence_detail="Full source excerpt and field-level verification.",
    )


def make_client(engine: MockEngine) -> TestClient:
    repository = SQLAlchemyTaskRepository("sqlite:///:memory:")
    return TestClient(create_app(repository=repository, engine_client=engine))


def test_search_persists_engine_results_and_returns_task_status() -> None:
    engine = MockEngine([sample_result()])
    with make_client(engine) as client:
        accepted = client.post(
            "/api/v1/search", json={"query": "source query", "candidate_limit": 4}
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
        assert body["total_results"] == 1
        assert body["page"] == 1
        assert len(body["stages"]) == 4
        assert all(stage["status"] == "skipped" for stage in body["stages"])
        card = body["results"][0]
        assert set(card) == {
            "id", "title", "category", "summary", "rank", "recommendation_reason",
            "attributes", "source_url", "source_name", "additional_sources",
            "published_at", "last_verified_at", "verification_status",
            "matching_status", "freshness_status",
        }
        assert card["id"] == "source-1"
        assert card["category"] == "other"
        assert card["rank"] == 1
        assert card["source_url"] == "https://example.org/source"
        assert card["verification_status"] == "verified"
        assert body["error"] is None
        assert body["results"][0]["attributes"][0]["key"] == "deadline"


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


def test_streaming_task_persists_stage_snapshots_and_pages_full_result_set() -> None:
    results = [sample_result().model_copy(update={"id": f"source-{i}"}) for i in range(3)]
    with make_client(StreamingMockEngine(results)) as client:
        accepted = client.post("/api/v1/search", json={"query": "question"})
        task_id = accepted.json()["task_id"]
        page = client.get(f"/api/v1/search/{task_id}?page=2&page_size=2")

    assert page.status_code == 200
    body = page.json()
    assert body["status"] == "completed"
    assert body["total_results"] == 3
    assert body["page"] == 2
    assert body["page_size"] == 2
    assert [result["id"] for result in body["results"]] == ["source-2"]
    assert [stage["key"] for stage in body["stages"]] == [
        "collect", "clean", "verify", "judge"
    ]
    assert all(stage["status"] == "completed" for stage in body["stages"])
    assert body["stages"][2]["duration_ms"] == 19


def test_default_page_of_ten_does_not_truncate_candidate_collection() -> None:
    candidates = [
        sample_result().model_copy(update={"id": f"source-{index}"})
        for index in range(12)
    ]
    engine = MockEngine(candidates)
    repository = SQLAlchemyTaskRepository("sqlite:///:memory:")
    with TestClient(create_app(repository=repository, engine_client=engine)) as client:
        accepted = client.post(
            "/api/v1/search", json={"query": "source query", "limit": 10}
        )
        task_id = accepted.json()["task_id"]
        first_page = client.get(f"/api/v1/search/{task_id}").json()
        second_page = client.get(f"/api/v1/search/{task_id}?page=2").json()

    assert engine.calls == [("source query", 50)]
    assert first_page["page_size"] == 10
    assert first_page["total_results"] == 12
    assert len(first_page["results"]) == 10
    assert len(second_page["results"]) == 2
    row = repository.get(task_id)
    assert row is not None and len(row.source_records) == 12


def test_source_records_are_persisted_privately_and_not_returned_with_cards() -> None:
    repository = SQLAlchemyTaskRepository("sqlite:///:memory:")
    with TestClient(create_app(repository=repository, engine_client=SourceRecordEngine())) as client:
        accepted = client.post("/api/v1/search", json={"query": "source query"})
        task_id = accepted.json()["task_id"]
        body = client.get(f"/api/v1/search/{task_id}").json()

    row = repository.get(task_id)
    assert row is not None
    assert row.source_records[0]["raw_text"] == "Full captured public source body for evidence review."
    assert row.source_records[0]["metadata"]["content_type"] == "text/html"
    assert "source_records" not in body
    assert "Full captured public source body" not in str(body)


def test_streaming_partial_results_are_marked_partial_and_error_is_sanitized() -> None:
    with make_client(
        StreamingMockEngine(
            results=[sample_result()],
            errors=["secret=must-not-escape"],
        )
    ) as client:
        accepted = client.post("/api/v1/search", json={"query": "question"})
        body = client.get(f"/api/v1/search/{accepted.json()['task_id']}").json()

    assert body["status"] == "partial"
    assert len(body["results"]) == 1
    assert body["errors"] == ["AI engine returned partial results."]
    assert "must-not-escape" not in str(body)


def test_partial_stage_marks_task_partial_without_fabricating_an_error() -> None:
    with make_client(
        StreamingMockEngine(results=[sample_result()], partial_stage="clean")
    ) as client:
        accepted = client.post("/api/v1/search", json={"query": "question"})
        body = client.get(f"/api/v1/search/{accepted.json()['task_id']}").json()

    assert body["status"] == "partial"
    assert body["errors"] == []
    assert body["stages"][1]["status"] == "partial"


def test_stream_failure_marks_stage_and_hides_upstream_message() -> None:
    with make_client(
        StreamingMockEngine(
            failure=EngineStreamError("api_key=secret must stay private", stage="verify")
        )
    ) as client:
        accepted = client.post("/api/v1/search", json={"query": "question"})
        body = client.get(f"/api/v1/search/{accepted.json()['task_id']}").json()

    assert body["status"] == "failed"
    assert body["stages"][2]["status"] == "failed"
    assert body["stages"][2]["input_count"] == 3
    assert body["stages"][2]["duration_ms"] == 19
    assert body["stages"][2]["error"] == "AI engine stage failed."
    assert body["stages"][3]["status"] == "skipped"
    assert body["error"] == "AI engine stage failed."
    assert "secret" not in str(body)


def test_engine_unavailable_marks_collection_failed_and_later_stages_skipped() -> None:
    with make_client(MockEngine(error=EngineClientError("AI engine is unavailable."))) as client:
        accepted = client.post("/api/v1/search", json={"query": "question"})
        body = client.get(f"/api/v1/search/{accepted.json()['task_id']}").json()

    assert body["status"] == "failed"
    assert [stage["status"] for stage in body["stages"]] == [
        "failed", "skipped", "skipped", "skipped"
    ]


def test_stage_error_payload_is_sanitized_even_from_injected_engine_client() -> None:
    with make_client(
        StreamingMockEngine(stage_error="api_key=secret-value")
    ) as client:
        accepted = client.post("/api/v1/search", json={"query": "question"})
        body = client.get(f"/api/v1/search/{accepted.json()['task_id']}").json()

    assert body["stages"][0]["error"] == "Stage reported an error."
    assert "secret-value" not in str(body)


def test_pagination_validates_page_and_page_size() -> None:
    with make_client(MockEngine([sample_result()])) as client:
        accepted = client.post("/api/v1/search", json={"query": "question"})
        task_id = accepted.json()["task_id"]
        page = client.get(f"/api/v1/search/{task_id}?page=0")
        size = client.get(f"/api/v1/search/{task_id}?page_size=101")

    assert page.status_code == 422
    assert size.status_code == 422


def test_search_result_accepts_recruitment_verification_fields_and_application_url() -> None:
    payload = sample_result().model_dump(mode="json")
    payload["verified_fields"] = ["graduate_cohort", "major", "application_url"]
    payload["application_url"] = "https://careers.example.org/apply/role-7"

    result = SearchResult.model_validate(payload)

    assert result.verified_fields == ["graduate_cohort", "major", "application_url"]
    assert result.application_url == "https://careers.example.org/apply/role-7"
