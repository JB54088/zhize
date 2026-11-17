from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_contract(filename: str) -> dict:
    path = ROOT / "contracts" / filename
    with path.open(encoding="utf-8") as contract_file:
        return yaml.safe_load(contract_file)


def resolve_ref(document: dict, schema: dict) -> dict:
    while "$ref" in schema:
        target = document
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        schema = target
    return schema


def request_schema(document: dict, path: str) -> dict:
    operation = document["paths"][path]["post"]
    schema = operation["requestBody"]["content"]["application/json"]["schema"]
    return resolve_ref(document, schema)


def response_schema(document: dict, path: str, method: str, status: int) -> dict:
    operation = document["paths"][path][method]
    schema = operation["responses"][str(status)]["content"]["application/json"]["schema"]
    return resolve_ref(document, schema)


def result_schema(document: dict) -> dict:
    schema = document["components"]["schemas"]["SearchResult"]
    return resolve_ref(document, schema)


def test_api_contract_creates_and_polls_search_tasks() -> None:
    api = read_contract("api.openapi.yaml")
    create = response_schema(api, "/api/v1/search", "post", 202)
    poll = response_schema(api, "/api/v1/search/{task_id}", "get", 200)

    assert {"task_id", "status"} <= set(create["properties"])
    assert create["properties"]["status"]["enum"] == [
        "queued", "running", "completed", "partial", "failed"
    ]
    assert {
        "task_id", "query", "status", "results", "stages", "total_results",
        "page", "page_size", "errors",
    } <= set(poll["properties"])
    assert {"page", "page_size"} <= {
        parameter["name"]
        for parameter in api["paths"]["/api/v1/search/{task_id}"]["get"].get(
            "parameters", []
        )
    }


def test_ai_engine_contract_accepts_query_and_returns_results() -> None:
    engine = read_contract("ai-engine.openapi.yaml")
    request = request_schema(engine, "/api/v1/search")
    response = response_schema(engine, "/api/v1/search", "post", 200)

    assert "query" in request["properties"]
    assert {"query", "results"} <= set(response["properties"])
    stream = engine["paths"]["/api/v1/search/stream"]["post"]
    assert stream["responses"]["200"]["content"]["application/x-ndjson"]


def test_backend_and_engine_candidate_limits_match_and_are_separate_from_page_size() -> None:
    api = read_contract("api.openapi.yaml")
    engine = read_contract("ai-engine.openapi.yaml")
    api_candidate_limit = request_schema(api, "/api/v1/search")["properties"]["candidate_limit"]
    engine_candidate_limit = request_schema(engine, "/api/v1/search")["properties"]["candidate_limit"]

    assert api_candidate_limit["minimum"] == engine_candidate_limit["minimum"] == 1
    assert api_candidate_limit["maximum"] == engine_candidate_limit["maximum"] == 100
    assert api_candidate_limit["default"] == engine_candidate_limit["default"] == 50

    task = api["components"]["schemas"]["SearchTask"]["properties"]
    assert task["page_size"]["default"] == 10


def test_api_and_engine_share_grounded_search_result_schema() -> None:
    required_fields = {
        "id",
        "title",
        "url",
        "source_name",
        "snippet",
        "evidence",
        "match_reason",
        "verdict",
        "confidence",
        "verified_fields",
        "unverified_fields",
        "deadline_status",
        "official_source_status",
        "evidence_status",
        "procurement_category",
        "match_status",
        "match_score",
        "decision",
        "summary",
        "key_conditions",
        "evidence_sources",
        "evidence_detail",
        "application_url",
    }
    api = read_contract("api.openapi.yaml")
    engine = read_contract("ai-engine.openapi.yaml")
    api_result = result_schema(api)
    engine_result = result_schema(engine)

    for schema in (api_result, engine_result):
        assert schema["type"] == "object"
        assert required_fields <= set(schema["properties"])
        assert required_fields <= set(schema["required"])
        assert schema["properties"]["verdict"]["enum"] == [
            "supports",
            "contradicts",
            "context",
            "unverified",
        ]
        assert schema["properties"]["confidence"]["minimum"] == 0
        assert schema["properties"]["confidence"]["maximum"] == 1
        assert schema["properties"]["deadline_status"]["enum"] == [
            "expired",
            "not_expired",
            "not_stated",
            "unverified",
        ]
        assert schema["properties"]["official_source_status"]["enum"] == [
            "verified",
            "third_party",
            "insufficient_evidence",
            "unverified",
        ]
        assert schema["properties"]["evidence_status"]["enum"] == [
            "available",
            "missing",
        ]
        assert schema["properties"]["procurement_category"]["enum"] == [
            "ai_training",
            "ai_system",
            "digital_transformation",
            "compute_hardware",
            "other",
            "unclear",
        ]
        assert schema["properties"]["match_status"]["enum"] == [
            "matched",
            "partial",
            "keyword_only",
            "not_matched",
            "unverified",
        ]
        assert schema["properties"]["result_kind"]["enum"][:3] == [
            "specific_opportunity", "organization_lead", "unknown"
        ]
        assert set(schema["properties"]["result_kind"]["enum"]) <= {
            "specific_opportunity", "organization_lead", "unknown", None
        }

    assert set(api_result["properties"]) == set(engine_result["properties"])
    assert api_result["properties"]["verified_fields"] == engine_result["properties"]["verified_fields"]
    assert api_result["properties"]["unverified_fields"] == engine_result["properties"]["unverified_fields"]

    for document in (api, engine):
        result = result_schema(document)
        assert result["properties"]["decision"]["enum"] == [
            "recommended", "needs_verification", "not_matched"
        ]
        condition = resolve_ref(
            document,
            result["properties"]["key_conditions"]["items"],
        )
        assert condition["properties"]["status"]["enum"] == [
            "verified", "unverified", "conflict", "missing"
        ]


def test_stage_trace_contract_has_four_real_ordered_stages_and_truthful_metrics() -> None:
    expected_keys = ["collect", "clean", "verify", "judge"]
    expected_statuses = [
        "pending", "running", "completed", "partial", "failed", "skipped"
    ]
    api = read_contract("api.openapi.yaml")
    engine = read_contract("ai-engine.openapi.yaml")
    api_stage = resolve_ref(
        api,
        api["components"]["schemas"]["SearchTask"]["properties"]["stages"]["items"],
    )
    engine_stage = engine["components"]["schemas"]["StageSnapshot"]

    for stage in (api_stage, engine_stage):
        assert {"key", "status", "started_at", "completed_at", "duration_ms"} <= set(stage["required"])
        assert stage["properties"]["key"]["enum"] == expected_keys
        assert stage["properties"]["status"]["enum"] == expected_statuses
        for count in (
            "input_count", "output_count", "verified_count", "unverified_count",
            "conflict_count",
        ):
            assert stage["properties"][count]["type"] in ("integer", ["integer", "null"])
            assert stage["properties"][count]["minimum"] == 0
    assert api_stage["properties"] == engine_stage["properties"]

    stream_content = engine["paths"]["/api/v1/search/stream"]["post"]["responses"]["200"]["content"]
    assert "application/x-ndjson" in stream_content
    stream_schema = stream_content["application/x-ndjson"]["schema"]
    assert stream_schema["type"] == "string"


def test_v04_research_loop_events_and_budgets_share_the_same_bounded_contract() -> None:
    api = read_contract("api.openapi.yaml")
    engine = read_contract("ai-engine.openapi.yaml")
    api_task = api["components"]["schemas"]["SearchTask"]["properties"]
    engine_response = engine["components"]["schemas"]["SearchResponse"]["properties"]

    assert {"research", "research_rounds", "tool_calls", "search_metrics"} <= set(api_task)
    assert {"research", "search_metrics"} <= set(engine_response)
    research = api["components"]["schemas"]["ResearchOutcome"]
    round_schema = api["components"]["schemas"]["ResearchRound"]
    assert research["properties"]["rounds"]["maxItems"] == 3
    assert round_schema["properties"]["round_number"]["maximum"] == 3
    assert api["components"]["schemas"]["ResearchToolCall"]["properties"]["name"]["enum"] == [
        "search_web", "search_official", "read_webpage", "inspect_evidence"
    ]
    assert api["components"]["schemas"]["SearchProviderMetric"]["properties"]["configured"]["type"] == "boolean"

    stream = engine["paths"]["/api/v1/search/stream"]["post"]["responses"]["200"]["content"]["application/x-ndjson"]
    # The event union is documented alongside representative NDJSON examples.
    schemas = engine["components"]["schemas"]["SearchStreamEvent"]["oneOf"]
    event_types = {
        branch["properties"]["type"]["const"]
        for branch in schemas
        if "properties" in branch and "type" in branch["properties"]
    }
    assert {"partial_results", "research_round", "tool_call", "complete"} <= event_types
    assert stream["schema"]["type"] == "string"
