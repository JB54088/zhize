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
        "queued",
        "running",
        "completed",
        "failed",
    ]
    assert {"task_id", "query", "status", "results"} <= set(poll["properties"])


def test_ai_engine_contract_accepts_query_and_returns_results() -> None:
    engine = read_contract("ai-engine.openapi.yaml")
    request = request_schema(engine, "/api/v1/search")
    response = response_schema(engine, "/api/v1/search", "post", 200)

    assert "query" in request["properties"]
    assert {"query", "results"} <= set(response["properties"])


def test_backend_and_engine_search_limits_match() -> None:
    api = read_contract("api.openapi.yaml")
    engine = read_contract("ai-engine.openapi.yaml")
    api_limit = request_schema(api, "/api/v1/search")["properties"]["limit"]
    engine_limit = request_schema(engine, "/api/v1/search")["properties"]["limit"]

    assert api_limit["minimum"] == engine_limit["minimum"]
    assert api_limit["maximum"] == engine_limit["maximum"]
    assert api_limit["default"] == engine_limit["default"]


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
    }
    api = read_contract("api.openapi.yaml")
    engine = read_contract("ai-engine.openapi.yaml")
    api_result = result_schema(api)
    engine_result = result_schema(engine)

    for schema in (api_result, engine_result):
        assert schema["type"] == "object"
        assert required_fields <= set(schema["properties"])
        assert schema["properties"]["verdict"]["enum"] == [
            "supports",
            "contradicts",
            "context",
            "unverified",
        ]
        assert schema["properties"]["confidence"]["minimum"] == 0
        assert schema["properties"]["confidence"]["maximum"] == 1

    assert set(api_result["properties"]) == set(engine_result["properties"])
