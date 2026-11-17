from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CARD_FIELDS = {
    "id",
    "title",
    "category",
    "summary",
    "rank",
    "recommendation_reason",
    "attributes",
    "source_url",
    "source_name",
    "additional_sources",
    "published_at",
    "last_verified_at",
    "verification_status",
    "matching_status",
    "freshness_status",
}


def load_contract(name: str) -> dict:
    with (ROOT / "contracts" / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve(document: dict, schema: dict) -> dict:
    while "$ref" in schema:
        target = document
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        schema = target
    return schema


def test_api_and_engine_define_the_same_generic_result_card() -> None:
    api = load_contract("api.openapi.yaml")
    engine = load_contract("ai-engine.openapi.yaml")
    api_card = api["components"]["schemas"]["ResultCard"]
    engine_card = engine["components"]["schemas"]["ResultCard"]

    assert CARD_FIELDS <= set(api_card["properties"])
    assert CARD_FIELDS <= set(engine_card["properties"])
    assert CARD_FIELDS <= set(api_card["required"])
    assert CARD_FIELDS <= set(engine_card["required"])
    for field in CARD_FIELDS - {"attributes", "additional_sources"}:
        assert api_card["properties"][field] == engine_card["properties"][field]
    assert set(api_card["properties"]) == set(engine_card["properties"])

    for document, card in ((api, api_card), (engine, engine_card)):
        attribute = resolve(document, card["properties"]["attributes"]["items"])
        assert {"key", "label", "value", "status", "evidence_url", "evidence_quote"} <= set(attribute["properties"])
        assert attribute["properties"]["status"]["enum"] == [
            "verified", "pending", "unknown", "conflict"
        ]


def test_search_endpoints_return_cards_without_redefining_candidate_limit_as_display_page() -> None:
    api = load_contract("api.openapi.yaml")
    engine = load_contract("ai-engine.openapi.yaml")
    api_request = resolve(
        api,
        api["paths"]["/api/v1/search"]["post"]["requestBody"]["content"]["application/json"]["schema"],
    )
    engine_request = resolve(
        engine,
        engine["paths"]["/api/v1/search"]["post"]["requestBody"]["content"]["application/json"]["schema"],
    )
    assert api_request["properties"]["candidate_limit"]["default"] == 50
    assert engine_request["properties"]["candidate_limit"]["default"] == 50
    assert api_request["properties"]["candidate_limit"]["maximum"] == 100
    assert engine_request["properties"]["candidate_limit"]["maximum"] == 100

    task = api["components"]["schemas"]["SearchTask"]["properties"]
    api_cards = resolve(api, task["results"]["items"])
    engine_response = engine["components"]["schemas"]["SearchResponse"]["properties"]
    engine_cards = resolve(engine, engine_response["cards"]["items"])

    assert api_cards["properties"] == api["components"]["schemas"]["ResultCard"]["properties"]
    assert engine_cards["properties"] == engine["components"]["schemas"]["ResultCard"]["properties"]
    assert task["page_size"]["default"] == 10
    assert task["total_results"]["type"] in ("integer", ["integer", "null"])


def test_source_links_and_domain_attributes_allow_missing_evidence_without_fabrication() -> None:
    for filename in ("api.openapi.yaml", "ai-engine.openapi.yaml"):
        document = load_contract(filename)
        card = document["components"]["schemas"]["ResultCard"]
        assert card["properties"]["source_url"].get("nullable") or "null" in card["properties"]["source_url"].get("type", [])
        assert card["properties"]["source_name"].get("nullable") or "null" in card["properties"]["source_name"].get("type", [])
        assert set(card["properties"]["verification_status"]["enum"]) == {
            "verified", "pending", "conflict", "unknown"
        }
        assert set(card["properties"]["matching_status"]["enum"]) == {
            "recommended", "needs_verification", "not_matched", "unknown"
        }
        assert set(card["properties"]["freshness_status"]["enum"]) == {
            "fresh", "stale", "unknown"
        }
