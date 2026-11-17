import ast
from pathlib import Path
from typing import get_args

import yaml

from services.api.app.schemas import SearchResult, VerificationField


ROOT = Path(__file__).resolve().parents[3]


def _literal_values_from_ai_schema() -> set[str]:
    path = ROOT / "services" / "ai-engine" / "app" / "schemas.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if not isinstance(target, ast.Name) or target.id != "VerificationField":
            continue
        if not isinstance(value, ast.Subscript):
            raise AssertionError("AI Engine VerificationField must be a Literal")
        literal = value.slice
        values = literal.elts if isinstance(literal, ast.Tuple) else [literal]
        return {
            item.value
            for item in values
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
    raise AssertionError("AI Engine VerificationField was not found")


def test_api_verification_fields_match_engine_and_openapi_contract() -> None:
    api_fields = set(get_args(VerificationField))
    contract_path = ROOT / "contracts" / "api.openapi.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    result = contract["components"]["schemas"]["SearchResult"]["properties"]
    verified_enum = set(result["verified_fields"]["items"]["enum"])
    unverified_enum = set(result["unverified_fields"]["items"]["enum"])

    assert api_fields == _literal_values_from_ai_schema()
    assert verified_enum == api_fields
    assert unverified_enum == api_fields


def test_region_verification_field_is_accepted_by_api_result_model() -> None:
    payload = {
        "id": "region-result",
        "title": "Regional role",
        "url": "https://example.org/role",
        "source_name": "Example",
        "snippet": "Regional role",
        "evidence": "Region: Fujian",
        "match_reason": "Region verified from source.",
        "verdict": "supports",
        "confidence": 0.8,
        "verified_fields": ["region"],
    }

    assert SearchResult.model_validate(payload).verified_fields == ["region"]
