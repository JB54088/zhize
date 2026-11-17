"""Shared safe LLM invocation and execution-log helpers."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from .models import AgentAction, AgentName, AgentRunLog, AgentStatus


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def llm_is_enabled(client: Any) -> bool:
    if client is None:
        return False
    value = getattr(client, "is_enabled", True)
    return bool(value() if callable(value) else value)


def completion_data(completion: Any) -> Any:
    value = getattr(completion, "data", completion)
    return value.model_dump(mode="python") if isinstance(value, BaseModel) else value


def completion_metadata(completion: Any) -> dict[str, Any]:
    return {
        "actual_model": getattr(completion, "model", None),
        "request_id": getattr(completion, "request_id", None),
        "latency_ms": getattr(completion, "latency_ms", None),
        "prompt_tokens": getattr(completion, "prompt_tokens", None),
        "completion_tokens": getattr(completion, "completion_tokens", None),
        "total_tokens": getattr(completion, "total_tokens", None),
        "estimated_cost_usd": getattr(completion, "estimated_cost_usd", None),
        "retry_count": getattr(completion, "retry_count", None),
    }


async def invoke_json(
    client: Any,
    *,
    agent_name: str,
    messages: list[dict[str, str]],
    response_model: type[BaseModel],
) -> tuple[Any | None, dict[str, Any], str | None]:
    """Call the shared OpenAI-compatible client without logging prompt content."""
    if not llm_is_enabled(client):
        return None, {}, None
    try:
        completion = await client.complete_json(
            agent_name=agent_name,
            messages=messages,
            response_model=response_model,
        )
        return completion_data(completion), completion_metadata(completion), None
    except Exception as exc:  # client exceptions are sanitized by the infra layer
        code = getattr(exc, "code", "llm_request_failed")
        candidate_code = str(code).casefold() if isinstance(code, str) else ""
        safe_code = (
            candidate_code
            if re.fullmatch(r"[a-z0-9_]{1,64}", candidate_code)
            else "llm_request_failed"
        )
        metadata = {
            name: getattr(exc, name, None)
            for name in (
                "actual_model",
                "request_id",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "estimated_cost_usd",
                "retry_count",
            )
            if getattr(exc, name, None) is not None
        }
        metadata["retry_count"] = _safe_nonnegative_int(metadata.get("retry_count"))
        return None, metadata, safe_code


def make_log(
    *,
    agent: AgentName,
    action: AgentAction,
    status: AgentStatus,
    started_iso: str,
    started_clock: float,
    input_summary: str,
    output_summary: str,
    metadata: dict[str, Any] | None = None,
    error_code: str | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    payload = {
        "agent": agent,
        "action": action,
        "status": status,
        "started_at": started_iso,
        "completed_at": now_iso(),
        "duration_ms": max(0, int((time.perf_counter() - started_clock) * 1000)),
        "input_summary": input_summary,
        "output_summary": output_summary,
        "error_code": error_code,
        "skip_reason": skip_reason,
    }
    for key in (
        "actual_model",
        "request_id",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "retry_count",
    ):
        value = metadata.get(key)
        if value is not None:
            payload[key] = value
    return AgentRunLog.model_validate(payload).model_dump(mode="json")


def json_message(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def literal(value: Any, source: str) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value in source


def normalize_literal(value: str) -> str:
    return "".join(value.split()).casefold()


def value_is_grounded(value: str, quote: str) -> bool:
    if literal(value, quote):
        return True
    normalized_value = normalize_literal(value)
    normalized_quote = normalize_literal(quote)
    return bool(normalized_value and normalized_value in normalized_quote)


def _safe_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted if converted >= 0 else None
