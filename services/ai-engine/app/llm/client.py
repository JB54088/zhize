"""Small, auditable OpenAI-compatible client used by the AI Engine agents.

The client deliberately knows nothing about search records or business decisions.
It sends structured requests, validates JSON responses and returns call metadata.
It never logs prompts, completions, credentials, or response bodies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)
T = TypeVar("T")
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class LLMConfig:
    """Runtime settings for an OpenAI-compatible Chat Completions endpoint."""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = ""
    thinking_mode: Literal["enabled", "disabled"] | None = None
    timeout_seconds: float = 45.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.25
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    max_prompt_tokens: int = 24_000
    max_completion_tokens: int = 8_192

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        if self.max_prompt_tokens <= 0 or self.max_completion_tokens <= 0:
            raise ValueError("LLM token budgets must be positive")
        if self.thinking_mode not in (None, "enabled", "disabled"):
            raise ValueError("thinking_mode must be enabled, disabled, or None")
        for rate in (self.input_usd_per_million, self.output_usd_per_million):
            if rate is not None and rate < 0:
                raise ValueError("token rates cannot be negative")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key.strip() and self.model.strip())


@dataclass(frozen=True)
class LLMCompletion(Generic[T]):
    """Validated output and provider-reported metadata for one successful call."""

    data: T
    model: str
    request_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: int
    estimated_cost_usd: float | None
    retry_count: int


@dataclass(frozen=True)
class LLMToolCall:
    """A model-proposed function call. It is data only; the client never dispatches it."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMToolCompletion:
    """Tool-call response and provider telemetry from one Chat Completions request."""

    content: str | None
    tool_calls: tuple[LLMToolCall, ...]
    model: str
    request_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: int
    estimated_cost_usd: float | None
    retry_count: int


class LLMClientError(RuntimeError):
    """Sanitized provider/configuration/validation error safe to persist or display."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        retry_count: int = 0,
        latency_ms: int | None = None,
        actual_model: str | None = None,
        request_id: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retry_count = retry_count
        self.latency_ms = latency_ms
        self.actual_model = actual_model
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.estimated_cost_usd = estimated_cost_usd


class LLMClient:
    """Async JSON-mode client for DeepSeek and compatible APIs.

    `complete_json` accepts a Pydantic model class for parsed model output, or a
    JSON Schema dictionary for a plain dictionary result. The API response's model
    identifier is returned verbatim; it is never inferred from the configured one.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport

    @property
    def is_enabled(self) -> bool:
        """True only when both an API key and model name are configured."""
        return self.config.enabled

    def require_enabled(self) -> None:
        if not self.is_enabled:
            raise LLMClientError(
                "not_configured",
                "LLM is not configured; set LLM_API_KEY and LLM_MODEL.",
            )

    async def complete_json(
        self,
        agent_name: str,
        messages: list[dict[str, str]],
        response_model: type[BaseModel] | dict[str, Any],
        *,
        max_prompt_tokens: int | None = None,
        max_completion_tokens: int | None = None,
    ) -> LLMCompletion[Any]:
        """Call Chat Completions in JSON mode and validate the returned object.

        Retryable transport errors and HTTP statuses are retried up to
        `config.max_retries`. Invalid JSON or schema output is surfaced directly;
        there is no repair prompt, rule-based fallback, or source fabrication.
        """
        self.require_enabled()
        normalized_messages = _prepare_messages(messages, response_model)
        prompt_budget = _effective_budget(
            max_prompt_tokens, self.config.max_prompt_tokens, "prompt"
        )
        completion_budget = _effective_budget(
            max_completion_tokens, self.config.max_completion_tokens, "completion"
        )
        estimated_prompt_tokens = estimate_prompt_tokens(normalized_messages)
        if estimated_prompt_tokens > prompt_budget:
            logger.error(
                "LLM prompt rejected by token budget",
                extra={
                    "llm_agent": _safe_agent_name(agent_name),
                    "llm_estimated_prompt_tokens": estimated_prompt_tokens,
                    "llm_max_prompt_tokens": prompt_budget,
                },
            )
            raise LLMClientError(
                "prompt_budget_exceeded",
                "LLM prompt exceeds the configured token budget.",
            )
        endpoint = _chat_completions_url(self.config.base_url)
        headers = {
            "Authorization": f"Bearer {self.config.api_key.strip()}",
            "Content-Type": "application/json",
        }
        request_body = {
            "model": self.config.model.strip(),
            "messages": normalized_messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": completion_budget,
        }
        if self.config.thinking_mode is not None:
            request_body["thinking"] = {"type": self.config.thinking_mode}

        started_ns = time.monotonic_ns()
        retry_count = 0
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            transport=self._transport,
        ) as client:
            while True:
                try:
                    response = await client.post(
                        endpoint, headers=headers, json=request_body
                    )
                except httpx.TimeoutException:
                    if retry_count < self.config.max_retries:
                        await self._wait_before_retry(retry_count, agent_name)
                        retry_count += 1
                        continue
                    raise self._error(
                        "timeout",
                        "LLM API request timed out.",
                        retry_count,
                        started_ns,
                    ) from None
                except httpx.RequestError:
                    if retry_count < self.config.max_retries:
                        await self._wait_before_retry(retry_count, agent_name)
                        retry_count += 1
                        continue
                    raise self._error(
                        "connection_error",
                        "Could not connect to the configured LLM API.",
                        retry_count,
                        started_ns,
                    ) from None

                if response.status_code >= 400:
                    if (
                        response.status_code in _RETRYABLE_STATUS_CODES
                        and retry_count < self.config.max_retries
                    ):
                        await self._wait_before_retry(retry_count, agent_name)
                        retry_count += 1
                        continue
                    raise self._error(
                        "http_error",
                        _safe_http_message(response.status_code),
                        retry_count,
                        started_ns,
                        status_code=response.status_code,
                    ) from None

                response_metadata: dict[str, Any] = {}
                try:
                    body = response.json()
                    content, actual_model, request_id, usage, finish_reason = _extract_completion(body, response)
                    response_metadata = _usage_metadata(
                        actual_model,
                        request_id,
                        usage,
                        self.config.input_usd_per_million,
                        self.config.output_usd_per_million,
                    )
                    if finish_reason == "length":
                        raise self._error(
                            "completion_truncated",
                            "LLM API stopped at the completion token limit before returning complete JSON.",
                            retry_count,
                            started_ns,
                            status_code=response.status_code,
                            response_metadata=response_metadata,
                        )
                    decoded = json.loads(content)
                    if not isinstance(decoded, dict):
                        raise _InvalidCompletion("JSON response must be an object")
                    validated = _validate_output(decoded, response_model)
                except ValidationError:
                    raise self._error(
                        "schema_validation",
                        "LLM API returned JSON that does not match the required schema.",
                        retry_count,
                        started_ns,
                        status_code=response.status_code,
                        response_metadata=response_metadata,
                    ) from None
                except (ValueError, TypeError, KeyError, IndexError, _InvalidCompletion):
                    raise self._error(
                        "invalid_response",
                        "LLM API returned an invalid JSON completion.",
                        retry_count,
                        started_ns,
                        status_code=response.status_code,
                        response_metadata=response_metadata,
                    ) from None

                latency_ms = _elapsed_ms(started_ns)
                prompt_tokens = response_metadata["prompt_tokens"]
                completion_tokens = response_metadata["completion_tokens"]
                total_tokens = response_metadata["total_tokens"]
                cost = response_metadata["estimated_cost_usd"]
                result = LLMCompletion(
                    data=validated,
                    model=actual_model,
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    estimated_cost_usd=cost,
                    retry_count=retry_count,
                )
                logger.info(
                    "LLM call completed",
                    extra={
                        "llm_agent": _safe_agent_name(agent_name),
                        "llm_model": actual_model,
                        "llm_request_id": request_id,
                        "llm_latency_ms": latency_ms,
                        "llm_prompt_tokens": prompt_tokens,
                        "llm_estimated_prompt_tokens": estimated_prompt_tokens,
                        "llm_completion_tokens": completion_tokens,
                        "llm_total_tokens": total_tokens,
                        "llm_retry_count": retry_count,
                        "llm_estimated_cost_usd": cost,
                    },
                )
                return result

    async def complete_with_tools(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_prompt_tokens: int | None = None,
        max_completion_tokens: int | None = None,
    ) -> LLMToolCompletion:
        """Ask the model to choose from declared tools without executing any.

        Tool arguments are parsed as JSON objects here, then must be validated by
        the application before dispatch. Unknown tool names, malformed argument
        JSON, malformed call envelopes, and truncated completions fail closed.
        """
        self.require_enabled()
        normalized_messages = _prepare_tool_messages(messages)
        normalized_tools = _prepare_tools(tools)
        prompt_budget = _effective_budget(
            max_prompt_tokens, self.config.max_prompt_tokens, "prompt"
        )
        completion_budget = _effective_budget(
            max_completion_tokens, self.config.max_completion_tokens, "completion"
        )
        token_messages = [
            {"role": item["role"], "content": _message_for_token_estimate(item)}
            for item in normalized_messages
        ]
        token_messages[0]["content"] += "\n\nDeclared tools: " + json.dumps(
            normalized_tools, ensure_ascii=False, separators=(",", ":")
        )
        estimated_prompt_tokens = estimate_prompt_tokens(token_messages)
        if estimated_prompt_tokens > prompt_budget:
            logger.error(
                "LLM tool prompt rejected by token budget",
                extra={
                    "llm_agent": _safe_agent_name(agent_name),
                    "llm_estimated_prompt_tokens": estimated_prompt_tokens,
                    "llm_max_prompt_tokens": prompt_budget,
                },
            )
            raise LLMClientError(
                "prompt_budget_exceeded",
                "LLM tool prompt exceeds the configured token budget.",
            )

        endpoint = _chat_completions_url(self.config.base_url)
        headers = {
            "Authorization": f"Bearer {self.config.api_key.strip()}",
            "Content-Type": "application/json",
        }
        request_body = {
            "model": self.config.model.strip(),
            "messages": normalized_messages,
            "tools": normalized_tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": completion_budget,
        }
        if self.config.thinking_mode is not None:
            request_body["thinking"] = {"type": self.config.thinking_mode}

        started_ns = time.monotonic_ns()
        retry_count = 0
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            transport=self._transport,
        ) as client:
            while True:
                try:
                    response = await client.post(
                        endpoint, headers=headers, json=request_body
                    )
                except httpx.TimeoutException:
                    if retry_count < self.config.max_retries:
                        await self._wait_before_retry(retry_count, agent_name)
                        retry_count += 1
                        continue
                    raise self._error(
                        "timeout", "LLM API request timed out.", retry_count, started_ns
                    ) from None
                except httpx.RequestError:
                    if retry_count < self.config.max_retries:
                        await self._wait_before_retry(retry_count, agent_name)
                        retry_count += 1
                        continue
                    raise self._error(
                        "connection_error",
                        "Could not connect to the configured LLM API.",
                        retry_count,
                        started_ns,
                    ) from None

                if response.status_code >= 400:
                    if (
                        response.status_code in _RETRYABLE_STATUS_CODES
                        and retry_count < self.config.max_retries
                    ):
                        await self._wait_before_retry(retry_count, agent_name)
                        retry_count += 1
                        continue
                    raise self._error(
                        "http_error",
                        _safe_http_message(response.status_code),
                        retry_count,
                        started_ns,
                        status_code=response.status_code,
                    ) from None

                response_metadata: dict[str, Any] = {}
                try:
                    body = response.json()
                    content, calls, actual_model, request_id, usage, finish_reason = (
                        _extract_tool_completion(body, response, normalized_tools)
                    )
                    response_metadata = _usage_metadata(
                        actual_model,
                        request_id,
                        usage,
                        self.config.input_usd_per_million,
                        self.config.output_usd_per_million,
                    )
                    if finish_reason == "length":
                        raise self._error(
                            "completion_truncated",
                            "LLM API stopped at the completion token limit before completing the tool response.",
                            retry_count,
                            started_ns,
                            status_code=response.status_code,
                            response_metadata=response_metadata,
                            request_max_completion_tokens=completion_budget,
                        )
                except (ValueError, TypeError, KeyError, IndexError, _InvalidCompletion):
                    raise self._error(
                        "invalid_response",
                        "LLM API returned an invalid tool-call response.",
                        retry_count,
                        started_ns,
                        status_code=response.status_code,
                        response_metadata=response_metadata,
                    ) from None

                latency_ms = _elapsed_ms(started_ns)
                actual_model = _safe_model_name(actual_model)
                cost = response_metadata["estimated_cost_usd"]
                result = LLMToolCompletion(
                    content=content,
                    tool_calls=tuple(calls),
                    model=actual_model,
                    request_id=request_id,
                    prompt_tokens=response_metadata["prompt_tokens"],
                    completion_tokens=response_metadata["completion_tokens"],
                    total_tokens=response_metadata["total_tokens"],
                    latency_ms=latency_ms,
                    estimated_cost_usd=cost,
                    retry_count=retry_count,
                )
                logger.info(
                    "LLM tool call completed",
                    extra={
                        "llm_agent": _safe_agent_name(agent_name),
                        "llm_model": actual_model,
                        "llm_request_id": request_id,
                        "llm_latency_ms": latency_ms,
                        "llm_prompt_tokens": result.prompt_tokens,
                        "llm_estimated_prompt_tokens": estimated_prompt_tokens,
                        "llm_completion_tokens": result.completion_tokens,
                        "llm_total_tokens": result.total_tokens,
                        "llm_retry_count": retry_count,
                        "llm_estimated_cost_usd": cost,
                        "llm_tool_count": len(calls),
                        "llm_max_completion_tokens": completion_budget,
                    },
                )
                return result

    async def _wait_before_retry(self, retry_count: int, agent_name: str) -> None:
        logger.warning(
            "Retrying LLM API request",
            extra={
                "llm_agent": _safe_agent_name(agent_name),
                "llm_retry_count": retry_count + 1,
            },
        )
        delay = min(self.config.retry_backoff_seconds * (2**retry_count), 2.0)
        if delay:
            await asyncio.sleep(delay)

    def _error(
        self,
        code: str,
        message: str,
        retry_count: int,
        started_ns: int,
        *,
        status_code: int | None = None,
        response_metadata: dict[str, Any] | None = None,
        request_max_completion_tokens: int | None = None,
    ) -> LLMClientError:
        latency_ms = _elapsed_ms(started_ns)
        response_metadata = response_metadata or {}
        logger.error(
            "LLM call failed",
            extra={
                "llm_error_code": code,
                "llm_http_status": status_code,
                "llm_retry_count": retry_count,
                "llm_latency_ms": latency_ms,
                "llm_model": response_metadata.get("actual_model"),
                "llm_request_id": response_metadata.get("request_id"),
                "llm_prompt_tokens": response_metadata.get("prompt_tokens"),
                "llm_completion_tokens": response_metadata.get("completion_tokens"),
                "llm_total_tokens": response_metadata.get("total_tokens"),
                "llm_estimated_cost_usd": response_metadata.get("estimated_cost_usd"),
                "llm_max_completion_tokens": request_max_completion_tokens,
            },
        )
        return LLMClientError(
            code,
            message,
            status_code=status_code,
            retry_count=retry_count,
            latency_ms=latency_ms,
            **response_metadata,
        )


class _InvalidCompletion(Exception):
    pass


def _prepare_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise LLMClientError("invalid_request", "At least one LLM message is required.")
    prepared: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system", "user", "assistant", "tool"
        }:
            raise LLMClientError(
                "invalid_request", "LLM messages must contain supported roles and text content."
            )
        role = message["role"]
        content = message.get("content", "")
        if content is None and role == "assistant":
            content = ""
        if not isinstance(content, str):
            raise LLMClientError(
                "invalid_request", "LLM message content must be text."
            )
        normalized: dict[str, Any] = {"role": role, "content": content}
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise LLMClientError(
                    "invalid_request", "Tool result messages must include a tool_call_id."
                )
            normalized["tool_call_id"] = call_id[:128]
        if role == "assistant" and "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list) or not calls or len(calls) > 16:
                raise LLMClientError("invalid_request", "Assistant tool-call history is invalid.")
            normalized["tool_calls"] = calls
        prepared.append(normalized)
    return prepared


def _message_for_token_estimate(message: dict[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def _prepare_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(tools, list) or not tools or len(tools) > 16:
        raise LLMClientError("invalid_request", "At least one valid function tool is required.")
    prepared: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise LLMClientError("invalid_request", "Only declared function tools are supported.")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise LLMClientError("invalid_request", "Function tool definition is invalid.")
        name = function.get("name")
        parameters = function.get("parameters")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 64
            or not all(character.isalnum() or character in "_-" for character in name)
            or name in names
            or not isinstance(parameters, dict)
            or parameters.get("type") != "object"
        ):
            raise LLMClientError("invalid_request", "Function tool definition is invalid.")
        names.add(name)
        normalized_function: dict[str, Any] = {"name": name, "parameters": parameters}
        description = function.get("description")
        if isinstance(description, str):
            normalized_function["description"] = description[:1000]
        prepared.append({"type": "function", "function": normalized_function})
    return prepared


def _extract_tool_completion(
    body: Any,
    response: httpx.Response,
    tools: list[dict[str, Any]],
) -> tuple[
    str | None,
    list[LLMToolCall],
    str,
    str | None,
    dict[str, Any],
    str | None,
]:
    if not isinstance(body, dict):
        raise _InvalidCompletion()
    actual_model = body.get("model")
    request_id = body.get("id") or response.headers.get("x-request-id")
    choices = body.get("choices")
    if (
        not isinstance(actual_model, str)
        or not actual_model.strip()
        or not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
    ):
        raise _InvalidCompletion()
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
        raise _InvalidCompletion()
    finish_reason = _safe_finish_reason(choice.get("finish_reason"))
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise _InvalidCompletion()
    raw_calls = message.get("tool_calls", [])
    if not isinstance(raw_calls, list) or len(raw_calls) > 16:
        raise _InvalidCompletion()
    declared_names = {
        item["function"]["name"] for item in tools if isinstance(item.get("function"), dict)
    }
    parsed_calls: list[LLMToolCall] = []
    seen_ids: set[str] = set()
    for item in raw_calls:
        if not isinstance(item, dict) or item.get("type") != "function":
            raise _InvalidCompletion()
        raw_id = item.get("id")
        call_id = _safe_request_id(raw_id)
        function = item.get("function")
        if not call_id or call_id in seen_ids or not isinstance(function, dict):
            raise _InvalidCompletion()
        name = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(name, str) or name not in declared_names:
            raise _InvalidCompletion()
        if not isinstance(raw_arguments, str) or len(raw_arguments) > 20_000:
            raise _InvalidCompletion()
        try:
            arguments = json.loads(raw_arguments)
        except (ValueError, TypeError):
            raise _InvalidCompletion() from None
        if not isinstance(arguments, dict):
            raise _InvalidCompletion()
        seen_ids.add(call_id)
        parsed_calls.append(LLMToolCall(call_id, name, arguments))
    if finish_reason == "length":
        pass  # The caller reports truncation with usage metadata.
    elif finish_reason not in {"stop", "tool_calls", None}:
        raise _InvalidCompletion()
    elif finish_reason == "tool_calls" and not parsed_calls:
        raise _InvalidCompletion()
    elif parsed_calls and content is None:
        content = ""
    elif not parsed_calls and content is None:
        raise _InvalidCompletion()
    usage = body.get("usage")
    return (
        content,
        parsed_calls,
        actual_model,
        _safe_request_id(request_id),
        usage if isinstance(usage, dict) else {},
        finish_reason,
    )


def _prepare_messages(
    messages: list[dict[str, str]], response_model: type[BaseModel] | dict[str, Any]
) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise LLMClientError("invalid_request", "At least one LLM message is required.")
    prepared: list[dict[str, str]] = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            raise LLMClientError(
                "invalid_request", "LLM messages must contain supported roles and text content."
            )
        prepared.append({"role": message["role"], "content": message["content"]})

    schema = _response_schema(response_model)
    instruction = (
        "Return exactly one JSON object. The JSON object must follow this schema: "
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )
    if prepared[0]["role"] == "system":
        prepared[0] = {
            "role": "system",
            "content": prepared[0]["content"] + "\n\n" + instruction,
        }
    else:
        prepared.insert(0, {"role": "system", "content": instruction})
    return prepared


def _response_schema(response_model: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    if isinstance(response_model, type) and issubclass(response_model, BaseModel):
        return response_model.model_json_schema()
    if isinstance(response_model, dict):
        return response_model
    raise LLMClientError(
        "invalid_request", "response_model must be a Pydantic model or JSON Schema object."
    )


def _validate_output(
    payload: dict[str, Any], response_model: type[BaseModel] | dict[str, Any]
) -> Any:
    if isinstance(response_model, type) and issubclass(response_model, BaseModel):
        return response_model.model_validate(payload)
    schema = _response_schema(response_model)
    _validate_json_schema(payload, schema, schema, "$" )
    return payload


def _validate_json_schema(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str,
) -> None:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        target: Any = root_schema
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        _validate_json_schema(value, target, root_schema, path)
        return

    if "anyOf" in schema:
        for candidate in schema["anyOf"]:
            try:
                _validate_json_schema(value, candidate, root_schema, path)
                return
            except (ValueError, KeyError, TypeError):
                continue
        raise ValueError(f"{path} does not match any allowed schema")

    expected_type = schema.get("type")
    allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
    if expected_type is not None and not any(
        _matches_json_type(value, type_name) for type_name in allowed_types
    ):
        raise ValueError(f"{path} has an invalid type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is outside its enum")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} has an invalid constant")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        for key, item in value.items():
            if key in properties:
                _validate_json_schema(item, properties[key], root_schema, f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ValueError(f"{path}.{key} is not allowed")
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate_json_schema(
                    item, schema["additionalProperties"], root_schema, f"{path}.{key}"
                )
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_json_schema(item, schema["items"], root_schema, f"{path}[{index}]")


def _matches_json_type(value: Any, type_name: Any) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return False


def _extract_completion(
    body: Any, response: httpx.Response
) -> tuple[str, str, str | None, dict[str, Any], str | None]:
    if not isinstance(body, dict):
        raise _InvalidCompletion()
    actual_model = body.get("model")
    request_id = body.get("id") or response.headers.get("x-request-id")
    choices = body.get("choices")
    if not isinstance(actual_model, str) or not actual_model.strip():
        raise _InvalidCompletion()
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _InvalidCompletion()
    choice = choices[0]
    message = choice.get("message")
    finish_reason = choice.get("finish_reason")
    if not isinstance(message, dict):
        raise _InvalidCompletion()
    content = message.get("content")
    if not isinstance(content, str):
        # Some compatible APIs omit the partial text when a length cap fires;
        # preserve usage/request metadata and report truncation below.
        if finish_reason != "length":
            raise _InvalidCompletion()
        content = ""
    usage = body.get("usage")
    return (
        content,
        actual_model,
        _safe_request_id(request_id),
        usage if isinstance(usage, dict) else {},
        _safe_finish_reason(finish_reason),
    )


def _usage_int(usage: dict[str, Any], field: str) -> int | None:
    value = usage.get(field)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _usage_metadata(
    actual_model: str,
    request_id: str | None,
    usage: dict[str, Any],
    input_rate: float | None,
    output_rate: float | None,
) -> dict[str, Any]:
    prompt_tokens = _usage_int(usage, "prompt_tokens")
    completion_tokens = _usage_int(usage, "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "actual_model": _safe_model_name(actual_model),
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": _estimate_cost(
            prompt_tokens, completion_tokens, input_rate, output_rate
        ),
    }


def _safe_request_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    safe = "".join(
        character
        for character in value
        if character.isalnum() or character in "-_.:"
    )[:128]
    return safe or None


def _safe_model_name(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:/"
    safe = "".join(character for character in value if character in allowed)
    return safe[:128] or "unknown"


def _safe_finish_reason(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    safe = "".join(character for character in value if character.isalnum() or character in "-_")
    return safe[:32] or None


def estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """Return a conservative, tokenizer-independent prompt budget estimate.

    Provider usage remains the source of truth. This estimate is only used to
    reject prompts that exceed a configured safety budget before transmission.
    Counting both Unicode characters and UTF-8 bytes/3 is intentionally
    conservative for mixed Chinese/Latin JSON prompts.
    """
    content = "\n".join(message.get("content", "") for message in messages)
    encoded_bytes = len(content.encode("utf-8"))
    base = max(len(content), (encoded_bytes + 2) // 3)
    return (base * 6 + 4) // 5  # 20% headroom for tokenizer variation.


def estimate_request_prompt_tokens(
    messages: list[dict[str, str]], response_model: type[BaseModel] | dict[str, Any]
) -> int:
    """Estimate the exact normalized prompt shape, including the output schema."""
    return estimate_prompt_tokens(_prepare_messages(messages, response_model))


def _effective_budget(requested: int | None, configured: int, kind: str) -> int:
    if requested is None:
        return configured
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise LLMClientError(
            "invalid_request", f"LLM {kind} token budget must be a positive integer."
        )
    return min(requested, configured)


def _estimate_cost(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    input_rate: float | None,
    output_rate: float | None,
) -> float | None:
    if (
        prompt_tokens is None
        or completion_tokens is None
        or input_rate is None
        or output_rate is None
    ):
        return None
    return round(
        (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000,
        10,
    )


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise LLMClientError("invalid_configuration", "LLM_BASE_URL is empty.")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _safe_http_message(status_code: int) -> str:
    if status_code == 401 or status_code == 403:
        return "LLM API rejected the configured credentials or request permissions."
    if status_code == 429:
        return "LLM API rate limit was exceeded."
    if status_code >= 500:
        return "LLM API returned a server error."
    return "LLM API rejected the request."


def _safe_agent_name(agent_name: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    sanitized = "".join(character for character in str(agent_name) if character in allowed)
    return sanitized[:64] or "unknown"


def _elapsed_ms(started_ns: int) -> int:
    return max(0, int((time.monotonic_ns() - started_ns) / 1_000_000))
