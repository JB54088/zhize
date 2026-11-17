import json
import logging
import unittest

import httpx
from pydantic import BaseModel

from app.llm import LLMClient, LLMClientError, LLMConfig


class Decision(BaseModel):
    status: str
    reasons: list[str]


def _api_response(
    content: str,
    *,
    model: str = "deepseek-flash-2026-09-10",
    usage: dict | None = None,
    request_id: str = "req-safe-test-id",
    status_code: int = 200,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "id": request_id,
            "model": model,
            "choices": [{"message": {"content": content}}],
            "usage": usage
            or {"prompt_tokens": 21, "completion_tokens": 8, "total_tokens": 29},
        },
        headers={"x-request-id": request_id},
    )


class LLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_json_returns_validated_data_and_provider_metadata(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("authorization")
            captured["payload"] = json.loads(request.content)
            return _api_response(
                '{"status":"needs_verification","reasons":["source quote absent"]}'
            )

        client = LLMClient(
            LLMConfig(
                api_key="secret-test-key",
                base_url="https://api.deepseek.com",
                model="deepseek-flash",
                thinking_mode="disabled",
                input_usd_per_million=0.2,
                output_usd_per_million=0.4,
            ),
            transport=httpx.MockTransport(handler),
        )
        result = await client.complete_json(
            "judge",
            [{"role": "user", "content": "Judge the supplied evidence."}],
            Decision,
        )

        self.assertEqual(
            result.data,
            Decision(status="needs_verification", reasons=["source quote absent"]),
        )
        self.assertEqual(result.model, "deepseek-flash-2026-09-10")
        self.assertEqual(result.request_id, "req-safe-test-id")
        self.assertEqual(
            (result.prompt_tokens, result.completion_tokens, result.total_tokens),
            (21, 8, 29),
        )
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(result.retry_count, 0)
        self.assertAlmostEqual(
            result.estimated_cost_usd,
            (21 * 0.2 + 8 * 0.4) / 1_000_000,
        )
        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["auth"], "Bearer secret-test-key")
        self.assertEqual(
            captured["payload"]["response_format"], {"type": "json_object"}
        )
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertIn("schema", captured["payload"]["messages"][0]["content"])

    async def test_json_schema_dict_is_validated_and_missing_usage_cost_is_unknown(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": False,
        }
        client = LLMClient(
            LLMConfig(api_key="key", model="m"),
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "id": "req-2",
                        "model": "actual-compatible-model",
                        "choices": [{"message": {"content": '{"count":2}'}}],
                    },
                )
            ),
        )
        result = await client.complete_json(
            "clean", [{"role": "user", "content": "Extract."}], schema
        )
        self.assertEqual(result.data, {"count": 2})
        self.assertEqual(result.model, "actual-compatible-model")
        self.assertIsNone(result.total_tokens)
        self.assertIsNone(result.estimated_cost_usd)

    async def test_disabled_client_is_explicit_and_does_not_call_transport(self):
        called = False

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return _api_response("{}")

        client = LLMClient(
            LLMConfig(api_key="", model=""), transport=httpx.MockTransport(handler)
        )
        self.assertFalse(client.is_enabled)
        with self.assertRaises(LLMClientError) as context:
            await client.complete_json(
                "intent", [{"role": "user", "content": "x"}], Decision
            )
        self.assertEqual(context.exception.code, "not_configured")
        self.assertFalse(called)

    async def test_configured_http_failure_has_sanitized_error_and_bounded_retry(self):
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                429,
                json={"error": {"message": "should never be copied into logs or errors"}},
            )

        client = LLMClient(
            LLMConfig(
                api_key="secret-never-echo",
                model="m",
                max_retries=2,
                retry_backoff_seconds=0,
            ),
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(LLMClientError) as context:
            await client.complete_json(
                "verify", [{"role": "user", "content": "secret prompt"}], Decision
            )
        error = context.exception
        self.assertEqual(calls, 3)
        self.assertEqual(error.code, "http_error")
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.retry_count, 2)
        self.assertNotIn("should never", str(error))
        self.assertNotIn("secret-never-echo", str(error))
        self.assertNotIn("secret prompt", str(error))

    async def test_nonretryable_http_failure_does_not_retry(self):
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                401, json={"error": {"message": "sensitive provider detail"}}
            )

        client = LLMClient(
            LLMConfig(
                api_key="secret", model="m", max_retries=3, retry_backoff_seconds=0
            ),
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(LLMClientError) as context:
            await client.complete_json(
                "collect", [{"role": "user", "content": "x"}], Decision
            )
        self.assertEqual(calls, 1)
        self.assertEqual(context.exception.code, "http_error")
        self.assertNotIn("sensitive provider detail", str(context.exception))

    async def test_invalid_or_schema_mismatched_completion_is_not_retried(self):
        responses = [
            ("not json", "invalid_response"),
            ('{"status":3,"reasons":[]}', "schema_validation"),
        ]
        for content, expected_code in responses:
            with self.subTest(expected_code=expected_code):
                calls = 0

                async def handler(request: httpx.Request) -> httpx.Response:
                    nonlocal calls
                    calls += 1
                    return _api_response(content)

                client = LLMClient(
                    LLMConfig(
                        api_key="key",
                        model="m",
                        max_retries=3,
                        retry_backoff_seconds=0,
                    ),
                    transport=httpx.MockTransport(handler),
                )
                with self.assertRaises(LLMClientError) as context:
                    await client.complete_json(
                        "answer", [{"role": "user", "content": "x"}], Decision
                    )
                self.assertEqual(calls, 1)
                self.assertEqual(context.exception.code, expected_code)

    async def test_success_log_excludes_prompt_completion_and_api_key(self):
        client = LLMClient(
            LLMConfig(
                api_key="do-not-log-this-key", model="m", retry_backoff_seconds=0
            ),
            transport=httpx.MockTransport(
                lambda request: _api_response('{"status":"ok","reasons":[]}')
            ),
        )
        with self.assertLogs("app.llm.client", level=logging.INFO) as captured:
            await client.complete_json(
                "answer",
                [{"role": "user", "content": "private user query"}],
                Decision,
            )
        logs = " ".join(captured.output)
        self.assertNotIn("private user query", logs)
        self.assertNotIn("do-not-log-this-key", logs)
        self.assertIn("LLM call completed", logs)


if __name__ == "__main__":
    unittest.main()
