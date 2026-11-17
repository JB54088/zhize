"""Regression tests for bounded, evidence-preserving clean-agent LLM prompts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.clean import (
    _MAX_BATCH_PROMPT_ESTIMATED_TOKENS,
    _MAX_CLEAN_COMPLETION_TOKENS,
    _MAX_SOURCE_CHARS,
    _clean_completion_budget,
    CleanAgent,
)
from app.agents.models import ExtractionResponse
from app.llm.client import (
    LLMClient,
    LLMClientError,
    LLMConfig,
    estimate_request_prompt_tokens,
)


class CapturingFakeClient:
    is_enabled = True

    def __init__(self):
        self.calls = []

    async def complete_json(self, *, agent_name, messages, response_model):
        self.calls.append({"agent_name": agent_name, "messages": messages})
        return SimpleNamespace(
            data=response_model.model_validate({"items": []}),
            model="deepseek-test-model",
            request_id="clean-budget-test",
            prompt_tokens=321,
            completion_tokens=0,
            total_tokens=321,
            latency_ms=17,
            estimated_cost_usd=None,
            retry_count=0,
        )


def _raw_records(count=7):
    records = []
    for index in range(count):
        body = (
            f"2027届招聘公告 {index}。专业要求电子信息工程。"
            + ("与招聘无关的历史介绍和公司宣传内容。" * 400)
            + f"岗位职责为嵌入式开发工程师 {index}。"
            + f"申请截止日期：2026年12月31日。"
        )
        records.append({
            "title": f"示例企业 {index} 2027校园招聘岗位",
            "url": f"https://jobs.example.test/campus/{index}",
            "content": body,
            "raw_content": body,
        })
    return records


def test_clean_agent_sends_compact_missing_field_excerpts_and_preserves_full_evidence():
    async def scenario():
        raw = _raw_records()
        client = CapturingFakeClient()

        execution = await CleanAgent().run(
            "2027届电子信息工程校园招聘", raw, 10, llm_client=client
        )

        assert execution.run_log["status"] == "completed"
        assert execution.run_log["prompt_tokens"] == 321
        assert execution.run_log["duration_ms"] >= 0
        assert len(execution.output["records"]) == len(raw)
        assert all(
            len(result["raw_text"]) == len(source["raw_content"])
            for result, source in zip(execution.output["records"], raw)
        )

        submitted = json.loads(client.calls[0]["messages"][-1]["content"])
        assert len(submitted["sources"]) == 3
        assert all(len(source["source_text"]) <= _MAX_SOURCE_CHARS for source in submitted["sources"])
        assert all(len(source["missing_fields"]) <= 1 for source in submitted["sources"])
        assert set(submitted["sources"][0]["already_extracted_fields"]) >= {
            "graduate_cohort", "majors", "role", "deadline"
        }
        assert "majors" not in submitted["sources"][0]["missing_fields"]
        assert "graduate_cohort" not in submitted["sources"][0]["missing_fields"]
        assert estimate_request_prompt_tokens(
            client.calls[0]["messages"], ExtractionResponse
        ) <= _MAX_BATCH_PROMPT_ESTIMATED_TOKENS

        # Earlier implementation sent up to 6,000 body characters/source.
        # This fixture reproduces the old upper-bound payload and proves the
        # optimized body portion shrinks while source originals remain intact.
        old_body_chars = sum(min(len(item["raw_content"]), 6_000) for item in raw[:3])
        new_body_chars = sum(len(item["source_text"]) for item in submitted["sources"])
        assert new_body_chars <= 5 * _MAX_SOURCE_CHARS
        assert new_body_chars < old_body_chars // 5

    asyncio.run(scenario())


def test_clean_agent_accepts_only_exact_quote_from_original_full_source():
    async def scenario():
        original = "2027届电子信息工程校招。此公告由示例科技发布，负责嵌入式开发岗位招聘。" + ("背景介绍。" * 500)
        raw = [{
            "title": "2027届招聘信息",
            "url": "https://jobs.example.test/notice",
            "content": original,
            "raw_content": original,
        }]

        class QuoteClient(CapturingFakeClient):
            async def complete_json(self, *, agent_name, messages, response_model):
                request = json.loads(messages[-1]["content"])
                source = request["sources"][0]
                assert "organization" in source["missing_fields"]
                return SimpleNamespace(
                    data=response_model.model_validate({"items": [{
                        "id": source["id"],
                        "fields": [
                            {"field": "application_url", "value": "https://fake.test/apply", "quote": "https://fake.test/apply"},
                        ],
                    }]}),
                    model="deepseek-test-model", request_id="quote-check", prompt_tokens=20,
                    completion_tokens=10, total_tokens=30, latency_ms=1,
                    estimated_cost_usd=None, retry_count=0,
                )

        execution = await CleanAgent().run("2027届电子信息工程校招", raw, 3, llm_client=QuoteClient())
        record = execution.output["records"][0]
        assert "organization" not in record["extracted_fields"]
        assert "application_url" not in record["extracted_fields"]
        assert record["raw_text"] == original

    asyncio.run(scenario())


def test_clean_completion_budget_scales_with_extraction_scope_and_has_hard_cap():
    one_field = _clean_completion_budget([{"missing_fields": ["role"]}])
    six_fields = _clean_completion_budget([{
        "missing_fields": ["organization", "role", "graduate_cohort", "majors", "deadline", "application_url"]
    }])
    maximum_scope = _clean_completion_budget([
        {"missing_fields": ["organization", "role", "graduate_cohort"]}
        for _ in range(3)
    ])

    assert one_field == six_fields < maximum_scope
    assert maximum_scope == 2_112
    assert maximum_scope < _MAX_CLEAN_COMPLETION_TOKENS == 4_096


def test_client_enforces_prompt_budget_before_network_and_caps_completion_tokens():
    async def scenario():
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            body = json.loads(request.content)
            assert body["max_tokens"] == 300
            return httpx.Response(200, json={
                "id": "budget-ok", "model": "deepseek-budget-test",
                "choices": [{"message": {"content": '{"items": []}'}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 0, "total_tokens": 30},
            })

        client = LLMClient(
            LLMConfig(api_key="test-key", model="test-model", max_prompt_tokens=5_000, max_completion_tokens=900),
            transport=httpx.MockTransport(handler),
        )
        messages = [{"role": "user", "content": "short source passage"}]
        result = await client.complete_json(
            "clean", messages, ExtractionResponse,
            max_prompt_tokens=4_000, max_completion_tokens=300,
        )
        assert result.prompt_tokens == 30
        assert calls == 1

        rejected = LLMClient(
            LLMConfig(api_key="test-key", model="test-model", max_prompt_tokens=100),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(LLMClientError) as error:
            await rejected.complete_json("clean", messages, ExtractionResponse)
        assert error.value.code == "prompt_budget_exceeded"
        assert calls == 1

    asyncio.run(scenario())


def test_clean_uses_field_scaled_output_budget_and_keeps_provider_usage_metadata():
    async def scenario():
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={
                "id": "clean-real-client-test", "model": "deepseek-test-model",
                "choices": [{"message": {"content": '{"items": []}'}}],
                "usage": {"prompt_tokens": 432, "completion_tokens": 12, "total_tokens": 444},
            })

        client = LLMClient(
            LLMConfig(
                api_key="test-key", model="deepseek-test-model",
                max_prompt_tokens=20_000, max_completion_tokens=10_000,
            ),
            transport=httpx.MockTransport(handler),
        )
        execution = await CleanAgent().run(
            "2027届电子信息工程校招", _raw_records(1), 2, llm_client=client
        )

        original_messages = captured["payload"]["messages"]
        request = json.loads(original_messages[-1]["content"])
        expected_budget = _clean_completion_budget(request["sources"])
        assert captured["payload"]["max_tokens"] == expected_budget
        assert 512 <= expected_budget <= _MAX_CLEAN_COMPLETION_TOKENS
        assert execution.run_log["actual_model"] == "deepseek-test-model"
        assert execution.run_log["prompt_tokens"] == 432
        assert execution.run_log["completion_tokens"] == 12
        assert execution.run_log["total_tokens"] == 444
        assert execution.run_log["latency_ms"] >= 0

    asyncio.run(scenario())


def test_length_finish_reason_surfaces_sanitized_usage_and_request_metadata(caplog):
    async def scenario():
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json={
                "id": "truncated-request-42",
                "model": "deepseek-test-model",
                "choices": [{
                    "message": {"content": '{"items":[{"id":"partial-secret-source"'},
                    "finish_reason": "length",
                }],
                "usage": {"prompt_tokens": 111, "completion_tokens": 300, "total_tokens": 411},
            })

        client = LLMClient(
            LLMConfig(
                api_key="test-secret-key",
                model="deepseek-test-model",
                max_retries=0,
                input_usd_per_million=0.3,
                output_usd_per_million=1.2,
            ),
            transport=httpx.MockTransport(handler),
        )
        messages = [{"role": "user", "content": "private source text must not enter logs"}]
        with pytest.raises(LLMClientError) as caught:
            await client.complete_json("clean", messages, ExtractionResponse, max_completion_tokens=300)

        error = caught.value
        assert error.code == "completion_truncated"
        assert error.actual_model == "deepseek-test-model"
        assert error.request_id == "truncated-request-42"
        assert (error.prompt_tokens, error.completion_tokens, error.total_tokens) == (111, 300, 411)
        assert error.estimated_cost_usd == pytest.approx((111 * 0.3 + 300 * 1.2) / 1_000_000)
        assert error.latency_ms is not None
        assert str(error) == "LLM API stopped at the completion token limit before returning complete JSON."
        assert captured["authorization"] == "Bearer test-secret-key"
        assert "private source text" not in caplog.text
        assert "partial-secret-source" not in caplog.text
        assert "test-secret-key" not in caplog.text
        record = next(item for item in caplog.records if item.getMessage() == "LLM call failed")
        assert record.llm_error_code == "completion_truncated"
        assert record.llm_request_id == "truncated-request-42"
        assert record.llm_prompt_tokens == 111
        assert record.llm_completion_tokens == 300
        assert record.llm_total_tokens == 411
        assert record.llm_estimated_cost_usd == pytest.approx(error.estimated_cost_usd)

    import logging
    with caplog.at_level(logging.ERROR, logger="app.llm.client"):
        asyncio.run(scenario())


def test_clean_run_log_retains_usage_when_model_response_is_truncated():
    async def scenario():
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={
                "id": "clean-truncated-42",
                "model": "deepseek-test-model",
                "choices": [{"message": {"content": '{"items":[{'}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 2048, "total_tokens": 2548},
            })

        client = LLMClient(
            LLMConfig(
                api_key="test-secret-key", model="deepseek-test-model", max_retries=0,
                input_usd_per_million=0.3, output_usd_per_million=1.2,
            ),
            transport=httpx.MockTransport(handler),
        )
        execution = await CleanAgent().run(
            "2027届电子信息工程校招", _raw_records(1), 2, llm_client=client
        )

        expected = _clean_completion_budget(
            json.loads(captured["payload"]["messages"][-1]["content"])["sources"]
        )
        assert captured["payload"]["max_tokens"] == expected
        assert execution.run_log["status"] == "failed"
        assert execution.run_log["error_code"] == "completion_truncated"
        assert execution.run_log["actual_model"] == "deepseek-test-model"
        assert execution.run_log["request_id"] == "clean-truncated-42"
        assert execution.run_log["prompt_tokens"] == 500
        assert execution.run_log["completion_tokens"] == 2048
        assert execution.run_log["total_tokens"] == 2548
        assert execution.run_log["estimated_cost_usd"] == pytest.approx(
            (500 * 0.3 + 2048 * 1.2) / 1_000_000
        )

    asyncio.run(scenario())
