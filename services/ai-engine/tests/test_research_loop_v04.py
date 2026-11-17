from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from app.llm.client import LLMClient, LLMClientError, LLMConfig
from app.research.config import ResearchLoopConfig, research_budget_from_env
from app.research.contracts import ResearchBudget
from app.research.loop import ResearchLoop, _source_summary
from app.search_tools import SearchToolbox


class FakeLLM:
    is_enabled = True

    def __init__(self, completions):
        self.completions = list(completions)
        self.requests = []

    async def complete_with_tools(self, **kwargs):
        self.requests.append(kwargs)
        if not self.completions:
            raise AssertionError("unexpected model call")
        value = self.completions.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _completion(tool_calls=(), *, total_tokens=25, model="deepseek-flash-2026-09-10"):
    return SimpleNamespace(
        content=None,
        tool_calls=tuple(tool_calls),
        model=model,
        request_id="req-v04-test",
        prompt_tokens=20,
        completion_tokens=5,
        total_tokens=total_tokens,
        latency_ms=10,
    )


def _tool(name, arguments, call_id="call_1"):
    return SimpleNamespace(id=call_id, name=name, arguments=arguments)


class ResearchLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_round_loop_emits_real_tool_and_round_events(self):
        llm = FakeLLM([
            _completion([_tool("search_web", {"query": "Example role official careers", "limit": 4})]),
            _completion([]),
        ])
        dispatched = []
        refreshed = []
        events = []

        async def dispatch(name, arguments):
            dispatched.append((name, arguments))
            return {
                "records": [{
                    "title": "Example embedded systems role",
                    "url": "https://jobs.example.org/roles/17",
                    "snippet": "Applications close 30 November 2026.",
                    "provider": "tavily",
                }],
                "provider": "tavily",
                "source_count": 1,
                "search_requests": 1,
                "providers": [
                    {"provider": "brave", "configured": False, "requests": 0,
                     "returned_sources": 0, "failures": 0, "latency_ms": 0},
                    {"provider": "tavily", "configured": True, "requests": 1,
                     "returned_sources": 1, "failures": 0, "latency_ms": 23},
                ],
                "evidence_gaps": ["graduate cohort not confirmed"],
            }

        async def refresh(records):
            refreshed.append(list(records))
            return {
                "candidates": [{"id": "role-17", "title": "Example role", "url": records[0]["url"]}],
                "evidence_gaps": ["graduate cohort not confirmed"],
            }

        loop = ResearchLoop(
            llm,
            ResearchLoopConfig(
                budget=ResearchBudget(max_rounds=3, max_search_requests=4, max_model_calls=4,
                                      max_total_tokens=5_000, max_candidates=10,
                                      max_elapsed_seconds=5, max_results_per_query=5),
                max_prompt_tokens=4_000,
                max_completion_tokens=300,
            ),
        )
        outcome, records = await loop.run(
            query="Find an active graduate role",
            initial_search_queries=["graduate role Example"],
            candidates=[],
            evidence_gaps=["no official role page"],
            tool_dispatcher=dispatch,
            on_event=events.append,
            refresh_context=refresh,
        )

        self.assertEqual(outcome.status, "partial")
        self.assertEqual(outcome.model_calls, 2)
        self.assertEqual(outcome.total_tokens, 50)
        self.assertEqual(len(records), 1)
        self.assertEqual([event["type"] for event in events], [
            "tool_call", "research_round", "research_round"
        ])
        self.assertEqual(outcome.rounds[0].new_sources, 1)
        self.assertIn("继续", outcome.rounds[0].reason)
        self.assertEqual(dispatched[0][1]["_max_requests"], 4)
        self.assertEqual(len(refreshed), 1)
        provider_metrics = {item.provider: item for item in outcome.providers}
        self.assertEqual(provider_metrics["tavily"].requests, 1)
        self.assertEqual(provider_metrics["brave"].requests, 0)
        self.assertEqual(outcome.tool_calls[0].model, "deepseek-flash-2026-09-10")
        self.assertEqual(outcome.tool_calls[0].provider, "tavily")

    async def test_compacted_evidence_context_still_allows_followup_planning(self):
        llm = FakeLLM([
            _completion([_tool("read_webpage", {"url": "https://jobs.example.org/roles/1"})]),
            _completion([]),
        ])
        candidate = {"id": "role-1", "title": "Example role", "url": "https://jobs.example.org/roles/1"}

        async def dispatch(name, arguments):
            return {
                "evidence": {
                    "url": arguments["url"],
                    "title": "Official role details",
                    "text": "公开招聘正文。" * 2_000,
                    "claims": [
                        {"field": f"field-{index}", "value": "2027届", "quote": "公开原文证据。" * 30,
                         "url": arguments["url"], "status": "unverified"}
                        for index in range(20)
                    ],
                },
                "source_count": 1,
                "evidence_gaps": ["开放状态仍未核实"],
            }

        loop = ResearchLoop(llm, ResearchLoopConfig(
            budget=ResearchBudget(max_rounds=2, max_search_requests=2, max_model_calls=2,
                                  max_total_tokens=5_000, max_candidates=10,
                                  max_elapsed_seconds=5, max_results_per_query=4),
            max_prompt_tokens=8_000,
            max_completion_tokens=300,
        ))
        outcome, _ = await loop.run(
            query="寻找在招校招岗位",
            initial_search_queries=[],
            candidates=[candidate],
            evidence_gaps=["开放状态尚未核实"],
            tool_dispatcher=dispatch,
            refresh_context=lambda _: {"candidates": [candidate], "evidence_gaps": ["开放状态尚未核实"]},
        )

        self.assertEqual(outcome.model_calls, 2)
        self.assertEqual(len(llm.requests), 2)
        self.assertIn("开放状态尚未核实", json.dumps(llm.requests[1]["messages"], ensure_ascii=False))

    def test_source_summary_bounds_long_bodies_and_claim_lists(self):
        summary = _source_summary({
            "url": "https://jobs.example.org/roles/1",
            "content": "A" * 2_000,
            "text": "B" * 2_000,
            "evidence": {"status": "available", "claims": [
                {"field": "graduate_cohort", "value": "2027届", "quote": "C" * 1_000,
                 "url": "https://jobs.example.org/roles/1", "status": "verified"}
                for _ in range(20)
            ]},
        }, max_text_chars=180)

        self.assertLessEqual(len(summary["content"]), 180)
        self.assertNotIn("text", summary)
        self.assertEqual(len(summary["evidence"]["claims"]), 2)
        self.assertLessEqual(len(summary["evidence"]["claims"][0]["quote"]), 180)

    async def test_known_url_can_add_new_evidence_without_double_counting_source(self):
        llm = FakeLLM([_completion([
            _tool("read_webpage", {"url": "https://jobs.example.org/roles/17"})
        ])])
        initial = [{"id": "role-17", "url": "https://jobs.example.org/roles/17", "title": "Example role"}]
        dispatched = []

        async def dispatch(name, arguments):
            dispatched.append((name, arguments))
            return {
                "evidence": [{
                    "field": "graduate_cohort",
                    "value": "2027届",
                    "quote": "面向2027届毕业生招聘。",
                    "evidence_url": "https://jobs.example.org/roles/17",
                }],
                "source_count": 1,
                "evidence_gaps": [],
            }

        loop = ResearchLoop(llm, ResearchLoopConfig(
            budget=ResearchBudget(max_rounds=1, max_search_requests=2, max_model_calls=2,
                                  max_total_tokens=5_000, max_candidates=10,
                                  max_elapsed_seconds=5, max_results_per_query=4),
        ))
        outcome, records = await loop.run(
            query="Find active graduate roles",
            initial_search_queries=[],
            candidates=initial,
            evidence_gaps=["graduate cohort"],
            tool_dispatcher=dispatch,
            refresh_context=lambda _: {"candidates": initial, "evidence_gaps": []},
        )
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(outcome.rounds[0].new_sources, 0)
        self.assertEqual(outcome.rounds[0].status, "completed")
        self.assertEqual(outcome.status, "completed")

    async def test_search_request_budget_is_passed_and_second_search_rejected(self):
        llm = FakeLLM([_completion([
            _tool("search_web", {"query": "specific role search", "limit": 5}, "call_1"),
            _tool("search_web", {"query": "official second query", "limit": 5}, "call_2"),
        ])])
        dispatched = []

        async def dispatch(name, arguments):
            dispatched.append(arguments)
            return {
                "records": [{"title": "role", "url": "https://jobs.example.org/role/1", "snippet": "Role text."}],
                "source_count": 1,
                "search_requests": 1,
                "providers": [{"provider": "tavily", "configured": True, "requests": 1,
                               "returned_sources": 1, "failures": 0, "latency_ms": 5}],
            }

        loop = ResearchLoop(llm, ResearchLoopConfig(
            budget=ResearchBudget(max_rounds=2, max_search_requests=1, max_model_calls=2,
                                  max_total_tokens=5_000, max_candidates=10,
                                  max_elapsed_seconds=5, max_results_per_query=5),
        ))
        outcome, _ = await loop.run(
            query="Find roles", initial_search_queries=[], candidates=[], evidence_gaps=["official source"],
            tool_dispatcher=dispatch,
        )
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0]["_max_requests"], 1)
        self.assertEqual(outcome.rounds[0].search_requests, 1)
        self.assertEqual(outcome.tool_calls[1].status, "rejected")
        self.assertEqual(outcome.tool_calls[1].error_code, "search_budget_exhausted")

    async def test_inspect_evidence_claims_are_preserved_as_traceable_new_evidence(self):
        candidate = {
            "id": "role-17",
            "url": "https://jobs.example.org/roles/17",
            "title": "Example role",
        }
        llm = FakeLLM([_completion([
            _tool("inspect_evidence", {"candidate_id": "role-17", "field": "major"})
        ])])

        async def dispatch(name, arguments):
            return {
                "records": [],
                "evidence": {
                    "candidate_id": "role-17",
                    "field": "major",
                    "status": "available",
                    "claims": [{
                        "field": "major",
                        "value": "電子信息工程",
                        "quote": "電子信息工程及相關專業優先。",
                        "url": "https://jobs.example.org/roles/17",
                        "status": "unverified",
                    }],
                },
                "source_count": 1,
                "evidence_gaps": [],
            }

        loop = ResearchLoop(llm, ResearchLoopConfig(
            budget=ResearchBudget(max_rounds=1, max_search_requests=2, max_model_calls=2,
                                  max_total_tokens=5_000, max_candidates=10,
                                  max_elapsed_seconds=5, max_results_per_query=4),
        ))
        outcome, records = await loop.run(
            query="Find matching role", initial_search_queries=[], candidates=[candidate],
            evidence_gaps=["major evidence"], tool_dispatcher=dispatch,
            refresh_context=lambda _: {"candidates": [candidate], "evidence_gaps": []},
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["claims"][0]["quote"], "電子信息工程及相關專業優先。")
        self.assertEqual(outcome.rounds[0].new_sources, 0)
        self.assertEqual(outcome.rounds[0].sources_found, 1)
        self.assertEqual(outcome.status, "completed")

    async def test_search_toolbox_cumulative_metrics_are_not_double_counted(self):
        class Provider:
            configured = True
            calls = 0

            async def search(self, query, limit, *, include_domains=None):
                self.calls += 1
                return [{
                    "title": f"Official role page {self.calls}",
                    "url": f"https://careers.example.org/role/{self.calls}",
                    "content": f"This page contains graduate role {self.calls}.",
                }]

        provider = Provider()
        toolbox = SearchToolbox(search_providers=[("stub", provider)])
        toolbox.record_external_search("stub", requests=6, returned_sources=60)
        llm = FakeLLM([
            _completion([_tool("search_web", {"query": "graduate role official page", "limit": 3})]),
            _completion([_tool("search_web", {"query": "another specific graduate role", "limit": 3}, "call_2")]),
        ])
        refreshed_batches = []

        def refresh_context(records):
            refreshed_batches.append([item["url"] for item in records])
            return {"candidates": [], "evidence_gaps": ["official role page"]}

        loop = ResearchLoop(llm, ResearchLoopConfig(
            budget=ResearchBudget(max_rounds=2, max_search_requests=10, max_model_calls=2,
                                  max_total_tokens=5_000, max_candidates=10,
                                  max_elapsed_seconds=5, max_results_per_query=4),
        ))
        outcome, records = await loop.run(
            query="Find a graduate role", initial_search_queries=[], candidates=[],
            evidence_gaps=["official role page"], tool_dispatcher=toolbox.dispatch,
            refresh_context=refresh_context,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(refreshed_batches, [
            ["https://careers.example.org/role/1"],
            ["https://careers.example.org/role/2"],
        ])
        self.assertEqual([item.search_requests for item in outcome.rounds], [1, 1])
        provider_metrics = {item.provider: item for item in outcome.providers}
        self.assertEqual(provider_metrics["stub"].requests, 8)
        self.assertEqual(provider_metrics["stub"].returned_sources, 62)
        toolbox_metrics = {
            item["provider"]: item for item in toolbox.provider_metrics()
        }
        self.assertEqual(
            provider_metrics["stub"].model_dump(mode="json"),
            toolbox_metrics["stub"],
        )

    async def test_reading_unknown_url_and_inspecting_unknown_candidate_are_rejected(self):
        llm = FakeLLM([_completion([
            _tool("read_webpage", {"url": "https://unknown.example.org/secret"}, "call_1"),
            _tool("inspect_evidence", {"candidate_id": "made-up", "field": "deadline"}, "call_2"),
        ])])
        dispatch_count = 0

        async def dispatch(name, arguments):
            nonlocal dispatch_count
            dispatch_count += 1
            return {"records": [], "evidence": [], "source_count": 0}

        loop = ResearchLoop(llm, ResearchLoopConfig(
            budget=ResearchBudget(max_rounds=1, max_search_requests=2, max_model_calls=2,
                                  max_total_tokens=5_000, max_candidates=10,
                                  max_elapsed_seconds=5, max_results_per_query=5),
        ))
        outcome, _ = await loop.run(
            query="Find roles", initial_search_queries=[],
            candidates=[{"id": "real", "url": "https://jobs.example.org/role/1"}],
            evidence_gaps=["deadline"], tool_dispatcher=dispatch,
        )
        self.assertEqual(dispatch_count, 0)
        self.assertEqual([item.status for item in outcome.tool_calls], ["rejected", "rejected"])
        self.assertEqual([item.error_code for item in outcome.tool_calls], ["url_not_discovered", "candidate_not_found"])

    async def test_not_configured_is_explicit_and_does_not_fallback_to_rules(self):
        llm = SimpleNamespace(is_enabled=False)
        called = False

        async def dispatch(*args):
            nonlocal called
            called = True
            return {}

        outcome, records = await ResearchLoop(llm).run(
            query="query", initial_search_queries=[], candidates=[], evidence_gaps=["missing official source"],
            tool_dispatcher=dispatch,
        )
        self.assertEqual(outcome.status, "failed")
        self.assertIn("LLM 未配置", outcome.stop_reason)
        self.assertFalse(called)
        self.assertEqual(records, [])

    async def test_truncated_planner_records_usage_and_never_dispatches_tools(self):
        error = LLMClientError(
            "completion_truncated",
            "LLM API stopped at the completion token limit before completing the tool response.",
            actual_model="deepseek-flash-2026-09-10",
            request_id="req-truncated-test",
            prompt_tokens=2_315,
            completion_tokens=1_200,
            total_tokens=3_515,
            latency_ms=100,
        )
        llm = FakeLLM([error])
        dispatch_count = 0

        async def dispatch(name, arguments):
            nonlocal dispatch_count
            dispatch_count += 1
            return {"records": []}

        loop = ResearchLoop(llm, ResearchLoopConfig(
            budget=ResearchBudget(max_rounds=3, max_search_requests=8, max_model_calls=8,
                                  max_total_tokens=12_000, max_candidates=30,
                                  max_elapsed_seconds=90, max_results_per_query=8),
            max_completion_tokens=1_200,
        ))
        with self.assertLogs("app.research.loop", level="WARNING") as logs:
            outcome, records = await loop.run(
                query="Find official sources", initial_search_queries=["already searched"],
                candidates=[], evidence_gaps=["official source missing"],
                tool_dispatcher=dispatch,
            )
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.model_calls, 1)
        self.assertEqual(outcome.total_tokens, 3_515)
        self.assertIn("completion_truncated", outcome.stop_reason)
        self.assertEqual(logs.records[0].research_prompt_tokens, 2_315)
        self.assertEqual(logs.records[0].research_completion_tokens, 1_200)
        self.assertEqual(logs.records[0].research_total_tokens, 3_515)
        self.assertEqual(logs.records[0].research_max_completion_tokens, 1_200)
        self.assertEqual(dispatch_count, 0)
        self.assertEqual(records, [])
        self.assertEqual(llm.requests[0]["max_completion_tokens"], 1_200)


class ToolCallingClientTests(unittest.IsolatedAsyncioTestCase):
    def _tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }]

    async def test_real_openai_compatible_tool_call_shape_and_usage(self):
        captured = {}

        async def handler(request):
            captured["request"] = json.loads(request.content)
            captured["authorization"] = request.headers["authorization"]
            return httpx.Response(200, json={
                "id": "req-tool-01",
                "model": "deepseek-flash-2026-09-10",
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "call_tool_01", "type": "function", "function": {
                        "name": "search_web", "arguments": "{\"query\":\"official role\"}"
                    }}],
                }}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 11, "total_tokens": 41},
            })

        client = LLMClient(
            LLMConfig(api_key="secret-test-key", model="deepseek-flash", thinking_mode="disabled", max_retries=0),
            transport=httpx.MockTransport(handler),
        )
        completion = await client.complete_with_tools(
            "research_planner", [{"role": "user", "content": "Plan a search"}], self._tools()
        )
        self.assertEqual(completion.model, "deepseek-flash-2026-09-10")
        self.assertEqual(completion.tool_calls[0].arguments, {"query": "official role"})
        self.assertEqual(completion.total_tokens, 41)
        self.assertEqual(captured["authorization"], "Bearer secret-test-key")
        self.assertEqual(captured["request"]["tool_choice"], "auto")
        self.assertEqual(captured["request"]["thinking"], {"type": "disabled"})
        self.assertNotIn("response_format", captured["request"])

    async def test_unsupported_tool_and_malformed_arguments_fail_closed(self):
        for function in (
            {"name": "invented_tool", "arguments": "{}"},
            {"name": "search_web", "arguments": "not json"},
            {"name": "search_web", "arguments": "[]"},
        ):
            async def handler(request, function=function):
                return httpx.Response(200, json={
                    "id": "req-bad",
                    "model": "deepseek-flash",
                    "choices": [{"finish_reason": "tool_calls", "message": {
                        "role": "assistant", "content": None,
                        "tool_calls": [{"id": "call_bad", "type": "function", "function": function}],
                    }}],
                })

            client = LLMClient(
                LLMConfig(api_key="test-secret", model="deepseek-flash", max_retries=0),
                transport=httpx.MockTransport(handler),
            )
            with self.assertRaises(LLMClientError) as caught:
                await client.complete_with_tools("planner", [{"role": "user", "content": "x"}], self._tools())
            self.assertEqual(caught.exception.code, "invalid_response")
            self.assertNotIn("test-secret", str(caught.exception))

    async def test_tool_http_failure_message_is_redacted(self):
        async def handler(request):
            return httpx.Response(401, text="invalid bearer secret-test-key")

        client = LLMClient(
            LLMConfig(api_key="secret-test-key", model="deepseek-flash", max_retries=0),
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(LLMClientError) as caught:
            await client.complete_with_tools("planner", [{"role": "user", "content": "x"}], self._tools())
        self.assertEqual(caught.exception.code, "http_error")
        self.assertNotIn("secret-test-key", str(caught.exception))

    async def test_tool_completion_truncation_preserves_separated_usage(self):
        captured = {}

        async def handler(request):
            captured["max_tokens"] = json.loads(request.content)["max_tokens"]
            return httpx.Response(200, json={
                "id": "req-truncated",
                "model": "deepseek-flash-2026-09-10",
                "choices": [{"finish_reason": "length", "message": {
                    "role": "assistant", "content": None, "tool_calls": []
                }}],
                "usage": {"prompt_tokens": 2_315, "completion_tokens": 1_200, "total_tokens": 3_515},
            })

        client = LLMClient(
            LLMConfig(api_key="secret-test-key", model="deepseek-flash", max_retries=0),
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(LLMClientError) as caught:
            await client.complete_with_tools(
                "research_planner", [{"role": "user", "content": "x"}], self._tools(),
                max_completion_tokens=1_200,
            )
        self.assertEqual(caught.exception.code, "completion_truncated")
        self.assertEqual(caught.exception.prompt_tokens, 2_315)
        self.assertEqual(caught.exception.completion_tokens, 1_200)
        self.assertEqual(caught.exception.total_tokens, 3_515)
        self.assertEqual(captured["max_tokens"], 1_200)


class ResearchConfigTests(unittest.TestCase):
    def test_context_defaults_keep_followup_prompt_within_measured_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            config = ResearchLoopConfig.from_env()
            budget = research_budget_from_env()
        self.assertEqual(config.max_prompt_tokens, 8_000)
        self.assertEqual(config.max_context_candidates, 6)
        self.assertEqual(config.max_context_sources, 3)
        self.assertEqual(budget.max_search_requests, 12)
        self.assertEqual(budget.max_rounds, 3)
        self.assertEqual(budget.max_total_tokens, 16_000)
        self.assertEqual(budget.max_candidates, 60)

    def test_budget_environment_values_are_clamped_to_contract(self):
        with patch.dict(os.environ, {
            "RESEARCH_MAX_ROUNDS": "99",
            "RESEARCH_MAX_SEARCH_REQUESTS": "-4",
            "RESEARCH_MAX_MODEL_CALLS": "5",
            "RESEARCH_MAX_TOTAL_TOKENS": "not-a-number",
            "RESEARCH_MAX_CANDIDATES": "200",
            "RESEARCH_MAX_ELAPSED_SECONDS": "9999",
            "RESEARCH_MAX_RESULTS_PER_QUERY": "0",
        }, clear=False):
            budget = research_budget_from_env()
        self.assertEqual(budget.max_rounds, 3)
        self.assertEqual(budget.max_search_requests, 1)
        self.assertEqual(budget.max_model_calls, 5)
        self.assertEqual(budget.max_total_tokens, 16_000)
        self.assertEqual(budget.max_candidates, 100)
        self.assertEqual(budget.max_elapsed_seconds, 600)
        self.assertEqual(budget.max_results_per_query, 1)


if __name__ == "__main__":
    unittest.main()
