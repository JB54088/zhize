import os
import json
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient

from app.main import _merge_research_extractions, _project_name_for_verification, app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_does_not_require_search_credentials(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "ok",
            "service": "zhize-ai-engine",
        })

    def test_search_without_provider_returns_explicit_configuration_error(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post(
                "/api/v1/search", json={"query": "test search"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("SEARCH_PROVIDER=tavily", response.json()["detail"])
        self.assertNotIn("results", response.json())

    def test_project_name_drops_notice_and_publisher_suffixes(self):
        self.assertEqual(
            _project_name_for_verification(
                "福建省公共资源交易中心政府采购业务人工智能辅助服务项目单一来源公告 _ 通知公告 _ 省发改委"
            ),
            "福建省公共资源交易中心政府采购业务人工智能辅助服务项目",
        )

    def test_research_extraction_merges_grounded_fields_without_replacing_existing(self):
        records = [{
            "url": "https://careers.example.com/job/1",
            "extracted_fields": {"role": {"value": "工程师", "quote": "工程师"}},
            "field_sources": {"role": {"url": "https://careers.example.com/job/1", "quote": "工程师"}},
            "raw_text": "short body",
        }]
        extractions = [{
            "url": "https://careers.example.com/job/1?utm_source=test",
            "extracted_fields": {
                "role": {"value": "覆盖字段", "quote": "覆盖字段"},
                "organization": {"value": "Example", "quote": "Example careers"},
            },
            "field_sources": {
                "organization": {"url": "https://careers.example.com/job/1", "quote": "Example careers"},
            },
            "raw_text": "a longer official page body",
        }]

        merged = _merge_research_extractions(records, extractions)

        self.assertEqual(merged[0]["extracted_fields"]["role"]["value"], "工程师")
        self.assertEqual(merged[0]["extracted_fields"]["organization"]["value"], "Example")
        self.assertIn("role", merged[0]["field_sources"])
        self.assertIn("organization", merged[0]["field_sources"])
        self.assertEqual(merged[0]["raw_text"], "a longer official page body")

    def test_search_without_tavily_key_returns_no_results(self):
        with patch.dict(os.environ, {"SEARCH_PROVIDER": "tavily"}, clear=True):
            response = self.client.post(
                "/api/v1/search", json={"query": "test search"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("TAVILY_API_KEY", response.json()["detail"])
        self.assertNotIn("results", response.json())

    def test_search_returns_provider_sources_with_grounded_fields(self):
        class ProviderStub:
            async def search(self, query, limit):
                self.query = query
                self.limit = limit
                return [{
                    "title": "A source",
                    "url": "https://example.org/source",
                    "content": "The decision process was reviewed.",
                }]

        with patch.dict(
            os.environ,
            {"SEARCH_PROVIDER": "tavily", "TAVILY_API_KEY": "test-key"},
            clear=True,
        ), patch("app.main.create_search_provider", return_value=ProviderStub()):
            response = self.client.post(
                "/api/v1/search", json={"query": "decision process", "limit": 3}
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"], "decision process")
        self.assertEqual(len(payload["results"]), 1)
        result = payload["results"][0]
        self.assertEqual(result["url"], "https://example.org/source")
        self.assertEqual(result["source_name"], "example.org")
        self.assertEqual(result["evidence"], "The decision process was reviewed.")
        self.assertEqual(result["verdict"], "context")
        self.assertEqual(result["deadline_status"], "unverified")
        self.assertEqual(result["official_source_status"], "third_party")
        self.assertEqual(result["evidence_status"], "available")
        self.assertEqual(len(payload["cards"]), 1)
        self.assertEqual(payload["cards"][0]["category"], "other")
        self.assertEqual(payload["cards"][0]["source_url"], None)
        self.assertEqual(payload["cards"][0]["additional_sources"][0]["url"], result["url"])
        self.assertEqual(len(payload["source_records"]), 1)
        self.assertEqual(payload["source_records"][0]["raw_text"], "The decision process was reviewed.")
        runs = {item["action"]: item for item in payload["agent_runs"]}
        self.assertEqual(set(runs), {
            "intent_understanding", "search", "extraction",
            "source_verification", "semantic_match", "final_answer",
        })
        self.assertEqual(runs["intent_understanding"]["status"], "skipped_unconfigured")
        self.assertEqual(runs["search"]["status"], "completed")
        self.assertEqual(runs["extraction"]["status"], "skipped_unconfigured")
        self.assertEqual(runs["source_verification"]["status"], "skipped_no_evidence")
        self.assertEqual(runs["semantic_match"]["status"], "skipped_no_evidence")
        self.assertEqual(runs["final_answer"]["status"], "skipped_no_evidence")
        self.assertTrue(all(item["model"] is None for item in runs.values()))
        self.assertEqual(payload["answer"], "")
        self.assertEqual(payload["answer_claims"], [])

    def test_candidate_limit_applies_after_normalization_and_keeps_source_archive(self):
        class ProviderStub:
            async def search(self, query, limit):
                return [
                    {
                        "title": f"Distinct source {index}",
                        "url": f"https://example.org/source/{index}",
                        "content": f"Evidence for source {index}.",
                        "raw_content": f"Evidence for source {index}.",
                    }
                    for index in range(3)
                ]

        with patch.dict(
            os.environ,
            {"SEARCH_PROVIDER": "tavily", "TAVILY_API_KEY": "test-key"},
            clear=True,
        ), patch("app.main.create_search_provider", return_value=ProviderStub()):
            response = self.client.post(
                "/api/v1/search", json={"query": "source evidence", "candidate_limit": 1}
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(len(payload["cards"]), 1)
        self.assertEqual(len(payload["source_records"]), 3)

    def test_stream_emits_ordered_real_stage_snapshots_and_a_complete_result(self):
        class ProviderStub:
            async def search(self, query, limit):
                return [{
                    "title": "Decision research",
                    "url": "https://example.org/research",
                    "content": "The decision process was reviewed.",
                    "raw_content": "The decision process was reviewed.",
                }]

        with patch.dict(
            os.environ,
            {"SEARCH_PROVIDER": "tavily", "TAVILY_API_KEY": "test-key"},
            clear=True,
        ), patch("app.main.create_search_provider", return_value=ProviderStub()):
            with self.client.stream(
                "POST", "/api/v1/search/stream", json={"query": "decision process", "limit": 3}
            ) as response:
                self.assertEqual(response.status_code, 200)
                events = [json.loads(line) for line in response.iter_lines() if line]

        stages = [event["stage"] for event in events if event["type"] == "stage"]
        agent_events = [event["run"] for event in events if event["type"] == "agent"]
        terminal = next(event for event in events if event["type"] == "complete")
        completed = [stage for stage in stages if stage["status"] == "completed"]
        self.assertEqual([stage["key"] for stage in completed], ["collect", "clean", "verify", "judge"])
        self.assertEqual([stage["key"] for stage in stages if stage["status"] == "pending"], ["collect", "clean", "verify", "judge"])
        for stage in stages:
            if stage["status"] == "running":
                self.assertIsNone(stage["input_count"])
                self.assertIsNone(stage["output_count"])
                self.assertIsNone(stage["duration_ms"])
        self.assertEqual(completed[0]["output_count"], 1)
        self.assertEqual(completed[1]["input_count"], 1)
        self.assertEqual(terminal["query"], "decision process")
        self.assertEqual(terminal["results"][0]["decision"], "needs_verification")
        self.assertEqual(terminal["cards"][0]["matching_status"], "needs_verification")
        self.assertEqual(terminal["source_records"][0]["url"], "https://example.org/research")
        self.assertIsNone(terminal["results"][0]["application_url"])
        self.assertTrue(any(run["status"] == "running" for run in agent_events))
        final_runs = {run["action"]: run for run in terminal["agent_runs"]}
        self.assertEqual(len(final_runs), 6)
        self.assertTrue(all(run["status"] != "running" for run in final_runs.values()))


if __name__ == "__main__":
    unittest.main()
