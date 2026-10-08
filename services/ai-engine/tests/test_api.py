import os
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient

from app.main import app


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


if __name__ == "__main__":
    unittest.main()
