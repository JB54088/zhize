import unittest
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.provider import TavilySearchProvider
from app.settings import Settings


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_tavily_adapter_uses_configured_real_search_request(self):
        captured = {}

        async def handler(request):
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers.get("Authorization")
            captured["payload"] = await request.aread()
            return httpx.Response(
                200,
                json={"results": [{"title": "Live source", "url": "https://example.org"}]},
            )

        settings = Settings(
            search_provider="tavily",
            search_api_key="test-secret",
            search_base_url="https://search.example.test/v2/search",
            search_timeout_seconds=5,
            official_source_domains=(),
            llm_api_key="",
            llm_model="",
            llm_base_url="https://api.openai.com/v1",
        )
        provider = TavilySearchProvider(
            settings, transport=httpx.MockTransport(handler)
        )

        results = await provider.search("a decision query", 4)

        self.assertEqual(captured["url"], "https://search.example.test/v2/search")
        self.assertEqual(captured["authorization"], "Bearer test-secret")
        self.assertNotIn(b"test-secret", captured["payload"])
        self.assertIn(b'"query":"a decision query"', captured["payload"])
        self.assertEqual(results[0]["title"], "Live source")


if __name__ == "__main__":
    unittest.main()
