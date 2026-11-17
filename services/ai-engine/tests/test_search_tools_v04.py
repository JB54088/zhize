from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.provider import (  # noqa: E402
    BraveSearchProvider,
    ProviderError,
    create_search_providers,
)
from app.research.contracts import SearchProviderMetric  # noqa: E402
from app.search_tools import SafeWebReader, SearchToolbox, WebReadError  # noqa: E402
from app.settings import Settings  # noqa: E402


PUBLIC_TEST_IP = "93.184.216.34"


def settings(**overrides) -> Settings:
    values = {
        "search_provider": "tavily",
        "search_api_key": "tavily-test-key",
        "search_base_url": "https://api.tavily.com/search",
        "search_timeout_seconds": 5,
        "official_source_domains": (),
        "llm_api_key": "",
        "llm_model": "",
        "llm_base_url": "https://api.deepseek.com",
        "search_providers": (),
        "brave_search_api_key": "",
        "brave_search_base_url": BraveSearchProvider.DEFAULT_BASE_URL,
    }
    values.update(overrides)
    return Settings(**values)


class _StubProvider:
    configured = True

    def __init__(self, results=None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.calls = []

    async def search(self, query, limit, *, include_domains=None):
        self.calls.append({"query": query, "limit": limit, "domains": include_domains})
        if self.error:
            raise self.error
        return self.results[:limit]


class SearchProviderV04Tests(unittest.IsolatedAsyncioTestCase):
    def test_auto_thinking_mode_disables_only_for_deepseek(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://api.deepseek.com", "LLM_THINKING": "auto"}, clear=True):
            self.assertEqual(Settings.from_env().llm_thinking_mode, "disabled")
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://api.example.test/v1", "LLM_THINKING": "auto"}, clear=True):
            self.assertIsNone(Settings.from_env().llm_thinking_mode)

    async def test_brave_adapter_uses_documented_endpoint_and_normalizes_results(self):
        captured = {}

        async def handler(request):
            captured["url"] = str(request.url)
            captured["token"] = request.headers.get("X-Subscription-Token")
            return httpx.Response(200, json={"web": {"results": [{
                "title": "Official role", "url": "https://jobs.example.com/role/42",
                "description": "Role details", "page_age": "2026-09-01",
            }]}})

        config = settings(brave_search_api_key="brave-test-token")
        provider = BraveSearchProvider(config, transport=httpx.MockTransport(handler))
        results = await provider.search("graduate role", 5, include_domains=["jobs.example.com"])

        self.assertEqual(captured["token"], "brave-test-token")
        self.assertIn("site%3Ajobs.example.com", captured["url"])
        self.assertEqual(results[0]["source_provider"], "brave")
        self.assertEqual(results[0]["content"], "Role details")
        self.assertEqual(results[0]["published_at"], "2026-09-01")

    async def test_missing_brave_key_is_skipped_when_tavily_is_available(self):
        config = settings(
            search_providers=("tavily", "brave"),
            brave_search_api_key="",
        )
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}):
            providers = create_search_providers(config)
            toolbox = SearchToolbox(config)

        self.assertEqual([name for name, _ in providers], ["tavily"])
        by_name = {item["provider"]: item for item in toolbox.provider_metrics()}
        self.assertTrue(by_name["tavily"]["configured"])
        self.assertFalse(by_name["brave"]["configured"])
        self.assertEqual(by_name["brave"]["requests"], 0)
        self.assertEqual(by_name["brave"]["failures"], 0)

    async def test_only_unconfigured_brave_fails_clearly(self):
        config = settings(search_providers=("brave",), brave_search_api_key="")
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}):
            with self.assertRaisesRegex(ProviderError, "BRAVE_SEARCH_API_KEY"):
                create_search_providers(config)

    async def test_official_search_uses_site_domains_and_records_traceback(self):
        provider = _StubProvider(results=[
            {"title": "Company campus hiring", "url": "https://careers.example.com/2027", "content": "A role"},
            {"title": "Wrong host", "url": "https://aggregator.example.net/jobs", "content": "Copied role"},
        ])
        toolbox = SearchToolbox(search_providers=[("tavily", provider)])

        result = await toolbox.dispatch("search_official", {
            "query": "Company 2027 campus hiring",
            "domains": ["www.example.com", "localhost", "127.0.0.1"],
            "limit": 5,
            "discovered_from": "https://aggregator.example.net/job/1",
            "_max_requests": 1,
        })

        self.assertEqual(len(provider.calls), 1)
        self.assertIn("site:example.com", provider.calls[0]["query"])
        self.assertEqual(provider.calls[0]["domains"], ("example.com",))
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["source_role"], "official_candidate")
        self.assertEqual(result["records"][0]["field_verification"], "not_verified")
        self.assertEqual(result["records"][0]["discovered_from"], "https://aggregator.example.net/job/1")
        self.assertEqual(result["search_requests"], 1)

    async def test_dispatch_enforces_provider_request_budget_and_deduplicates_tracking_urls(self):
        first = _StubProvider(results=[
            {"title": "Role", "url": "https://jobs.example.com/role/1?utm_source=search", "content": "A"},
            {"title": "Role mirror", "url": "https://jobs.example.com/role/1", "content": "B"},
        ])
        second = _StubProvider(results=[{"title": "Other", "url": "https://other.example.com/", "content": "C"}])
        toolbox = SearchToolbox(search_providers=[("tavily", first), ("brave", second)])

        result = await toolbox.dispatch("search_web", {
            "query": "active graduate jobs", "limit": 8, "_max_requests": 1,
        })

        self.assertEqual(len(first.calls), 1)
        self.assertEqual(second.calls, [])
        self.assertEqual(result["search_requests"], 1)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["url"], "https://jobs.example.com/role/1")
        metrics = {metric["provider"]: metric for metric in result["providers"]}
        self.assertEqual(metrics["tavily"]["requests"], 1)
        self.assertEqual(metrics["tavily"]["returned_sources"], 2)
        self.assertEqual(metrics["brave"]["requests"], 0)

    async def test_failed_provider_error_is_redacted_and_metrics_count_it(self):
        provider = _StubProvider(error=RuntimeError("credential leaked should never show"))
        toolbox = SearchToolbox(search_providers=[("tavily", provider)])

        result = await toolbox.search_web("query")

        self.assertEqual(result["error_code"], "all_search_providers_failed")
        self.assertNotIn("credential", str(result))
        metric = next(item for item in toolbox.provider_metrics() if item["provider"] == "tavily")
        self.assertEqual(metric["requests"], 1)
        self.assertEqual(metric["failures"], 1)

    async def test_external_initial_search_metrics_can_be_merged(self):
        toolbox = SearchToolbox(search_providers=[("tavily", _StubProvider())])
        toolbox.record_external_search("tavily", requests=3, returned_sources=17, latency_ms=205)

        metric = next(item for item in toolbox.provider_metrics() if item["provider"] == "tavily")
        SearchProviderMetric.model_validate(metric)
        self.assertEqual(metric["requests"], 3)
        self.assertEqual(metric["returned_sources"], 17)
        self.assertEqual(metric["latency_ms"], 205)


class SafeWebReaderV04Tests(unittest.IsolatedAsyncioTestCase):
    def reader(self, handler, *, resolve_host=None, max_page_bytes=512 * 1024):
        return SafeWebReader(
            transport=httpx.MockTransport(handler),
            resolve_host=resolve_host or (lambda _host: [PUBLIC_TEST_IP]),
            max_page_bytes=max_page_bytes,
            timeout_seconds=2,
        )

    async def test_reads_public_html_after_robots_and_removes_script_text(self):
        requested = []

        async def handler(request):
            requested.append((request.url.path, request.headers.get("host")))
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nAllow: /\n")
            return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=(
                "<html><head><title>Opening</title><script>secret()</script></head>"
                "<body><h1>Open role</h1><p>Accepts the 2027 graduating class.</p></body></html>"
            ))

        evidence = await self.reader(handler).read("https://jobs.example.com/opening")

        self.assertEqual(evidence["title"], "Opening")
        self.assertIn("Accepts the 2027 graduating class", evidence["text"])
        self.assertNotIn("secret()", evidence["text"])
        self.assertEqual(evidence["access"], "public_anonymous")
        self.assertEqual(requested, [
            ("/robots.txt", "jobs.example.com"),
            ("/opening", "jobs.example.com"),
        ])

    async def test_https_sni_extension_is_a_string_for_httpcore_anyio_tls(self):
        seen_sni = []

        async def handler(request):
            seen_sni.append(request.extensions.get("sni_hostname"))
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><body>Public evidence</body></html>",
            )

        evidence = await self.reader(handler).read("https://jobs.example.com/opening")

        self.assertEqual(evidence["text"], "Public evidence")
        self.assertEqual(seen_sni, ["jobs.example.com", "jobs.example.com"])
        self.assertTrue(all(isinstance(value, str) for value in seen_sni))

    async def test_robots_disallow_prevents_page_request(self):
        requested = []

        async def handler(request):
            requested.append(request.url.path)
            return httpx.Response(200, text="User-agent: *\nDisallow: /private")

        with self.assertRaisesRegex(WebReadError, "robots.txt") as caught:
            await self.reader(handler).read("https://jobs.example.com/private/role")
        self.assertEqual(caught.exception.code, "robots_disallowed")
        self.assertEqual(requested, ["/robots.txt"])

    async def test_private_direct_destination_is_rejected_before_network(self):
        requested = []

        async def handler(request):
            requested.append(request)
            return httpx.Response(200, text="ignored")

        with self.assertRaises(WebReadError) as caught:
            await self.reader(handler).read("http://127.0.0.1/admin")
        self.assertEqual(caught.exception.code, "ssrf_address_blocked")
        self.assertEqual(requested, [])

    async def test_dns_answer_with_any_private_address_is_rejected(self):
        async def handler(request):
            return httpx.Response(200, text="ignored")

        reader = self.reader(handler, resolve_host=lambda _host: [PUBLIC_TEST_IP, "10.0.0.4"])
        with self.assertRaises(WebReadError) as caught:
            await reader.read("https://jobs.example.com/opening")
        self.assertEqual(caught.exception.code, "ssrf_address_blocked")

    async def test_redirect_to_local_destination_is_rejected(self):
        requested = []

        async def handler(request):
            requested.append(request.url.host)
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(302, headers={"location": "http://internal.local/admin"})

        with self.assertRaises(WebReadError) as caught:
            await self.reader(handler).read("https://public.example.com/redirect")
        self.assertEqual(caught.exception.code, "ssrf_host_blocked")
        self.assertEqual(requested, [PUBLIC_TEST_IP, PUBLIC_TEST_IP])

    async def test_download_size_is_limited_even_without_content_length(self):
        async def handler(request):
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 1200)

        with self.assertRaises(WebReadError) as caught:
            await self.reader(handler, max_page_bytes=1024).read("https://public.example.com/page")
        self.assertEqual(caught.exception.code, "download_too_large")

    async def test_redirect_count_and_non_html_content_are_limited(self):
        async def redirect_handler(request):
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(302, headers={"location": "/again"})

        with self.assertRaises(WebReadError) as caught:
            await self.reader(redirect_handler).read("https://public.example.com/page")
        self.assertEqual(caught.exception.code, "too_many_redirects")

        async def unsupported_handler(request):
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(200, headers={"content-type": "application/zip"}, content=b"zip")

        with self.assertRaises(WebReadError) as unsupported:
            await self.reader(unsupported_handler).read("https://public.example.com/archive")
        self.assertEqual(unsupported.exception.code, "unsupported_content_type")

    async def test_pdf_unavailable_is_reported_instead_of_returning_fake_text(self):
        if "pypdf" in sys.modules:
            self.skipTest("pypdf is installed in this test environment")

        async def handler(request):
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 sample")

        with patch.dict(sys.modules, {"pypdf": None}):
            with self.assertRaises(WebReadError) as caught:
                await self.reader(handler).read("https://public.example.com/file.pdf")
        self.assertEqual(caught.exception.code, "pdf_reader_unavailable")


class EvidenceInspectionV04Tests(unittest.TestCase):
    def test_only_exact_source_quotes_are_returned_for_field_inspection(self):
        toolbox = SearchToolbox(search_providers=[])
        toolbox.set_candidates([{
            "id": "role-1",
            "evidence_detail": "The role accepts 2027 graduates.",
            "field_verification": {
                "graduate_cohort": {
                    "value": "2027届",
                    "status": "verified",
                    "quote": "accepts 2027 graduates",
                    "evidence_url": "https://careers.example.com/role-1",
                },
                "major": {
                    "value": "电子信息工程",
                    "status": "verified",
                    "quote": "This quote does not exist in the page",
                    "evidence_url": "https://careers.example.com/role-1",
                },
            },
        }])

        good = toolbox.inspect_evidence("role-1", "graduate_cohort")
        bad = toolbox.inspect_evidence("role-1", "major")

        self.assertEqual(good["evidence"]["status"], "available")
        self.assertEqual(good["evidence"]["claims"][0]["url"], "https://careers.example.com/role-1")
        self.assertEqual(bad["evidence"]["status"], "missing")
        self.assertTrue(bad["evidence_gaps"])


if __name__ == "__main__":
    unittest.main()
