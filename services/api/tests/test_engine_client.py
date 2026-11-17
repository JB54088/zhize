import json
import unittest

import httpx

from services.api.app.engine_client import EngineClientError, EngineStreamError, HttpEngineClient
from services.api.app.schemas import StageProgress


def ndjson(*events: dict[str, object]) -> bytes:
    return ("\n".join(json.dumps(event) for event in events) + "\n").encode()


class EngineClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_stream_posts_contract_and_delivers_stage_snapshots(self) -> None:
        seen: list[StageProgress] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert request.url.path == "/api/v1/search/stream"
            assert json.loads(request.content) == {"query": "query", "candidate_limit": 7}
            return httpx.Response(
                200,
                content=ndjson(
                    {
                        "type": "stage",
                        "stage": {
                            "key": "collect",
                            "status": "completed",
                            "started_at": "2026-10-08T00:00:00Z",
                            "completed_at": "2026-10-08T00:00:01Z",
                            "duration_ms": 1000,
                            "input_count": 1,
                            "output_count": 1,
                            "verified_count": 0,
                            "unverified_count": 1,
                            "conflict_count": 0,
                            "error": None,
                        },
                    },
                    {
                        "type": "complete",
                        "query": "query",
                        "results": [],
                        "cards": [],
                        "source_records": [
                            {
                                "id": "source-1",
                                "title": "Source page",
                                "url": "https://example.org/source",
                                "source_name": "Example",
                                "provider": "tavily",
                                "captured_at": "2026-10-09T00:00:00Z",
                                "published_at": None,
                                "search_query_index": 0,
                                "search_query": "query",
                                "raw_text": "Original public source body",
                                "raw_text_truncated": False,
                                "metadata": {"score": 0.9},
                            }
                        ],
                        "errors": [],
                    },
                ),
            )

        async def on_stage(stage: StageProgress) -> None:
            seen.append(stage)

        client = HttpEngineClient("http://engine.local", transport=httpx.MockTransport(handler))
        result = await client.search_stream("query", 7, on_stage)

        self.assertEqual(result.query, "query")
        self.assertEqual(result.results, [])
        self.assertEqual(result.source_records[0].raw_text, "Original public source body")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].key, "collect")
        self.assertEqual(seen[0].duration_ms, 1000)


    async def test_search_stream_sanitizes_stage_errors(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ndjson(
                    {
                        "type": "error",
                        "stage": "verify",
                        "message": "api_key=secret-value",
                    }
                ),
            )

        async def on_stage(_: StageProgress) -> None:
            return None

        client = HttpEngineClient("http://engine.local", transport=httpx.MockTransport(handler))
        with self.assertRaises(EngineStreamError) as caught:
            await client.search_stream("query", 10, on_stage)

        self.assertEqual(caught.exception.stage, "verify")
        self.assertEqual(str(caught.exception), "AI engine stage failed.")
        self.assertNotIn("secret-value", str(caught.exception))


    async def test_search_stream_rejects_missing_terminal_event(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ndjson(
                    {"type": "stage", "stage": {"key": "collect", "status": "running"}}
                ),
            )

        async def on_stage(_: StageProgress) -> None:
            return None

        client = HttpEngineClient("http://engine.local", transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(EngineClientError, "stream ended"):
            await client.search_stream("query", 10, on_stage)

    async def test_one_shot_search_retains_source_records_and_cards(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/search"
            assert json.loads(request.content) == {"query": "query", "candidate_limit": 50}
            return httpx.Response(
                200,
                json={
                    "query": "query",
                    "results": [],
                    "cards": [],
                    "source_records": [
                        {
                            "id": "source-1",
                            "title": "Source page",
                            "url": "https://example.org/source",
                            "raw_text": "Original source text",
                        }
                    ],
                    "errors": [],
                },
            )

        client = HttpEngineClient("http://engine.local", transport=httpx.MockTransport(handler))
        result = await client.search("query", 50)

        self.assertEqual(result.source_records[0].id, "source-1")
        self.assertEqual(result.source_records[0].raw_text, "Original source text")
