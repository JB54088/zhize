from __future__ import annotations

import httpx

from .schemas import EngineSearchResponse, SearchResult


class EngineClientError(Exception):
    """A sanitized upstream AI-engine error safe to store on a search task."""


class HttpEngineClient:
    def __init__(self, base_url: str, timeout_seconds: float = 45.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/search",
                    json={"query": query, "limit": limit},
                )
        except httpx.RequestError as exc:
            raise EngineClientError("AI engine is unavailable.") from exc

        if response.is_error:
            raise EngineClientError("AI engine could not complete the search.")

        try:
            payload = EngineSearchResponse.model_validate(response.json())
        except (ValueError, TypeError) as exc:
            raise EngineClientError("AI engine returned an invalid search response.") from exc
        return payload.results
