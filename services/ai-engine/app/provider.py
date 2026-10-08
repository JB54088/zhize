from typing import Any, Optional

import httpx

from .settings import Settings


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class TavilySearchProvider:
    """Search adapter for Tavily's real web-search API."""

    def __init__(
        self,
        settings: Settings,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.search_timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    self.settings.search_base_url,
                    json={
                        "api_key": self.settings.search_api_key,
                        "query": query,
                        "max_results": limit,
                        "search_depth": "advanced",
                        "include_raw_content": True,
                        "include_answer": False,
                    },
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                "Configured search provider returned an error response."
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(
                "Could not connect to the configured search provider."
            ) from exc

        try:
            payload = response.json()
            results = payload["results"]
            if not isinstance(results, list):
                raise TypeError("results must be a list")
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                "Configured search provider returned an invalid response."
            ) from exc
        return [item for item in results if isinstance(item, dict)]


def create_search_provider(settings: Settings) -> TavilySearchProvider:
    if not settings.search_provider:
        raise ProviderError(
            "Search is not configured. Set SEARCH_PROVIDER=tavily and TAVILY_API_KEY.",
            status_code=503,
        )
    if settings.search_provider != "tavily":
        raise ProviderError(
            f"Unsupported SEARCH_PROVIDER '{settings.search_provider}'. Supported provider: tavily.",
            status_code=503,
        )
    if not settings.search_api_key:
        raise ProviderError(
            "Tavily search is selected but TAVILY_API_KEY is not configured.",
            status_code=503,
        )
    if not settings.search_base_url:
        raise ProviderError(
            "Tavily search is selected but SEARCH_BASE_URL is empty.",
            status_code=503,
        )
    return TavilySearchProvider(settings)
