import os
import re
from typing import Any, Callable, Optional, Sequence

import httpx

from .settings import Settings


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


_SEARCH_PROVIDER_FACTORIES: dict[str, Callable[[Settings], Any]] = {}


def register_search_provider(name: str, factory: Callable[[Settings], Any]) -> None:
    """Register a named search adapter factory for SEARCH_PROVIDERS selection."""
    normalized = name.strip().lower()
    if not normalized or not normalized.replace("-", "").replace("_", "").isalnum():
        raise ValueError("provider name must be a simple alphanumeric identifier")
    _SEARCH_PROVIDER_FACTORIES[normalized] = factory


class TavilySearchProvider:
    """Search adapter for Tavily's real web-search API."""

    def __init__(
        self,
        settings: Settings,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def search(
        self,
        query: str,
        limit: int,
        *,
        include_domains: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        request_body: dict[str, Any] = {
            "query": query,
            "max_results": limit,
            "search_depth": "advanced",
            "include_raw_content": True,
            "include_answer": False,
        }
        if include_domains:
            request_body["include_domains"] = list(include_domains)
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.search_timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    self.settings.search_base_url,
                    headers={
                        "Authorization": f"Bearer {self.settings.search_api_key}"
                    },
                    json=request_body,
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


class BraveSearchProvider:
    """Optional adapter for Brave's documented Web Search API."""

    DEFAULT_BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        settings: Settings,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.api_key = (
            getattr(settings, "brave_search_api_key", "")
            or os.getenv("BRAVE_SEARCH_API_KEY", "")
        ).strip()
        self.base_url = (
            getattr(settings, "brave_search_base_url", "")
            or os.getenv("BRAVE_SEARCH_BASE_URL", "")
            or self.DEFAULT_BASE_URL
        ).strip()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def search(
        self,
        query: str,
        limit: int,
        *,
        include_domains: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.configured:
            raise ProviderError(
                "Brave Search is not configured. Set BRAVE_SEARCH_API_KEY.",
                status_code=503,
            )
        targeted_query = _targeted_query(query, include_domains)
        params = {"q": targeted_query, "count": max(1, min(20, int(limit))), "search_lang": "en"}
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.search_timeout_seconds,
                transport=self.transport,
                follow_redirects=False,
            ) as client:
                response = await client.get(
                    self.base_url,
                    params=params,
                    headers={
                        "X-Subscription-Token": self.api_key,
                        "Accept": "application/json",
                    },
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError("Brave Search returned an error response.") from exc
        except httpx.RequestError as exc:
            raise ProviderError("Could not connect to Brave Search.") from exc

        try:
            payload = response.json()
            web = payload.get("web", {})
            results = web.get("results", []) if isinstance(web, dict) else []
            if not isinstance(results, list):
                raise TypeError("web.results must be a list")
        except (ValueError, AttributeError, TypeError) as exc:
            raise ProviderError("Brave Search returned an invalid response.") from exc

        normalized: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            title = item.get("title")
            if not isinstance(url, str) or not isinstance(title, str):
                continue
            description = item.get("description")
            normalized.append({
                **item,
                "title": title,
                "url": url,
                "content": description if isinstance(description, str) else "",
                "raw_content": description if isinstance(description, str) else "",
                "source_provider": "brave",
                "published_at": item.get("page_age") or item.get("published_at"),
            })
        return normalized[:limit]


class BaiduSearchProvider:
    """Adapter for Baidu Qianfan's documented web-search API."""

    BASE_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"
    _DOMAIN_RE = re.compile(
        r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$",
        re.I,
    )

    def __init__(
        self,
        settings: Settings,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.api_key = (
            getattr(settings, "baidu_search_api_key", "")
            or os.getenv("BAIDU_SEARCH_API_KEY", "")
        ).strip()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def search(
        self,
        query: str,
        limit: int,
        *,
        include_domains: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.configured:
            raise ProviderError(
                "Baidu Search is not configured. Set BAIDU_SEARCH_API_KEY.",
                status_code=503,
            )
        clean_query = " ".join(str(query).split())[:2_000]
        if not clean_query:
            raise ProviderError("Baidu Search requires a non-empty query.", status_code=400)

        request_body: dict[str, Any] = {
            "messages": [{"role": "user", "content": clean_query}],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "web", "top_k": max(1, min(50, int(limit)))}],
        }
        domains = self._valid_domains(include_domains)
        if domains:
            request_body["search_filter"] = {"match": {"site": domains}}

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.search_timeout_seconds,
                transport=self.transport,
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    self.BASE_URL,
                    headers={
                        "X-Appbuilder-Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=request_body,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError("Baidu Search returned an error response.") from exc
        except httpx.RequestError as exc:
            raise ProviderError("Could not connect to Baidu Search.") from exc

        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("response must be an object")
            references = payload.get("references")
            if not isinstance(references, list):
                raise TypeError("references must be a list")
            api_code = payload.get("code")
            if api_code not in (None, "", 0, "0") and not references:
                raise ValueError("Baidu Search returned an API error")
        except (ValueError, TypeError) as exc:
            raise ProviderError("Baidu Search returned an invalid or error response.") from exc

        normalized: list[dict[str, Any]] = []
        for item in references:
            if not isinstance(item, dict) or item.get("type", "web") != "web":
                continue
            title = item.get("title")
            url = item.get("url")
            if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
                continue
            content = item.get("content")
            normalized.append({
                "title": title.strip(),
                "url": url,
                "content": content if isinstance(content, str) else "",
                "raw_content": content if isinstance(content, str) else "",
                "source_provider": "baidu",
                "source_name": item.get("website") or item.get("web_anchor") or "百度搜索",
                "published_at": item.get("date"),
                "score": item.get("rerank_score"),
                "authority_score": item.get("authority_score"),
            })
            if len(normalized) >= max(1, min(50, int(limit))):
                break
        return normalized

    @classmethod
    def _valid_domains(cls, values: Sequence[str] | None) -> list[str]:
        domains: list[str] = []
        for value in values or ():
            if not isinstance(value, str):
                continue
            domain = value.strip().lower().removeprefix("www.").rstrip(".")
            try:
                domain = domain.encode("idna").decode("ascii")
            except UnicodeError:
                continue
            if cls._DOMAIN_RE.fullmatch(domain) and domain not in domains:
                domains.append(domain)
            if len(domains) >= 20:
                break
        return domains


def create_search_provider(settings: Settings) -> Any:
    if not settings.search_provider:
        raise ProviderError(
            "Search is not configured. Set SEARCH_PROVIDER and the selected provider's local credentials.",
            status_code=503,
        )
    providers = create_search_providers(settings)
    if len(providers) != 1:
        raise ProviderError("create_search_provider requires exactly one configured search source.")
    return providers[0][1]


def create_search_providers(settings: Settings) -> list[tuple[str, Any]]:
    """Build configured providers; adapters are injected through a registry.

    SEARCH_PROVIDERS is a comma-separated list. SEARCH_PROVIDER remains the
    V0.1-compatible default when the plural setting is absent. Baidu, Tavily,
    and Brave are bundled adapters; applications can register additional ones.
    """
    names = settings.search_providers or ((settings.search_provider,) if settings.search_provider else ())
    if not names:
        raise ProviderError(
            "Search is not configured. Set SEARCH_PROVIDER and the selected provider's local credentials.",
            status_code=503,
        )
    providers: list[tuple[str, Any]] = []
    for name in names:
        normalized = name.strip().lower()
        if normalized == "tavily":
            if not settings.search_api_key:
                if len(names) == 1:
                    raise ProviderError("Tavily search is selected but TAVILY_API_KEY is not configured.", status_code=503)
                continue
            if not settings.search_base_url:
                if len(names) == 1:
                    raise ProviderError("Tavily search is selected but SEARCH_BASE_URL is empty.", status_code=503)
                continue
            factory = _SEARCH_PROVIDER_FACTORIES.get("tavily", TavilySearchProvider)
        elif normalized == "brave":
            factory = _SEARCH_PROVIDER_FACTORIES.get("brave", BraveSearchProvider)
            provider = factory(settings)
            if not getattr(provider, "configured", False):
                # Another configured source may still be useful. The toolbox
                # reports this skip as `configured: false` in its metrics.
                if len(names) == 1:
                    raise ProviderError("Brave Search is selected but BRAVE_SEARCH_API_KEY is not configured.", status_code=503)
                continue
            providers.append((normalized, provider))
            continue
        elif normalized == "baidu":
            factory = _SEARCH_PROVIDER_FACTORIES.get("baidu", BaiduSearchProvider)
            provider = factory(settings)
            if not getattr(provider, "configured", False):
                if len(names) == 1:
                    raise ProviderError("Baidu Search is selected but BAIDU_SEARCH_API_KEY is not configured.", status_code=503)
                continue
            providers.append((normalized, provider))
            continue
        else:
            factory = _SEARCH_PROVIDER_FACTORIES.get(normalized)
            if factory is None:
                raise ProviderError(
                    f"Unsupported SEARCH_PROVIDERS entry '{normalized}'. Register a search adapter for this name.",
                    status_code=503,
                )
        providers.append((normalized, factory(settings)))
    if not providers:
        raise ProviderError("No selected search provider has usable local credentials.", status_code=503)
    return providers


_SEARCH_PROVIDER_FACTORIES.setdefault("tavily", TavilySearchProvider)
_SEARCH_PROVIDER_FACTORIES.setdefault("brave", BraveSearchProvider)
_SEARCH_PROVIDER_FACTORIES.setdefault("baidu", BaiduSearchProvider)


def provider_configuration(settings: Settings) -> list[dict[str, Any]]:
    """Return safe provider availability metadata without exposing credentials."""
    names = settings.search_providers or ((settings.search_provider,) if settings.search_provider else ("baidu",))
    result: list[dict[str, Any]] = []
    for name in names:
        normalized = name.strip().lower()
        if normalized == "tavily":
            configured = bool(settings.search_api_key and settings.search_base_url)
        elif normalized == "brave":
            configured = bool(
                getattr(settings, "brave_search_api_key", "")
                or os.getenv("BRAVE_SEARCH_API_KEY", "")
            )
        elif normalized == "baidu":
            configured = bool(
                getattr(settings, "baidu_search_api_key", "")
                or os.getenv("BAIDU_SEARCH_API_KEY", "")
            )
        else:
            configured = normalized in _SEARCH_PROVIDER_FACTORIES
        result.append({"provider": normalized, "configured": configured})
    return result


def _targeted_query(query: str, domains: Sequence[str] | None) -> str:
    clean_query = " ".join(str(query).split())
    if re_search_site_clause(clean_query):
        return clean_query[:600]
    valid = []
    for domain in domains or ():
        normalized = str(domain).strip().lower().removeprefix("www.").rstrip(".")
        if normalized and normalized not in valid:
            valid.append(normalized)
    if not valid:
        return clean_query[:600]
    site_filter = " OR ".join(f"site:{domain}" for domain in valid)
    prefix = f"{site_filter} "
    return (prefix + clean_query[: max(0, 600 - len(prefix))]).strip()


def re_search_site_clause(query: str) -> bool:
    # The toolbox already constructed a site-filtered query. Avoid duplicating
    # the operator when the adapter also receives include_domains.
    return "site:" in query.lower()
