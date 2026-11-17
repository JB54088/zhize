"""Safe, bounded search and evidence tools used by autonomous research."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..provider import (
    ProviderError,
    create_search_providers,
    provider_configuration,
)
from ..settings import Settings
from .web_reader import SafeWebReader, WebReadError


_TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "mc_cid", "mc_eid", "ref", "source",
    "spm", "from", "share_token",
}
_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$", re.I)


class SearchToolbox:
    """Dispatch the four fixed research tools and expose honest provider metrics.

    Accepts either `SearchToolbox(settings)` or already-created provider pairs
    such as `SearchToolbox(search_providers=[("tavily", adapter)])`.
    """

    def __init__(
        self,
        settings: Settings | Sequence[tuple[str, Any]] | None = None,
        *,
        search_providers: Sequence[tuple[str, Any]] | None = None,
        max_candidates: int = 60,
        web_reader: SafeWebReader | None = None,
    ) -> None:
        if settings is not None and not isinstance(settings, Settings):
            if search_providers is not None:
                raise TypeError("Pass provider pairs in only one constructor argument.")
            search_providers = settings
            settings = None
        self.settings = settings
        self.max_candidates = min(max(int(max_candidates), 1), 100)
        self.reader = web_reader or SafeWebReader(
            timeout_seconds=(settings.search_timeout_seconds if settings else 12.0)
        )
        self.candidates: dict[str, dict[str, Any]] = {}
        self._provider_objects: dict[str, Any] = {}
        self._metrics: dict[str, dict[str, Any]] = {}
        if search_providers is None and settings is not None:
            configs = provider_configuration(settings)
            self._metrics = {item["provider"]: _metric(item["provider"], item["configured"]) for item in configs}
            try:
                search_providers = create_search_providers(settings)
            except ProviderError:
                # Keep a usable toolbox object so the research caller can
                # expose a structured configuration error and metric.
                search_providers = ()
        elif search_providers is None:
            search_providers = ()

        for name, provider in search_providers:
            normalized = str(name).strip().lower()
            configured = bool(getattr(provider, "configured", True))
            self._provider_objects[normalized] = provider
            self._metrics.setdefault(normalized, _metric(normalized, configured))
            self._metrics[normalized]["configured"] = configured

        # Report optional bundled providers when this instance was supplied
        # prebuilt adapters without settings. Presence is a boolean only.
        if settings is None:
            for name, key in (
                ("tavily", "TAVILY_API_KEY"),
                ("brave", "BRAVE_SEARCH_API_KEY"),
                ("baidu", "BAIDU_SEARCH_API_KEY"),
            ):
                self._metrics.setdefault(name, _metric(name, bool(os.getenv(key, "").strip())))

    def set_candidates(self, candidates: Sequence[Any] | Mapping[str, Any]) -> None:
        """Register existing candidate data for the inspect_evidence tool."""
        if isinstance(candidates, Mapping):
            items = candidates.items()
        else:
            items = (
                (str(_as_mapping(item).get("id", "")), item)
                for item in candidates
            )
        indexed: dict[str, dict[str, Any]] = {}
        for candidate_id, candidate in items:
            data = _as_mapping(candidate)
            normalized_id = str(data.get("id") or candidate_id or "").strip()
            if normalized_id:
                indexed[normalized_id] = data
        self.candidates = indexed

    def record_external_search(
        self,
        provider_name: str,
        *,
        requests: int = 1,
        returned_sources: int = 0,
        failures: int = 0,
        latency_ms: int = 0,
    ) -> None:
        """Merge measured provider usage from a search stage run before dispatch."""
        name = provider_name.strip().lower()
        if not name:
            return
        metric = self._metrics.setdefault(name, _metric(name, True))
        metric["requests"] += max(0, int(requests))
        metric["returned_sources"] += max(0, int(returned_sources))
        metric["failures"] += max(0, int(failures))
        metric["latency_ms"] += max(0, int(latency_ms))

    def provider_metrics(self) -> list[dict[str, Any]]:
        return [dict(self._metrics[name]) for name in sorted(self._metrics)]

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch only declared tools; return JSON-safe output and metrics."""
        args = arguments if isinstance(arguments, dict) else {}
        if tool_name == "search_web":
            return await self.search_web(
                str(args.get("query", "")),
                _bounded_int(args.get("limit", 8), 8, 1, 20),
                max_requests=_bounded_int(args.get("_max_requests", 30), 30, 0, 30),
            )
        if tool_name == "search_official":
            domains = args.get("domains", [])
            return await self.search_official(
                str(args.get("query", "")),
                domains=domains if isinstance(domains, list) else [],
                limit=_bounded_int(args.get("limit", 8), 8, 1, 20),
                discovered_from=args.get("discovered_from"),
                max_requests=_bounded_int(args.get("_max_requests", 30), 30, 0, 30),
            )
        if tool_name == "read_webpage":
            return await self.read_webpage(str(args.get("url", "")))
        if tool_name == "inspect_evidence":
            return self.inspect_evidence(
                str(args.get("candidate_id", "")), str(args.get("field", ""))
            )
        return {
            "records": [],
            "evidence": None,
            "evidence_gaps": ["Unknown research tool was rejected."],
            "error_code": "unknown_tool",
            "provider": None,
            "source_count": 0,
            "providers": self.provider_metrics(),
        }

    async def search_web(
        self,
        query: str,
        limit: int = 8,
        *,
        max_requests: int | None = None,
    ) -> dict[str, Any]:
        clean_query = _clean_query(query)
        if not clean_query:
            return self._search_error("empty_query", "A non-empty search query is required.")
        return await self._search(
            clean_query, limit, domains=(), discovered_from=None,
            max_requests=max_requests,
        )

    async def search_official(
        self,
        query: str,
        *,
        domains: Sequence[str] = (),
        limit: int = 8,
        discovered_from: str | None = None,
        max_requests: int | None = None,
    ) -> dict[str, Any]:
        clean_query = _clean_query(query)
        if not clean_query:
            return self._search_error("empty_query", "A non-empty official-source query is required.")
        normalized_domains = _normalize_domains(domains)
        if not normalized_domains and self.settings:
            normalized_domains = _normalize_domains(self.settings.official_source_domains)
        # Without a known domain, search can still discover a candidate public
        # official site, but no result is ever declared verified from that fact.
        targeted_query = clean_query
        if normalized_domains:
            targeted_query = _site_query(clean_query, normalized_domains)
        elif not re.search(r"\b(official|官网|官方|政府|gov|career|招聘)\b", clean_query, re.I):
            targeted_query = f"{clean_query} 官网 官方来源"
        return await self._search(
            targeted_query,
            limit,
            domains=normalized_domains,
            discovered_from=discovered_from,
            max_requests=max_requests,
            official_candidate=True,
        )

    async def read_webpage(self, url: str) -> dict[str, Any]:
        try:
            evidence = await self.reader.read(url)
        except WebReadError as exc:
            return {
                "records": [],
                "evidence": None,
                "evidence_gaps": [str(exc)],
                "error_code": exc.code,
                "provider": None,
                "source_count": 0,
                "providers": self.provider_metrics(),
            }
        return {
            "records": [],
            "evidence": evidence,
            "evidence_gaps": [],
            "provider": None,
            "source_count": 1,
            "providers": self.provider_metrics(),
        }

    def inspect_evidence(self, candidate_id: str, field: str) -> dict[str, Any]:
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            return {
                "records": [], "evidence": None,
                "evidence_gaps": ["Candidate was not present in the current research context."],
                "error_code": "candidate_not_found", "provider": None, "source_count": 0,
                "providers": self.provider_metrics(),
            }
        evidence: list[dict[str, str]] = []
        checks = candidate.get("field_verification")
        field_check = checks.get(field) if isinstance(checks, Mapping) else None
        if isinstance(field_check, Mapping):
            quote = field_check.get("quote") or field_check.get("evidence")
            source_url = field_check.get("source_url") or field_check.get("evidence_url")
            value = field_check.get("value")
            text = _candidate_evidence_text(candidate)
            if isinstance(quote, str) and quote and quote in text:
                evidence.append({
                    "field": field,
                    "value": str(value or ""),
                    "quote": quote,
                    "url": str(source_url or candidate.get("official_source_url") or candidate.get("url") or ""),
                    "status": str(field_check.get("status", "unverified")),
                })
        label_aliases = {
            "graduate_cohort": ("毕业届", "毕业年份", "graduate cohort", "graduation year"),
            "major": ("专业", "major", "discipline"),
            "role": ("岗位", "职位", "role", "position"),
            "deadline": ("截止", "截止日期", "deadline", "closing date"),
            "open_status": ("招聘状态", "开放状态", "open status", "currently hiring"),
            "application_url": ("投递", "申请链接", "application", "apply"),
            "organization": ("企业", "公司", "单位", "organization", "company"),
            "region": ("地区", "区域", "region", "location"),
        }
        source_text = _candidate_evidence_text(candidate)
        for condition in _as_list(candidate.get("key_conditions")):
            if not isinstance(condition, Mapping):
                continue
            label = str(condition.get("field") or condition.get("label") or "").casefold()
            if field != condition.get("field") and not any(alias.casefold() in label for alias in label_aliases.get(field, (field.casefold(),))):
                continue
            quote = condition.get("quote")
            if isinstance(quote, str) and quote and quote in source_text:
                evidence.append({
                    "field": field,
                    "value": str(condition.get("value", "")),
                    "quote": quote,
                    "url": str(condition.get("evidence_url") or candidate.get("official_source_url") or candidate.get("url") or ""),
                    "status": str(condition.get("status", "unverified")),
                })
        for fact in _as_list(candidate.get("llm_verification", {}).get("facts") if isinstance(candidate.get("llm_verification"), Mapping) else []):
            if not isinstance(fact, Mapping) or fact.get("field") != field:
                continue
            quote = fact.get("quote")
            if isinstance(quote, str) and quote and quote in _candidate_evidence_text(candidate):
                evidence.append({
                    "field": field,
                    "value": str(fact.get("value", "")),
                    "quote": quote,
                    "url": str(fact.get("evidence_url", "")),
                    "status": str(fact.get("assessment", "unverified")),
                })
        unique = _dedupe_dicts(evidence)
        return {
            "records": [],
            "evidence": {
                "candidate_id": candidate_id,
                "field": field,
                "status": "available" if unique else "missing",
                "claims": unique,
            },
            "evidence_gaps": [] if unique else [f"No source-backed quote is available for field '{field}'."],
            "provider": None,
            "source_count": len(unique),
            "providers": self.provider_metrics(),
        }

    async def _search(
        self,
        query: str,
        limit: int,
        *,
        domains: Sequence[str],
        discovered_from: str | None,
        max_requests: int | None,
        official_candidate: bool = False,
    ) -> dict[str, Any]:
        ceiling = len(self._provider_objects) if max_requests is None else min(len(self._provider_objects), max(0, max_requests))
        selected = list(self._provider_objects.items())[:ceiling]
        if not selected:
            unconfigured = [name for name, metric in self._metrics.items() if not metric["configured"]]
            code = "search_budget_exhausted" if max_requests == 0 else "search_provider_unconfigured"
            reason = (
                "The search request budget is exhausted."
                if max_requests == 0
                else "No configured search provider is available; configure the selected provider's credentials locally."
            )
            return self._search_error(code, reason, source_count=0, provider=unconfigured[0] if len(unconfigured) == 1 else None)

        started = time.perf_counter()

        async def invoke(name: str, provider: Any) -> tuple[str, list[dict[str, Any]], str | None, int]:
            began = time.perf_counter()
            try:
                try:
                    records = await provider.search(query, limit, include_domains=domains or None)
                except TypeError:
                    records = await provider.search(query, limit)
                return name, [item for item in records if isinstance(item, Mapping)], None, max(0, int((time.perf_counter() - began) * 1000))
            except Exception as exc:
                # Only the exception class is retained; remote error bodies may
                # contain request data and are intentionally not logged.
                return name, [], type(exc).__name__, max(0, int((time.perf_counter() - began) * 1000))

        outputs = await asyncio.gather(*(invoke(name, provider) for name, provider in selected))
        records: list[dict[str, Any]] = []
        failures: list[str] = []
        source_count = 0
        providers_used: list[str] = []
        for name, items, error, latency in outputs:
            metric = self._metrics[name]
            metric["requests"] += 1
            metric["latency_ms"] += latency
            metric["returned_sources"] += len(items)
            source_count += len(items)
            providers_used.append(name)
            if error:
                metric["failures"] += 1
                metric["error_code"] = _safe_error_code(error)
                failures.append(name)
            for item in items:
                normalized = _normalize_source(item, name)
                if not normalized:
                    continue
                if domains and not _domain_matches(_url_host(normalized["url"]), domains):
                    continue
                if official_candidate:
                    normalized["source_role"] = "official_candidate"
                    normalized["official_domain_match"] = bool(domains)
                    normalized["field_verification"] = "not_verified"
                if discovered_from:
                    normalized["discovered_from"] = discovered_from[:2048]
                records.append(normalized)
        records = _dedupe_sources(records)[: min(max(1, limit), self.max_candidates)]
        if failures and len(failures) == len(selected):
            return {
                "records": [], "evidence_gaps": ["All configured search providers failed for this request."],
                "error_code": "all_search_providers_failed", "provider": None,
                "source_count": source_count, "search_requests": len(selected),
                "providers": self.provider_metrics(),
            }
        gaps: list[str] = []
        if failures:
            gaps.append("One or more configured search providers failed; partial results are returned.")
        if not records:
            gaps.append("No new public sources were returned for this query.")
        if official_candidate and domains and not records:
            gaps.append("No result matched the requested official domains; no official source was verified.")
        return {
            "records": records,
            "evidence_gaps": gaps,
            "provider": providers_used[0] if len(providers_used) == 1 else None,
            "source_count": source_count,
            "search_requests": len(selected),
            "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
            "providers": self.provider_metrics(),
            "partial_results": bool(failures and records),
        }

    def _search_error(
        self,
        code: str,
        message: str,
        *,
        source_count: int = 0,
        provider: str | None = None,
    ) -> dict[str, Any]:
        return {
            "records": [], "evidence_gaps": [message], "error_code": code,
            "provider": provider, "source_count": source_count,
            "search_requests": 0, "providers": self.provider_metrics(),
        }


def _metric(name: str, configured: bool) -> dict[str, Any]:
    return {
        "provider": name,
        "configured": bool(configured),
        "requests": 0,
        "returned_sources": 0,
        "failures": 0,
        "latency_ms": 0,
        "error_code": None,
    }


def _clean_query(value: str) -> str:
    return " ".join(str(value or "").split())[:600]


def _bounded_int(value: Any, fallback: int, low: int, high: int) -> int:
    try:
        return min(high, max(low, int(value)))
    except (TypeError, ValueError):
        return fallback


def _normalize_domains(domains: Sequence[Any]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in domains:
        if not isinstance(value, str):
            continue
        domain = value.strip().lower().removeprefix("www.").rstrip(".")
        try:
            domain = domain.encode("idna").decode("ascii")
        except UnicodeError:
            continue
        try:
            ipaddress.ip_address(domain)
            continue
        except ValueError:
            pass
        if domain in {"localhost", "local", "internal"} or not _DOMAIN_RE.fullmatch(domain):
            continue
        if domain not in normalized:
            normalized.append(domain)
    return tuple(normalized[:4])


def _site_query(query: str, domains: Sequence[str]) -> str:
    prefix = ""
    chosen: list[str] = []
    for domain in domains:
        candidate = " OR ".join([*(f"site:{item}" for item in chosen), f"site:{domain}"])
        if len(candidate) > 350:
            break
        chosen.append(domain)
        prefix = candidate
    if not prefix:
        return query[:600]
    prefix += " "
    return (prefix + query[: max(0, 600 - len(prefix))]).strip()


def _canonical_public_http_url(value: str) -> str | None:
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except (ValueError, AttributeError):
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme.lower() not in {"http", "https"} or not host or parsed.username or parsed.password:
        return None
    if port not in {None, 80, 443}:
        return None
    try:
        address = ipaddress.ip_address(host)
        if not address.is_global:
            return None
    except ValueError:
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home")):
            return None
    query = urlencode([
        (key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMS
    ], doseq=True)
    netloc = host.encode("idna").decode("ascii")
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", query, ""))


def _normalize_source(item: Mapping[str, Any], provider: str) -> dict[str, Any] | None:
    url = item.get("url")
    title = item.get("title")
    if not isinstance(url, str) or not isinstance(title, str) or not title.strip():
        return None
    canonical = _canonical_public_http_url(url)
    if not canonical:
        return None
    result: dict[str, Any] = {
        "title": title.strip()[:1000],
        "url": canonical,
        "source_provider": provider,
    }
    for field in ("content", "raw_content", "description", "snippet"):
        if isinstance(item.get(field), str):
            result[field] = item[field][:12_000]
    for field in ("score", "authority_score", "published_at", "published_date", "page_age", "date", "author", "source_name", "language"):
        value = item.get(field)
        if isinstance(value, (str, int, float, bool)):
            result[field] = str(value)[:500] if isinstance(value, str) else value
    return result


def _url_host(value: str) -> str:
    try:
        return (urlsplit(value).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _domain_matches(host: str, domains: Sequence[str]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _dedupe_sources(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for record in records:
        key = record["url"]
        if key not in seen:
            seen.add(key)
            output.append(record)
    return output


def _candidate_evidence_text(candidate: Mapping[str, Any]) -> str:
    values = [candidate.get("evidence_detail"), candidate.get("official_evidence_detail"), candidate.get("evidence")]
    sources = candidate.get("evidence_sources")
    for source in _as_list(sources):
        if isinstance(source, Mapping):
            values.extend((source.get("quote"), source.get("text"), source.get("content")))
    return "\n".join(value for value in values if isinstance(value, str))


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        data = dump(mode="json")
        return data if isinstance(data, dict) else {}
    return {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dedupe_dicts(items: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    output = []
    for item in items:
        key = tuple(sorted(item.items()))
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _safe_error_code(exception_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", exception_name)[:80].lower() or "provider_error"
