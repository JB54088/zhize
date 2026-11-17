"""Build bounded, allowlisted source records for the API's private archive."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from .schemas import SourceRecord


MAX_SOURCE_RECORDS = 500
MAX_RAW_TEXT_CHARS = 12_000
_METADATA_KEYS = (
    "score",
    "rank",
    "language",
    "content_type",
    "content_hash",
    "provider_record_index",
    "search_query_index",
)


def build_source_records(records: Sequence[Mapping[str, Any]]) -> list[SourceRecord]:
    """Preserve each collected URL/body and safe acquisition metadata.

    Arbitrary provider payload keys are deliberately omitted so credentials,
    headers, and provider-specific opaque objects cannot enter persistence.
    """
    output: list[SourceRecord] = []
    seen: set[str] = set()
    for item in records:
        if len(output) >= MAX_SOURCE_RECORDS:
            break
        if not isinstance(item, Mapping):
            continue
        title = _text(item.get("title"))
        url = _http_url(item.get("url"))
        if not title or not url:
            continue
        provenance = item.get("_provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        provider = _optional_text(provenance.get("provider") or item.get("source_provider"), 80)
        provider_index = _optional_int(provenance.get("provider_record_index"))
        query_index = _optional_int(provenance.get("search_query_index"))
        query = _optional_text(provenance.get("search_query"), 4000)
        identity = hashlib.sha256(
            f"{provider or ''}|{url}|{provider_index}|{query_index}".encode("utf-8")
        ).hexdigest()[:32]
        if identity in seen:
            continue
        seen.add(identity)

        raw_value = item.get("raw_content")
        if not isinstance(raw_value, str) or not raw_value.strip():
            raw_value = item.get("content")
        raw_text = raw_value.strip() if isinstance(raw_value, str) else ""
        truncated = len(raw_text) > MAX_RAW_TEXT_CHARS
        raw_text = raw_text[:MAX_RAW_TEXT_CHARS]
        metadata = _safe_metadata(item, provenance)
        hostname = (urlparse(url).hostname or "").removeprefix("www.")
        output.append(SourceRecord(
            id=identity,
            title=title[:1000],
            url=url,
            source_name=_optional_text(item.get("source_name") or hostname, 240),
            provider=provider,
            captured_at=_optional_text(provenance.get("captured_at") or item.get("captured_at"), 80),
            published_at=_optional_text(item.get("published_date") or item.get("published_at"), 120),
            search_query_index=query_index,
            search_query=query,
            raw_text=raw_text,
            raw_text_truncated=truncated,
            metadata=metadata,
        ))
    return output


def _safe_metadata(item: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, str | int | float | bool | None]:
    metadata: dict[str, str | int | float | bool | None] = {}
    for key in _METADATA_KEYS:
        value = provenance.get(key, item.get(key))
        if isinstance(value, bool) or value is None:
            metadata[key] = value
        elif isinstance(value, int):
            metadata[key] = value
        elif isinstance(value, float):
            metadata[key] = value if value == value and abs(value) != float("inf") else None
        elif isinstance(value, str):
            metadata[key] = value.strip()[:500]
    return metadata


def _http_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    parsed = urlparse(cleaned)
    return cleaned if parsed.scheme in {"http", "https"} and parsed.hostname else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:limit]


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
