import hashlib
import re
from datetime import date
from typing import Any, Optional
from urllib.parse import urlparse

from .schemas import SearchResult


def clean_and_ground_results(
    query: str,
    raw_results: list[dict[str, Any]],
    limit: int,
    official_source_domains: tuple[str, ...] = (),
) -> list[SearchResult]:
    """小洗: normalize provider records and retain only source-grounded content."""
    cleaned: list[SearchResult] = []
    seen_urls: set[str] = set()

    for item in raw_results:
        title = _clean_text(item.get("title"))
        url = _clean_url(item.get("url"))
        if not title or not url or url in seen_urls:
            continue
        seen_urls.add(url)

        content = _provider_text(item.get("raw_content")) or _provider_text(
            item.get("content")
        )
        snippet_source = _clean_text(item.get("content")) or content
        snippet = snippet_source[:500]
        # Evidence is copied verbatim from provider-returned content. A source
        # record without textual content remains visible but is explicitly
        # marked unverified.
        evidence = _evidence_excerpt(query, content)
        matched_terms = _matching_terms(query, evidence)
        if evidence and matched_terms:
            verdict = "context"
            confidence = min(0.72, 0.35 + 0.08 * len(matched_terms))
            match_reason = (
                "检索来源文本包含与问题相关的词项："
                + "、".join(matched_terms[:6])
                + "。该匹配表示相关上下文，尚不单独证明结论。"
            )
        elif evidence:
            verdict = "unverified"
            confidence = 0.2
            match_reason = "搜索源返回了文本，但未发现明确的查询词项匹配；需要进一步核验。"
        else:
            verdict = "unverified"
            confidence = 0.0
            match_reason = "搜索源未返回可引用的正文片段，当前无法核验。"

        parsed = urlparse(url)
        cleaned.append(
            SearchResult(
                id=hashlib.sha256(url.encode("utf-8")).hexdigest()[:20],
                title=title,
                url=url,
                source_name=(parsed.hostname or "unknown source").removeprefix("www."),
                snippet=snippet,
                evidence=evidence,
                match_reason=match_reason,
                verdict=verdict,
                confidence=confidence,
                published_at=_clean_optional_text(item.get("published_date")),
                deadline_status=_deadline_status(content),
                official_source_status=_official_source_status(
                    parsed.hostname or "", official_source_domains
                ),
                evidence_status="available" if evidence else "missing",
            )
        )
        if len(cleaned) >= limit:
            break
    return cleaned


def apply_grounded_annotations(
    results: list[SearchResult], annotations: list[dict[str, Any]]
) -> list[SearchResult]:
    """小验/小判: accept model labels only for retrieved IDs and exact quotes."""
    by_id = {result.id: result for result in results}
    allowed_verdicts = {"supports", "contradicts", "context", "unverified"}
    for annotation in annotations:
        result_id = annotation.get("id")
        result = by_id.get(result_id) if isinstance(result_id, str) else None
        if result is None:
            continue

        quote = annotation.get("evidence")
        if not isinstance(quote, str) or not quote or quote not in result.evidence:
            # A model cannot add evidence that was absent from provider content.
            result.verdict = "unverified"
            result.confidence = 0.0
            result.evidence = ""
            result.evidence_status = "missing"
            result.match_reason = "模型未能提供与检索来源正文完全一致的证据；当前无法核验。"
            continue

        verdict = annotation.get("verdict")
        try:
            confidence = float(annotation.get("confidence"))
        except (TypeError, ValueError):
            continue
        if verdict not in allowed_verdicts or not 0 <= confidence <= 1:
            continue

        result.evidence = quote
        result.evidence_status = "available"
        result.verdict = verdict
        result.confidence = confidence
        result.match_reason = {
            "supports": "模型将来源中的这段原文判为支持查询；证据引用已逐字匹配来源正文。",
            "contradicts": "模型将来源中的这段原文判为与查询相矛盾；证据引用已逐字匹配来源正文。",
            "context": "模型将来源中的这段原文判为相关上下文；证据引用已逐字匹配来源正文。",
            "unverified": "来源正文有匹配引用，但模型认为该引用不足以核验查询。",
        }[verdict]
    return results


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _provider_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clean_optional_text(value: Any) -> Optional[str]:
    text = _clean_text(value)
    return text or None


def _official_source_status(
    hostname: str, configured_domains: tuple[str, ...]
) -> str:
    host = hostname.lower().removeprefix("www.").rstrip(".")
    if host.endswith((".gov", ".gov.cn", ".mil")):
        return "verified"
    if any(host == domain or host.endswith(f".{domain}") for domain in configured_domains):
        return "verified"
    return "unverified"


def _deadline_status(content: str, today: date | None = None) -> str:
    cues = (
        "截止", "截至", "有效期至", "最后申请日期", "最后报名日期",
        "deadline", "due date", "expires on", "expiration date", "closes on",
    )
    lowered = content.lower()
    positions = [position for cue in cues if (position := lowered.find(cue)) >= 0]
    if not positions:
        return "not_stated"

    dates = _source_dates(content)
    if not dates:
        return "unverified"

    # Require an explicit deadline cue and a date close to it. This avoids
    # treating an unrelated publication date as an expiry date.
    nearest: tuple[int, date] | None = None
    for cue_position in positions:
        for date_position, parsed_date in dates:
            distance = abs(date_position - cue_position)
            if distance <= 40 and (nearest is None or distance < nearest[0]):
                nearest = (distance, parsed_date)
    if nearest is None:
        return "unverified"
    return "expired" if nearest[1] < (today or date.today()) else "not_expired"


def _source_dates(content: str) -> list[tuple[int, date]]:
    patterns = (
        r"(?P<year>20\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})",
        r"(?P<year>20\d{2})年\s*(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日?",
    )
    found: list[tuple[int, date]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, content):
            try:
                parsed = date(
                    int(match.group("year")),
                    int(match.group("month")),
                    int(match.group("day")),
                )
            except ValueError:
                continue
            found.append((match.start(), parsed))
    return found


def _clean_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    url = value.strip()
    parsed = urlparse(url)
    return url if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def _evidence_excerpt(query: str, content: str, max_chars: int = 700) -> str:
    if not content:
        return ""
    # Select a relevant source passage while preserving it as a literal slice
    # of the provider-returned text.
    passages = [
        passage.strip()
        for passage in re.split(r"\n\s*\n|(?<=[.!?。！？])\s+", content)
        if passage.strip()
    ]
    if not passages:
        return ""
    passage = max(passages, key=lambda item: len(_matching_terms(query, item)))
    if len(passage) <= max_chars:
        return passage

    lowered = passage.lower()
    positions = [
        position
        for term in _terms(query)
        if (position := lowered.find(term.lower())) >= 0
    ]
    start = max(0, min(positions, default=0) - 160)
    return passage[start : start + max_chars].strip()


def _matching_terms(query: str, content: str) -> list[str]:
    query_terms = _terms(query)
    content_terms = set(_terms(content))
    return [term for term in query_terms if term in content_terms]


def _terms(text: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}|[\u3400-\u9fff]", text.lower())
    expanded: list[str] = []
    for term in terms:
        if len(term) == 1 and "\u3400" <= term <= "\u9fff":
            continue
        expanded.append(term)
    # Add adjacent CJK bigrams, which work for Chinese queries without an
    # external tokenizer while remaining deterministic.
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        expanded.extend(run[index : index + 2] for index in range(len(run) - 1))
    return list(dict.fromkeys(expanded))
