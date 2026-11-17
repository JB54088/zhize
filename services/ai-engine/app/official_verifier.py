"""Conservative, source-grounded verification for procurement notices.

This module only consumes records returned by an injected search callback. It
never requests result URLs or follows redirects, which keeps arbitrary search
result links out of the network-fetch path.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo


SearchRecord = Mapping[str, Any]
SearchCallback = Callable[
    [str, Sequence[str], int], Awaitable[Sequence[SearchRecord]]
]

_DEFAULT_OFFICIAL_DOMAINS = ("ccgp.gov.cn", "gov.cn")
_PROCUREMENT_NOTICE_MARKERS = (
    "采购公告",
    "招标公告",
    "竞争性磋商公告",
    "竞争性谈判公告",
    "询价公告",
    "单一来源采购公告",
    "单一来源公告",
    "采购意向公告",
    "采购结果公告",
    "采购成交公告",
    "中标公告",
    "成交公告",
)
_FIELD_ORDER = (
    "project_name",
    "purchasing_unit",
    "published_at",
    "budget",
    "deadline",
    "evidence",
)
_PURCHASING_UNIT_LABELS = (
    "采购单位名称",
    "采购人名称",
    "采购单位",
    "采购人",
    "招标人",
)
_BUDGET_LABELS = (
    "采购预算金额",
    "项目预算金额",
    "最高投标限价",
    "最高限价",
    "预算金额",
    "采购预算",
    "项目预算",
    "预算",
)
_PUBLISHED_LABELS = (
    "公告发布时间",
    "采购公告时间",
    "公告发布日期",
    "发布日期",
    "发布时间",
    "公告日期",
    "公告时间",
    "信息时间",
)
_DEADLINE_LABELS = (
    "提交首次响应文件截止时间及地点",
    "响应文件递交截止时间及地点",
    "响应文件递交截止时间",
    "响应文件提交截止时间",
    "提交投标文件截止时间",
    "投标文件递交截止时间",
    "投标文件提交截止时间",
    "投标截止时间",
    "响应文件截止时间",
    "响应截止时间",
    "报价截止时间",
    "报名截止时间",
    "截止时间",
)
_VALUE_STOP_LABELS = tuple(sorted(set(
    _PURCHASING_UNIT_LABELS
    + _BUDGET_LABELS
    + _PUBLISHED_LABELS
    + _DEADLINE_LABELS
    + ("采购代理机构", "代理机构", "联系人", "联系电话", "联系方式", "联系地址", "联系电话")
), key=len, reverse=True))
_DATE = (
    r"(?:\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"(?:\s*(?:\d{1,2}\s*(?:时|点)\s*\d{1,2}\s*分"
    r"(?:\s*\d{1,2}\s*秒)?|\d{1,2}:\d{2}(?::\d{2})?))?"
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?)"
)
_BUDGET_RE = re.compile(
    r"(?P<label>采购预算金额|项目预算金额|最高投标限价|最高限价|预算金额|采购预算|项目预算|预算)"
    r"\s*[:：]?\s*(?P<value>(?:(?:人民币|RMB|CNY)\s*)?[¥￥]?\s*"
    r"\d[\d,]*(?:\.\d+)?\s*(?:元|万元|亿元)?)",
    re.IGNORECASE,
)


async def verify_official_source(
    *,
    project_name: str,
    support_claim: str,
    search: SearchCallback,
    official_domains: Sequence[str] = (),
    limit: int = 10,
) -> dict[str, Any]:
    """Find and verify an official procurement notice using source text only.

    The callback signature is ``await search(query, include_domains, limit)``.
    The ``include_domains`` list is a search restriction, not proof of source
    authenticity. Every returned record is independently URL- and text-checked.

    ``support_claim`` is an optional search hint. ``evidence`` is populated
    only when an exact source quote is found in the official notice;
    paraphrases remain unverified because this module does not infer semantics.
    """
    project_name = project_name.strip()
    support_claim = support_claim.strip()
    if not project_name:
        raise ValueError("project_name must not be empty")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    include_domains = _search_domains(official_domains)
    claim_term = f' "{support_claim}"' if support_claim else ""
    search_query = f'"{project_name}"{claim_term} 采购公告'
    records = await search(search_query, include_domains, limit)

    official_candidates: list[dict[str, Any]] = []
    relevant_third_party = False
    for record in records:
        if not isinstance(record, Mapping):
            continue
        title = _text(record.get("title"))
        url = _text(record.get("url"))
        body = _record_body(record)
        if not body:
            continue

        # A procurement notice's official title can establish the project
        # name even when the extracted body begins below the page heading.
        project_quote = _find_quote(body, project_name) or _find_quote(title, project_name)
        claim_quote = _find_quote(body, support_claim) if support_claim else None
        if not _is_official_candidate_url(url, include_domains):
            if project_quote:
                relevant_third_party = True
            continue
        parsed_url = urlsplit(url)
        if parsed_url.path in {"", "/"} and _is_portal_homepage_title(title):
            # Search-engine snippets from a government portal homepage can
            # contain unrelated notices. The homepage itself is not the
            # original procurement announcement.
            continue
        if not project_quote or not _is_procurement_notice(title, body):
            # A government-looking result for another project is not a
            # matching original procurement notice and must not shadow a
            # relevant third-party hit.
            continue

        extracted = _extract_fields(body)
        official_candidates.append(
            {
                "title": title,
                "url": url,
                "body": body,
                "project_quote": project_quote,
                "claim_quote": claim_quote,
                "fields": extracted,
                "deadline_label_present": _has_labeled_date(body, _DEADLINE_LABELS),
            }
        )

    if not official_candidates:
        status = "third_party" if relevant_third_party else "insufficient_evidence"
        return _empty_result(status)

    # Prefer a source that directly names the project and claim, then one that
    # contains more labeled procurement fields. Stable order breaks ties.
    selected = max(
        official_candidates,
        key=lambda item: (
            bool(item["project_quote"]),
            bool(item["claim_quote"]),
            len(item["fields"]),
        ),
    )
    return _result_for_candidate(selected)


def _empty_result(status: str) -> dict[str, Any]:
    return {
        "official_source_status": status,
        "official_source_url": None,
        "verified_fields": [],
        "unverified_fields": list(_FIELD_ORDER),
        "purchasing_unit": None,
        "budget": None,
        "deadline_at": None,
        "deadline_status": "unverified",
        "published_at": None,
        "evidence": "",
    }


def _result_for_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    # A result on an official-looking host is not the requested notice unless
    # the body itself contains the exact project name.
    if not candidate.get("project_quote"):
        return _empty_result("insufficient_evidence")

    fields = candidate["fields"]
    values: dict[str, Optional[str]] = {
        "purchasing_unit": fields.get("purchasing_unit", {}).get("value"),
        "budget": fields.get("budget", {}).get("value"),
        "deadline_at": fields.get("deadline_at", {}).get("value"),
        "published_at": fields.get("published_at", {}).get("value"),
    }
    deadline = values["deadline_at"]
    if deadline:
        deadline_status = _deadline_status(deadline)
    elif candidate.get("deadline_label_present"):
        deadline_status = "unverified"
    else:
        deadline_status = "not_stated"
    quotes: dict[str, str] = {}
    if candidate.get("project_quote"):
        quotes["project_name"] = candidate["project_quote"]
    if candidate.get("claim_quote"):
        quotes["evidence"] = _quote_context(
            candidate["body"], candidate["claim_quote"]
        )
    for field_name, field in fields.items():
        quotes[field_name] = field["quote"]

    region = _verified_region_quote(candidate["project_quote"], candidate["body"])
    if region:
        values["region"] = region["value"]
        quotes["region"] = region["quote"]

    if "deadline_at" in quotes:
        quotes["deadline"] = quotes.pop("deadline_at")
    verified_fields = [field for field in _FIELD_ORDER if field in quotes]
    if "region" in quotes:
        verified_fields.append("region")
    unverified_fields = [field for field in _FIELD_ORDER if field not in quotes]
    enough_to_verify = "project_name" in quotes
    return {
        "official_source_status": "verified" if enough_to_verify else "insufficient_evidence",
        "official_source_url": candidate["url"] if "project_name" in quotes else None,
        "official_source_title": candidate["title"],
        "evidence_detail": candidate["body"],
        "verified_fields": verified_fields,
        "unverified_fields": unverified_fields,
        **values,
        "deadline_status": deadline_status,
        "evidence": quotes.get("evidence", ""),
        # Internal field-level contract consumed by the V0.2 verifier. The
        # existing procurement response fields remain unchanged.
        "field_evidence": {
            field: {"value": ({"deadline": values.get("deadline_at")}.get(field) or values.get(field) or quote), "quote": quote}
            for field, quote in quotes.items()
            if field != "evidence"
        } | ({"evidence": {"value": quotes["evidence"], "quote": quotes["evidence"]}} if quotes.get("evidence") else {}),
    }


def _verified_region_quote(title: str, body: str) -> Optional[dict[str, str]]:
    region_pattern = re.compile(
        r"(?:北京|天津|上海|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆)(?:省|市|自治区|特别行政区)?"
    )
    for match in region_pattern.finditer(title):
        return {"value": match.group(), "quote": match.group()}
    for line in body.splitlines():
        if not any(label in line for label in ("项目地点", "实施地点", "服务地点", "采购区域", "项目所在地")):
            continue
        match = region_pattern.search(line)
        if match:
            return {"value": match.group(), "quote": line.strip()[:240]}
    return None


def _search_domains(official_domains: Sequence[str]) -> tuple[str, ...]:
    values: set[str] = set(_DEFAULT_OFFICIAL_DOMAINS)
    if isinstance(official_domains, str):
        official_domains = (official_domains,)
    for domain in official_domains:
        normalized = _normalize_domain(domain)
        if normalized:
            values.add(normalized)
    return tuple(sorted(values))


def _normalize_domain(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    domain = value.strip().lower().rstrip(".").lstrip(".")
    if not domain or ":" in domain or "/" in domain or "@" in domain:
        return None
    try:
        domain = domain.encode("idna").decode("ascii")
        ipaddress.ip_address(domain)
    except ValueError:
        pass
    except UnicodeError:
        return None
    else:
        return None
    if not re.fullmatch(r"(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", domain):
        return None
    return domain


def _is_official_candidate_url(url: str, domains: Sequence[str]) -> bool:
    """Validate a candidate without ever dereferencing it."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme.lower() != "https" or not host or parsed.username or parsed.password:
        return False
    if port not in (None, 443):
        return False
    host = host.lower().rstrip(".")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        # IP literals are rejected even when globally routable. Official
        # candidate discovery should be domain based, never IP based.
        return False
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _record_body(record: SearchRecord) -> str:
    raw_content = record.get("raw_content")
    content = record.get("content")
    if isinstance(raw_content, str) and raw_content.strip():
        return raw_content.strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def _is_procurement_notice(title: str, body: str) -> bool:
    text = title + "\n" + body
    return any(marker in text for marker in _PROCUREMENT_NOTICE_MARKERS)


def _is_portal_homepage_title(title: str) -> bool:
    normalized = _normalize(title).strip(" _-｜|")
    return normalized in {
        "中央政府采购网",
        "中国政府采购网",
        "福建省政府采购网",
        "政府采购网",
    }


def _find_quote(source: str, phrase: str) -> Optional[str]:
    normalized_source, positions = _normalized_with_positions(source)
    normalized_phrase = _normalize(phrase)
    if not normalized_phrase:
        return None
    start = normalized_source.find(normalized_phrase)
    if start < 0:
        return None
    end = start + len(normalized_phrase) - 1
    raw_start = positions[start]
    raw_end = positions[end] + 1
    return source[raw_start:raw_end]


def _quote_context(source: str, phrase: str, limit: int = 400) -> str:
    """Return the source sentence containing an exactly matched phrase."""
    exact_phrase = _find_quote(source, phrase)
    if not exact_phrase:
        return ""
    start = source.find(exact_phrase)
    if start < 0:
        return exact_phrase
    end = start + len(exact_phrase)
    boundaries = "。！？!?；;\n\r"
    sentence_start = max(
        (source.rfind(mark, 0, start) for mark in boundaries), default=-1
    ) + 1
    stops = [source.find(mark, end) for mark in boundaries]
    stops = [position for position in stops if position >= 0]
    sentence_end = min(stops) + 1 if stops else len(source)
    excerpt = source[sentence_start:sentence_end].strip()
    if len(excerpt) <= limit:
        return excerpt
    left = max(sentence_start, start - max(0, (limit - len(exact_phrase)) // 2))
    right = min(sentence_end, left + limit)
    left = min(left, max(sentence_start, right - limit))
    return source[left:right].strip()


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalized_with_positions(source: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    positions: list[int] = []
    for index, original in enumerate(source):
        for character in unicodedata.normalize("NFKC", original).casefold():
            if character.isspace():
                if characters and characters[-1] != " ":
                    characters.append(" ")
                    positions.append(index)
            else:
                characters.append(character)
                positions.append(index)
    while characters and characters[-1] == " ":
        characters.pop()
        positions.pop()
    while characters and characters[0] == " ":
        characters.pop(0)
        positions.pop(0)
    return "".join(characters), positions


def _extract_fields(text: str) -> dict[str, dict[str, str]]:
    fields: dict[str, dict[str, str]] = {}
    unit = _extract_purchasing_unit(text)
    if unit:
        fields["purchasing_unit"] = unit

    budget = _BUDGET_RE.search(text)
    if budget:
        fields["budget"] = {
            "value": budget.group("value").strip(),
            "quote": budget.group(0).strip(),
        }

    published = _extract_labeled_date(text, _PUBLISHED_LABELS)
    if published is None:
        published = _extract_source_published_date(text)
    if published:
        fields["published_at"] = published

    deadline = _extract_labeled_date(text, _DEADLINE_LABELS)
    if deadline:
        fields["deadline_at"] = deadline
    return fields


def _extract_source_published_date(text: str) -> Optional[dict[str, str]]:
    match = re.search(
        rf"来源\s*[:：]\s*本网\s*时间\s*[:：]?\s*(?P<value>{_DATE})",
        text,
    )
    if not match:
        return None
    value = match.group("value").strip()
    if not _valid_date_value(value):
        return None
    return {"value": value, "quote": match.group(0).strip()}


def _extract_labeled_value(
    text: str,
    labels: Sequence[str],
    label_pattern: Optional[str] = None,
) -> Optional[dict[str, str]]:
    labels_pattern = label_pattern or "|".join(
        re.escape(label) for label in sorted(labels, key=len, reverse=True)
    )
    pattern = re.compile(rf"(?P<label>{labels_pattern})\s*[:：]\s*(?P<value>[^\r\n;；。]+)")
    stop_pattern = re.compile(
        r"\s+(?:" + "|".join(re.escape(label) for label in _VALUE_STOP_LABELS) + r")\s*[:：]"
    )
    for segment in re.split(r"[\r\n;；。]+", text):
        match = pattern.search(segment)
        if not match:
            continue
        raw_value = match.group("value")
        next_label = stop_pattern.search(raw_value)
        if next_label:
            raw_value = raw_value[:next_label.start()]
        value = raw_value.strip(" \t,，、")
        if not value or len(value) > 160:
            continue
        quote = segment[match.start():match.start("value") + len(raw_value)].strip()
        return {"value": value, "quote": quote}
    return None


def _extract_purchasing_unit(text: str) -> Optional[dict[str, str]]:
    labels_pattern = (
        r"采购人信息\s*[/／]\s*名称|"
        + "|".join(
            re.escape(label)
            for label in sorted(_PURCHASING_UNIT_LABELS, key=len, reverse=True)
        )
    )
    labeled = _extract_labeled_value(text, _PURCHASING_UNIT_LABELS, labels_pattern)
    if labeled:
        return labeled

    entrusted = re.search(r"受\s*(?P<value>[^，,；;\r\n。]{2,100}?)\s*委托", text)
    if not entrusted:
        return None
    value = entrusted.group("value").strip(" \t,，、")
    if not value:
        return None
    return {"value": value, "quote": entrusted.group(0).strip()}


def _extract_labeled_date(text: str, labels: Sequence[str]) -> Optional[dict[str, str]]:
    labels_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    pattern = re.compile(rf"(?P<label>{labels_pattern})\s*[:：]?\s*(?P<value>{_DATE})")
    for match in pattern.finditer(text):
        value = match.group("value").strip()
        if _valid_date_value(value):
            return {"value": value, "quote": match.group(0).strip()}
    return None


def _has_labeled_date(text: str, labels: Sequence[str]) -> bool:
    labels_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    return bool(re.search(rf"(?:{labels_pattern})\s*[:：]?", text))


def _valid_date_value(value: str) -> bool:
    patterns = (
        re.compile(
            r"(?P<year>\d{4})\s*年\s*(?P<month>\d{1,2})\s*月\s*"
            r"(?P<day>\d{1,2})\s*日(?:\s*(?P<hour>\d{1,2})\s*(?:时|点)\s*"
            r"(?P<minute>\d{1,2})\s*分(?:\s*(?P<second>\d{1,2})\s*秒)?)?"
        ),
        re.compile(
            r"(?P<year>\d{4})\s*年\s*(?P<month>\d{1,2})\s*月\s*"
            r"(?P<day>\d{1,2})\s*日(?:\s*(?P<hour>\d{1,2}):"
            r"(?P<minute>\d{2})(?::(?P<second>\d{2}))?)?"
        ),
        re.compile(
            r"(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})"
            r"(?:[ T](?P<hour>\d{1,2}):(?P<minute>\d{2})"
            r"(?::(?P<second>\d{2}))?)?"
        ),
    )
    for pattern in patterns:
        match = pattern.fullmatch(value)
        if not match:
            continue
        try:
            year = int(match.group("year"))
            month = int(match.group("month"))
            day = int(match.group("day"))
            if match.group("hour") is None:
                hour, minute, second, microsecond = 23, 59, 59, 999999
            else:
                hour = int(match.group("hour"))
                minute = int(match.group("minute"))
                second = int(match.group("second") or 0)
                microsecond = 0
            datetime(
                year,
                month,
                day,
                hour,
                minute,
                second,
                microsecond,
            )
        except (ValueError, TypeError):
            return False
        return True
    return False


def _deadline_status(value: str) -> str:
    deadline = _parse_deadline(value)
    if deadline is None:
        return "unverified"
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return "expired" if deadline < now else "not_expired"


def _parse_deadline(value: str) -> Optional[datetime]:
    if not _valid_date_value(value):
        return None
    patterns = (
        re.compile(
            r"(?P<year>\d{4})\s*年\s*(?P<month>\d{1,2})\s*月\s*"
            r"(?P<day>\d{1,2})\s*日(?:\s*(?:(?P<hour>\d{1,2})\s*(?:时|点)\s*"
            r"(?P<minute>\d{1,2})\s*分(?:\s*(?P<second>\d{1,2})\s*秒)?|"
            r"(?P<hour_colon>\d{1,2}):(?P<minute_colon>\d{2})"
            r"(?::(?P<second_colon>\d{2}))?))?"
        ),
        re.compile(
            r"(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})"
            r"(?:[ T](?P<hour>\d{1,2}):(?P<minute>\d{2})"
            r"(?::(?P<second>\d{2}))?)?"
        ),
    )
    for pattern in patterns:
        match = pattern.fullmatch(value)
        if not match:
            continue
        try:
            hour = match.group("hour")
            minute = match.group("minute")
            second = match.group("second")
            if hour is None and match.groupdict().get("hour_colon") is not None:
                hour = match.group("hour_colon")
                minute = match.group("minute_colon")
                second = match.group("second_colon")
            if hour is None:
                hour_value, minute_value, second_value, microsecond = 23, 59, 59, 999999
            else:
                hour_value = int(hour)
                minute_value = int(minute)
                second_value = int(second or 0)
                microsecond = 0
            return datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                hour_value,
                minute_value,
                second_value,
                microsecond,
                tzinfo=ZoneInfo("Asia/Shanghai"),
            )
        except (ValueError, TypeError):
            return None
    return None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
