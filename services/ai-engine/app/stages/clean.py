"""小洗: normalize search records, deduplicate, and extract grounded attributes."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..pipeline import clean_and_ground_results

_TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_MAJOR_TERMS = (
    "电子信息工程", "电子科学与技术", "通信工程", "计算机科学与技术",
    "软件工程", "人工智能", "自动化", "信息工程", "网络工程",
)
_QUALIFICATION_TERMS = (
    "学历", "学位", "专业要求", "资格要求", "资格证", "工作经验", "应届生",
    "毕业届别", "毕业生", "本科", "硕士", "博士", "英语", "证书",
)
_BUSINESS_TERMS = (
    "岗位职责", "工作内容", "采购内容", "项目概况", "业务范围", "服务内容",
    "招聘人数", "职位类别", "岗位类别",
)
_REGION_TERMS = (
    "北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
    "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海",
    "台湾", "内蒙古", "广西", "西藏", "宁夏", "新疆",
)
_REGION_LABELS = (
    "项目地点", "项目地址", "实施地点", "服务地点", "采购区域", "项目所在地",
    "项目区域", "服务区域",
)


def canonicalize_url(url: str) -> str:
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError:
        return ""
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not scheme or not host:
        return ""
    netloc = host if port is None or (scheme, port) in {("http", 80), ("https", 443)} else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(sorted(
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in _TRACKING_KEYS and not key.casefold().startswith("utm_")
    ))
    return urlunsplit((scheme, netloc, path, query, ""))


def clean_records(
    query: str,
    raw_records: Sequence[Mapping[str, Any]],
    limit: int,
    official_source_domains: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Normalize records while retaining raw fields and exact field provenance."""
    raw_list = [dict(item) for item in raw_records]
    # Normalize every collected source before applying any display or candidate
    # page limit. The orchestrator owns the bounded candidate-pool limit.
    base_results = clean_and_ground_results(query, raw_list, len(raw_list), official_source_domains)
    raw_by_canonical: dict[str, dict[str, Any]] = {}
    for raw in raw_list:
        url = raw.get("url")
        if isinstance(url, str):
            canonical = canonicalize_url(url)
            current = raw_by_canonical.get(canonical)
            if current is None or len(_raw_text(raw)) > len(_raw_text(current)):
                raw_by_canonical[canonical] = raw

    output: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_records: list[tuple[str, str, str]] = []
    for base in base_results:
        canonical_url = canonicalize_url(base.url)
        if not canonical_url or canonical_url in seen_urls:
            continue
        normalized_title = _normalize_title(base.title)
        raw = raw_by_canonical.get(canonical_url, {})
        content = _raw_text(raw)
        if any(
            _records_are_duplicate(
                normalized_title,
                content,
                base.url,
                prior_title,
                prior_content,
                prior_url,
            )
            for prior_title, prior_content, prior_url in seen_records
        ):
            continue
        seen_urls.add(canonical_url)
        seen_records.append((normalized_title, content, base.url))
        extracted = _extract_fields(base.title, content)
        source_id = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:20]
        field_sources = {
            name: {
                "source_id": source_id,
                "url": base.url,
                "raw_field": "raw_content" if raw.get("raw_content") else "content",
                "quote": detail["quote"],
            }
            for name, detail in extracted.items()
        }
        output.append({
            "id": base.id,
            "canonical_url": canonical_url,
            "source_id": source_id,
            "source_provenance": raw.get("_provenance", {}),
            "raw_record": raw,
            "title": base.title,
            "url": base.url,
            "source_name": base.source_name,
            "snippet": base.snippet,
            "evidence": base.evidence,
            "match_reason": base.match_reason,
            "verdict": base.verdict,
            "confidence": base.confidence,
            "published_at": base.published_at,
            "official_source_url": base.official_source_url,
            "verified_fields": base.verified_fields,
            "unverified_fields": base.unverified_fields,
            "purchasing_unit": base.purchasing_unit,
            "budget": base.budget,
            "deadline_at": base.deadline_at,
            "deadline_status": base.deadline_status,
            "official_source_status": base.official_source_status,
            "evidence_status": base.evidence_status,
            "procurement_category": base.procurement_category,
            "match_status": base.match_status,
            "match_score": base.match_score,
            "mismatch_reason": base.mismatch_reason,
            "extracted_fields": extracted,
            "field_sources": field_sources,
            "raw_text": content,
        })
    return output


def _raw_text(raw: Mapping[str, Any]) -> str:
    body = raw.get("raw_content") or raw.get("content")
    return body.strip() if isinstance(body, str) else ""


def _extract_fields(title: str, body: str) -> dict[str, dict[str, str]]:
    text = f"{title}\n{body}"
    fields: dict[str, dict[str, str]] = {}
    cohort = re.search(r"20\d{2}\s*届", text)
    if cohort:
        fields["graduate_cohort"] = {"value": re.sub(r"\s+", "", cohort.group()), "quote": cohort.group()}
    majors = _non_overlapping_terms(text, _MAJOR_TERMS)
    if majors:
        fields["majors"] = {"value": "、".join(dict.fromkeys(majors)), "quote": majors[0]}
    company = re.search(r"(?:公司名称|企业名称|招聘单位|采购人|采购单位)\s*[:：]\s*([^\n，。；;]{2,60})", body)
    if company:
        fields["organization"] = {"value": company.group(1).strip(), "quote": company.group(0).strip()}
    elif cohort:
        # Keep a title-derived employer as an unverified lead. The verifier
        # must independently confirm that the official channel names it.
        cohort_in_title = re.search(r"20\d{2}\s*届", title)
        if cohort_in_title:
            prefix = title[:cohort_in_title.start()].strip(" _-—|｜:：，,（）()【】[]")
            prefix = re.sub(r"(?:校园招聘|校招|招聘|应届生)$", "", prefix).strip(" _-—|｜:：，,（）()【】[]")
            if 2 <= len(prefix) <= 40:
                fields["organization"] = {"value": prefix, "quote": prefix}
    region = _extract_region(title, body)
    if region:
        fields["region"] = region
    date_match = re.search(r"(?:截止(?:时间|日期)?|报名截止|网申截止|投递截止|响应文件递交截止时间)\s*[:：]?\s*((?:20\d{2}[年./-]\s*\d{1,2}[月./-]\s*\d{1,2}日?)(?:\s*\d{1,2}:\d{2})?)", body)
    if date_match:
        fields["deadline"] = {"value": date_match.group(1).strip(), "quote": date_match.group(0).strip()}
    application = re.search(
        r"(?:投递链接|投递入口|报名入口|网申入口|网申网址|申请链接|职位申请|点击申请)\s*[:：]?\s*(https?://[^\s<>\]\[（）()，。；;]+)",
        body,
        re.IGNORECASE,
    )
    if application:
        link = application.group(1).rstrip(".,。；;，")
        fields["application_url"] = {"value": link, "quote": application.group(0).replace(application.group(1), link)}
    qualifications = [term for term in _QUALIFICATION_TERMS if term in body]
    if qualifications:
        term = qualifications[0]
        fields["qualifications"] = {"value": "、".join(qualifications), "quote": _context_for_term(body, term)}
    business = [term for term in _BUSINESS_TERMS if term in body]
    if business:
        term = business[0]
        fields["business_attributes"] = {"value": "、".join(business), "quote": _context_for_term(body, term)}
    role = next((line.strip() for line in re.split(r"[\n。；;]", text) if any(t in line for t in ("工程师", "开发", "分析师", "产品经理", "岗位", "职位"))), None)
    if role:
        fields["role"] = {"value": role[:120], "quote": role[:120]}
    return fields


def _extract_region(title: str, body: str) -> dict[str, str] | None:
    """Use a title or explicitly labeled location, never arbitrary page text."""
    region = _find_region(title)
    if region:
        return region

    for line_index, line in enumerate(body.splitlines()):
        label_match = re.search(
            "|".join(re.escape(label) for label in _REGION_LABELS),
            line,
        )
        if not label_match:
            continue
        location_text = line[label_match.end():]
        if not location_text.strip() and line_index + 1 < len(body.splitlines()):
            location_text = body.splitlines()[line_index + 1]
        region = _find_region(location_text)
        if region:
            return region
    return None


def _find_region(text: str) -> dict[str, str] | None:
    suffix = r"(?:省|市|(?:壮族|回族|维吾尔)?自治区|特别行政区)?"
    matches: list[tuple[int, int, str]] = []
    for term in _REGION_TERMS:
        match = re.search(re.escape(term) + suffix, text)
        if match:
            matches.append((match.start(), -len(term), match.group()))
    if not matches:
        return None
    value = min(matches)[2]
    return {"value": value, "quote": value}


def _normalize_title(title: str) -> str:
    title = re.sub(r"\s+", "", title).casefold()
    title = re.sub(r"[【】\[\]（）()_\-—,，。:：|｜]", "", title)
    return title


def _title_is_duplicate(current: str, previous: str) -> bool:
    if not current or not previous:
        return False
    if current == previous:
        return True
    left, right = _title_tokens(current), _title_tokens(previous)
    if not left or not right:
        return False
    return len(left & right) / len(left | right) >= 0.92


def _records_are_duplicate(
    current_title: str,
    current_body: str,
    current_url: str,
    previous_title: str,
    previous_body: str,
    previous_url: str,
) -> bool:
    """Collapse cross-URL duplicates only when source text corroborates it.

    A matching title alone is not enough: employers often reuse role names
    across cities or requisitions. Require similar titles and substantial
    body overlap, and preserve records when explicit job identity, location,
    or application-link evidence conflicts.
    """
    if not _title_is_duplicate(current_title, previous_title):
        return False
    current = _normalize_body(current_body)
    previous = _normalize_body(previous_body)
    # Short snippets and empty pages do not establish semantic identity.
    if min(len(current), len(previous)) < 100:
        return False
    left_identity = _distinct_posting_identity(current_body, current_url)
    right_identity = _distinct_posting_identity(previous_body, previous_url)
    if any(
        left_identity.get(key) != right_identity.get(key)
        for key in left_identity.keys() | right_identity.keys()
    ):
        return False
    left_tokens = _content_tokens(current)
    right_tokens = _content_tokens(previous)
    if not left_tokens or not right_tokens:
        return current == previous
    similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return similarity >= 0.94


def _normalize_body(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)


def _content_tokens(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", value))
    for run in re.findall(r"[\u3400-\u9fff]+", value):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _distinct_posting_identity(body: str, url: str) -> dict[str, str]:
    identity: dict[str, str] = {}
    parsed = urlsplit(url)
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.casefold() in {"jobid", "positionid", "requisitionid", "postingid"} and value:
            identity["posting_id"] = value.casefold()
            break
    if "posting_id" not in identity:
        match = re.search(r"/(?:jobs?|positions?|requisitions?)/(?:detail/)?([a-z0-9_-]{3,})(?:/|$)", parsed.path, re.I)
        if match:
            identity["posting_id"] = match.group(1).casefold()
    labelled_id = re.search(r"(?:岗位编号|职位编号|招聘编号|岗位ID|职位ID)\s*[:：]?\s*([a-z0-9_-]{3,})", body, re.I)
    if labelled_id:
        identity["posting_id"] = labelled_id.group(1).casefold()
    location = re.search(r"(?:工作地点|职位地点|岗位地点|招聘地点)\s*[:：]?\s*([^\n，,。；;]{2,30})", body)
    if location:
        identity["location"] = re.sub(r"\s+", "", location.group(1)).casefold()
    application = re.search(
        r"(?:投递链接|投递入口|网申入口|申请链接|职位申请)\s*[:：]?\s*(https?://[^\s<>，。；;]+)",
        body,
        re.I,
    )
    if application:
        identity["application_url"] = canonicalize_url(application.group(1).rstrip(".,，。；;"))
    return identity


def _context_for_term(text: str, term: str) -> str:
    for line in re.split(r"[\n。；;]", text):
        if term in line:
            return line.strip()[:240]
    return term


def _non_overlapping_terms(text: str, terms: Sequence[str]) -> list[str]:
    spans: list[tuple[int, int, str]] = []
    for term in sorted(terms, key=len, reverse=True):
        for match in re.finditer(re.escape(term), text):
            if any(match.start() < end and match.end() > start for start, end, _ in spans):
                continue
            spans.append((match.start(), match.end(), term))
    return [term for _, _, term in sorted(spans)]


def _title_tokens(text: str) -> set[str]:
    cjk = re.findall(r"[\u3400-\u9fff]+", text)
    tokens = set(re.findall(r"[a-z0-9]+", text))
    for run in cjk:
        tokens.update(run[i:i + 2] for i in range(len(run) - 1))
    return tokens
