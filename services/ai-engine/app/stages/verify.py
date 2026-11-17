"""小验: independent field evidence with explicit unknown/conflict handling."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit


def verify_records(
    records: Sequence[Mapping[str, Any]],
    official_results: Mapping[str, Mapping[str, Any]] | None = None,
    required_conditions: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Attach per-field verification. A URL/domain alone never verifies a field."""
    official_results = official_results or {}
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        fields = item.get("extracted_fields")
        fields = fields if isinstance(fields, Mapping) else {}
        field_sources = item.get("field_sources")
        field_sources = field_sources if isinstance(field_sources, Mapping) else {}
        official = official_results.get(str(item.get("url", "")), {})
        official_url = official.get("official_source_url")
        verified_names = set(official.get("verified_fields", ()))
        official_quotes = official.get("field_evidence", {})
        official_conflicts = official.get("field_conflicts", {})
        checks: dict[str, dict[str, Any]] = {}
        for field_name, raw_value in fields.items():
            result_field = _map_field(field_name)
            value = raw_value.get("value") if isinstance(raw_value, Mapping) else None
            source = field_sources.get(field_name, {})
            quote = source.get("quote") if isinstance(source, Mapping) else None
            evidence_url = source.get("url") if isinstance(source, Mapping) else None
            status = "unverified"
            official_field = result_field
            if official_url and isinstance(official_conflicts, Mapping) and official_field in official_conflicts:
                conflict = official_conflicts[official_field]
                status = "conflict"
                if isinstance(conflict, Mapping):
                    quote = str(conflict.get("quote", quote or "")) or None
                    evidence_url = str(conflict.get("evidence_url", official_url))
            elif official_field in verified_names and official_url:
                official_quote = official_quotes.get(official_field) if isinstance(official_quotes, Mapping) else None
                if isinstance(official_quote, Mapping):
                    status = "verified"
                    quote = str(official_quote.get("quote", "")).strip() or None
                    official_value = str(official_quote.get("value", value or "")) or None
                    if value and official_value and not _field_value_agrees(official_field, str(value), official_value):
                        status = "conflict"
                    value = official_value
                    evidence_url = official_url
                elif isinstance(official_quote, str) and official_quote.strip():
                    status, quote, evidence_url = "verified", official_quote.strip(), official_url
                elif field_name == "deadline" and official_field == "deadline" and official.get("deadline_at"):
                    status, evidence_url = "verified", official_url
                    quote = str(official.get("deadline_at"))
            checks[result_field] = {
                "value": value,
                "status": status,
                "evidence_url": evidence_url if isinstance(evidence_url, str) else None,
                "quote": quote if isinstance(quote, str) and quote else None,
            }
        # Some facts (notably open/closed status) are derived from an exact
        # labeled date or status phrase on the matched official page rather
        # than an initially extracted search snippet. Attach those too.
        if official_url and isinstance(official_quotes, Mapping):
            for official_field, official_quote in official_quotes.items():
                if official_field in checks or not isinstance(official_quote, Mapping):
                    continue
                checks[str(official_field)] = {
                    "value": official_quote.get("value"),
                    "status": "verified",
                    "evidence_url": official_url,
                    "quote": official_quote.get("quote"),
                }
        if official_url and official.get("deadline_status") in {"expired", "not_expired"}:
            deadline_evidence = official_quotes.get("deadline") if isinstance(official_quotes, Mapping) else None
            deadline_quote = deadline_evidence.get("quote") if isinstance(deadline_evidence, Mapping) else official.get("deadline_at")
            deadline_conflict = checks.get("deadline", {}).get("status") == "conflict"
            checks["open_status"] = {
                "value": "已截止" if official.get("deadline_status") == "expired" else "未截止",
                "status": "conflict" if deadline_conflict else "verified",
                "evidence_url": official_url,
                "quote": deadline_quote,
            }
        # Explicit query requirements are tracked even when a source does not
        # contain a corresponding field. This makes verification metrics count
        # missing facts instead of silently ignoring them.
        for condition in required_conditions:
            if not isinstance(condition, Mapping):
                continue
            field_name = str(condition.get("field", ""))
            result_field = _map_field(field_name)
            if not result_field or result_field in checks:
                continue
            checks[result_field] = {
                "value": None,
                "status": "missing",
                "evidence_url": None,
                "quote": None,
            }
        if official_url:
            item["official_source_url"] = official_url
            item["official_source_status"] = "verified"
            item["official_source_title"] = official.get("official_source_title")
            item["official_evidence_detail"] = official.get("evidence_detail", "")
            item["source_identity_quote"] = official.get("source_identity_quote")
        item["field_verification"] = checks
        output.append(item)
    return output


def verification_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"verified": 0, "unverified": 0, "conflict": 0}
    for record in records:
        checks = record.get("field_verification", {})
        if isinstance(checks, Mapping):
            for check in checks.values():
                if isinstance(check, Mapping) and check.get("status") in counts:
                    counts[check["status"]] += 1
                elif isinstance(check, Mapping) and check.get("status") == "missing":
                    counts["unverified"] += 1
    return counts


async def find_official_employment_evidence(
    records: Sequence[Mapping[str, Any]],
    search: Any,
    *,
    limit: int = 3,
    required_conditions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, dict[str, Any]]:
    """Search for a matching employer careers page and verify quoted fields.

    Career path/domain markers only narrow candidates. A candidate is accepted
    only when its retrieved body also contains the candidate role/cohort anchor;
    each field is then verified independently by an exact literal quote.
    """
    found: dict[str, dict[str, Any]] = {}
    for record in records:
        extracted = record.get("extracted_fields", {})
        if not isinstance(extracted, Mapping):
            continue
        title = str(record.get("title", "")).strip()
        cohort = extracted.get("graduate_cohort", {})
        cohort_value = cohort.get("value", "") if isinstance(cohort, Mapping) else ""
        search_terms = [term for term in (title, str(cohort_value)) if term]
        if not search_terms:
            continue
        query = " ".join(f'"{term}"' for term in search_terms) + " 官方 校园招聘"
        try:
            candidates = await search(query, (), limit)
        except Exception:
            continue
        accepted: list[tuple[Mapping[str, Any], dict[str, Any], dict[str, dict[str, str]]]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            url = candidate.get("url")
            body = _body(candidate)
            candidate_title = str(candidate.get("title", ""))
            if not isinstance(url, str) or not body:
                continue
            parsed = urlsplit(url)
            career_marker = any(
                token in (parsed.hostname or "").casefold()
                or token in parsed.path.casefold()
                for token in ("career", "campus", "jobs", "recruit", "talent", "校招", "招聘")
            )
            # A marker alone never authenticates a page: the page must carry
            # the specific job title or a cohort plus a recruitment context.
            role = extracted.get("role", {})
            role_value = role.get("value", "") if isinstance(role, Mapping) else ""
            anchor = _literal_quote(body, title)
            if not anchor and isinstance(role_value, str) and role_value:
                anchor = _literal_quote(body, role_value)
            recruitment_context = any(term in body for term in ("校园招聘", "社会招聘", "岗位", "职位", "招聘简章"))
            organization = extracted.get("organization", {})
            organization_value = organization.get("value", "") if isinstance(organization, Mapping) else ""
            if not career_marker or not recruitment_context or not anchor or not organization_value:
                continue
            ownership_quote = _employer_channel_quote(body, str(organization_value))
            if not ownership_quote:
                continue
            field_evidence: dict[str, dict[str, str]] = {
                "organization": {
                    "value": str(organization_value),
                    "quote": ownership_quote,
                }
            }
            for field_name, raw_field in extracted.items():
                if field_name in {"majors", "organization"}:
                    # Major evidence is verified only in a position requirement
                    # section, never from an unrelated page mention.
                    continue
                quote = raw_field.get("quote") if isinstance(raw_field, Mapping) else None
                if not isinstance(quote, str) or not quote:
                    continue
                exact = _literal_quote(body, quote)
                if exact:
                    field_evidence[_map_field(field_name)] = {
                        "value": str(raw_field.get("value", "")),
                        "quote": exact,
                    }
            # Candidate facts are parsed independently. A matching page may
            # contradict the search snippet; preserve the official value.
            requested_major = next((str(item.get("value")) for item in required_conditions if isinstance(item, Mapping) and item.get("field") == "major"), None)
            official_facts = _employment_facts(body, requested_major=requested_major)
            local_conflicts = official_facts.pop("_field_conflicts", {})
            accepted.append((candidate, {**field_evidence, **official_facts}, local_conflicts))
        if not accepted:
            continue
        field_values: dict[str, list[tuple[str, str, str]]] = {}
        for candidate, facts, _local_conflicts in accepted:
            url = str(candidate.get("url", ""))
            body = _body(candidate)
            for field_name, fact in facts.items():
                value = fact.get("value", "")
                quote = fact.get("quote", "")
                if value and quote and quote in body:
                    field_values.setdefault(field_name, []).append((value, quote, url))
        conflicts: dict[str, dict[str, str]] = {}
        for candidate, _facts, local_conflicts in accepted:
            for field_name, conflict in local_conflicts.items():
                conflicts[field_name] = {
                    "quote": conflict.get("quote", ""),
                    "evidence_url": str(candidate.get("url", "")),
                }
        evidence: dict[str, dict[str, str]] = {}
        for field_name, facts in field_values.items():
            if field_name in conflicts:
                continue
            distinct = {value for value, _, _ in facts}
            if len(distinct) > 1:
                conflicts[field_name] = {
                    "quote": "；".join(dict.fromkeys(quote for _, quote, _ in facts)),
                    "evidence_url": facts[0][2],
                }
            else:
                value, quote, _ = facts[0]
                evidence[field_name] = {"value": value, "quote": quote}
        if "deadline" in conflicts and "open_status" in field_values:
            conflicts["open_status"] = {
                "quote": conflicts["deadline"]["quote"],
                "evidence_url": conflicts["deadline"]["evidence_url"],
            }
            evidence.pop("open_status", None)
        first_url = str(accepted[0][0].get("url", ""))
        found[str(record.get("url", ""))] = {
            "official_source_url": first_url,
            "official_source_title": str(accepted[0][0].get("title", "")),
            "verified_fields": list(evidence),
            "field_evidence": evidence,
            "field_conflicts": conflicts,
            "evidence_detail": _body(accepted[0][0]),
            "source_identity_quote": _employer_channel_quote(
                _body(accepted[0][0]),
                str((extracted.get("organization") or {}).get("value", "")),
            ),
        }
    return found


def _map_field(field_name: str) -> str:
    return {"deadline": "deadline", "graduate_cohort": "graduate_cohort", "majors": "major"}.get(field_name, field_name)


def _body(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("raw_content") or candidate.get("content")
    return value.strip() if isinstance(value, str) else ""


def _literal_quote(text: str, quote: str) -> str | None:
    quote = quote.strip()
    if quote and quote in text:
        return quote
    return None


def _employment_facts(text: str, *, requested_major: str | None = None) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    field_conflicts: dict[str, dict[str, str]] = {}
    cohorts = list(dict.fromkeys(re.sub(r"\s+", "", match.group()) for match in re.finditer(r"20\d{2}\s*届", text)))
    if len(cohorts) > 1:
        field_conflicts["graduate_cohort"] = {"quote": "；".join(cohorts)}
    elif cohorts:
        cohort = re.search(r"20\d{2}\s*届", text)
        facts["graduate_cohort"] = {"value": cohorts[0], "quote": cohort.group() if cohort else cohorts[0]}
    deadline_matches = list(re.finditer(
        r"(?:截止(?:时间|日期)?|网申截止|投递截止)\s*[:：]?\s*((?:20\d{2}[年./-]\s*\d{1,2}[月./-]\s*\d{1,2}日?)(?:\s*\d{1,2}:\d{2})?)",
        text,
    ))
    deadline_values = list(dict.fromkeys(match.group(1).strip() for match in deadline_matches))
    if len(deadline_values) > 1:
        field_conflicts["deadline"] = {"quote": "；".join(match.group(0).strip() for match in deadline_matches)}
    elif deadline_matches:
        deadline = deadline_matches[0]
        facts["deadline"] = {"value": deadline_values[0], "quote": deadline.group(0).strip()}
        parsed = re.search(r"(20\d{2})[年./-]\s*(\d{1,2})[月./-]\s*(\d{1,2})", deadline.group(1))
        if parsed:
            try:
                from datetime import date
                state = "招聘中" if date(*(int(part) for part in parsed.groups())) >= date.today() else "已截止"
                facts["open_status"] = {"value": state, "quote": deadline.group(0).strip()}
            except ValueError:
                pass
    explicit_open = next((term for term in ("正在招聘", "招聘中", "网申开放", "接受投递") if term in text), None)
    explicit_closed = next((term for term in ("招聘结束", "已截止", "停止招聘", "网申关闭") if term in text), None)
    if explicit_open:
        facts["open_status"] = {"value": "招聘中", "quote": explicit_open}
    elif explicit_closed:
        facts["open_status"] = {"value": "已截止", "quote": explicit_closed}
    application = re.search(
        r"(?:投递链接|投递入口|报名入口|网申入口|网申网址|申请链接|职位申请|点击申请)\s*[:：]?\s*(https?://[^\s<>\]\[（）()，。；;]+)",
        text,
        re.IGNORECASE,
    )
    if application:
        url = application.group(1).rstrip(".,。；;，")
        facts["application_url"] = {"value": url, "quote": application.group(0).replace(application.group(1), url)}
    major_terms = (
        "电子信息工程", "电子科学与技术", "通信工程", "计算机科学与技术",
        "软件工程", "人工智能", "自动化", "信息工程", "网络工程",
    )
    requirement_markers = ("专业要求", "相关专业", "招聘条件", "任职资格", "岗位要求", "职位要求", "所学专业", "资格条件")
    major_evidence: list[tuple[list[str], str]] = []
    for line in re.split(r"[\n。；;]", text):
        line = line.strip()
        if not any(marker in line for marker in requirement_markers):
            continue
        majors = [item for item in major_terms if item in line]
        if requested_major:
            majors = [item for item in majors if item == requested_major]
        if majors:
            major_evidence.append((majors, line[:240]))
    if major_evidence:
        # Keep the fact value aligned with its single quote. Combining majors
        # collected from different requirement lines made later cards imply
        # that one citation proved every listed major.
        major_values, major_quote = major_evidence[0]
        facts["major"] = {
            "value": "、".join(dict.fromkeys(major_values)),
            "quote": major_quote,
        }
    if field_conflicts:
        facts["_field_conflicts"] = field_conflicts
    return facts


def _employer_channel_quote(body: str, organization: str) -> str | None:
    """Require a source-owned channel statement tied to the named employer."""
    organization = re.sub(r"[\s\-—_（）()【】\[\],，。:：]", "", organization).casefold()
    if not organization:
        return None
    ownership_markers = (
        "官方招聘", "招聘官网", "官方人才招聘", "校园招聘官网", "官方招聘网站",
        "官方招聘平台", "官方发布招聘", "官方网站招聘", "招聘门户",
    )
    for line in re.split(r"[\n。；;]", body):
        normalized = re.sub(r"[\s\-—_（）()【】\[\],，。:：]", "", line).casefold()
        if organization in normalized and any(marker in normalized for marker in ownership_markers):
            return line.strip()[:240]
    return None


def _normalized_value(value: str) -> str:
    return re.sub(r"[\s\-—_（）()【】\[\],，。:：]", "", value).casefold()


def _field_value_agrees(field_name: str, original: str, official: str) -> bool:
    if field_name == "major":
        return _normalized_value(official) in _normalized_value(original)
    if field_name == "region":
        return _normalized_region(original) == _normalized_region(official)
    return _normalized_value(original) == _normalized_value(official)


def _normalized_region(value: str) -> str:
    normalized = _normalized_value(value)
    return re.sub(
        r"(?:壮族自治区|回族自治区|维吾尔自治区|特别行政区|自治区|省|市)$",
        "",
        normalized,
    )
