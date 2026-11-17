"""小判: condition-based decisions; unknown facts never satisfy hard constraints."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from ..semantic_matching import classify_procurement, employment_source_scope


def judge_records(
    query: str,
    intent: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    domain = str(intent.get("domain", "general"))
    hard = intent.get("hard_conditions", [])
    hard = hard if isinstance(hard, list) else []
    employment_mode = domain == "employment" or any(
        isinstance(condition, Mapping)
        and str(condition.get("field", "")) in {"graduate_cohort", "major", "application_url"}
        for condition in hard
    )
    for record in records:
        item = dict(record)
        if employment_mode:
            scope = employment_source_scope(item)
            item["employment_source_scope"] = scope
            if scope == "corporate_campus_page":
                checks = item.get("field_verification")
                if isinstance(checks, Mapping):
                    downgraded_checks = dict(checks)
                    for field_name in ("graduate_cohort", "major", "role", "open_status", "deadline", "application_url"):
                        check = downgraded_checks.get(field_name)
                        if not isinstance(check, Mapping) or check.get("status") != "verified":
                            continue
                        downgraded = dict(check)
                        downgraded["status"] = "unverified"
                        downgraded["scope_note"] = "官方来源不是具体岗位公告，无法确认该岗位的届别、专业、招聘状态、截止日期或投递入口。"
                        downgraded_checks[field_name] = downgraded
                    item["field_verification"] = downgraded_checks
        title = _text(item.get("title"))
        body = " ".join((_text(item.get("snippet")), _text(item.get("evidence")), _text(item.get("raw_text"))))
        checks: list[dict[str, Any]] = []
        failed = False
        unknown = False
        failed_labels: list[str] = []
        for condition in hard:
            if not isinstance(condition, Mapping):
                continue
            field = str(condition.get("field", ""))
            value = str(condition.get("value", ""))
            extracted = item.get("field_verification", {}).get(field) if isinstance(item.get("field_verification"), Mapping) else None
            if field == "graduate_cohort":
                extracted = extracted or _exact_fact(item, "graduate_cohort", value, value)
            elif field == "major":
                extracted = extracted or _exact_fact(item, "majors", value, value)
            elif field == "open_status":
                if (
                    domain == "procurement"
                    and item.get("official_source_status") == "verified"
                    and item.get("deadline_status") in {"expired", "not_expired"}
                ):
                    deadline_check = item.get("field_verification", {}).get("deadline") if isinstance(item.get("field_verification"), Mapping) else None
                    if not isinstance(extracted, Mapping) or extracted.get("status") != "conflict":
                        extracted = {
                            "value": "已截止" if item.get("deadline_status") == "expired" else "未截止",
                            "status": "verified",
                            "evidence_url": item.get("official_source_url"),
                            "quote": deadline_check.get("quote") if isinstance(deadline_check, Mapping) else item.get("deadline_at"),
                        }
                extracted = extracted or _open_status_fact(body)
            elif field == "region":
                extracted = extracted or _exact_fact(item, "region", value, value)
            if isinstance(extracted, Mapping) and extracted.get("value"):
                fact_value = str(extracted.get("value"))
                fact_status = str(extracted.get("status", "unverified"))
                is_match = _condition_match(field, value, fact_value)
                if fact_status == "conflict":
                    unknown = True
                    status = "conflict"
                elif fact_status == "verified":
                    status = "verified"
                    if not is_match:
                        failed = True
                        failed_labels.append(str(condition.get("label", field)))
                else:
                    unknown = True
                    status = "unverified"
                checks.append({
                    "label": str(condition.get("label", field)),
                    "value": fact_value,
                    "status": status,
                    "evidence_url": extracted.get("evidence_url"),
                    "quote": extracted.get("quote"),
                })
            else:
                unknown = True
                checks.append({
                    "label": str(condition.get("label", field)),
                    "value": "未找到证据",
                    "status": "missing",
                    "evidence_url": None,
                    "quote": None,
                })

        if domain == "procurement":
            # Procurement intent must be derived from the matched official
            # notice. Third-party snippets are useful for discovery, but they
            # cannot prove what the official project actually buys.
            official_url = _text(item.get("official_source_url"))
            official_status = item.get("official_source_status")
            official_title = _text(item.get("official_source_title")) if official_status == "verified" else ""
            verified_category_text = _verified_procurement_category_text(item, official_url)
            category_evidence = bool(
                official_status == "verified"
                and official_url
                and (official_title or verified_category_text)
                and _has_procurement_category(query, official_title, verified_category_text)
            )
            classification = classify_procurement(
                query,
                official_title if category_evidence else "",
                verified_category_text if category_evidence else "",
            )
            if not category_evidence:
                classification.update({
                    "procurement_category": "unclear",
                    "match_status": "unverified",
                    "match_score": 0.12,
                    "match_reason": "第三方搜索内容或其他未核验信息提到了相关类别，但官方公告中尚无可追溯的类别证据。",
                    "mismatch_reason": "官方公告未能核实采购业务类别。",
                })
            item.update({
                "procurement_category": classification["procurement_category"],
                "match_status": classification["match_status"],
                "match_score": classification["match_score"],
                "match_reason": classification["match_reason"],
                "mismatch_reason": classification["mismatch_reason"],
            })
            category_quote, category_url = _procurement_category_evidence(
                item, official_url, official_title, verified_category_text
            )
            category_verified = category_evidence and bool(category_quote and category_url)
            checks.append({
                "label": "官方公告业务类别",
                "value": classification["procurement_category"] if category_verified else "未核实",
                "status": "verified" if category_verified else "missing",
                "evidence_url": category_url if category_verified else None,
                "quote": category_quote if category_verified else None,
            })
            if not category_verified:
                unknown = True
            elif classification["match_status"] == "not_matched":
                failed = True
            elif classification["match_status"] in {"keyword_only", "partial", "unverified"}:
                unknown = True
        else:
            verified_count = sum(
                1 for check in checks
                if check.get("status") == "verified"
            )
            unresolved_labels = [
                str(check.get("label", "条件"))
                for check in checks
                if check.get("status") != "verified"
            ]
            if failed:
                item.update({
                    "match_status": "not_matched",
                    "match_score": 0.04,
                    "match_reason": "已核验证据与明确条件冲突，不能推荐。",
                    "mismatch_reason": "已核实不符合：" + "、".join(failed_labels),
                })
            elif checks and not unresolved_labels:
                item.update({
                    "match_status": "matched",
                    "match_score": 0.96,
                    "match_reason": "所有明确条件均有可核验证据并满足。",
                    "mismatch_reason": None,
                })
            elif checks:
                score = round(0.12 + 0.26 * verified_count / len(checks), 4)
                item.update({
                    "match_status": "partial" if verified_count else "unverified",
                    "match_score": score,
                    "match_reason": (
                        f"已核实 {verified_count}/{len(checks)} 项明确条件；"
                        "其余未知条件不视为满足。"
                    ),
                    "mismatch_reason": "待核实条件：" + "、".join(unresolved_labels),
                })
            else:
                item.update({
                    "match_status": "unverified",
                    "match_score": 0.12,
                    "match_reason": "当前没有足够的可核验证据判断实际需求匹配度。",
                    "mismatch_reason": "匹配条件或来源证据不足。",
                })

        if failed:
            decision = "not_matched"
            status = "未满足明确条件，或业务类别不匹配。"
        elif (
            unknown
            or not checks and domain != "procurement"
            or domain == "procurement" and not any(
                isinstance(value, Mapping) and value.get("status") == "verified"
                for value in item.get("field_verification", {}).values()
            )
        ):
            decision = "needs_verification"
            status = "部分条件缺少可核验证据，未知条件不视为满足。"
        else:
            decision = "recommended"
            status = "已找到与明确条件相符且逐项有证据的来源。"
        item["decision"] = decision
        item["summary"] = f"{status} {title}".strip()
        item["key_conditions"] = checks
        item["evidence_sources"] = _evidence_sources(item)
        item["evidence_detail"] = _evidence_detail(item)
        application = item.get("field_verification", {}).get("application_url") if isinstance(item.get("field_verification"), Mapping) else None
        item["application_url"] = (
            application.get("value")
            if isinstance(application, Mapping) and application.get("status") == "verified"
            else None
        )
        output.append(item)
    output.sort(key=lambda result: (
        {"recommended": 0, "needs_verification": 1, "not_matched": 2}.get(result.get("decision"), 3),
        -float(result.get("match_score", 0) or 0),
        -float(result.get("confidence", 0) or 0),
    ))
    return output


def _verified_procurement_category_text(record: Mapping[str, Any], official_url: str) -> str:
    """Return only exact quotes attached to verified official procurement fields."""
    checks = record.get("field_verification")
    if not isinstance(checks, Mapping) or not official_url:
        return ""
    quotes: list[str] = []
    for field in ("evidence", "business_attributes", "project_name"):
        check = checks.get(field)
        if not isinstance(check, Mapping):
            continue
        if check.get("status") != "verified" or check.get("evidence_url") != official_url:
            continue
        quote = _text(check.get("quote"))
        if quote and quote not in quotes:
            quotes.append(quote)
    return "\n".join(quotes)


def _has_procurement_category(query: str, official_title: str, official_text: str) -> bool:
    if not official_title and not official_text:
        return False
    classification = classify_procurement(query, official_title, official_text)
    return classification.get("procurement_category") != "unclear"


def _procurement_category_evidence(
    record: Mapping[str, Any], official_url: str, official_title: str, official_text: str
) -> tuple[str | None, str | None]:
    if not official_url:
        return None, None
    checks = record.get("field_verification")
    if isinstance(checks, Mapping):
        for field in ("evidence", "business_attributes", "project_name"):
            check = checks.get(field)
            if (
                isinstance(check, Mapping)
                and check.get("status") == "verified"
                and check.get("evidence_url") == official_url
                and _text(check.get("quote"))
                and _text(check.get("quote")) in official_text
            ):
                return _text(check.get("quote")), official_url
    if official_title and _has_procurement_category("", official_title, official_text):
        return official_title, official_url
    return None, None


def _exact_fact(record: Mapping[str, Any], field_name: str, wanted: str, fallback: str) -> dict[str, Any] | None:
    extracted = record.get("extracted_fields", {})
    sources = record.get("field_sources", {})
    field = extracted.get(field_name) if isinstance(extracted, Mapping) else None
    source = sources.get(field_name) if isinstance(sources, Mapping) else None
    if not isinstance(field, Mapping) or not isinstance(source, Mapping):
        return None
    fact_value = str(field.get("value", ""))
    if wanted and wanted.casefold() not in fact_value.casefold():
        return {"value": fact_value, "status": "unverified", "evidence_url": source.get("url"), "quote": source.get("quote")}
    return {"value": fact_value or fallback, "status": "unverified", "evidence_url": source.get("url"), "quote": source.get("quote")}


def _open_status_fact(text: str) -> dict[str, Any] | None:
    if any(term in text for term in ("已截止", "招聘结束", "停止招聘", "网申已关闭")):
        return {"value": "已截止", "status": "unverified", "evidence_url": None, "quote": next(term for term in ("已截止", "招聘结束", "停止招聘", "网申已关闭") if term in text)}
    if any(term in text for term in ("正在招聘", "招聘中", "投递中", "立即投递", "网申入口")):
        return {"value": "招聘中", "status": "unverified", "evidence_url": None, "quote": next(term for term in ("正在招聘", "招聘中", "投递中", "立即投递", "网申入口") if term in text)}
    deadline = _deadline_from_text(text)
    if deadline:
        opened = deadline >= date.today()
        return {"value": "招聘中" if opened else "已截止", "status": "unverified", "evidence_url": None, "quote": deadline.isoformat()}
    return None


def _deadline_from_text(text: str) -> date | None:
    match = re.search(r"(?:截止(?:时间|日期)?|网申截止|投递截止)\s*[:：]?\s*(20\d{2})[-年./]\s*(\d{1,2})[-月./]\s*(\d{1,2})", text)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _condition_match(field: str, wanted: str, actual: str) -> bool:
    if field == "application_url":
        return actual.startswith(("https://", "http://"))
    if field == "graduate_cohort":
        return re.sub(r"\s+", "", wanted) in re.sub(r"\s+", "", actual)
    if field == "major":
        return wanted.casefold() in actual.casefold()
    if field == "open_status":
        if wanted in {"未截止", "仍在招标", "正在招标", "仍在采购"}:
            return actual in {"招聘中", "未截止", "not_expired"}
        return actual == "招聘中"
    if field == "region":
        return _text(wanted) in _text(actual)
    return wanted.casefold() in actual.casefold()


def _evidence_sources(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    url = _text(record.get("official_source_url")) or _text(record.get("url"))
    if not url:
        return []
    official = record.get("official_source_status") == "verified" and url == record.get("official_source_url")
    third_party = record.get("official_source_status") == "third_party"
    return [{
        "title": _text(record.get("official_source_title")) if official else _text(record.get("title")),
        "url": url,
        "source_type": "official" if official else "third_party" if third_party else "unknown",
        "status": "verified" if official else "unverified",
    }]


def _evidence_detail(record: Mapping[str, Any]) -> str:
    official_body = record.get("official_evidence_detail")
    if isinstance(official_body, str) and official_body.strip():
        return official_body.strip()
    source = record.get("raw_record", {})
    if isinstance(source, Mapping):
        body = source.get("raw_content") or source.get("content")
        if isinstance(body, str):
            return body
    return _text(record.get("evidence"))


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
