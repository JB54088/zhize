"""Evidence-preserving adapters from the V0.3 result record to V0.9 cards."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from .schemas import (
    CardAttributeStatus,
    ResultCard,
    ResultCardAttribute,
    ResultCardSource,
    SearchResult,
)


_FIELD_LABELS: dict[str, tuple[str, str]] = {
    "purchasing_unit": ("采购单位", "采购单位"),
    "budget": ("预算金额", "预算"),
    "deadline": ("响应截止时间", "截止时间"),
    "published_at": ("发布时间", "发布时间"),
    "graduate_cohort": ("毕业届别", "毕业届别"),
    "major": ("专业要求", "专业"),
    "open_status": ("招聘状态", "招聘状态"),
    "application_url": ("投递链接", "投递链接"),
    "role": ("岗位", "岗位"),
    "organization": ("单位", "单位"),
    "qualifications": ("资格要求", "资格要求"),
    "region": ("地区", "地区"),
    "business_attributes": ("需求内容", "需求内容"),
}

_LABEL_KEYS = {
    "采购单位": "purchasing_unit",
    "采购人": "purchasing_unit",
    "预算": "budget",
    "预算金额": "budget",
    "截止时间": "deadline",
    "响应截止时间": "response_deadline",
    "投递截止时间": "application_deadline",
    "报名截止时间": "application_deadline",
    "发布时间": "published_at",
    "毕业届别": "graduate_cohort",
    "毕业年份": "graduate_cohort",
    "专业": "major",
    "专业要求": "major",
    "资格要求": "qualifications",
    "招聘资格": "qualifications",
    "学历": "education",
    "薪资": "salary",
    "薪酬": "salary",
    "招聘状态": "job_status",
    "营业状态": "restaurant_status",
    "投递链接": "application_url",
    "申请链接": "application_url",
    "岗位": "role",
    "职位": "role",
    "公司": "organization",
    "企业": "organization",
    "采购单位名称": "purchasing_unit",
    "地区": "region",
}
_PROCUREMENT_KEYS = {
    "purchasing_unit", "budget", "response_deadline", "business_attributes"
}
_EMPLOYMENT_KEYS = {
    "graduate_cohort", "major", "job_status", "application_deadline", "application_url",
    "role", "organization", "salary", "education", "qualifications"
}
_DINING_KEYS = {
    "per_capita", "rating", "recent_reviews", "opening_hours", "cuisine", "restaurant_status"
}


def _category_specific_key(label: str, category: str) -> str | None:
    normalized = label.strip()
    if category == "dining":
        dining_labels = {
            "预算": "per_capita",
            "人均": "per_capita",
            "人均价格": "per_capita",
            "价格": "per_capita",
            "位置": "location",
            "地址": "location",
            "地点": "location",
            "评分": "rating",
            "评价": "recent_reviews",
            "近期评价": "recent_reviews",
            "营业状态": "restaurant_status",
            "营业时间": "opening_hours",
        }
        if normalized in dining_labels:
            return dining_labels[normalized]
    if category == "employment":
        employment_labels = {
            "公司": "organization",
            "企业": "organization",
            "公司名称": "organization",
            "岗位": "role",
            "职位": "role",
            "薪资": "salary",
            "薪酬": "salary",
            "学历": "education",
            "专业": "major",
            "专业要求": "major",
            "毕业届别": "graduate_cohort",
            "毕业年份": "graduate_cohort",
            "投递链接": "application_url",
            "招聘状态": "job_status",
            "截止时间": "application_deadline",
            "报名截止时间": "application_deadline",
            "投递截止时间": "application_deadline",
        }
        if normalized in employment_labels:
            return employment_labels[normalized]
    if category == "procurement" and normalized in {"截止时间", "响应截止时间"}:
        return "response_deadline"
    dining_labels = {
        "人均": "per_capita", "人均价格": "per_capita", "评分": "rating",
        "评价": "recent_reviews", "近期评价": "recent_reviews", "营业时间": "opening_hours",
        "菜品": "cuisine", "营业状态": "restaurant_status",
    }
    if normalized in dining_labels:
        return dining_labels[normalized]
    return _LABEL_KEYS.get(normalized)


def _status(value: str | None) -> CardAttributeStatus:
    return {
        "verified": "verified",
        "conflict": "conflict",
        "unverified": "pending",
        "pending": "pending",
        "missing": "unknown",
        "unknown": "unknown",
    }.get(value or "", "unknown")  # type: ignore[return-value]


def _field_status(result: SearchResult, field: str) -> CardAttributeStatus:
    field_alias = "deadline" if field in {"deadline", "response_deadline", "application_deadline"} else field
    if field_alias in result.verified_fields:
        return "verified"
    if field_alias in result.unverified_fields:
        return "pending"
    if field in {"deadline", "response_deadline", "application_deadline"} and result.deadline_status == "unverified":
        return "pending" if result.deadline_at else "unknown"
    return "unknown"


def _condition_attributes(result: SearchResult, category: str) -> list[ResultCardAttribute]:
    attributes: list[ResultCardAttribute] = []
    seen: set[str] = set()
    for index, condition in enumerate(result.key_conditions):
        value = condition.value.strip()
        if not value:
            continue
        normalized_label = condition.label.strip()
        key = _category_specific_key(normalized_label, category)
        if key is None:
            key = re.sub(r"[^a-z0-9]+", "_", normalized_label.casefold()).strip("_")
            key = key or f"condition_{index + 1}"
        if key in _PROCUREMENT_KEYS and category != "procurement":
            continue
        if key in _EMPLOYMENT_KEYS and category != "employment":
            continue
        if key in _DINING_KEYS and category != "dining":
            continue
        if key in seen:
            continue
        seen.add(key)
        attributes.append(
            ResultCardAttribute(
                key=key,
                label=normalized_label or key,
                value=value,
                status=_status(condition.status),
                evidence_url=condition.evidence_url,
                evidence_quote=condition.quote,
            )
        )
    return attributes


def _direct_attributes(result: SearchResult, category: str) -> list[ResultCardAttribute]:
    if category == "procurement":
        values: list[tuple[str, str | None, str | None]] = [
            ("purchasing_unit", result.purchasing_unit, result.official_source_url),
            ("budget", result.budget, result.official_source_url),
            ("response_deadline", result.deadline_at, result.official_source_url),
        ]
    elif category == "employment":
        values = [
            ("application_deadline", result.deadline_at, result.official_source_url),
            ("application_url", result.application_url, result.application_url),
        ]
    else:
        values = []
    existing = {attribute.key for attribute in _condition_attributes(result, category)}
    attributes: list[ResultCardAttribute] = []
    for key, value, evidence_url in values:
        if not value or key in existing:
            continue
        label = {
            "response_deadline": "响应截止时间",
            "application_deadline": "投递截止时间",
        }.get(key, _FIELD_LABELS.get(key, (key, key))[0])
        actual_key = key
        attributes.append(
            ResultCardAttribute(
                key=actual_key,
                label=label,
                value=value,
                status=_field_status(result, key),
                evidence_url=evidence_url,
                evidence_quote=None,
            )
        )
    return attributes


def _category(result: SearchResult, query: str) -> str:
    query_text = query.casefold()
    result_text = " ".join((result.title, result.summary, result.snippet, result.evidence)).casefold()
    if any(term in query_text for term in ("餐厅", "餐饮", "午餐", "人均", "台江区")) and any(
        term in result_text for term in ("餐厅", "餐饮", "人均", "菜品", "饭店", "午餐")
    ):
        return "dining"
    if any(term in query_text for term in ("校招", "毕业生", "毕业届", "校园招聘", "招聘岗位")) and any(
        term in result_text for term in ("岗位", "职位", "招聘", "校招", "简历", "投递")
    ):
        return "employment"
    procurement_text = any(
        term in result_text for term in ("政府采购", "采购项目", "招标公告", "采购公告", "采购人")
    )
    if result.procurement_category != "unclear" or (result.purchasing_unit and procurement_text) or procurement_text:
        return "procurement"
    if any(field in result.verified_fields + result.unverified_fields for field in ("graduate_cohort", "major", "role", "organization", "application_url")):
        return "employment"
    if any(term in result_text for term in ("餐厅", "餐饮", "人均", "饭店", "营业时间", "菜品")):
        return "dining"
    if any(term in result_text for term in ("酒店", "景点", "旅游", "行程", "机票")):
        return "travel"
    if any(term in result_text for term in ("政策", "条例", "补贴", "申报")):
        return "policy"
    return "other"


def _source_status(result_status: str) -> CardAttributeStatus:
    return _status(result_status)


def _source_type(value: str) -> str:
    return value if value in {"official", "third_party", "unknown"} else "unknown"


def _source_name(url: str, fallback: str | None) -> str:
    if fallback and fallback.strip():
        return fallback.strip()[:300]
    host = urlparse(url).hostname
    return host or "来源"


def _sources(result: SearchResult, primary_url: str | None) -> list[ResultCardSource]:
    sources: list[ResultCardSource] = []
    seen: set[str] = set()
    for source in result.evidence_sources:
        if source.url == primary_url or source.url in seen:
            continue
        seen.add(source.url)
        sources.append(
            ResultCardSource(
                url=source.url,
                name=_source_name(source.url, source.title),
                type=_source_type(source.source_type),
                status=_source_status(source.status),
            )
        )
    if result.official_source_url and result.official_source_url != primary_url and result.official_source_url not in seen:
        seen.add(result.official_source_url)
        sources.append(
            ResultCardSource(
                url=result.official_source_url,
                name=_source_name(result.official_source_url, None),
                type="official",
                status=(
                    "verified"
                    if result.official_source_status == "verified"
                    else "pending"
                ),
            )
        )
    if result.url != primary_url and result.url not in seen:
        sources.append(
            ResultCardSource(
                url=result.url,
                name=_source_name(result.url, result.source_name),
                type="unknown",
                status="unknown",
            )
        )
    return sources


def _verification_status(attributes: list[ResultCardAttribute], result: SearchResult) -> str:
    if any(attribute.status == "conflict" for attribute in attributes):
        return "conflict"
    if any(attribute.status == "pending" for attribute in attributes):
        return "pending"
    if attributes and all(attribute.status == "verified" for attribute in attributes):
        return "verified"
    if result.official_source_status == "verified" and result.evidence_status == "available":
        # Source identity is validated, but this does not mark every field verified.
        return "pending"
    return "unknown"


def _matching_status(result: SearchResult) -> str:
    if result.decision == "recommended":
        return "recommended"
    if result.decision == "not_matched":
        return "not_matched"
    if result.decision == "needs_verification":
        return "needs_verification"
    if result.match_status == "matched":
        return "recommended"
    if result.match_status == "not_matched":
        return "not_matched"
    if result.match_status in {"partial", "keyword_only", "unverified"}:
        return "needs_verification"
    return "unknown"


def _freshness(category: str, result: SearchResult, *, now: datetime) -> str:
    if category == "dining":
        # Old publication dates do not make a restaurant cease to exist.
        return "unknown" if not result.published_at else _age_freshness(result.published_at, now)
    if category in {"employment", "procurement"}:
        if result.deadline_status == "expired":
            return "stale"
        if result.deadline_status == "not_expired":
            return "fresh"
    if not result.published_at:
        return "unknown"
    return _age_freshness(result.published_at, now)


def _age_freshness(published_at: str, now: datetime) -> str:
    try:
        date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return "unknown"
    return "fresh" if date >= now - timedelta(days=90) else "stale"


def _ranking_key(card: ResultCard) -> tuple[int, int, int, int]:
    matching = {
        "recommended": 0,
        "needs_verification": 1,
        "unknown": 2,
        "not_matched": 3,
    }[card.matching_status]
    verified = {"verified": 0, "pending": 1, "unknown": 2, "conflict": 3}[
        card.verification_status
    ]
    freshness = {"fresh": 0, "unknown": 1, "stale": 2}[card.freshness_status]
    return (matching, verified, freshness, card.rank)


def rank_cards(cards: list[ResultCard]) -> list[ResultCard]:
    """Sort cards by decision/evidence first, then assign stable global ranks."""
    ordered = sorted(cards, key=_ranking_key)
    return [card.model_copy(update={"rank": index}) for index, card in enumerate(ordered, start=1)]


def legacy_result_to_card(
    result: SearchResult, *, query: str, rank: int, now: datetime | None = None
) -> ResultCard:
    """Map only source-backed legacy fields; absent values stay absent."""
    current_time = now or datetime.now(timezone.utc)
    category = _category(result, query)
    attributes = _condition_attributes(result, category)
    known = {attribute.key for attribute in attributes}
    for attribute in _direct_attributes(result, category):
        if attribute.key not in known:
            attributes.append(attribute)
            known.add(attribute.key)
    primary_url: str | None = None
    if result.official_source_url and result.official_source_status == "verified":
        primary_url = result.official_source_url
    else:
        primary_url = next(
            (
                source.url for source in result.evidence_sources
                if source.status == "verified" and source.source_type == "official"
            ),
            None,
        )
    source_title = next(
        (source.title for source in result.evidence_sources if source.url == primary_url),
        None,
    )
    summary = result.summary.strip() or result.snippet.strip()
    matching_status = _matching_status(result)
    verification_status = _verification_status(attributes, result)
    return ResultCard(
        id=result.id,
        title=result.title,
        category=category,  # type: ignore[arg-type]
        summary=summary,
        rank=max(1, rank),
        recommendation_reason=(result.match_reason or result.mismatch_reason or "").strip(),
        attributes=attributes,
        source_url=primary_url,
        source_name=_source_name(primary_url, source_title) if primary_url else None,
        additional_sources=_sources(result, primary_url),
        published_at=result.published_at,
        last_verified_at=result.last_verified_at,
        verification_status=verification_status,  # type: ignore[arg-type]
        matching_status=matching_status,  # type: ignore[arg-type]
        freshness_status=_freshness(category, result, now=current_time),  # type: ignore[arg-type]
    )


def legacy_results_to_cards(
    results: list[SearchResult], *, query: str, now: datetime | None = None
) -> list[ResultCard]:
    cards = [
        legacy_result_to_card(result, query=query, rank=index, now=now)
        for index, result in enumerate(results, start=1)
    ]
    # Use the legacy engine's relevance/confidence signals without inventing
    # numeric scores in the public card schema.
    score_by_id = {result.id: (result.match_score, result.confidence) for result in results}
    ordered = sorted(
        cards,
        key=lambda card: (
            *_ranking_key(card)[:3],
            -score_by_id.get(card.id, (0.0, 0.0))[0],
            -score_by_id.get(card.id, (0.0, 0.0))[1],
            card.rank,
        ),
    )
    return [card.model_copy(update={"rank": index}) for index, card in enumerate(ordered, start=1)]
