"""Evidence-bound normalization from legacy workflow records to generic cards."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

from .schemas import ResultCard, ResultCardAttribute, ResultCardSource


_CATEGORY_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("procurement", ("政府采购", "采购项目", "采购公告", "招标", "投标", "采购单位")),
    ("employment", ("校招", "招聘", "招聘中", "岗位", "职位", "求职", "毕业生", "届毕业")),
    ("dining", ("餐厅", "餐馆", "美食", "午餐", "晚餐", "人均", "营业时间", "饭店")),
    ("travel", ("旅游", "旅行", "景点", "酒店", "民宿", "行程")),
    ("product", ("商品", "购买", "型号", "价格", "多少钱", "推荐买", "产品")),
    ("policy", ("政策", "法规", "条例", "补贴", "申报指南")),
)

_ATTRIBUTE_DEFINITIONS: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    "dining": (
        ("location", "位置", ("位置", "地点", "区域", "附近", "台江", "福州")),
        ("price_per_person", "人均价格", ("人均", "每人", "预算", "元以内", "价格")),
        ("rating", "评分", ("评分", "评价高", "好评", "口碑")),
        ("recent_reviews", "近期评价", ("近期评价", "近期评论", "最近评价", "最近评论")),
        ("open_status", "营业状态", ("营业", "营业中", "营业状态", "是否营业")),
    ),
    "employment": (
        ("company", "公司", ("公司", "企业")),
        ("role", "岗位", ("岗位", "职位", "工作")),
        ("location", "工作地点", ("地点", "地区", "工作地")),
        ("education", "学历", ("学历", "学位", "本科", "硕士")),
        ("major", "专业", ("专业",)),
        ("cohort", "毕业届别", ("届", "毕业年份")),
        ("salary", "薪资", ("薪资", "工资", "月薪", "年薪")),
        ("deadline", "截止日期", ("截止", "报名期限", "投递期限")),
        ("application_url", "投递链接", ("投递", "网申", "申请")),
    ),
    "procurement": (
        ("purchasing_unit", "采购单位", ("采购单位", "采购人", "单位")),
        ("region", "项目地区", ("地区", "区域", "省", "市", "福建")),
        ("budget", "项目预算", ("预算", "预算金额", "金额")),
        ("need", "需求内容", ("需求", "建设", "采购内容", "AI", "人工智能")),
        ("response_deadline", "响应截止时间", ("截止", "响应时间", "投标时间")),
    ),
    "product": (
        ("brand", "品牌", ("品牌",)),
        ("model", "型号", ("型号", "规格")),
        ("price", "价格", ("价格", "预算", "多少钱", "元以内")),
        ("availability", "在售状态", ("在售", "库存", "购买")),
    ),
    "travel": (
        ("location", "位置", ("位置", "地区", "目的地", "附近")),
        ("price", "价格", ("预算", "价格", "人均")),
        ("dates", "出行时间", ("时间", "日期", "几天")),
        ("availability", "开放状态", ("开放", "营业", "开放时间")),
    ),
    "policy": (
        ("issuing_authority", "发布单位", ("发布单位", "发布机关", "主管部门")),
        ("effective_date", "生效时间", ("生效", "实施日期")),
        ("deadline", "申报截止", ("申报截止", "截止日期")),
        ("eligibility", "适用条件", ("适用对象", "申请条件", "申报条件")),
    ),
}

_FIELD_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "dining": {
        "location": ("location", "region"),
        "price_per_person": ("price_per_person", "per_person_price"),
        "rating": ("rating", "score"),
        "recent_reviews": ("recent_reviews", "reviews"),
        "open_status": ("open_status",),
    },
    "employment": {
        "company": ("organization", "company"),
        "role": ("role",),
        "location": ("location", "region"),
        "education": ("education", "qualifications"),
        "major": ("major", "majors"),
        "cohort": ("graduate_cohort", "cohort"),
        "salary": ("salary",),
        "deadline": ("deadline",),
        "application_url": ("application_url",),
    },
    "procurement": {
        "purchasing_unit": ("purchasing_unit",),
        "region": ("region",),
        "budget": ("budget",),
        "need": ("business_attributes", "evidence", "need"),
        "response_deadline": ("deadline", "response_deadline"),
    },
    "product": {
        "brand": ("brand",),
        "model": ("model",),
        "price": ("price",),
        "availability": ("availability",),
    },
    "travel": {
        "location": ("region", "location"),
        "price": ("price",),
        "dates": ("dates",),
        "availability": ("availability", "open_status"),
    },
    "policy": {
        "issuing_authority": ("organization", "issuing_authority"),
        "effective_date": ("effective_date",),
        "deadline": ("deadline",),
        "eligibility": ("qualifications", "eligibility"),
    },
}


def classify_category(query: str, records: Sequence[Mapping[str, Any]] = ()) -> str:
    """Classify the user need first; use candidate evidence only as a fallback."""
    text = query.casefold()
    for category, terms in _CATEGORY_TERMS:
        if any(term.casefold() in text for term in terms):
            return category
    evidence = " ".join(
        " ".join(_source_texts(record))
        for record in records[:3]
        if isinstance(record, Mapping)
    ).casefold()
    for category, terms in _CATEGORY_TERMS:
        if any(term.casefold() in evidence for term in terms):
            return category
    return "other"


def build_result_cards(query: str, records: Sequence[Mapping[str, Any]]) -> list[ResultCard]:
    """Map all judged records to cards, conservatively merging only known entities."""
    category = classify_category(query, records)
    groups: OrderedDict[str, list[tuple[int, Mapping[str, Any]]]] = OrderedDict()
    for rank, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            continue
        key = _entity_key(category, record)
        groups.setdefault(key or f"record:{record.get('id', rank)}:{rank}", []).append((rank, record))

    cards: list[ResultCard] = []
    for group in groups.values():
        cards.append(_card_for_group(query, category, group))
    return cards


def _card_for_group(
    query: str,
    category: str,
    group: Sequence[tuple[int, Mapping[str, Any]]],
) -> ResultCard:
    rank, primary = group[0]
    all_records = [record for _, record in group]
    source_items = _collect_sources(all_records)
    primary_source = next((item for item in source_items if item.status == "verified" and item.type == "official"), None)
    source_url = primary_source.url if primary_source else None
    source_name = primary_source.name if primary_source else None
    additional_sources = [item for item in source_items if item.url != source_url]

    attributes = _build_attributes(query, category, all_records)
    # Re-check after cross-source aggregation as a final guard. Field extractors
    # and legacy projections can evolve independently of this card mapper.
    attributes = _validate_card_attributes(query, category, attributes, all_records)
    identity = _entity_key(category, primary)
    stable_id = identity or str(primary.get("id") or hashlib.sha256(f"{rank}:{primary.get('title', '')}".encode()).hexdigest()[:20])
    verification_status = _verification_status(all_records)
    matching_status = _matching_status(primary)
    freshness_status = _freshness_status(category, primary)
    summary = _summary(primary)
    recommendation_reason = _first_text(primary.get("match_reason"), primary.get("mismatch_reason"))

    return ResultCard(
        id=stable_id,
        title=_first_text(primary.get("title"), primary.get("official_source_title")) or "未命名信息",
        category=category,
        summary=summary,
        rank=min(item_rank for item_rank, _ in group),
        recommendation_reason=recommendation_reason,
        attributes=attributes,
        source_url=source_url,
        source_name=source_name,
        additional_sources=additional_sources,
        published_at=_nullable_text(primary.get("published_at")),
        last_verified_at=_nullable_text(_first_text(primary.get("last_verified_at"), primary.get("verification_checked_at"))),
        verification_status=verification_status,
        matching_status=matching_status,
        freshness_status=freshness_status,
    )


def _build_attributes(
    query: str,
    category: str,
    records: Sequence[Mapping[str, Any]],
) -> list[ResultCardAttribute]:
    definitions = _ATTRIBUTE_DEFINITIONS.get(category, ())
    if not definitions:
        definitions = _generic_definitions(records)
    attributes: list[ResultCardAttribute] = []
    used_keys: set[str] = set()
    for key, label, _terms in definitions:
        observations = [
            fact
            for record in records
            if (fact := _fact_for_attribute(query, category, key, record)) is not None
        ]
        attribute = _merge_facts(key, label, observations)
        if attribute is not None:
            attributes.append(attribute)
            used_keys.add(key)
            continue
        explicit = next((
            fact for record in records
            if (fact := _explicit_fact_for_category(query, category, key, label, record)) is not None
        ), None)
        if explicit:
            attributes.append(_fact_attribute(key, label, explicit))
            used_keys.add(key)
        elif _query_requests_attribute(query, key, label, _terms):
            attributes.append(ResultCardAttribute(
                key=key, label=label, value=None, status="unknown",
                evidence_url=None, evidence_quote=None,
            ))
            used_keys.add(key)

    if category == "dining":
        for record in records:
            for key, label in (("location", "位置"), ("price_per_person", "人均价格"), ("rating", "评分"),
                               ("recent_reviews", "近期评价"), ("open_status", "营业状态")):
                if key in used_keys:
                    continue
                fact = _explicit_dining_fact(key, record)
                if fact:
                    attributes.append(_fact_attribute(key, label, fact))
                    used_keys.add(key)
    if category == "employment":
        for record in records:
            for key, label in (("salary", "薪资"), ("education", "学历")):
                if key in used_keys:
                    continue
                fact = _explicit_text_fact("", key, label, record)
                if fact:
                    attributes.append(_fact_attribute(key, label, fact))
                    used_keys.add(key)
    if category == "product":
        for record in records:
            for key, label in (("brand", "品牌"), ("model", "型号"), ("price", "价格"), ("availability", "在售状态")):
                if key in used_keys:
                    continue
                fact = _explicit_text_fact("", key, label, record)
                if fact:
                    attributes.append(_fact_attribute(key, label, fact))
                    used_keys.add(key)
    return attributes


def _fact_for_attribute(query: str, category: str, key: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
    aliases = _FIELD_ALIASES.get(category, {}).get(key, (key,) if category == "other" else ())
    checks = record.get("field_verification")
    for field in aliases:
        check = checks.get(field) if isinstance(checks, Mapping) else None
        if isinstance(check, Mapping) and check.get("value") is not None:
            quote = _nullable_text(check.get("quote"))
            url = _valid_url(check.get("evidence_url"))
            status = _status(str(check.get("status", "unverified")))
            if (
                not quote or not url or not _quote_is_grounded(record, url, quote)
                or not _fact_value_supported(query, category, key, str(check["value"]), quote)
            ):
                continue
            return {"value": str(check["value"]), "status": status, "url": url, "quote": quote}

    extracted = record.get("extracted_fields")
    sources = record.get("field_sources")
    for field in aliases:
        fact = extracted.get(field) if isinstance(extracted, Mapping) else None
        source = sources.get(field) if isinstance(sources, Mapping) else None
        if not isinstance(fact, Mapping) or not isinstance(source, Mapping):
            continue
        value = _nullable_text(fact.get("value"))
        quote = _nullable_text(source.get("quote") or fact.get("quote"))
        url = _valid_url(source.get("url"))
        if (
            value and quote and _quote_is_grounded(record, url, quote)
            and _fact_value_supported(query, category, key, value, quote)
        ):
            return {"value": value, "status": "pending", "url": url, "quote": quote}

    # Legacy, evidence-backed fields are retained as cards without inventing
    # field verification. Only use them when the legacy verifier emitted a quote.
    legacy = {
        "purchasing_unit": ("purchasing_unit", "采购单位"),
        "budget": ("budget", "预算"),
        "deadline": ("deadline_at", "deadline"),
        "response_deadline": ("deadline_at", "deadline"),
        "company": ("organization", "organization"),
        "cohort": ("graduate_cohort", "graduate_cohort"),
        "major": ("major", "major"),
        "role": ("role", "role"),
        "application_url": ("application_url", "application_url"),
    }.get(key)
    if legacy:
        value_field, check_field = legacy
        value = _nullable_text(record.get(value_field))
        check = checks.get(check_field) if isinstance(checks, Mapping) else None
        if value and isinstance(check, Mapping):
            quote = _nullable_text(check.get("quote"))
            url = _valid_url(check.get("evidence_url"))
            status = _status(str(check.get("status", "unverified")))
            if (
                quote and url and _quote_is_grounded(record, url, quote)
                and _fact_value_supported(query, category, key, value, quote)
            ):
                return {"value": value, "status": status, "url": url, "quote": quote}
    return None


def _explicit_dining_fact(key: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
    text, url = _primary_text(record)
    if not text:
        return None
    patterns: dict[str, tuple[str, ...]] = {
        "price_per_person": (
            r"(?:人均(?:消费)?|每人消费|均价)\s*[:：]?\s*[¥￥]?\s*\d+(?:\.\d+)?\s*(?:元)?",
            r"[¥￥]\s*\d+(?:\.\d+)?\s*/\s*(?:人|位)",
        ),
        "rating": (r"(?:评分|口碑评分)\s*[:：]?\s*\d(?:\.\d+)?\s*(?:分|/\s*5)?",),
        "recent_reviews": (r"(?:近期评价|最近评价|近期评论|最近评论)\s*[:：]?[^\n。；;]{2,100}",),
        "open_status": (r"(?:营业中|暂停营业|暂时停业|正常营业|已停业)",),
        "location": (r"(?:地址|位置|所在区域|商圈)\s*[:：]\s*[^\n。；;]{2,100}",),
    }
    for pattern in patterns.get(key, ()):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            quote = match.group(0).strip()
            value = _fact_value(key, quote)
            if value:
                return {"value": value, "status": _source_status(record), "url": url, "quote": quote}
    return None


def _explicit_fact_for_category(
    query: str, category: str, key: str, label: str, record: Mapping[str, Any]
) -> dict[str, Any] | None:
    if category == "dining":
        return _explicit_dining_fact(key, record)
    if category in {"employment", "product"}:
        return _explicit_text_fact(query, key, label, record)
    return None


def _explicit_text_fact(query: str, key: str, label: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
    text, url = _primary_text(record)
    if not text:
        return None
    patterns = {
        "salary": r"(?:薪资|工资|月薪|年薪|薪酬)\s*[:：]?\s*[^\n。；;]{1,50}",
        "education": r"(?:学历要求|学历|学位要求)\s*[:：]?\s*[^\n。；;]{1,80}",
        "brand": r"(?:品牌)\s*[:：]\s*[^\n，,。；;]{1,40}",
        "model": r"(?:型号|规格型号)\s*[:：]\s*[^\n，,。；;]{1,60}",
        "price": r"(?:价格|售价|参考价)\s*[:：]?\s*[¥￥]?\s*\d+(?:\.\d+)?\s*(?:元)?",
        "availability": r"(?:有货|现货|在售|暂时缺货|缺货|售罄)",
    }
    match = re.search(patterns.get(key, r"(?!)"), text, re.IGNORECASE)
    if not match:
        return None
    quote = match.group(0).strip()
    value = _fact_value(key, quote) or quote
    if not _fact_value_supported(query, "employment" if key in {"salary", "education"} else "product", key, value, quote):
        return None
    return {"value": value, "status": _source_status(record), "url": url, "quote": quote}


def _fact_value_supported(query: str, category: str, key: str, value: str, quote: str) -> bool:
    """Require this exact field value, or a narrow normalized form, in its cited quote."""
    value = value.strip()
    quote = quote.strip()
    if not value or not quote:
        return False

    normalized_value = _evidence_text(value)
    normalized_quote = _evidence_text(quote)

    if key == "role":
        role = _clean_role_value(value)
        return bool(role) and _evidence_text(role) in normalized_quote

    if key == "company":
        company = _clean_company_value(value)
        return bool(company) and _evidence_text(company) in normalized_quote

    if key == "education":
        return _normalize_education_value(value, quote) is not None

    if key == "salary":
        # Generic labels such as “薪酬体系” are not salary evidence. Require one
        # or more actual amounts with a monetary unit in both the value and quote.
        amounts = _salary_amounts(value)
        quoted_amounts = _salary_amounts(quote)
        return bool(amounts) and all(amount in quoted_amounts for amount in amounts)

    if key == "price_per_person":
        amounts = _salary_amounts(value)
        quoted_amounts = _salary_amounts(quote)
        has_per_person_context = any(term in quote for term in ("人均", "每人消费", "每人", "元/人", "元/位"))
        return bool(amounts) and amounts.issubset(quoted_amounts) and has_per_person_context

    if key == "major":
        majors = [part.strip() for part in re.split(r"[、,，;；|/]+", value) if part.strip()]
        return bool(majors) and all(_evidence_text(major) in normalized_quote for major in majors)

    if key in {"cohort", "graduate_cohort"}:
        years = set(re.findall(r"20\d{2}", value))
        quote_years = set(re.findall(r"20\d{2}", quote))
        has_cohort_context = any(term in quote for term in ("届", "毕业", "应届"))
        return bool(years) and years.issubset(quote_years) and has_cohort_context

    if key in {"deadline", "response_deadline", "effective_date"}:
        value_date = _parse_date(value)
        quote_dates = _dates_in_text(quote)
        return value_date is not None and value_date in quote_dates

    if key == "dates":
        value_dates = _dates_in_text(value)
        quote_dates = _dates_in_text(quote)
        return bool(value_dates) and value_dates.issubset(quote_dates)

    if key == "need":
        if category != "procurement":
            return False
        generic_need_labels = {"项目概况", "项目需求", "采购需求", "需求内容", "采购内容", "建设需求"}
        if _evidence_text(value) in {_evidence_text(term) for term in generic_need_labels}:
            return False
        procurement_terms = (
            "采购", "需求", "建设", "系统", "平台", "软件", "设备", "服务", "应用",
            "培训", "模型", "算力", "数据", "算法", "智能", "人工智能", "AI",
        )
        if not any(term.casefold() in value.casefold() for term in procurement_terms):
            return False
        ai_query = any(term.casefold() in query.casefold() for term in ("AI", "人工智能", "大模型", "机器学习"))
        if ai_query:
            ai_terms = ("AI", "人工智能", "大模型", "机器学习", "智能")
            if not any(term.casefold() in value.casefold() for term in ai_terms):
                return False
            if not any(term.casefold() in quote.casefold() for term in ai_terms):
                return False

    return bool(normalized_value) and normalized_value in normalized_quote


def _normalize_card_attribute_value(category: str, key: str, value: str, quote: str) -> str | None:
    if key == "role":
        return _clean_role_value(value)
    if key == "company":
        return _clean_company_value(value)
    if key == "education":
        return _normalize_education_value(value, quote)
    return value.strip()


def _clean_role_value(value: str) -> str | None:
    role = value.strip()
    # Extracted snippets can accidentally be Markdown navigation, bullets,
    # instructions, or a truncated link instead of a concrete job title.
    if (
        len(role) > 80
        or "\n" in role or "\r" in role
        or re.match(r"^(?:[#>*+\-]|\d+[.)、])", role)
        or re.search(r"\[[^\]]*\]\s*\(|!?\[[^\]]*\]\s*\(|https?://|javascript:|mailto:", role, re.IGNORECASE)
        or any(mark in role for mark in ("<", ">", "`", "。", "；", ";", "！", "？"))
        or any(term in role for term in ("请您", "请在", "请于", "请填写", "请标明", "点击", "简历上", "意向地点"))
    ):
        return None
    role = re.sub(r"^(?:岗位|职位)(?:名称|信息)?\s*[:：]\s*", "", role).strip()
    generic_headings = {
        "招聘公告", "招聘公告与岗位信息", "招聘公告及岗位信息", "招聘信息",
        "岗位信息", "岗位职责", "岗位要求", "招聘岗位", "招聘职位", "岗位及任职要求",
        "职位信息", "职位要求", "职位职责", "岗位列表", "职位列表", "岗位名称", "职位名称",
    }
    if not role or role.startswith("#") or _evidence_text(role) in {_evidence_text(x) for x in generic_headings}:
        return None
    return role


def _clean_company_value(value: str) -> str | None:
    company = value.strip()
    if (
        len(company) > 100
        or "\n" in company or "\r" in company
        or re.match(r"^(?:[#>*+\-]|\d+[.)、])", company)
        or re.search(r"\[[^\]]*\]\s*\(|!?\[[^\]]*\]\s*\(|https?://|javascript:|mailto:|`|\*\*", company, re.IGNORECASE)
        or any(mark in company for mark in ("。", "；", ";", "！", "？"))
    ):
        return None
    pairs = {"(": ")", "（": "）", "[": "]", "【": "】", "{": "}", "〔": "〕"}
    closers = {value: key for key, value in pairs.items()}
    stack: list[str] = []
    for character in company:
        if character in pairs:
            stack.append(character)
        elif character in closers:
            if not stack or stack.pop() != closers[character]:
                return None
    if stack or company.endswith(("：", ":", "，", ",", "、", "（", "(", "[", "【", "{")):
        return None
    company = re.sub(r"^(?:公司名称|企业名称|招聘单位|公司|企业)\s*[:：]\s*", "", company).strip()
    return company or None


def _education_levels(value: str) -> list[str]:
    pattern = r"博士研究生|硕士研究生|研究生|博士|硕士|本科|大专|专科|中专|中技|高中|高职"
    levels: list[str] = []
    for match in re.finditer(rf"({pattern})(?:\s*(及以上|以上))?", value):
        level = match.group(1) + (match.group(2) or "")
        if level not in levels:
            levels.append(level)
    return levels


def _normalize_education_value(value: str, quote: str) -> str | None:
    value_levels = _education_levels(value)
    quote_levels = set(_education_levels(quote))
    supported = [level for level in value_levels if level in quote_levels]
    return "、".join(supported) if supported else None


def _evidence_text(value: str) -> str:
    value = value.casefold().replace("人工智能", "ai")
    value = re.sub(r"\s+", "", value)
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)


def _salary_amounts(value: str) -> set[str]:
    pattern = re.compile(
        r"(?<!\d)(\d+(?:,\d{3})*(?:\.\d+)?)(?:\s*[-~至到]\s*(\d+(?:,\d{3})*(?:\.\d+)?))?\s*(万|千|k|元)",
        re.IGNORECASE,
    )
    amounts: set[str] = set()
    for match in pattern.finditer(value):
        unit = match.group(3).casefold()
        for number in (match.group(1), match.group(2)):
            if number:
                amounts.add(f"{number.replace(',', '')}{unit}")
    return amounts


def _dates_in_text(value: str) -> set[date]:
    matches = re.finditer(
        r"20\d{2}\s*(?:年|[-/.])\s*\d{1,2}\s*(?:月|[-/.])\s*\d{1,2}",
        value,
    )
    result: set[date] = set()
    for match in matches:
        parsed = _parse_date(match.group(0))
        if parsed:
            result.add(parsed)
    return result


def _fact_value(key: str, quote: str) -> str:
    if key in {"price_per_person", "price"}:
        match = re.search(r"(?:[¥￥]\s*)?(\d+(?:\.\d+)?)\s*(?:元)?", quote)
        return f"{match.group(1)}元/人" if match and key == "price_per_person" else f"{match.group(1)}元" if match else ""
    if key == "rating":
        match = re.search(r"\d(?:\.\d+)?", quote)
        return f"{match.group(0)}分" if match else ""
    if key == "location":
        return re.sub(r"^(?:地址|位置|所在区域|商圈)\s*[:：]\s*", "", quote).strip()
    if key == "brand":
        return re.sub(r"^品牌\s*[:：]\s*", "", quote).strip()
    if key == "model":
        return re.sub(r"^(?:规格型号|型号)\s*[:：]\s*", "", quote).strip()
    if key == "education":
        return re.sub(r"^(?:学历要求|学历|学位要求)\s*[:：]\s*", "", quote).strip()
    return quote


def _generic_definitions(records: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    found: OrderedDict[str, str] = OrderedDict()
    for record in records:
        checks = record.get("field_verification")
        extracted = record.get("extracted_fields")
        for source in (checks, extracted):
            if not isinstance(source, Mapping):
                continue
            for key, fact in source.items():
                if key.startswith("_") or not isinstance(fact, Mapping):
                    continue
                if not fact.get("value"):
                    continue
                label = str(fact.get("label") or _generic_label(key))
                found.setdefault(key, label)
    return tuple((key, label, ()) for key, label in found.items())


def _merge_facts(key: str, label: str, facts: Sequence[Mapping[str, Any]]) -> ResultCardAttribute | None:
    if not facts:
        return None
    values = {str(item.get("value")) for item in facts if item.get("value") is not None}
    verified_values = {str(item.get("value")) for item in facts if item.get("status") == "verified"}
    if len(verified_values) > 1 or len(values) > 1 and any(item.get("status") == "conflict" for item in facts):
        selected = next((item for item in facts if item.get("status") == "conflict"), facts[0])
        return _fact_attribute(key, label, {**selected, "status": "conflict"})
    selected = max(facts, key=lambda item: _status_strength(str(item.get("status"))))
    return _fact_attribute(key, label, selected)


def _fact_attribute(key: str, label: str, fact: Mapping[str, Any]) -> ResultCardAttribute:
    return ResultCardAttribute(
        key=key,
        label=label,
        value=_nullable_text(fact.get("value")),
        status=_status(str(fact.get("status", "unknown"))),
        evidence_url=_valid_url(fact.get("url")),
        evidence_quote=_nullable_text(fact.get("quote")),
    )


def _validate_card_attributes(
    query: str,
    category: str,
    attributes: Sequence[ResultCardAttribute],
    records: Sequence[Mapping[str, Any]],
) -> list[ResultCardAttribute]:
    validated: list[ResultCardAttribute] = []
    for attribute in attributes:
        value = attribute.value
        quote = attribute.evidence_quote
        url = _valid_url(attribute.evidence_url)
        normalized_value = (
            _normalize_card_attribute_value(category, attribute.key, value, quote)
            if value and quote else None
        )
        grounded = bool(
            normalized_value and quote and url
            and any(_quote_is_grounded(record, url, quote) for record in records)
            and _fact_value_supported(query, category, attribute.key, normalized_value, quote)
        )
        if grounded or value is None:
            if grounded and normalized_value != value:
                status = attribute.status
                if attribute.key == "education":
                    source_levels = _education_levels(value)
                    retained_levels = _education_levels(normalized_value)
                    if len(retained_levels) < len(source_levels):
                        status = "pending"
                validated.append(ResultCardAttribute(
                    key=attribute.key,
                    label=attribute.label,
                    value=normalized_value,
                    status=status,
                    evidence_url=attribute.evidence_url,
                    evidence_quote=quote,
                ))
            else:
                validated.append(attribute)
            continue
        validated.append(ResultCardAttribute(
            key=attribute.key,
            label=attribute.label,
            value=None,
            status="unknown",
            evidence_url=None,
            evidence_quote=None,
        ))
    return validated


def _collect_sources(records: Sequence[Mapping[str, Any]]) -> list[ResultCardSource]:
    sources: OrderedDict[str, ResultCardSource] = OrderedDict()
    for record in records:
        official_url = _valid_url(record.get("official_source_url"))
        if official_url:
            official_verified = _official_source_verified(record, official_url)
            _append_source(
                sources,
                official_url,
                record.get("official_source_name"),
                "official" if official_verified else "unknown",
                "verified" if official_verified else "pending",
            )
        url = _valid_url(record.get("url"))
        if url:
            status = _source_status(record)
            source_type = "official" if url == official_url and _official_source_verified(record, url) else (
                "third_party" if record.get("official_source_status") == "third_party" else "unknown"
            )
            if source_type == "official" and not _official_source_verified(record, url):
                status = "pending"
            _append_source(sources, url, record.get("source_name"), source_type, status)
        evidence_sources = record.get("evidence_sources")
        if isinstance(evidence_sources, Sequence) and not isinstance(evidence_sources, (str, bytes)):
            for source in evidence_sources:
                if not isinstance(source, Mapping):
                    continue
                source_url = _valid_url(source.get("url"))
                if source_url:
                    source_type = str(source.get("source_type", "unknown"))
                    source_status = _status(str(source.get("status", "unknown")))
                    if source_type == "official" and not _official_source_verified(record, source_url):
                        source_type = "unknown"
                        source_status = "pending" if source_status != "conflict" else source_status
                    _append_source(
                        sources, source_url, source.get("title") or source.get("name"),
                        source_type, source_status,
                    )
    return list(sources.values())


def _append_source(
    sources: OrderedDict[str, ResultCardSource],
    url: str,
    name: Any,
    source_type: str,
    status: str,
) -> None:
    current = sources.get(url)
    normalized_type = source_type if source_type in {"official", "third_party", "unknown"} else "unknown"
    normalized_status = _status(status)
    candidate = ResultCardSource(
        url=url,
        name=_source_name(name, url),
        type=normalized_type,
        status=normalized_status,
    )
    if current is None or _status_strength(candidate.status) > _status_strength(current.status):
        sources[url] = candidate


def _entity_key(category: str, record: Mapping[str, Any]) -> str | None:
    """Return a merge key only when source evidence establishes same entity."""
    fields = record.get("extracted_fields")
    fields = fields if isinstance(fields, Mapping) else {}
    verification = record.get("field_verification")
    verification = verification if isinstance(verification, Mapping) else {}

    def fact_value(*names: str) -> str:
        for name in names:
            fact = verification.get(name)
            if isinstance(fact, Mapping) and fact.get("value") and fact.get("status") == "verified":
                return str(fact["value"])
            fact = fields.get(name)
            if isinstance(fact, Mapping) and fact.get("value"):
                source_map = record.get("field_sources")
                source = source_map.get(name) if isinstance(source_map, Mapping) else None
                quote = source.get("quote") if isinstance(source, Mapping) else None
                url = source.get("url") if isinstance(source, Mapping) else None
                if isinstance(quote, str) and quote and _quote_is_grounded(record, _valid_url(url), quote):
                    return str(fact["value"])
        return ""

    explicit_ids = _posting_identity(record)
    if category == "employment":
        company = fact_value("organization", "company")
        role = fact_value("role")
        application = fact_value("application_url")
        cohort = fact_value("graduate_cohort")
        location = fact_value("location", "region")
        if company and role and application:
            return _digest_key(category, company, role, application, cohort, location)
        posting_id = explicit_ids
        if company and role and posting_id:
            return _digest_key(category, company, role, posting_id, cohort, location)
    elif category == "procurement":
        official_url = _valid_url(record.get("official_source_url"))
        official_url = official_url if official_url and _official_source_verified(record, official_url) else ""
        official_project = fact_value("project_name")
        official_title = _first_text(record.get("official_source_title")) if official_url else ""
        candidate_title = _first_text(record.get("title"))
        lot_id = _project_lot_identity(record)
        if official_url and (official_project or official_title or candidate_title):
            unit = fact_value("purchasing_unit")
            budget = fact_value("budget")
            deadline = fact_value("deadline")
            return _digest_key(
                category,
                official_url,
                _normalize_identity(official_project),
                _normalize_identity(official_title),
                _normalize_identity(candidate_title),
                lot_id,
                _normalize_identity(unit),
                _normalize_identity(budget),
                _normalize_identity(deadline),
            )
        unit = fact_value("purchasing_unit") or _nullable_text(record.get("purchasing_unit")) or ""
        title = _first_text(record.get("official_source_title"), record.get("title"))
        deadline = fact_value("deadline") or _nullable_text(record.get("deadline_at")) or ""
        budget = fact_value("budget") or _nullable_text(record.get("budget")) or ""
        if unit and title and deadline and budget:
            return _digest_key(category, _normalize_identity(title), _normalize_identity(unit), deadline, budget, lot_id)
    elif category == "dining":
        name = fact_value("restaurant_name", "merchant_name", "organization") or _restaurant_name(record)
        location = fact_value("location", "region") or _restaurant_location(record)
        merchant_id = _restaurant_source_id(record)
        if name and location:
            return _digest_key(category, _normalize_identity(name), _normalize_identity(location))
        if merchant_id:
            return _digest_key(category, merchant_id)
    elif category == "product":
        model = fact_value("model", "product_model")
        name = fact_value("product_name") or _first_text(record.get("title"))
        if model:
            return _digest_key(category, _normalize_identity(name), _normalize_identity(model))
        if name:
            return _digest_key(category, _normalize_identity(name))
    elif category == "policy":
        url = _valid_url(record.get("official_source_url"))
        if url and record.get("official_source_status") == "verified":
            return _digest_key(category, url)
    return None


def _restaurant_name(record: Mapping[str, Any]) -> str:
    text, _url = _primary_text(record)
    for pattern in (
        r"(?:餐厅名称|商户名称|店铺名称|餐馆名称)\s*[:：]\s*([^\n，,。；;]{2,60})",
        r"(?:餐厅|饭店|餐馆)\s*[:：]\s*([^\n，,。；;]{2,60})",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    title = _first_text(record.get("title"))
    title = re.split(r"\s*[|｜｜·•]\s*", title, maxsplit=1)[0]
    title = re.sub(r"(?:大众点评|美团|小红书|携程|攻略|评价|点评|菜单|地址|电话|营业时间).*$", "", title).strip(" -_—|｜:：")
    return title if any(term in title for term in ("餐厅", "餐馆", "饭店")) and len(title) < 80 else ""


def _restaurant_location(record: Mapping[str, Any]) -> str:
    text, _url = _primary_text(record)
    match = re.search(r"(?:门店地址|店铺地址|地址|位置|商圈)\s*[:：]\s*([^\n。；;]{2,120})", text)
    return match.group(1).strip() if match else ""


def _restaurant_source_id(record: Mapping[str, Any]) -> str:
    url = _valid_url(record.get("url"))
    if not url:
        return ""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    for key in ("merchant_id", "store_id", "shop_id", "restaurant_id", "poi_id"):
        values = params.get(key)
        if values and values[0].strip():
            return f"{parsed.hostname or ''}:{key}:{values[0].casefold()}"
    match = re.search(r"/(?:store|shop|merchant|restaurant|poi)/([a-z0-9_-]{3,})(?:/|$)", parsed.path, re.I)
    return f"{parsed.hostname or ''}:path:{match.group(1).casefold()}" if match else ""


def _posting_identity(record: Mapping[str, Any]) -> str:
    url = _valid_url(record.get("url")) or ""
    text, _ = _primary_text(record)
    parsed = urlparse(url)
    query_id = re.search(r"(?:job|position|requisition|posting)[_-]?id=([a-z0-9_-]+)", parsed.query, re.I)
    if query_id:
        return query_id.group(1).casefold()
    path_id = re.search(r"/(?:jobs?|positions?|requisitions?)/(?:detail/)?([a-z0-9_-]{3,})(?:/|$)", parsed.path, re.I)
    if path_id:
        return path_id.group(1).casefold()
    body_id = re.search(r"(?:岗位编号|职位编号|招聘编号|岗位ID|职位ID)\s*[:：]?\s*([a-z0-9_-]{3,})", text, re.I)
    return body_id.group(1).casefold() if body_id else ""


def _project_lot_identity(record: Mapping[str, Any]) -> str:
    text, _url = _primary_text(record)
    for pattern in (
        r"(?:采购包编号|包件编号|标段编号|项目编号)\s*[:：]?\s*([a-z0-9_-]{2,})",
        r"(?:第\s*)?([一二三四五六七八九十\d]+)\s*(?:包|标段)",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).casefold()
    return ""


def _digest_key(*parts: str) -> str:
    return "entity:" + hashlib.sha256("|".join(part.casefold() for part in parts).encode()).hexdigest()[:24]


def _verification_status(records: Sequence[Mapping[str, Any]]) -> str:
    statuses = [
        _status(str(check.get("status", "unknown")))
        for record in records
        for check in (record.get("field_verification", {}).values() if isinstance(record.get("field_verification"), Mapping) else ())
        if isinstance(check, Mapping)
        and isinstance(check.get("quote"), str)
        and bool(check["quote"].strip())
        and _valid_url(check.get("evidence_url"))
        and _quote_is_grounded(record, _valid_url(check.get("evidence_url")), check["quote"].strip())
    ]
    if "conflict" in statuses:
        return "conflict"
    if "verified" in statuses and any(
        (url := _valid_url(record.get("official_source_url"))) and _official_source_verified(record, url)
        for record in records
    ):
        return "verified"
    if any(_source_texts(record) for record in records):
        return "pending"
    return "unknown"


def _matching_status(record: Mapping[str, Any]) -> str:
    decision = str(record.get("decision", ""))
    if decision in {"recommended", "needs_verification", "not_matched"}:
        return decision
    match = str(record.get("match_status", ""))
    return {
        "matched": "recommended",
        "partial": "needs_verification",
        "keyword_only": "needs_verification",
        "unverified": "needs_verification",
        "not_matched": "not_matched",
    }.get(match, "unknown")


def _freshness_status(category: str, record: Mapping[str, Any], today: date | None = None) -> str:
    today = today or datetime.now(timezone.utc).date()
    field_checks = record.get("field_verification")
    field_checks = field_checks if isinstance(field_checks, Mapping) else {}
    deadline = field_checks.get("deadline")
    deadline_date = None
    if isinstance(deadline, Mapping) and _grounded_verified_fact(record, deadline):
        deadline_date = _parse_date(deadline.get("value"))
    if category in {"procurement", "employment"}:
        open_check = field_checks.get("open_status")
        if isinstance(open_check, Mapping) and _grounded_verified_fact(record, open_check):
            value = str(open_check.get("value", ""))
            if any(term in value for term in ("已截止", "招聘结束", "已关闭", "已结束", "expired", "closed")):
                return "stale"
            if any(term in value for term in ("招聘中", "正在招聘", "投递中", "未截止", "开放", "not_expired", "open")):
                return "fresh"
        if deadline_date:
            return "fresh" if deadline_date >= today else "stale"
    if category == "dining":
        text = " ".join(_source_texts(record))
        review_date = re.search(r"(?:近期评价|最近评价|评论时间|评价时间)\s*[:：]?\s*(20\d{2}[-年./]\d{1,2}[-月./]\d{1,2})", text)
        if review_date:
            parsed = _parse_date(review_date.group(1))
            return _age_status(parsed, today) if parsed else "unknown"
        # Venue age/listing time says nothing about current operating relevance.
        return "unknown"
    published_check = field_checks.get("published_at")
    if not isinstance(published_check, Mapping) or not _grounded_verified_fact(record, published_check):
        return "unknown"
    published = _parse_date(published_check.get("value"))
    return _age_status(published, today) if published else "unknown"


def _grounded_verified_fact(record: Mapping[str, Any], fact: Mapping[str, Any]) -> bool:
    evidence_url = _valid_url(fact.get("evidence_url"))
    quote = fact.get("quote")
    return (
        fact.get("status") == "verified"
        and evidence_url is not None
        and isinstance(quote, str)
        and bool(quote.strip())
        and _quote_is_grounded(record, evidence_url, quote.strip())
    )


def _age_status(value: date | None, today: date) -> str:
    if value is None:
        return "unknown"
    delta = (today - value).days
    if delta < -1:
        return "unknown"
    return "fresh" if delta <= 90 else "stale"


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    match = re.search(r"(20\d{2})\s*(?:年|[-/.])\s*(\d{1,2})\s*(?:月|[-/.])\s*(\d{1,2})", value)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _query_requests_attribute(query: str, key: str, label: str, terms: Sequence[str]) -> bool:
    text = query.casefold()
    if any(term.casefold() in text for term in terms if term):
        return True
    aliases = {
        "price_per_person": ("200元以内", "人均", "预算"),
        "rating": ("评价高", "评分高", "高评分", "好评"),
        "recent_reviews": ("近期评价", "最近评价"),
        "cohort": ("届",),
        "major": ("专业",),
        "open_status": ("正在招聘", "招聘中", "未截止", "仍在招标"),
        "response_deadline": ("未截止", "响应截止"),
        "need": ("AI", "人工智能", "建设需求"),
    }
    return any(term.casefold() in text for term in aliases.get(key, ())) or label.casefold() in text


def _source_status(record: Mapping[str, Any]) -> str:
    official_url = _valid_url(record.get("official_source_url"))
    if official_url and _official_source_verified(record, official_url):
        return "verified"
    if _source_texts(record):
        return "pending"
    return "unknown"


def _official_source_verified(record: Mapping[str, Any], official_url: str) -> bool:
    """Do not infer verification from a URL or a status label alone."""
    if record.get("official_source_status") != "verified":
        return False
    checks = record.get("field_verification")
    if not isinstance(checks, Mapping):
        return False
    return any(
        isinstance(check, Mapping)
        and check.get("status") == "verified"
        and _valid_url(check.get("evidence_url")) == official_url
        and isinstance(check.get("quote"), str)
        and bool(check["quote"].strip())
        and _quote_is_grounded(record, official_url, check["quote"].strip())
        for check in checks.values()
    )


def _status(value: str) -> str:
    normalized = value.casefold()
    if normalized == "verified":
        return "verified"
    if normalized == "conflict":
        return "conflict"
    if normalized in {"unverified", "pending", "available", "third_party", "insufficient_evidence"}:
        return "pending"
    return "unknown"


def _status_strength(value: str) -> int:
    return {"unknown": 0, "pending": 1, "verified": 2, "conflict": 3}.get(_status(value), 0)


def _quote_is_grounded(record: Mapping[str, Any], url: str | None, quote: str) -> bool:
    if not quote:
        return False
    sources = _source_map(record)
    if url:
        return url in sources and quote in sources[url]
    return any(quote in text for text in sources.values())


def _source_map(record: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    raw_record = record.get("raw_record")
    if isinstance(raw_record, Mapping):
        url = _valid_url(raw_record.get("url") or record.get("url"))
        text = _nullable_text(raw_record.get("raw_content") or raw_record.get("content"))
        if url and text:
            result[url] = text
    url = _valid_url(record.get("url"))
    text = _nullable_text(record.get("raw_text") or record.get("evidence_detail") or record.get("evidence"))
    if url and text:
        result[url] = text
    official_url = _valid_url(record.get("official_source_url"))
    official_text = _nullable_text(record.get("official_evidence_detail") or record.get("official_source_text"))
    if official_url and official_text:
        result[official_url] = official_text
    if official_url and not official_text:
        evidence_detail = _nullable_text(record.get("evidence_detail"))
        if evidence_detail:
            result[official_url] = evidence_detail
    return result


def _source_texts(record: Mapping[str, Any]) -> list[str]:
    return list(_source_map(record).values())


def _primary_text(record: Mapping[str, Any]) -> tuple[str, str | None]:
    source_map = _source_map(record)
    official_url = _valid_url(record.get("official_source_url"))
    if official_url and official_url in source_map:
        return source_map[official_url], official_url
    url = _valid_url(record.get("url"))
    return (source_map.get(url, ""), url)


def _source_name(value: Any, url: str) -> str:
    name = _nullable_text(value)
    if name:
        return name
    return (urlparse(url).hostname or "来源").removeprefix("www.")


def _valid_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return value.strip()


def _nullable_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _first_text(*values: Any) -> str:
    return next((value.strip() for value in values if isinstance(value, str) and value.strip()), "")


def _summary(record: Mapping[str, Any]) -> str:
    text = _first_text(record.get("summary"), record.get("snippet"), record.get("evidence"))
    return re.sub(r"\s+", " ", text)[:320]


def _normalize_identity(value: str) -> str:
    value = value.casefold().strip()
    return re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE)


def _generic_label(key: str) -> str:
    labels = {
        "organization": "机构或公司",
        "role": "岗位",
        "graduate_cohort": "毕业届别",
        "majors": "专业",
        "major": "专业",
        "deadline": "截止日期",
        "application_url": "申请链接",
        "qualifications": "资格要求",
        "business_attributes": "业务内容",
        "region": "地区",
        "project_name": "项目名称",
        "purchasing_unit": "采购单位",
        "budget": "预算",
    }
    return labels.get(key, key.replace("_", " ").title())
