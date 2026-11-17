"""Deterministic procurement-intent classification and result ranking.

This module is deliberately self-contained: it performs no network or model
calls, and missing result metadata never implies a deadline or source status.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit


_CATEGORY_CUES: dict[str, tuple[str, ...]] = {
    "ai_training": (
        "人工智能教育培训",
        "人工智能拔尖创新人才培养",
        "生成式人工智能培训",
        "生成式ai培训",
        "人工智能编程培训",
        "人工智能培训",
        "ai培训",
        "aigc培训",
        "大模型培训",
        "人工智能课程",
        "大模型课程",
        "ai课程",
        "教师素养提升服务",
        "ai training",
        "artificial intelligence training",
        "人工智能培训服务",
    ),
    "ai_system": (
        "人工智能系统建设需求",
        "人工智能系统建设",
        "人工智能应用系统",
        "人工智能辅助服务",
        "人工智能辅助系统",
        "人工智能系统",
        "ai系统建设需求",
        "ai系统建设",
        "ai系统",
        "ai智能服务",
        "智能辅助系统",
        "人工智能平台",
        "大模型平台",
        "ai平台",
        "大模型部署",
        "ai部署",
        "ai软件",
        "ai software",
        "ai platform",
        "ai system",
        "人工智能解决方案",
        "ai解决方案",
        "智能体平台",
        "agent platform",
    ),
    "digital_transformation": (
        "数字化转型",
        "数字化改造",
        "数字化建设",
        "数字化升级",
        "业务数字化",
        "企业数字化",
        "信息化改造",
        "信息化建设",
        "智慧化改造",
        "智能化改造",
        "数字化 transformation",
        "digital transformation",
    ),
    "compute_hardware": (
        "gpu服务器",
        "ai服务器",
        "算力设备",
        "计算设备",
        "计算集群",
        "gpu集群",
        "高性能计算",
        "计算节点",
        "服务器",
        "存储阵列",
        "算力硬件",
        "gpu server",
        "gpu cluster",
        "compute hardware",
    ),
    "other": (
        "办公家具",
        "办公设备",
        "物业服务",
        "保洁服务",
        "食堂承包",
        "食堂服务",
        "绿化工程",
        "道路施工",
        "医疗设备",
        "课桌椅",
        "消防工程",
        "印刷服务",
        "会议服务",
        "校服采购",
        "餐饮服务",
        "车辆采购",
    ),
}

_CATEGORY_LABELS = {
    "ai_training": "AI 培训与课程服务",
    "ai_system": "AI 系统与软件平台",
    "digital_transformation": "数字化转型与改造",
    "compute_hardware": "算力与计算硬件",
    "other": "其他采购类别",
    "unclear": "未能明确的采购类别",
}

_OVERLAP_CUES = (
    "人工智能",
    "ai",
    "大模型",
    "数字化",
    "培训",
    "课程",
    "系统",
    "平台",
    "算力",
    "gpu",
    "服务器",
)

_STATUS_SCORES = {
    "matched": 0.96,
    "partial": 0.68,
    "keyword_only": 0.38,
    "unverified": 0.12,
    "not_matched": 0.04,
}


def employment_source_scope(record: Mapping[str, Any]) -> str:
    """Classify official employment evidence at company-page or job scope.

    An employer-owned campus home page is not evidence that a particular role
    is open. For a job-level classification require a role heading/title and a
    qualification section, or a role-detail URL with those same facts quoted.
    """
    official_url = record.get("official_source_url")
    if record.get("official_source_status") != "verified" or not isinstance(official_url, str) or not official_url:
        return "unknown"
    raw_body = record.get("official_evidence_detail")
    body = raw_body if isinstance(raw_body, str) else ""
    title = str(record.get("official_source_title") or "")
    fields = record.get("extracted_fields")
    role = fields.get("role") if isinstance(fields, Mapping) else None
    role_value = role.get("value", "") if isinstance(role, Mapping) else ""
    role_anchor = isinstance(role_value, str) and bool(role_value.strip()) and role_value.strip() in f"{title}\n{body}"
    role_heading = bool(re.search(r"(?:岗位名称|职位名称|招聘岗位|岗位类别|职位类别)\s*[:：]", body))
    requirement_section = bool(re.search(r"(?:专业要求|岗位要求|职位要求|任职资格|资格条件|所学专业)\s*[:：]", body))

    # If full official text is unavailable, a preserved literal field quote
    # plus a role-specific official title can still identify the page scope.
    checks = record.get("field_verification")
    verified_quotes: list[str] = []
    if isinstance(checks, Mapping):
        for check in checks.values():
            if (
                isinstance(check, Mapping)
                and check.get("status") == "verified"
                and check.get("evidence_url") == official_url
            ):
                quote = check.get("quote")
                if isinstance(quote, str):
                    verified_quotes.append(quote)
    grounded_text = "\n".join((body, *verified_quotes))
    requirement_section = requirement_section or bool(
        re.search(r"(?:专业要求|岗位要求|职位要求|任职资格|资格条件|所学专业)\s*[:：]", "\n".join(verified_quotes))
    )
    role_title = bool(re.search(
        r"(?:工程师|开发|算法|产品经理|分析师|研究员|技术员|专员|运营|设计师|工程岗位)",
        title,
    ))
    parsed = urlsplit(str(record.get("official_source_url") or ""))
    detail_path = bool(re.search(r"(?:/job/|/jobs/|/position/|/positions/|/requisition/|/vacancy/)", parsed.path, re.I))
    homepage_path = bool(re.search(r"(?:^|/)(?:campus(?:-recruiting)?|recruitment|careers?)(?:/|$)", parsed.path, re.I))
    role_anchor = role_anchor or bool(role_value and role_value.strip() in grounded_text)
    if requirement_section and (role_heading or detail_path and role_anchor or role_title and not homepage_path):
        return "specific_job_posting"
    explicit_homepage = any(term in f"{title}\n{body}" for term in (
        "招聘官网", "招聘门户", "人才招聘官网", "官方人才招聘", "校园招聘首页", "人才招聘网站",
    ))
    if not detail_path and (homepage_path or explicit_homepage):
        return "corporate_campus_page"
    return "unknown"


def classify_procurement(query: str, title: str, content: str) -> dict[str, Any]:
    """Classify a query and one candidate by procurement intent.

    The candidate title is the strongest indication of what is being bought;
    source content is used when the title is inconclusive and to confirm a
    title-level classification. Generic overlap (for example, ``AI`` in both
    an AI-training query and an AI-system notice) is not treated as a semantic
    match.
    """
    query_text = _normalize(query)
    title_text = _normalize(title)
    content_text = _normalize(content)

    query_category = _detect_category(query_text)
    title_category = _detect_category(title_text)
    content_category = _detect_category(content_text)

    candidate_category = title_category
    candidate_source = "标题"
    if candidate_category == "unclear":
        candidate_category = content_category
        candidate_source = "来源正文"

    if not query_text or query_category == "unclear":
        status = "unverified"
        reason = "查询没有表达清楚的采购类别或服务意图，暂时无法判断候选是否匹配。"
        mismatch_reason = "查询意图不明确。"
    elif not title_text and not content_text:
        status = "unverified"
        reason = "候选缺少标题和来源正文，无法核实采购意图。"
        mismatch_reason = "候选信息不足，无法确认类别。"
    elif candidate_category == "unclear":
        if _has_overlap(query_text, title_text, content_text):
            status = "keyword_only"
            reason = "候选与查询只共享宽泛关键词，现有标题和正文无法确认采购意图一致。"
            mismatch_reason = "仅有关键词重合，候选类别未能确认。"
        else:
            status = "unverified"
            reason = "候选信息不足以识别采购类别，无法核实与查询的关系。"
            mismatch_reason = "候选采购意图不明确。"
    elif candidate_category == query_category:
        if title_category == "unclear" or not content_text:
            status = "partial"
            reason = (
                f"查询和候选均指向{_CATEGORY_LABELS[query_category]}，"
                f"但目前主要依据{candidate_source}判断，正文佐证不足。"
            )
            mismatch_reason = "采购类别相符，但可用于核实的正文信息不足。"
        elif (
            query_category == "ai_system"
            and _query_explicitly_requests_system_build(query_text)
            and not _title_explicitly_names_a_system(title_text)
        ):
            if _text_explicitly_names_a_system(content_text):
                status = "partial"
                reason = "标题概括为 AI 服务，但来源正文明确描述了 AI 系统或平台建设内容；采购范围仍需结合完整公告判断。"
                mismatch_reason = "标题未直接写明系统建设；正文有明确系统建设内容，匹配结论为部分匹配。"
            else:
                status = "keyword_only"
                reason = "候选提到 AI 服务，但标题和来源正文没有明确显示系统、平台或软件建设，与查询的 AI 系统建设需求尚不能确认匹配。"
                mismatch_reason = "仅命中 AI 服务关键词；缺少系统、平台或软件建设的来源证据。"
        elif _has_competing_intent(query_text, title_text, content_text, query_category):
            status = "partial"
            reason = f"候选包含{_CATEGORY_LABELS[query_category]}内容，但也涉及其他采购意图。"
            mismatch_reason = "候选可能同时包含其他类别，需进一步确认采购主体内容。"
        else:
            status = "matched"
            reason = f"查询与候选的标题或来源正文均指向{_CATEGORY_LABELS[query_category]}。"
            mismatch_reason = None
    elif _has_overlap(query_text, title_text, content_text):
        status = "keyword_only"
        reason = (
            f"查询意图是{_CATEGORY_LABELS[query_category]}，候选主要属于"
            f"{_CATEGORY_LABELS[candidate_category]}；两者只有宽泛关键词重合。"
        )
        mismatch_reason = (
            f"采购意图不一致：查询为{_CATEGORY_LABELS[query_category]}，"
            f"候选为{_CATEGORY_LABELS[candidate_category]}。"
        )
    else:
        status = "not_matched"
        reason = (
            f"查询意图是{_CATEGORY_LABELS[query_category]}，候选主要属于"
            f"{_CATEGORY_LABELS[candidate_category]}，未发现足够的主题关联。"
        )
        mismatch_reason = (
            f"采购类别不同：查询为{_CATEGORY_LABELS[query_category]}，"
            f"候选为{_CATEGORY_LABELS[candidate_category]}。"
        )

    return {
        "query_category": query_category,
        "procurement_category": candidate_category,
        "match_status": status,
        "match_score": _STATUS_SCORES[status],
        "match_reason": reason,
        "mismatch_reason": mismatch_reason,
    }


def rank_and_explain_results(
    query: str, results: Sequence[Any]
) -> list[dict[str, Any]]:
    """Add semantic match and ranking explanations, then stably sort results.

    Explicit deadline and source-evidence metadata affects order. Absent fields
    have no effect: in particular, a missing deadline is not treated as either
    current or expired, and missing source metadata earns no reliability boost.
    """
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for index, raw_result in enumerate(results):
        result = _as_mapping(raw_result)
        title = _text(result.get("title"))
        evidence = _text(result.get("evidence"))
        snippet = _text(result.get("snippet"))
        content = " ".join(
            part
            for part in (evidence, snippet, _text(result.get("content")))
            if part
        )

        classification = classify_procurement(query, title, content)
        result.update(classification)
        # Keep the output name aligned with SearchResult while clarifying that
        # the score represents category/intent match, not factual confidence.
        result["procurement_category"] = classification["procurement_category"]
        result["match_status"] = classification["match_status"]
        result["match_score"] = classification["match_score"]
        result["match_reason"] = classification["match_reason"]
        result["mismatch_reason"] = classification["mismatch_reason"]

        ranking_score = classification["match_score"]
        ranking_factors: list[str] = []

        deadline_status = result.get("deadline_status")
        if deadline_status == "expired":
            ranking_score -= 0.35
            ranking_factors.append("已明确标记为过期，降低排序")
        elif deadline_status == "not_expired":
            ranking_score += 0.04
            ranking_factors.append("来源明确标记为未过期")
        elif deadline_status == "unverified":
            ranking_score -= 0.05
            ranking_factors.append("截止状态明确为未核实")

        evidence_status = result.get("evidence_status")
        if evidence_status == "available":
            ranking_score += 0.08
            ranking_factors.append("来源证据明确可用")
        elif evidence_status == "missing":
            ranking_score -= 0.25
            ranking_factors.append("来源证据明确缺失")

        official_status = result.get("official_source_status")
        if official_status == "verified":
            ranking_score += 0.05
            ranking_factors.append("官方来源状态已核实")
        elif official_status in {"unverified", "insufficient_evidence"}:
            ranking_score -= 0.03
            ranking_factors.append("来源可靠性尚未核实")
        elif official_status == "third_party":
            ranking_factors.append("来源标记为第三方")

        ranking_score = round(min(1.0, max(0.0, ranking_score)), 4)
        result["ranking_score"] = ranking_score
        result["ranking_reason"] = (
            "按采购意图匹配度排序。"
            if not ranking_factors
            else "按采购意图匹配度排序；" + "；".join(ranking_factors) + "。"
        )
        ranked.append((ranking_score, index, result))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked]


def _normalize(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # People commonly type the Latin abbreviation with a space before Chinese
    # nouns (for example, "AI 系统建设"). Canonicalize only that boundary.
    return re.sub(r"\bai\s+(?=[\u3400-\u9fff])", "ai", normalized)


def _query_explicitly_requests_system_build(query: str) -> bool:
    return any(
        cue in query
        for cue in ("系统建设", "平台建设", "软件系统", "系统开发", "system build", "system construction")
    )


def _title_explicitly_names_a_system(title: str) -> bool:
    return any(
        cue in title
        for cue in (
            "人工智能系统", "ai系统", "人工智能应用系统", "系统建设", "平台建设",
            "软件系统", "大模型平台", "智能体平台", "ai平台", "ai system",
        )
    )


def _text_explicitly_names_a_system(text: str) -> bool:
    return any(
        cue in text
        for cue in (
            "人工智能应用系统", "人工智能系统", "ai系统", "ai平台",
            "大模型平台", "系统建设", "平台建设", "智能体平台",
        )
    )


def _detect_category(text: str) -> str:
    if not text:
        return "unclear"

    scores: list[tuple[int, str]] = []
    for category, cues in _CATEGORY_CUES.items():
        score = max((_cue_score(text, cue) for cue in cues), default=0)
        if score:
            scores.append((score, category))
    if not scores:
        return "unclear"

    scores.sort(reverse=True)
    best_score, best_category = scores[0]
    if best_score < 5:
        return "unclear"
    if len(scores) > 1 and best_score - scores[1][0] < 2:
        return "unclear"
    return best_category


def _cue_score(text: str, cue: str) -> int:
    normalized_cue = _normalize(cue)
    if not normalized_cue:
        return 0
    if re.fullmatch(r"[a-z0-9 -]+", normalized_cue):
        pattern = re.escape(normalized_cue).replace(r"\ ", r"\s+")
        return 5 if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) else 0
    return 6 if normalized_cue in text else 0


def _has_overlap(query: str, title: str, content: str) -> bool:
    candidate = f"{title} {content}"
    return any(_cue_score(query, cue) and _cue_score(candidate, cue) for cue in _OVERLAP_CUES)


def _has_competing_intent(
    query: str, title: str, content: str, matched_category: str
) -> bool:
    candidate = f"{title} {content}"
    for category, cues in _CATEGORY_CUES.items():
        if category == matched_category:
            continue
        if any(_cue_score(query, cue) and _cue_score(candidate, cue) for cue in cues):
            return True
    return False


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
