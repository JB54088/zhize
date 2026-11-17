"""小采: query intent and provenance-preserving search adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol


class SearchAdapter(Protocol):
    name: str

    async def search(self, query: str, limit: int) -> Sequence[Mapping[str, Any]]: ...


class ProviderAdapter:
    """Adapter for the configured search provider; easy to replace or combine."""

    def __init__(self, provider: Any, name: str = "configured_search") -> None:
        self.provider = provider
        self.name = name

    async def search(self, query: str, limit: int) -> Sequence[Mapping[str, Any]]:
        return await self.provider.search(query, limit)


def parse_query_intent(query: str) -> dict[str, Any]:
    """Extract conservative hard/soft conditions without needing an LLM."""
    text = " ".join(query.split())
    lowered = text.casefold()
    if any(term in text for term in ("校招", "招聘", "毕业生", "求职", "实习")):
        domain = "employment"
    elif any(term in text for term in ("采购", "招标", "投标", "采购项目")):
        domain = "procurement"
    else:
        domain = "general"

    hard: list[dict[str, str]] = []
    soft: list[dict[str, str]] = []
    cohort = next((f"{year}届" for year in range(2020, 2041) if f"{year}届" in text), None)
    if cohort:
        hard.append({"field": "graduate_cohort", "label": "毕业届别", "value": cohort})
    major_match = None
    for major in (
        "电子信息工程", "电子科学与技术", "通信工程", "计算机科学与技术",
        "软件工程", "人工智能", "自动化", "信息工程", "网络工程",
    ):
        if major in text:
            major_match = major
            break
    if major_match:
        hard.append({"field": "major", "label": "专业", "value": major_match})
    province = next((name for name in (
        "北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
        "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
        "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海",
        "台湾", "内蒙古", "广西", "西藏", "宁夏", "新疆",
    ) if f"{name}省" in text or f"{name}市" in text or f"{name}自治区" in text), None)
    if province:
        hard.append({"field": "region", "label": "地区", "value": province})
    if domain == "employment" and any(term in text for term in ("正在招聘", "校招", "招聘中", "在招")):
        hard.append({"field": "open_status", "label": "招聘状态", "value": "正在招聘"})
        hard.append({"field": "application_url", "label": "官方投递链接", "value": "官方投递链接"})
    elif domain == "procurement" and any(term in text for term in ("未截止", "仍在招标", "仍在采购", "正在招标")):
        hard.append({"field": "open_status", "label": "项目状态", "value": "未截止"})
    if any(term in text for term in ("近期", "最近", "近期仍在")):
        soft.append({"field": "recency", "label": "时间偏好", "value": "近期"})
    if any(term in text for term in ("符合条件", "优先", "最好")):
        soft.append({"field": "fit", "label": "匹配偏好", "value": "优先满足明确条件"})
    return {
        "query": text,
        "domain": domain,
        "hard_conditions": hard,
        "soft_conditions": soft,
        "language": "zh" if any("\u3400" <= ch <= "\u9fff" for ch in text) else "en",
        "normalized_query": lowered,
    }


async def collect_records(
    query: str, adapters: Sequence[SearchAdapter], limit: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run configured search adapters concurrently and retain raw records."""
    if not adapters:
        raise ValueError("at least one search adapter is required")
    tasks = [adapter.search(query, limit) for adapter in adapters]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    captured_at = datetime.now(timezone.utc).isoformat()
    for adapter, response in zip(adapters, responses):
        if isinstance(response, BaseException):
            errors.append(f"{adapter.name}: {type(response).__name__}")
            continue
        for index, raw in enumerate(response):
            if not isinstance(raw, Mapping):
                continue
            records.append({
                **dict(raw),
                "_provenance": {
                    "provider": adapter.name,
                    "provider_record_index": index,
                    "captured_at": captured_at,
                },
            })
    if not records and errors:
        raise RuntimeError("all configured search adapters failed: " + "; ".join(errors))
    return records, errors
