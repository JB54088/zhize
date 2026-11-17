from datetime import datetime, timezone

from services.api.app.result_cards import legacy_results_to_cards, rank_cards
from services.api.app.schemas import ResultCard, SearchResult


def legacy_result(**updates: object) -> SearchResult:
    payload: dict[str, object] = {
        "id": "candidate-1",
        "title": "结果候选",
        "url": "https://third.example/item",
        "source_name": "第三方线索",
        "snippet": "有待核实的信息线索。",
        "evidence": "",
        "match_reason": "命中用户查询条件。",
        "verdict": "unverified",
        "confidence": 0.2,
    }
    payload.update(updates)
    return SearchResult.model_validate(payload)


def test_dining_attributes_are_domain_specific_and_old_page_is_not_removed() -> None:
    result = legacy_result(
        title="台江老街餐厅午餐",
        summary="餐厅提供午餐，地址与评价见来源。",
        published_at="2025-01-01T00:00:00Z",
        purchasing_unit="错误字段不应显示",
        budget="错误采购预算",
        key_conditions=[
            {"label": "预算", "value": "人均 80 元", "status": "verified", "evidence_url": "https://official.example/menu", "quote": "人均 80 元"},
            {"label": "评分", "value": "4.8 分", "status": "verified", "evidence_url": "https://official.example/menu", "quote": "评分 4.8 分"},
            {"label": "地址", "value": "福州市台江区", "status": "verified", "evidence_url": "https://official.example/menu", "quote": "地址：福州市台江区"},
            {"label": "响应截止时间", "value": "2026-10-30", "status": "verified", "evidence_url": "https://official.example/menu", "quote": "截止时间 2026-10-30"},
            {"label": "招聘资格", "value": "2027届", "status": "verified", "evidence_url": "https://official.example/menu", "quote": "2027届"},
        ],
    )
    card = legacy_results_to_cards(
        [result], query="福州台江区200元以内、评价高的午餐餐厅",
        now=datetime(2026, 10, 9, tzinfo=timezone.utc),
    )[0]

    assert card.category == "dining"
    assert {item.key for item in card.attributes} == {"per_capita", "rating", "location"}
    assert all(item.key not in {"purchasing_unit", "budget", "response_deadline", "application_deadline", "qualifications", "graduate_cohort", "major"} for item in card.attributes)
    assert card.freshness_status == "stale"
    assert card.id == "candidate-1"  # age affects rank/status, not candidate retention


def test_active_job_older_than_three_months_remains_fresh_and_unverified_fields_stay_pending() -> None:
    result = legacy_result(
        title="2027届电子信息工程校招岗位",
        summary="正在招聘的岗位详情。",
        published_at="2026-01-01T00:00:00Z",
        deadline_status="not_expired",
        deadline_at="2027-03-01T00:00:00Z",
        verified_fields=["graduate_cohort"],
        unverified_fields=["major"],
        key_conditions=[
            {"label": "毕业届别", "value": "2027届", "status": "verified", "evidence_url": "https://jobs.example/role", "quote": "招收2027届毕业生"},
            {"label": "专业要求", "value": "电子信息工程待核实", "status": "unverified", "evidence_url": None, "quote": None},
            {"label": "岗位", "value": "嵌入式软件工程师", "status": "verified", "evidence_url": "https://jobs.example/role", "quote": "嵌入式软件工程师"},
        ],
        evidence_sources=[
            {"title": "招聘官网", "url": "https://jobs.example/role", "source_type": "official", "status": "verified"}
        ],
    )
    card = legacy_results_to_cards(
        [result], query="2027届电子信息工程专业校园招聘",
        now=datetime(2026, 10, 9, tzinfo=timezone.utc),
    )[0]

    assert card.category == "employment"
    assert card.freshness_status == "fresh"
    assert card.source_url == "https://jobs.example/role"
    assert {attribute.key for attribute in card.attributes} == {"graduate_cohort", "major", "role", "application_deadline"}
    assert next(attribute for attribute in card.attributes if attribute.key == "major").status == "pending"
    assert all(attribute.key not in {"purchasing_unit", "budget"} for attribute in card.attributes)


def test_procurement_uses_procurement_fields_and_conflicts_are_not_hidden() -> None:
    a = legacy_result(
        id="project-a",
        title="AI系统建设采购项目",
        summary="政府采购项目公告。",
        procurement_category="ai_system",
        purchasing_unit="福建某采购单位",
        budget="100万元",
        deadline_at="2026-11-01T00:00:00Z",
        deadline_status="not_expired",
        verified_fields=["purchasing_unit", "budget"],
        unverified_fields=["deadline"],
    )
    b = a.model_copy(update={"id": "project-b", "title": "另一个AI采购项目"})
    cards = legacy_results_to_cards(
        [a, b], query="福建省近期未截止的AI政府采购项目",
        now=datetime(2026, 10, 9, tzinfo=timezone.utc),
    )

    assert [card.id for card in cards] == ["project-a", "project-b"]
    assert all(card.category == "procurement" for card in cards)
    assert all({"purchasing_unit", "budget", "response_deadline"} <= {item.key for item in card.attributes} for card in cards)
    assert all(card.freshness_status == "fresh" for card in cards)
    assert all(next(item for item in card.attributes if item.key == "response_deadline").status == "pending" for card in cards)


def test_third_party_candidate_never_becomes_primary_source_link() -> None:
    result = legacy_result(
        evidence_sources=[
            {"title": "转载页", "url": "https://third.example/copy", "source_type": "third_party", "status": "verified"}
        ],
        official_source_url="https://official.example/missing",
        official_source_status="insufficient_evidence",
    )
    card = legacy_results_to_cards([result], query="普通查询")[0]

    assert card.source_url is None
    assert card.source_name is None
    assert {source.url for source in card.additional_sources} == {
        "https://third.example/copy", "https://official.example/missing", "https://third.example/item"
    }


def test_rank_order_and_required_card_schema() -> None:
    cards = [
        ResultCard(
            id="stale", title="Stale", category="other", summary="", rank=1,
            recommendation_reason="", attributes=[], source_url=None, source_name=None,
            additional_sources=[], published_at=None, last_verified_at=None,
            verification_status="unknown", matching_status="needs_verification", freshness_status="stale",
        ),
        ResultCard(
            id="recommended", title="Recommended", category="other", summary="", rank=2,
            recommendation_reason="", attributes=[], source_url=None, source_name=None,
            additional_sources=[], published_at=None, last_verified_at=None,
            verification_status="verified", matching_status="recommended", freshness_status="fresh",
        ),
    ]

    ranked = rank_cards(cards)

    assert [(card.id, card.rank) for card in ranked] == [("recommended", 1), ("stale", 2)]
