from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.result_cards as result_cards_module
from app.result_cards import build_result_cards, classify_category, _freshness_status
from app.schemas import ResultCardAttribute, SearchRequest
from app.stages.clean import clean_records


def _record(record_id: str, title: str, url: str, text: str, **values):
    return {
        "id": record_id,
        "title": title,
        "url": url,
        "source_name": "example.test",
        "snippet": text[:160],
        "raw_text": text,
        "raw_record": {"url": url, "raw_content": text},
        "decision": "needs_verification",
        "match_reason": "来源包含部分相关信息，关键条件仍需核验。",
        **values,
    }


def test_dining_cards_include_evidence_bound_attributes_without_procurement_fields():
    text = (
        "商户名称：榕城小馆\n地址：福建省福州市台江区五一中路\n"
        "人均：68元\n评分：4.6分\n近期评价：2026-09-20 有顾客评价菜品新鲜。"
    )
    card = build_result_cards(
        "找福州台江区200元以内、评价高的午餐餐厅",
        [_record("d1", "榕城小馆餐厅 - 大众点评", "https://reviews.example.test/store/1", text)],
    )[0]

    attrs = {attribute.key: attribute for attribute in card.attributes}
    assert card.category == "dining"
    assert card.source_url is None  # third-party listing is retained, not promoted to an official source
    assert "procurement" not in {attribute.key for attribute in card.attributes}
    assert attrs["price_per_person"].value == "68元/人"
    assert attrs["price_per_person"].evidence_quote == "人均：68元"
    assert attrs["price_per_person"].evidence_url == "https://reviews.example.test/store/1"
    assert attrs["price_per_person"].status == "pending"
    assert attrs["rating"].value == "4.6分"
    assert attrs["rating"].evidence_quote == "评分：4.6分"
    assert attrs["location"].value == "福建省福州市台江区五一中路"
    assert attrs["location"].evidence_quote == "地址：福建省福州市台江区五一中路"
    assert attrs["recent_reviews"].value.startswith("近期评价：")
    assert any(source.url == "https://reviews.example.test/store/1" for source in card.additional_sources)


def test_dining_pages_merge_only_when_source_names_the_same_restaurant():
    first = _record(
        "d1", "榕城小馆餐厅 - 点评", "https://reviews.example.test/store/1",
        "商户名称：榕城小馆\n地址：福州台江区中亭街\n人均：68元",
    )
    second = _record(
        "d2", "榕城小馆餐厅地址和菜单", "https://guide.example.test/food/88",
        "餐厅名称：榕城小馆\n地址：福州台江区中亭街",
    )
    cards = build_result_cards("福州午餐餐厅", [first, second])

    assert len(cards) == 1
    assert cards[0].id.startswith("entity:")
    assert {source.url for source in cards[0].additional_sources} == {
        first["url"], second["url"],
    }


def test_employment_cards_keep_distinct_roles_and_do_not_add_dining_attributes():
    def job(role: str, posting_id: str):
        url = f"https://careers.example.test/positions/{posting_id}"
        values = {
            "extracted_fields": {
                "organization": {"value": "星海科技", "quote": "招聘单位：星海科技"},
                "role": {"value": role, "quote": f"岗位名称：{role}"},
                "graduate_cohort": {"value": "2027届", "quote": "招收2027届毕业生"},
                "application_url": {"value": f"https://careers.example.test/apply/{posting_id}", "quote": f"投递链接：https://careers.example.test/apply/{posting_id}"},
            },
            "field_sources": {
                "organization": {"url": url, "quote": "招聘单位：星海科技"},
                "role": {"url": url, "quote": f"岗位名称：{role}"},
                "graduate_cohort": {"url": url, "quote": "招收2027届毕业生"},
                "application_url": {"url": url, "quote": f"投递链接：https://careers.example.test/apply/{posting_id}"},
            },
        }
        body = (
            f"招聘单位：星海科技\n岗位名称：{role}\n招收2027届毕业生\n"
            f"投递链接：https://careers.example.test/apply/{posting_id}"
        )
        return _record(posting_id, f"星海科技 2027 校招 {role}", url, body, **values)

    cards = build_result_cards(
        "我是2027届电子信息工程专业毕业生，找正在招聘的校招岗位",
        [job("硬件工程师", "HW-1"), job("嵌入式软件工程师", "SW-2")],
    )

    assert len(cards) == 2
    assert {card.category for card in cards} == {"employment"}
    assert all("rating" not in {attribute.key for attribute in card.attributes} for card in cards)
    assert all("graduate_cohort" not in {attribute.key for attribute in card.attributes} for card in cards)
    assert all(card.matching_status == "needs_verification" for card in cards)


def test_procurement_fields_are_not_mixed_into_other_categories_and_official_link_needs_exact_proof():
    query = "寻找福建省近期未截止的AI政府采购项目"
    official_url = "https://procurement.example.test/notices/42"
    body = "采购单位：福建省示例局\n项目预算：100万元\n采购内容：人工智能系统建设\n响应截止时间：2026-10-31"
    record = _record(
        "p1", "人工智能系统建设项目", "https://aggregator.example.test/project/42", body,
        official_source_url=official_url,
        official_source_status="verified",
        official_source_title="人工智能系统建设项目",
        official_evidence_detail=body,
        purchasing_unit="福建省示例局",
        budget="100万元",
        deadline_at="2026-10-31",
        field_verification={
            "purchasing_unit": {"value": "福建省示例局", "status": "verified", "evidence_url": official_url, "quote": "采购单位：福建省示例局"},
            "budget": {"value": "100万元", "status": "verified", "evidence_url": official_url, "quote": "项目预算：100万元"},
            "business_attributes": {"value": "人工智能系统建设", "status": "verified", "evidence_url": official_url, "quote": "采购内容：人工智能系统建设"},
            "deadline": {"value": "2026-10-31", "status": "verified", "evidence_url": official_url, "quote": "响应截止时间：2026-10-31"},
        },
    )
    card = build_result_cards(query, [record])[0]
    attributes = {attribute.key: attribute for attribute in card.attributes}

    assert card.category == "procurement"
    assert card.source_url == official_url
    assert card.source_name
    assert attributes["purchasing_unit"].status == "verified"
    assert attributes["purchasing_unit"].evidence_url == official_url
    assert attributes["need"].evidence_quote == "采购内容：人工智能系统建设"
    assert "role" not in attributes and "cohort" not in attributes

    unproven = _record(
        "p2", "采购公告", "https://aggregator.example.test/p2", body,
        official_source_url="https://fake.example.test/guessed-official",
        official_source_status="verified",
    )
    unproven_card = build_result_cards(query, [unproven])[0]
    assert unproven_card.source_url is None
    assert all(source.type != "official" or source.status != "verified" for source in unproven_card.additional_sources)


def test_different_procurement_projects_do_not_merge_even_on_the_same_source_page():
    url = "https://procurement.example.test/notices"
    first = _record(
        "p1", "AI数据分析平台项目", url, "采购内容：AI数据分析平台项目",
        official_source_url=url,
        official_source_status="verified",
        official_source_title="AI数据分析平台项目",
        official_evidence_detail="采购内容：AI数据分析平台项目",
        field_verification={
            "business_attributes": {"value": "AI数据分析平台项目", "status": "verified", "evidence_url": url, "quote": "采购内容：AI数据分析平台项目"},
        },
    )
    second = _record(
        "p2", "AI政务服务系统项目", url, "采购内容：AI政务服务系统项目",
        official_source_url=url,
        official_source_status="verified",
        official_source_title="AI政务服务系统项目",
        official_evidence_detail="采购内容：AI政务服务系统项目",
        field_verification={
            "business_attributes": {"value": "AI政务服务系统项目", "status": "verified", "evidence_url": url, "quote": "采购内容：AI政务服务系统项目"},
        },
    )

    cards = build_result_cards("福建省AI政府采购项目", [first, second])
    assert len(cards) == 2


def test_missing_or_mismatched_quotes_do_not_emit_attribute_values():
    official_url = "https://jobs.example.test/post/1"
    record = _record(
        "j1", "校招工程师", official_url, "岗位名称：硬件工程师",
        field_verification={
            "role": {"value": "编造的高级岗位", "status": "verified", "evidence_url": official_url, "quote": "岗位名称：未出现的职位"},
        },
    )
    card = build_result_cards("校招岗位", [record])[0]
    role = next(attribute for attribute in card.attributes if attribute.key == "role")

    assert role.value is None
    assert role.status == "unknown"
    assert role.evidence_url is None
    assert role.evidence_quote is None


def test_recruitment_field_values_must_be_supported_by_their_own_quote():
    url = "https://jobs.example.test/post/evidence"
    bad_fields = {
        "role": ("## 招聘公告与岗位信息", "## 招聘公告与岗位信息"),
        "salary": ("薪酬体系", "薪酬体系"),
        "major": ("电子信息工程、通信工程", "专业要求：电子信息工程"),
        "graduate_cohort": ("2027届", "毕业届别：2026届"),
        "deadline": ("2026-11-30", "报名截止：2026-10-31"),
    }
    raw_text = "\n".join(quote for _, quote in bad_fields.values())
    record = _record(
        "job-bad-fields", "星海科技招聘信息", url, raw_text,
        field_verification={
            key: {
                "value": value,
                "status": "verified",
                "evidence_url": url,
                "quote": quote,
            }
            for key, (value, quote) in bad_fields.items()
        },
    )
    query = "寻找校招岗位，要求薪资、专业符合2027届毕业生且查看截止日期"
    card = build_result_cards(query, [record])[0]
    attributes = {attribute.key: attribute for attribute in card.attributes}

    for key in ("role", "salary", "major", "cohort", "deadline"):
        attribute = attributes[key]
        assert attribute.value is None, key
        assert attribute.status == "unknown", key
        assert attribute.evidence_url is None, key
        assert attribute.evidence_quote is None, key


def test_supported_recruitment_values_accept_narrow_normalized_evidence():
    url = "https://jobs.example.test/post/valid-evidence"
    fields = {
        "role": ("嵌入式软件工程师（驱动方向）", "岗位名称：嵌入式软件工程师（驱动方向）"),
        "salary": ("20k-30k/月", "月薪：20k-30k/月"),
        "major": ("电子信息工程", "专业要求：电子信息工程"),
        "graduate_cohort": ("2027届", "招收2027年毕业生"),
        "deadline": ("2026-10-31", "报名截止：2026年10月31日"),
    }
    raw_text = "\n".join(quote for _, quote in fields.values())
    record = _record(
        "job-valid-fields", "星海科技硬件工程师", url, raw_text,
        field_verification={
            key: {"value": value, "status": "verified", "evidence_url": url, "quote": quote}
            for key, (value, quote) in fields.items()
        },
    )
    query = "寻找校招硬件工程师，要求薪资、电子信息工程专业、2027届毕业生和截止日期"
    card = build_result_cards(query, [record])[0]
    attributes = {attribute.key: attribute for attribute in card.attributes}

    assert attributes["role"].value == "嵌入式软件工程师（驱动方向）"
    assert attributes["salary"].value == "20k-30k/月"
    assert attributes["major"].value == "电子信息工程"
    assert attributes["cohort"].value == "2027届"
    assert attributes["deadline"].value == "2026-10-31"


def test_role_rejects_markdown_link_snippet_but_keeps_real_parenthetical_titles():
    url = "https://jobs.example.test/post/markdown-role"
    for number, bad_role in enumerate((
        "+ [招聘职位](javascript:void(0)",
        "**请您在简历上标明意向地点和职位名称",
    )):
        bad_record = _record(
            f"job-markdown-role-{number}", "企业招聘页面", url, bad_role,
            field_verification={
                "role": {
                    "value": bad_role,
                    "status": "pending",
                    "evidence_url": url,
                    "quote": bad_role,
                },
            },
        )
        card = build_result_cards("搜索校招岗位", [bad_record])[0]
        role = next(attribute for attribute in card.attributes if attribute.key == "role")

        assert role.value is None
        assert role.status == "unknown"
        assert role.evidence_url is None
        assert role.evidence_quote is None


def test_education_is_normalized_to_levels_present_in_quote():
    url = "https://jobs.example.test/post/education"
    value = "学历要求 **本科、硕士**"
    record = _record(
        "job-education-markdown", "星海科技校招", url, value,
        field_verification={
            "education": {
                "value": value,
                "status": "verified",
                "evidence_url": url,
                "quote": value,
            },
        },
    )

    card = build_result_cards("搜索校招岗位，学历要求本科或硕士", [record])[0]
    education = next(attribute for attribute in card.attributes if attribute.key == "education")

    assert education.value == "本科、硕士"
    assert education.evidence_quote == value


def test_company_rejects_unbalanced_or_markdown_value():
    url = "https://jobs.example.test/post/company"
    invalid_values = ("杭州高新区（滨江", "**星海科技有限公司**")
    for number, value in enumerate(invalid_values):
        record = _record(
            f"job-company-invalid-{number}", "企业校招", url, value,
            field_verification={
                "organization": {
                    "value": value,
                    "status": "verified",
                    "evidence_url": url,
                    "quote": value,
                },
            },
        )
        card = build_result_cards("搜索校招公司", [record])[0]
        company = next(attribute for attribute in card.attributes if attribute.key == "company")

        assert company.value is None
        assert company.status == "unknown"
        assert company.evidence_url is None
        assert company.evidence_quote is None


def test_major_alias_rejects_joined_values_when_quote_supports_only_one_major():
    url = "https://careers.example.test/jobs/major-list"
    quote = "专业要求：通信工程"
    record = _record(
        "job-major-alias", "中网华通 2027届校招", url, quote,
        extracted_fields={"majors": {"value": "通信工程、人工智能、自动化", "quote": quote}},
        field_sources={"majors": {"url": url, "quote": quote}},
    )

    card = build_result_cards("2027届校招专业要求", [record])[0]
    major = next(attribute for attribute in card.attributes if attribute.key == "major")

    assert major.value is None
    assert major.status == "unknown"
    assert major.evidence_url is None
    assert major.evidence_quote is None


def test_final_card_validation_catches_bad_value_after_aggregation(monkeypatch):
    url = "https://careers.example.test/jobs/aggregated-major"
    quote = "专业要求：通信工程"
    record = _record("job-merged-major", "中网华通校招", url, quote)
    bad_aggregate = ResultCardAttribute(
        key="major", label="专业", value="通信工程、人工智能、自动化",
        status="verified", evidence_url=url, evidence_quote=quote,
    )
    monkeypatch.setattr(result_cards_module, "_build_attributes", lambda *_args: [bad_aggregate])

    card = build_result_cards("校招专业要求", [record])[0]
    major = card.attributes[0]

    assert major.value is None
    assert major.status == "unknown"
    assert major.evidence_url is None
    assert major.evidence_quote is None


def test_procurement_need_rejects_generic_section_title_even_when_quote_is_grounded():
    url = "https://procurement.example.test/project/overview"
    record = _record(
        "proc-generic-need", "AI建设项目", url, "项目概况",
        field_verification={
            "business_attributes": {
                "value": "项目概况",
                "status": "verified",
                "evidence_url": url,
                "quote": "项目概况",
            },
        },
    )
    card = build_result_cards("寻找AI系统建设需求明确的政府采购项目", [record])[0]
    need = next(attribute for attribute in card.attributes if attribute.key == "need")

    assert need.value is None
    assert need.status == "unknown"
    assert need.evidence_url is None
    assert need.evidence_quote is None


def test_product_results_do_not_receive_recruitment_or_procurement_attributes():
    text = "品牌：海岚\n型号：HL-20\n参考价：1299元\n现货"
    card = build_result_cards(
        "推荐一款海岚HL-20商品，价格不超过1500元",
        [_record("product-1", "海岚 HL-20 商品", "https://shop.example.test/item/20", text)],
    )[0]
    attributes = {attribute.key: attribute for attribute in card.attributes}

    assert card.category == "product"
    assert attributes["brand"].value == "海岚"
    assert attributes["brand"].evidence_quote == "品牌：海岚"
    assert attributes["model"].value == "HL-20"
    assert attributes["price"].value == "1299元"
    assert not {"cohort", "major", "role", "purchasing_unit", "budget"} & set(attributes)


def test_freshness_uses_verified_business_dates_and_does_not_expire_restaurants_by_listing_age():
    today = date(2026, 10, 9)
    dining = _record("d", "餐厅", "https://example.test/d", "营业信息", published_at="2020-01-01")
    assert _freshness_status("dining", dining, today) == "unknown"

    job = _record(
        "j", "校招岗位", "https://example.test/j", "招聘公告\n截止日期：2027-03-01",
        published_at="2025-01-01",
        field_verification={
            "deadline": {"value": "2027-03-01", "status": "verified", "evidence_url": "https://example.test/j", "quote": "截止日期：2027-03-01"},
        },
    )
    assert _freshness_status("employment", job, today) == "fresh"

    unverified_deadline = _record(
        "p", "采购公告", "https://example.test/p", "采购公告\n发布日期：2026-10-01", published_at="2026-10-01",
        field_verification={
            "deadline": {"value": "2020-01-01", "status": "unverified", "evidence_url": "https://example.test/p", "quote": "日期未核实"},
            "published_at": {"value": "2026-10-01", "status": "verified", "evidence_url": "https://example.test/p", "quote": "发布日期：2026-10-01"},
        },
    )
    assert _freshness_status("procurement", unverified_deadline, today) == "fresh"


def test_normalization_retains_all_sources_before_candidate_or_display_paging():
    raw = [
        {"title": f"Distinct source {number}", "url": f"https://example.test/{number}", "raw_content": f"Unique evidence {number}."}
        for number in range(3)
    ]

    assert len(clean_records("unique evidence", raw, limit=1)) == 3
    assert SearchRequest(query="query").candidate_limit == 50
    assert SearchRequest(query="query", candidate_limit=100, limit=10).candidate_limit == 100
