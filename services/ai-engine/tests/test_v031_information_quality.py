"""Small, manually labelled V0.3.1 regressions; not a population benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.verify import VerifyAgent
from app.stages.collect import parse_query_intent
from app.stages.judge import judge_records
from app.stages.clean import clean_records


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "v031_quality_gold.json"
GOLD = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _employment_record(case: dict) -> dict:
    official_url = case["official_source_url"]
    field_verification = {
        field_name: {
            "value": field["value"],
            "status": field["status"],
            "evidence_url": official_url if field["status"] == "verified" else None,
            "quote": field["quote"],
        }
        for field_name, field in case["fields"].items()
    }
    return {
        "id": case["id"],
        "title": case["title"],
        "url": f"https://search.example/{case['id']}",
        "source_name": "search.example",
        "snippet": "校招候选来源",
        "evidence": "校招候选来源",
        "raw_text": case["official_evidence_detail"],
        "extracted_fields": {"role": {"value": "软件工程师", "quote": "软件工程师"}},
        "field_verification": field_verification,
        "official_source_url": official_url,
        "official_source_status": "verified",
        "official_source_title": case["official_source_title"],
        "official_evidence_detail": case["official_evidence_detail"],
        "confidence": 0.5,
    }


def _employment_intent(query: str) -> dict:
    intent = parse_query_intent(query)
    intent["hard_conditions"].append({
        "field": "deadline",
        "label": "截止日期",
        "value": "2099-05-01",
    })
    return intent


def test_v031_manual_gold_set_employment_and_procurement_quality():
    """Report exact-decision accuracy on the hand-labelled regression sample."""
    employment_records = [_employment_record(case) for case in GOLD["employment_cases"]]
    employment_intent = _employment_intent(GOLD["employment_query"])

    async def run_verification():
        result = await VerifyAgent().run(
            employment_records,
            required_conditions=employment_intent["hard_conditions"],
            llm_client=None,
        )
        return result.output["records"]

    import asyncio
    verified = asyncio.run(run_verification())
    employment_results = judge_records(GOLD["employment_query"], employment_intent, verified)
    by_id = {record["id"]: record for record in employment_results}
    correct = 0
    employment_predictions: dict[str, str] = {}
    for case in GOLD["employment_cases"]:
        result = by_id[case["id"]]
        assert result["decision"] == case["expected_decision"], case["id"]
        employment_predictions[case["id"]] = result["decision"]
        states = {check["label"]: check["status"] for check in result["key_conditions"]}
        field_to_label = {
            "graduate_cohort": "毕业届别",
            "major": "专业",
            "open_status": "招聘状态",
            "deadline": "截止日期",
            "application_url": "官方投递链接",
        }
        for field_name, expected_state in case["expected_field_states"].items():
            assert states[field_to_label[field_name]] == expected_state, (case["id"], field_name)
        correct += 1

    procurement = GOLD["procurement_case"]
    procurement_intent = parse_query_intent(procurement["query"])
    procurement_result = judge_records(
        procurement["query"], procurement_intent, [procurement["record"]]
    )[0]
    assert procurement_result["decision"] == procurement["expected_decision"]
    procurement_states = {check["label"]: check["status"] for check in procurement_result["key_conditions"]}
    assert procurement_states["地区"] == procurement["expected_field_states"]["region"]
    assert procurement_states["项目状态"] == procurement["expected_field_states"]["open_status"]
    assert procurement_states["官方公告业务类别"] == procurement["expected_field_states"]["official_category"]

    positive_cases = [case for case in GOLD["employment_cases"] if case["expected_decision"] == "recommended"]
    true_positives = sum(
        1 for case in positive_cases if employment_predictions[case["id"]] == "recommended"
    )
    false_positives = sum(
        1 for case in GOLD["employment_cases"]
        if case["expected_decision"] != "recommended" and employment_predictions[case["id"]] == "recommended"
    )
    false_negatives = sum(
        1 for case in positive_cases if employment_predictions[case["id"]] != "recommended"
    )
    assert true_positives == 1 and false_positives == 0 and false_negatives == 0

    # Five exact labels matched. The sample intentionally has only one positive
    # employment recommendation, so this is a guard against regressions, not a
    # meaningful estimate of live-search accuracy or recall.
    total = len(GOLD["employment_cases"]) + 1
    correct += 1
    assert correct / total == 1.0


def test_v031_corporate_campus_page_cannot_verify_a_specific_open_job():
    query = GOLD["employment_query"]
    intent = _employment_intent(query)
    official_url = "https://jobs.xinghai.example/campus"
    candidate = {
        "id": "generic-campus-page",
        "title": "星海科技2027届软件工程师校招",
        "url": "https://search.example/generic-campus-page",
        "source_name": "search.example",
        "snippet": "星海科技2027届校园招聘",
        "evidence": "星海科技2027届校园招聘",
        "raw_text": "星海科技官方招聘官网\n星海科技2027届校园招聘现已开启，欢迎毕业生关注。",
        "extracted_fields": {"role": {"value": "软件工程师", "quote": "软件工程师"}},
        "field_verification": {
            "graduate_cohort": {"value": "2027届", "status": "verified", "evidence_url": official_url, "quote": "2027届校园招聘"},
            "major": {"value": "电子信息工程", "status": "verified", "evidence_url": official_url, "quote": "专业要求：电子信息工程专业"},
            "open_status": {"value": "招聘中", "status": "verified", "evidence_url": official_url, "quote": "招聘中"},
            "deadline": {"value": "2099-05-01", "status": "verified", "evidence_url": official_url, "quote": "截止时间：2099-05-01"},
            "application_url": {"value": "https://jobs.xinghai.example/apply", "status": "verified", "evidence_url": official_url, "quote": "网申入口：https://jobs.xinghai.example/apply"},
        },
        "official_source_url": official_url,
        "official_source_status": "verified",
        "official_source_title": "星海科技2027届软件工程师校园招聘",
        "official_evidence_detail": "星海科技官方招聘官网\n星海科技2027届软件工程师校园招聘现已开启，欢迎毕业生关注。\n专业要求：电子信息工程等相关专业。",
    }

    import asyncio
    execution = asyncio.run(VerifyAgent().run(
        [candidate], required_conditions=intent["hard_conditions"], llm_client=None
    ))
    checked = execution.output["records"][0]
    assert checked["employment_source_scope"] == "corporate_campus_page"
    for field_name in ("graduate_cohort", "major", "open_status", "deadline", "application_url"):
        assert checked["field_verification"][field_name]["status"] == "unverified"

    result = judge_records(query, intent, [checked])[0]
    assert result["decision"] == "needs_verification"
    assert result["application_url"] is None
    assert result["employment_source_scope"] == "corporate_campus_page"


def test_v031_same_title_distinct_postings_survive_but_copied_pages_collapse():
    common = (
        "星海科技2027届软件工程师岗位，负责软件系统研发与测试。"
        "岗位要求：本科及以上学历，熟悉软件工程、计算机基础和团队协作。"
        "本岗位属于星海科技校园招聘计划，候选人须按官方页面要求提交申请材料。"
        "岗位说明：工作内容包括需求分析、编码实现、联调和版本维护，需根据项目计划完成交付。"
    )
    distinct = clean_records(
        "星海科技软件工程师校招",
        [
            {"title": "星海科技2027届软件工程师校招", "url": "https://jobs.xinghai.example/jobs/REQ-BJ-101", "raw_content": common + "\n岗位编号：REQ-BJ-101\n工作地点：北京市。"},
            {"title": "星海科技2027届软件工程师校招", "url": "https://jobs.xinghai.example/jobs/REQ-SH-202", "raw_content": common + "\n岗位编号：REQ-SH-202\n工作地点：上海市。"},
        ],
        10,
    )
    assert len(distinct) == 2

    copied = clean_records(
        "星海科技软件工程师校招",
        [
            {"title": "星海科技2027届软件工程师校招", "url": "https://jobs.xinghai.example/jobs/REQ-101", "raw_content": common + "\n岗位编号：REQ-101"},
            {"title": "星海科技2027届软件工程师校招", "url": "https://mirror.example/jobs/REQ-101", "raw_content": common + "\n岗位编号：REQ-101"},
        ],
        10,
    )
    assert len(copied) == 1
