from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.stages.collect import parse_query_intent
from app.stages.judge import judge_records


QUERY = "我是2027届毕业生，电子信息工程专业，帮我寻找符合条件且正在招聘的校招公司。"


def test_unknown_hard_conditions_and_missing_official_link_never_recommend():
    intent = parse_query_intent(QUERY)
    candidate = {
        "id": "candidate-1", "title": "星海科技软件工程师校招",
        "url": "https://third.example/job", "source_name": "third.example",
        "snippet": "2027届校招，电子信息工程相关专业。",
        "evidence": "2027届校招，电子信息工程相关专业。",
        "raw_text": "招聘中。2027届校招，电子信息工程相关专业。",
        "raw_record": {"content": "2027届校招，电子信息工程相关专业。"},
        "extracted_fields": {
            "graduate_cohort": {"value": "2027届", "quote": "2027届"},
            "majors": {"value": "电子信息工程", "quote": "电子信息工程"},
            "application_url": {"value": "https://third.example/apply/1", "quote": "投递链接：https://third.example/apply/1"},
        },
        "field_sources": {},
        "field_verification": {
            "graduate_cohort": {"value": "2027届", "status": "verified", "evidence_url": "https://company.example/job", "quote": "2027届"},
            "major": {"value": "电子信息工程", "status": "verified", "evidence_url": "https://company.example/job", "quote": "专业要求：电子信息工程"},
            "open_status": {"value": "招聘中", "status": "verified", "evidence_url": "https://company.example/job", "quote": "招聘中"},
            "application_url": {"value": "https://third.example/apply/1", "status": "unverified", "evidence_url": "https://third.example/job", "quote": "投递链接：https://third.example/apply/1"},
        },
        "official_source_url": "https://company.example/job",
        "official_source_status": "verified",
        "decision": "needs_verification", "match_score": 0.4, "confidence": 0.8,
    }

    result = judge_records(QUERY, intent, [candidate])[0]

    assert result["decision"] == "needs_verification"
    assert result["application_url"] is None
    assert next(item for item in result["key_conditions"] if item["label"] == "官方投递链接")["status"] == "unverified"
    assert result["match_status"] == "partial"
    assert "采购" not in result["match_reason"]
    assert "待核实条件" in result["mismatch_reason"]


def test_verified_conflict_is_preserved_and_blocks_recommendation():
    intent = parse_query_intent(QUERY)
    candidate = {
        "id": "candidate-2", "title": "星海科技软件工程师校招",
        "url": "https://third.example/job", "source_name": "third.example",
        "snippet": "", "evidence": "", "raw_text": "", "raw_record": {},
        "extracted_fields": {}, "field_sources": {},
        "field_verification": {
            "graduate_cohort": {"value": "2026届", "status": "conflict", "evidence_url": "https://company.example/job", "quote": "2026届招聘简章"},
            "major": {"value": "电子信息工程", "status": "verified", "evidence_url": "https://company.example/job", "quote": "专业要求：电子信息工程"},
            "open_status": {"value": "招聘中", "status": "verified", "evidence_url": "https://company.example/job", "quote": "招聘中"},
            "application_url": {"value": "https://company.example/apply/1", "status": "verified", "evidence_url": "https://company.example/job", "quote": "投递链接：https://company.example/apply/1"},
        },
        "official_source_url": "https://company.example/job", "official_source_status": "verified",
        "match_score": 0.5, "confidence": 0.9,
    }

    result = judge_records(QUERY, intent, [candidate])[0]

    assert result["decision"] == "needs_verification"
    assert next(item for item in result["key_conditions"] if item["label"] == "毕业届别")["status"] == "conflict"
    assert result["application_url"] == "https://company.example/apply/1"


def test_procurement_region_and_open_deadline_can_match_without_recruitment_specific_rules():
    query = "寻找福建省近期明确 AI 系统建设需求且未截止的政府采购项目"
    intent = parse_query_intent(query)
    candidate = {
        "id": "procurement-1", "title": "福建省人工智能系统建设项目采购公告",
        "url": "https://third.example/project", "source_name": "third.example",
        "snippet": "采购AI系统建设。", "evidence": "采购AI系统建设。",
        "raw_text": "采购内容：人工智能系统建设。", "raw_record": {},
        "extracted_fields": {"region": {"value": "福建省", "quote": "福建省"}},
        "field_sources": {},
        "field_verification": {
            "region": {"value": "福建省", "status": "verified", "evidence_url": "https://official.gov.cn/notice", "quote": "福建省"},
            "open_status": {"value": "未截止", "status": "verified", "evidence_url": "https://official.gov.cn/notice", "quote": "投标截止时间：2099-01-01"},
            "project_name": {"value": "福建省人工智能系统建设项目", "status": "verified", "evidence_url": "https://official.gov.cn/notice", "quote": "项目名称：福建省人工智能系统建设项目"},
            "evidence": {"value": "采购内容：人工智能系统建设", "status": "verified", "evidence_url": "https://official.gov.cn/notice", "quote": "采购内容：人工智能系统建设"},
        },
        "official_source_url": "https://official.gov.cn/notice",
        "official_source_title": "福建省人工智能系统建设项目采购公告",
        "official_source_status": "verified", "match_score": 0.96, "confidence": 0.9,
    }

    result = judge_records(query, intent, [candidate])[0]

    assert result["decision"] == "recommended"
    assert {item["status"] for item in result["key_conditions"]} == {"verified"}


def test_verified_expired_procurement_fails_explicit_not_expired_condition():
    query = "寻找福建省近期明确 AI 系统建设需求且未截止的政府采购项目"
    intent = parse_query_intent(query)
    candidate = {
        "title": "福建省人工智能系统建设项目采购公告",
        "url": "https://third.example/project",
        "snippet": "",
        "evidence": "",
        "raw_text": "",
        "field_verification": {
            "region": {"value": "福建省", "status": "verified", "evidence_url": "https://official.gov.cn/notice", "quote": "福建省"},
            "deadline": {"value": "2020-01-01", "status": "verified", "evidence_url": "https://official.gov.cn/notice", "quote": "投标截止时间：2020-01-01"},
            "evidence": {"value": "采购内容：人工智能系统建设", "status": "verified", "evidence_url": "https://official.gov.cn/notice", "quote": "采购内容：人工智能系统建设"},
        },
        "official_source_url": "https://official.gov.cn/notice",
        "official_source_title": "福建省人工智能系统建设项目采购公告",
        "official_source_status": "verified",
        "deadline_status": "expired",
        "confidence": 0.9,
    }

    result = judge_records(query, intent, [candidate])[0]

    assert result["decision"] == "not_matched"
    open_check = next(item for item in result["key_conditions"] if item["label"] == "项目状态")
    assert open_check["status"] == "verified"
    assert open_check["value"] == "已截止"


def test_third_party_ai_claim_cannot_verify_official_procurement_category():
    query = "寻找福建省近期明确 AI 系统建设需求且未截止的政府采购项目"
    intent = parse_query_intent(query)
    candidate = {
        "title": "福建省综合服务采购公告",
        "url": "https://third.example/project",
        "snippet": "AI 系统建设项目采购。",
        "evidence": "采购内容包括人工智能系统建设。",
        "raw_text": "采购内容：人工智能系统建设。",
        "official_source_title": "福建省综合服务采购公告",
        "official_source_url": "https://official.gov.cn/notice",
        "official_source_status": "verified",
        "field_verification": {
            "region": {"value": "福建省", "status": "verified", "evidence_url": "https://official.gov.cn/notice", "quote": "福建省"},
            "open_status": {"value": "未截止", "status": "verified", "evidence_url": "https://official.gov.cn/notice", "quote": "投标截止时间：2099-01-01"},
            "project_name": {"value": "福建省综合服务项目", "status": "verified", "evidence_url": "https://official.gov.cn/notice", "quote": "项目名称：福建省综合服务项目"},
        },
        "confidence": 0.9,
    }

    result = judge_records(query, intent, [candidate])[0]

    assert result["decision"] == "needs_verification"
    category_check = next(item for item in result["key_conditions"] if item["label"] == "官方公告业务类别")
    assert category_check["status"] == "missing"
    assert category_check["evidence_url"] is None
    assert result["procurement_category"] == "unclear"
