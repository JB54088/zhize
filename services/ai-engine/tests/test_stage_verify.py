import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.stages.clean import clean_records
from app.stages.verify import _employment_facts, find_official_employment_evidence, verification_counts, verify_records


def test_a_gov_domain_or_third_party_quote_does_not_verify_fields():
    records = clean_records("2027届 电子信息工程", [{
        "title": "星海科技2027届软件工程师",
        "url": "https://careers.gov.cn/jobs/1",
        "raw_content": "招聘单位：星海科技\n2027届\n专业要求：电子信息工程",
    }], 5)

    checked = verify_records(records, required_conditions=[
        {"field": "graduate_cohort"}, {"field": "major"}, {"field": "open_status"},
    ])
    fields = checked[0]["field_verification"]

    assert fields["graduate_cohort"]["status"] == "unverified"
    assert fields["major"]["status"] == "unverified"
    assert fields["open_status"]["status"] == "missing"
    counts = verification_counts(checked)
    assert counts["verified"] == 0
    assert counts["unverified"] >= 3


def test_employment_page_requires_employer_owned_channel_and_job_context():
    async def scenario():
        records = clean_records("星海科技 2027届 软件工程师", [{
            "title": "星海科技2027届校园招聘软件工程师",
            "url": "https://jobs.third-party.example/posting/1",
            "raw_content": "招聘单位：星海科技\n岗位名称：软件工程师\n专业要求：电子信息工程",
        }], 5)
        generic = {
            "title": records[0]["title"],
            "url": "https://jobs.third-party.example/posting/1",
            "raw_content": "星海科技2027届校园招聘软件工程师\n招聘单位：星海科技\n专业要求：电子信息工程",
        }
        official = {
            "title": records[0]["title"],
            "url": "https://jobs.xinghai.example/campus/2027",
            "raw_content": (
                "星海科技官方招聘官网\n"
                "星海科技2027届校园招聘软件工程师\n"
                "招聘中\n"
                "任职资格：电子信息工程专业\n"
                "截止时间：2027-05-01\n"
                "网申入口：https://jobs.xinghai.example/apply/123"
            ),
        }
        calls = 0

        async def search(query, domains, limit):
            nonlocal calls
            calls += 1
            return [generic, official]

        none = await find_official_employment_evidence(records, lambda *_args: asyncio.sleep(0, result=[generic]), limit=3)
        assert not none
        found = await find_official_employment_evidence(records, search, limit=3)
        assert records[0]["url"] in found
        details = found[records[0]["url"]]
        assert details["official_source_url"] == official["url"]
        assert details["field_evidence"]["major"]["quote"] == "任职资格：电子信息工程专业"
        assert details["field_evidence"]["application_url"]["value"] == "https://jobs.xinghai.example/apply/123"
        assert "major" in details["verified_fields"]

    asyncio.run(scenario())


def test_official_facts_that_disagree_with_search_record_are_conflicts():
    record = {
        "url": "https://third.example/job",
        "extracted_fields": {"graduate_cohort": {"value": "2027届", "quote": "2027届"}},
        "field_sources": {"graduate_cohort": {"url": "https://third.example/job", "quote": "2027届"}},
    }
    verified = verify_records([record], {
        "https://third.example/job": {
            "official_source_url": "https://company.example/campus",
            "verified_fields": ["graduate_cohort"],
            "field_evidence": {"graduate_cohort": {"value": "2026届", "quote": "2026届招聘简章"}},
        }
    })[0]

    assert verified["field_verification"]["graduate_cohort"]["status"] == "conflict"
    assert verified["field_verification"]["graduate_cohort"]["value"] == "2026届"


def test_region_suffix_variants_agree_but_different_regions_remain_conflicts():
    records = [
        {
            "url": "https://third.example/match",
            "extracted_fields": {"region": {"value": "福建", "quote": "福建"}},
            "field_sources": {"region": {"url": "https://third.example/match", "quote": "福建"}},
        },
        {
            "url": "https://third.example/conflict",
            "extracted_fields": {"region": {"value": "福建", "quote": "福建"}},
            "field_sources": {"region": {"url": "https://third.example/conflict", "quote": "福建"}},
        },
    ]
    verified = verify_records(records, {
        "https://third.example/match": {
            "official_source_url": "https://official.example/fujian",
            "verified_fields": ["region"],
            "field_evidence": {"region": {"value": "福建省", "quote": "福建省"}},
        },
        "https://third.example/conflict": {
            "official_source_url": "https://official.example/jiangxi",
            "verified_fields": ["region"],
            "field_evidence": {"region": {"value": "江西省", "quote": "江西省"}},
        },
    })

    assert verified[0]["field_verification"]["region"]["status"] == "verified"
    assert verified[0]["field_verification"]["region"]["value"] == "福建省"
    assert verified[1]["field_verification"]["region"]["status"] == "conflict"
    assert verified[1]["field_verification"]["region"]["value"] == "江西省"


def test_official_deadline_independently_verifies_procurement_open_status():
    record = {
        "url": "https://third.example/project",
        "extracted_fields": {"region": {"value": "福建省", "quote": "福建省"}},
        "field_sources": {"region": {"url": "https://third.example/project", "quote": "福建省"}},
    }
    checked = verify_records([record], {
        "https://third.example/project": {
            "official_source_url": "https://procurement.gov.cn/notice/1",
            "verified_fields": ["region", "deadline"],
            "field_evidence": {
                "region": {"value": "福建省", "quote": "福建省"},
                "deadline": {"value": "2099-01-01", "quote": "投标截止时间：2099-01-01"},
            },
            "deadline_at": "2099-01-01",
            "deadline_status": "not_expired",
        }
    }, required_conditions=[
        {"field": "region"}, {"field": "open_status"},
    ])[0]

    assert checked["field_verification"]["region"]["status"] == "verified"
    assert checked["field_verification"]["open_status"] == {
        "value": "未截止",
        "status": "verified",
        "evidence_url": "https://procurement.gov.cn/notice/1",
        "quote": "投标截止时间：2099-01-01",
    }


def test_major_mention_outside_job_requirement_is_not_verified():
    facts = _employment_facts(
        "公司简介：电子信息工程领域经验丰富。\n岗位职责：负责系统开发。\n任职资格：计算机科学与技术专业。",
        requested_major="电子信息工程",
    )

    assert "major" not in facts


def test_major_fact_does_not_combine_values_from_different_requirement_quotes():
    facts = _employment_facts(
        "专业要求：通信工程\n岗位要求：人工智能、自动化",
    )

    assert facts["major"] == {
        "value": "通信工程",
        "quote": "专业要求：通信工程",
    }

    same_quote = _employment_facts("专业要求：通信工程、人工智能、自动化")
    assert same_quote["major"] == {
        "value": "通信工程、人工智能、自动化",
        "quote": "专业要求：通信工程、人工智能、自动化",
    }
