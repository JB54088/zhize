from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.stages.clean import canonicalize_url, clean_records


def test_canonical_url_removes_tracking_and_fragments():
    assert canonicalize_url("HTTPS://www.example.com/jobs/?utm_source=x&z=1#section") == "https://example.com/jobs?z=1"


def test_normalizes_fields_and_semantically_deduplicates_with_raw_mapping():
    raw = [
        {
            "title": "星海科技2027届校园招聘-软件工程师",
            "url": "https://jobs.example.com/posting/1?utm_source=tavily",
            "content": "招聘单位：星海科技\n专业要求：电子信息工程\n截止时间：2027-05-01\n投递链接：https://jobs.example.com/apply/1",
            "raw_content": "招聘单位：星海科技\n专业要求：电子信息工程\n截止时间：2027-05-01\n投递链接：https://jobs.example.com/apply/1",
            "_provenance": {"provider": "test", "provider_record_index": 0},
        },
        {
            "title": "星海科技 2027 届校园招聘 — 软件工程师",
            "url": "https://www.jobs.example.com/posting/1?utm_medium=organic#top",
            "content": "duplicate page",
            "raw_content": "duplicate page with less text",
            "_provenance": {"provider": "test", "provider_record_index": 1},
        },
    ]

    records = clean_records("2027届电子信息工程软件工程师招聘", raw, 10)

    assert len(records) == 1
    item = records[0]
    assert item["canonical_url"] == "https://jobs.example.com/posting/1"
    assert item["source_provenance"]["provider"] == "test"
    assert item["extracted_fields"]["graduate_cohort"]["value"] == "2027届"
    assert "电子信息工程" in item["extracted_fields"]["majors"]["value"]
    assert item["extracted_fields"]["deadline"]["value"] == "2027-05-01"
    assert item["extracted_fields"]["application_url"]["value"] == "https://jobs.example.com/apply/1"
    assert item["field_sources"]["deadline"]["quote"] in item["raw_text"]


def test_region_uses_title_before_unrelated_navigation_regions_in_body():
    records = clean_records("福建省人工智能系统建设采购", [{
        "title": "福建省人工智能系统建设项目采购公告",
        "url": "https://procurement.example/project/1",
        "raw_content": "北京 上海 浙江\n网站导航：广东、江苏\n采购内容：人工智能系统建设。",
    }], 5)

    assert records[0]["extracted_fields"]["region"] == {
        "value": "福建省", "quote": "福建省",
    }


def test_region_can_be_extracted_from_an_explicit_project_location_label():
    records = clean_records("人工智能系统建设采购", [{
        "title": "人工智能系统建设项目采购公告",
        "url": "https://procurement.example/project/2",
        "raw_content": "北京 上海 浙江\n项目地点：福建省福州市。\n采购内容：人工智能系统建设。",
    }], 5)

    assert records[0]["extracted_fields"]["region"] == {
        "value": "福建省", "quote": "福建省",
    }
