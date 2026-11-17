from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.source_records import MAX_RAW_TEXT_CHARS, MAX_SOURCE_RECORDS, build_source_records


def test_source_archive_keeps_raw_text_and_safe_acquisition_metadata():
    sources = build_source_records([{
        "title": "公开采购公告",
        "url": "https://procurement.example.test/notice/1",
        "source_name": "采购公告网",
        "raw_content": "项目正文",
        "published_date": "2026-10-01",
        "score": 0.89,
        "api_key": "must-not-be-copied",
        "_provenance": {
            "provider": "tavily",
            "provider_record_index": 4,
            "captured_at": "2026-10-09T08:00:00Z",
            "search_query_index": 2,
            "search_query": "福建省近期未截止AI政府采购项目",
            "authorization": "must-not-be-copied",
        },
    }])

    assert len(sources) == 1
    source = sources[0]
    assert source.url == "https://procurement.example.test/notice/1"
    assert source.raw_text == "项目正文"
    assert source.provider == "tavily"
    assert source.search_query_index == 2
    assert source.search_query == "福建省近期未截止AI政府采购项目"
    assert source.metadata["score"] == 0.89
    assert source.metadata["provider_record_index"] == 4
    assert "api_key" not in source.model_dump()
    assert "authorization" not in source.metadata


def test_source_archive_bounds_text_and_total_records():
    source = {
        "title": "公开页面",
        "url": "https://example.test/page",
        "raw_content": "x" * (MAX_RAW_TEXT_CHARS + 1),
    }
    result = build_source_records([source])[0]
    assert len(result.raw_text) == MAX_RAW_TEXT_CHARS
    assert result.raw_text_truncated is True

    sources = build_source_records([
        {"title": f"页面 {i}", "url": f"https://example.test/{i}", "content": f"正文 {i}"}
        for i in range(MAX_SOURCE_RECORDS + 1)
    ])
    assert len(sources) == MAX_SOURCE_RECORDS
