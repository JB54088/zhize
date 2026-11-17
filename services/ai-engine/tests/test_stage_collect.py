import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.stages.collect import collect_records, parse_query_intent


def test_parses_employment_hard_conditions_without_model_calls():
    intent = parse_query_intent("我是2027届毕业生，电子信息工程专业，帮我寻找符合条件且正在招聘的校招公司。")

    assert intent["domain"] == "employment"
    assert {item["field"] for item in intent["hard_conditions"]} == {
        "graduate_cohort", "major", "open_status", "application_url"
    }
    assert any(item["value"] == "2027届" for item in intent["hard_conditions"])


def test_parses_procurement_region_and_open_deadline_as_hard_conditions():
    intent = parse_query_intent("寻找福建省近期明确 AI 系统建设需求且未截止的政府采购项目")

    assert intent["domain"] == "procurement"
    assert {item["field"] for item in intent["hard_conditions"]} == {"region", "open_status"}
    assert next(item for item in intent["hard_conditions"] if item["field"] == "region")["value"] == "福建"


def test_collect_runs_adapters_concurrently_and_retains_provenance():
    async def scenario():
        active = 0
        max_active = 0

        class Adapter:
            def __init__(self, name):
                self.name = name

            async def search(self, query, limit):
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.01)
                active -= 1
                return [{"title": self.name, "url": f"https://{self.name}.example/item", "content": "source text"}]

        records, errors = await collect_records("query", [Adapter("one"), Adapter("two")], 4)
        assert not errors
        assert max_active == 2
        assert [item["_provenance"]["provider"] for item in records] == ["one", "two"]
        assert records[0]["_provenance"]["provider_record_index"] == 0

    asyncio.run(scenario())
