import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents import AnswerAgent, CleanAgent, CollectAgent, IntentAgent, JudgeAgent, VerifyAgent
from app.agents._runtime import invoke_json
from app.agents.models import IntentAnalysis
from app.agents.verify import _select_relevant_excerpt


class FakeLLMClient:
    is_enabled = True

    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls = []

    async def complete_json(self, *, agent_name, messages, response_model):
        self.calls.append({"agent_name": agent_name, "messages": messages})
        value = self.responses[agent_name]
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(
            data=response_model.model_validate(value),
            model="deepseek-test-model",
            request_id="req-test-1",
            prompt_tokens=101,
            completion_tokens=19,
            total_tokens=120,
            latency_ms=8,
            estimated_cost_usd=None,
            retry_count=0,
        )


class DisabledLLMClient:
    is_enabled = False


def test_verify_excerpt_prioritizes_query_relevant_exact_source_sentences():
    navigation = "网站导航和无关介绍。" * 300
    evidence = "采购内容：构建政府采购专属AI系统。提交首次响应文件截止时间为2025年10月29日9:00。"
    excerpt = _select_relevant_excerpt(
        navigation + evidence,
        [
            {"field": "business_attributes", "value": "AI系统建设"},
            {"field": "open_status", "value": "未截止"},
        ],
        500,
    )

    assert len(excerpt) <= 500
    assert "采购内容：构建政府采购专属AI系统。" in excerpt
    assert "截止时间为2025年10月29日9:00" in excerpt
    assert len(excerpt) < len(navigation)


def test_intent_llm_generates_queries_but_rule_hard_conditions_win():
    async def scenario():
        query = "我是2027届电子信息工程毕业生，寻找正在招聘的校招岗位"
        client = FakeLLMClient({"collect": {
            "domain": "procurement",
            "intent_summary": "寻找校招岗位",
            "search_queries": ["2027届电子信息工程 校园招聘", "校园招聘 2027届"],
            "soft_preferences": ["优先官方招聘来源"],
        }})
        execution = await IntentAgent().run(query, client)

        assert execution.output["domain"] == "employment"
        assert {item["field"] for item in execution.output["hard_conditions"]} == {
            "graduate_cohort", "major", "open_status", "application_url"
        }
        assert len(execution.output["search_queries"]) == 3
        assert execution.run_log["status"] == "completed"
        assert execution.run_log["actual_model"] == "deepseek-test-model"
        assert query not in execution.run_log["input_summary"]
        assert "api_key" not in str(execution.run_log).lower()

    asyncio.run(scenario())


def test_intent_explicitly_skips_unconfigured_llm_and_preserves_deterministic_intent():
    async def scenario():
        execution = await IntentAgent().run("寻找福建省未截止的政府采购项目", DisabledLLMClient())
        assert execution.run_log["status"] == "skipped_unconfigured"
        assert execution.output["domain"] == "procurement"
        assert execution.output["search_queries"] == ["寻找福建省未截止的政府采购项目"]

    asyncio.run(scenario())


def test_configured_intent_error_is_failed_instead_of_silent_fallback():
    async def scenario():
        error = RuntimeError("private response details must stay out of logs")
        error.code = "http_503"
        execution = await IntentAgent().run("招聘搜索", FakeLLMClient({"collect": error}))
        assert execution.run_log["status"] == "failed"
        assert execution.run_log["error_code"] == "http_503"
        assert "private response details" not in str(execution.run_log)

    asyncio.run(scenario())


def test_agent_error_log_preserves_provider_usage_metadata_without_response_body():
    async def scenario():
        error = RuntimeError("private provider response")
        error.code = "completion_truncated"
        error.actual_model = "deepseek-test-model"
        error.request_id = "request-42"
        error.latency_ms = 321
        error.prompt_tokens = 700
        error.completion_tokens = 4096
        error.total_tokens = 4796
        error.estimated_cost_usd = 0.01
        error.retry_count = 1

        class FailedClient:
            is_enabled = True

            async def complete_json(self, *, agent_name, messages, response_model):
                raise error

        data, metadata, code = await invoke_json(
            FailedClient(),
            agent_name="verify",
            messages=[{"role": "user", "content": "safe prompt"}],
            response_model=IntentAnalysis,
        )

        assert data is None
        assert code == "completion_truncated"
        assert metadata["actual_model"] == "deepseek-test-model"
        assert metadata["request_id"] == "request-42"
        assert metadata["prompt_tokens"] == 700
        assert metadata["completion_tokens"] == 4096
        assert metadata["total_tokens"] == 4796
        assert metadata["retry_count"] == 1
        assert "private provider response" not in str(metadata)

    asyncio.run(scenario())


def test_collect_dispatches_multiple_queries_and_keeps_provider_provenance():
    async def scenario():
        class Adapter:
            name = "fixture_search_source"

            async def search(self, query, limit):
                return [{
                    "title": f"Result for {query}",
                    "url": f"https://example.test/{len(query)}",
                    "content": "A source passage.",
                }]

        execution = await CollectAgent().run(
            "original", ["first query", "second query"], [Adapter()], 4
        )
        assert len(execution.output["records"]) == 2
        assert {item["_provenance"]["search_query_index"] for item in execution.output["records"]} == {0, 1}
        assert {item["_provenance"]["provider"] for item in execution.output["records"]} == {"fixture_search_source"}
        assert execution.run_log["status"] == "completed"
        assert "first query" not in execution.run_log["input_summary"]

    asyncio.run(scenario())


def test_clean_agent_only_adds_extractions_with_exact_source_quotes():
    async def scenario():
        body = "此公告由示例科技发布。招聘2027届毕业生，专业要求电子信息工程。"
        raw = [{
            "title": "校招信息",
            "url": "https://jobs.example.test/campus",
            "content": body,
            "raw_content": body,
        }]
        client = FakeLLMClient({"clean": {"items": [{
            "id": "a5a839dbb04e6ff1e4f9",
            "fields": [
                {"field": "organization", "value": "示例科技", "quote": "此公告由示例科技发布"},
            ],
        }]}})
        # The source id is deterministic from the canonical URL.
        from app.stages.clean import canonicalize_url
        import hashlib
        source_id = hashlib.sha256(canonicalize_url(raw[0]["url"]).encode()).hexdigest()[:20]
        client.responses["clean"]["items"][0]["id"] = source_id
        execution = await CleanAgent().run("2027届电子信息工程校招", raw, 5, llm_client=client)

        fields = execution.output["records"][0]["extracted_fields"]
        assert fields["organization"] == {"value": "示例科技", "quote": "此公告由示例科技发布"}
        assert fields.get("role", {}).get("value") != "人工智能算法岗"
        assert execution.run_log["status"] == "completed"
        assert "接受 1 个" in execution.run_log["output_summary"]

    asyncio.run(scenario())


def test_clean_agent_bounds_llm_prompt_but_keeps_all_cleaned_results():
    async def scenario():
        raw = [{
            "title": f"校招职位 {index}",
            "url": f"https://jobs.example.test/campus/{index}",
            "content": f"2027届电子信息工程职位 {index}。" + ("来源正文。" * 1200),
            "raw_content": f"2027届电子信息工程职位 {index}。" + ("来源正文。" * 1200),
        } for index in range(7)]
        client = FakeLLMClient({"clean": {"items": []}})

        execution = await CleanAgent().run("2027届电子信息工程校招", raw, 10, llm_client=client)
        import json
        submitted = json.loads(client.calls[0]["messages"][-1]["content"])

        assert len(execution.output["records"]) == 7
        assert len(submitted["sources"]) == 3
        assert all(len(source["source_text"]) <= 720 for source in submitted["sources"])
        assert execution.run_log["status"] == "completed"

    asyncio.run(scenario())


def test_verify_agent_preserves_deterministic_status_and_accepts_grounded_quote():
    async def scenario():
        record = {
            "id": "r1",
            "title": "示例科技 2027 校招",
            "url": "https://search.example.test/item",
            "official_source_status": "verified",
            "official_source_url": "https://careers.example.test/campus",
            "official_source_title": "示例科技 2027 校园招聘公告",
            "official_evidence_detail": "示例科技招聘2027届毕业生，专业要求：电子信息工程。",
            "field_verification": {
                "graduate_cohort": {
                    "value": "2027届", "status": "verified",
                    "evidence_url": "https://careers.example.test/campus", "quote": "招聘2027届毕业生",
                },
                "business_attributes": {
                    "value": "AI系统建设", "status": "unverified",
                    "evidence_url": None, "quote": None,
                },
            },
        }
        original_fields = {key: dict(value) for key, value in record["field_verification"].items()}
        client = FakeLLMClient({"verify": {"items": [{
            "id": "r1",
            "facts": [
                {
                    "field": "major", "value": "电子信息工程", "quote": "专业要求：电子信息工程。",
                    "evidence_url": "https://careers.example.test/campus", "assessment": "supports",
                },
            ],
        }]}})
        execution = await VerifyAgent().run(
            [record],
            required_conditions=[{"field": "business_attributes", "value": "AI系统建设"}],
            llm_client=client,
        )
        result = execution.output["records"][0]
        import json
        model_input = json.loads(client.calls[0]["messages"][-1]["content"])

        assert result["field_verification"] == original_fields
        assert result["official_source_status"] == "verified"
        assert result["llm_verification"]["status"] == "analyzed"
        assert len(result["llm_verification"]["facts"]) == 1
        assert result["llm_verification"]["facts"][0]["field"] == "major"
        assert set(model_input["sources"][0]["deterministic_fields"]) == {"business_attributes"}

    asyncio.run(scenario())


def test_verify_agent_rejects_model_quote_absent_from_official_source():
    async def scenario():
        record = {
            "id": "r1",
            "title": "官方公告",
            "url": "https://search.example.test/item",
            "official_source_status": "verified",
            "official_source_url": "https://official.example.test/notice",
            "official_source_title": "官方公告",
            "official_evidence_detail": "项目采购内容为人工智能辅助服务。",
            "field_verification": {"business_attributes": {"status": "unverified"}},
        }
        client = FakeLLMClient({"verify": {"items": [{
            "id": "r1",
            "facts": [{
                "field": "business_attributes", "value": "AI平台建设", "quote": "本项目建设人工智能平台。",
                "evidence_url": "https://official.example.test/notice", "assessment": "supports",
            }],
        }]}})

        execution = await VerifyAgent().run(
            [record],
            required_conditions=[{"field": "business_attributes", "value": "AI平台建设"}],
            llm_client=client,
        )

        assert execution.output["records"][0]["llm_verification"] == {"status": "rejected", "facts": []}

    asyncio.run(scenario())


def test_verify_agent_caps_model_source_context_without_truncating_stored_evidence():
    async def scenario():
        full_body = "官方公告正文。" * 900
        record = {
            "id": "r1",
            "title": "示例公司岗位公告",
            "url": "https://search.example.test/item",
            "official_source_status": "verified",
            "official_source_url": "https://careers.example.test/role",
            "official_source_title": "示例公司岗位公告",
            "official_evidence_detail": full_body,
            "field_verification": {},
        }
        client = FakeLLMClient({"verify": {"items": []}})

        execution = await VerifyAgent().run(
            [record],
            required_conditions=[{"field": "business_attributes", "value": "AI系统建设"}],
            llm_client=client,
        )
        import json
        submitted = json.loads(client.calls[0]["messages"][-1]["content"])

        assert len(submitted["sources"][0]["source_text"]) <= 2000
        assert execution.output["records"][0]["official_evidence_detail"] == full_body

    asyncio.run(scenario())


def test_verify_agent_caps_llm_assistance_to_one_source_but_keeps_all_records():
    async def scenario():
        records = [
            {
                "id": f"r{index}",
                "title": f"官方项目公告 {index}",
                "url": f"https://search.example.test/{index}",
                "official_source_status": "verified",
                "official_source_url": f"https://official.example.test/{index}",
                "official_source_title": f"官方项目公告 {index}",
                "official_evidence_detail": f"采购公告 {index}，项目内容为人工智能系统建设。",
                "field_verification": {"business_attributes": {"status": "unverified"}},
            }
            for index in range(2)
        ]
        client = FakeLLMClient({"verify": {"items": []}})

        execution = await VerifyAgent().run(
            records,
            required_conditions=[{"field": "business_attributes", "value": "人工智能系统建设"}],
            llm_client=client,
        )
        import json
        submitted = json.loads(client.calls[0]["messages"][-1]["content"])

        assert len(submitted["sources"]) == 1
        assert len(execution.output["records"]) == 2
        assert execution.output["records"][1]["llm_verification"]["status"] == "not_analyzed"

    asyncio.run(scenario())


def test_judge_llm_cannot_override_failed_hard_condition():
    async def scenario():
        query = "我是2027届电子信息工程毕业生，寻找正在招聘的校招岗位"
        intent = {
            "domain": "employment",
            "hard_conditions": [
                {"field": "graduate_cohort", "label": "毕业届别", "value": "2027届"},
                {"field": "major", "label": "专业", "value": "电子信息工程"},
            ],
            "soft_conditions": [],
        }
        record = {
            "id": "r1", "title": "某公司校招", "url": "https://search.test/item",
            "snippet": "", "evidence": "", "raw_text": "",
            "official_source_status": "verified",
            "official_source_url": "https://careers.example.test/notice",
            "official_source_title": "某公司校招公告",
            "field_verification": {
                "graduate_cohort": {"value": "2027届", "status": "verified", "evidence_url": "https://careers.example.test/notice", "quote": "招聘2027届毕业生"},
                "major": {"value": "机械工程", "status": "verified", "evidence_url": "https://careers.example.test/notice", "quote": "专业要求为机械工程"},
            },
        }
        client = FakeLLMClient({"judge": {"items": [{
            "id": "r1", "semantic_fit": "strong", "score": 0.99,
            "reason": "模型给出偏强语义相关判断。", "evidence_quotes": ["招聘2027届毕业生"],
        }]}})
        execution = await JudgeAgent().run(query, intent, [record], client)
        result = execution.output["records"][0]

        assert result["decision"] == "not_matched"
        assert result.get("match_score") != 0.99
        assert result["llm_semantic_match"]["fit"] == "strong"

    asyncio.run(scenario())


def test_answer_rejects_model_claim_without_exact_verified_quote():
    async def scenario():
        record = {
            "id": "r1", "title": "示例公司校招", "decision": "recommended",
            "official_source_status": "verified",
            "official_source_url": "https://careers.example.test/notice",
            "official_source_title": "示例公司校招公告",
            "field_verification": {
                "graduate_cohort": {"value": "2027届", "status": "verified", "evidence_url": "https://careers.example.test/notice", "quote": "招聘2027届毕业生"},
            },
        }
        client = FakeLLMClient({"judge": {"summary": "找到推荐岗位。", "claims": [{
            "result_id": "r1", "text": "专业要求为电子信息工程。", "evidence_quotes": ["岗位明确要求电子信息工程"],
        }]}})
        execution = await AnswerAgent().run("找校招", {}, [record], client)
        assert execution.run_log["status"] == "failed"
        assert execution.run_log["error_code"] == "citation_validation_failed"
        assert execution.output["answer"] is None

    asyncio.run(scenario())


def test_answer_accepts_exact_verified_quote_and_records_usage_metadata():
    async def scenario():
        record = {
            "id": "r1", "title": "示例公司校招", "decision": "needs_verification",
            "official_source_status": "verified",
            "official_source_url": "https://careers.example.test/notice",
            "official_source_title": "示例公司校招公告",
            "field_verification": {
                "graduate_cohort": {"value": "2027届", "status": "verified", "evidence_url": "https://careers.example.test/notice", "quote": "招聘2027届毕业生"},
            },
        }
        client = FakeLLMClient({"judge": {
            "summary": "来源显示该岗位面向2027届毕业生，其他条件仍需核实。",
            "claims": [{
                "result_id": "r1", "text": "公告写明招聘2027届毕业生。", "evidence_quotes": ["招聘2027届毕业生"],
            }],
        }})
        execution = await AnswerAgent().run("找校招", {}, [record], client)
        assert execution.run_log["status"] == "completed"
        assert execution.run_log["total_tokens"] == 120
        assert execution.run_log["latency_ms"] == 8
        assert execution.output["claims"][0]["evidence_quotes"] == ["招聘2027届毕业生"]

    asyncio.run(scenario())
