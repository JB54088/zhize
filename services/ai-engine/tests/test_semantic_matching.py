from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.semantic_matching import classify_procurement, rank_and_explain_results


def test_ai_training_does_not_match_an_ai_system_notice() -> None:
    match = classify_procurement(
        "高校人工智能培训服务采购",
        "人工智能系统平台建设项目",
        "建设用于业务办理的AI系统平台，不是培训课程或培训服务。",
    )

    assert match["query_category"] == "ai_training"
    assert match["procurement_category"] == "ai_system"
    assert match["match_status"] == "keyword_only"
    assert "采购意图不一致" in match["mismatch_reason"]
    assert "关键词重合" in match["match_reason"]


def test_real_training_notice_titles_are_classified_as_ai_training() -> None:
    candidates = [
        (
            "人工智能编程培训",
            "面向学生开展人工智能编程能力培训课程。",
        ),
        (
            "人工智能拔尖创新人才培养",
            "面向学生开展人工智能领域拔尖创新人才培养。",
        ),
        (
            "教师素养提升服务",
            "为教师提供素养提升、教学能力培养和专题培训服务。",
        ),
    ]

    for title, content in candidates:
        match = classify_procurement("人工智能培训服务采购", title, content)
        assert match["procurement_category"] == "ai_training", (title, match)
        assert match["match_status"] == "matched", (title, match)


def test_ai_system_query_does_not_match_the_real_training_notices() -> None:
    candidates = [
        (
            "人工智能编程培训",
            "面向学生开展人工智能编程能力培训课程。",
            "keyword_only",
        ),
        (
            "人工智能拔尖创新人才培养",
            "面向学生开展人工智能领域拔尖创新人才培养。",
            "keyword_only",
        ),
        (
            "教师素养提升服务",
            "为教师提供素养提升、教学能力培养和专题培训服务。",
            "not_matched",
        ),
    ]

    for title, content, expected_status in candidates:
        match = classify_procurement("人工智能应用系统采购", title, content)
        assert match["query_category"] == "ai_system", match
        assert match["procurement_category"] == "ai_training", match
        assert match["match_status"] == expected_status, (title, match)
        assert match["mismatch_reason"]


def test_spaced_ai_system_build_query_and_service_only_title_are_not_overmatched() -> None:
    match = classify_procurement(
        "寻找福建省近期有明确 AI 系统建设需求且未截止的政府采购项目",
        "福建省公共资源交易中心政府采购业务人工智能辅助服务项目",
        "采购人工智能辅助服务，用于政府采购智能政策解读和招标文件合规性辅助。",
    )

    assert match["query_category"] == "ai_system"
    assert match["procurement_category"] == "ai_system"
    assert match["match_status"] == "keyword_only"
    assert "缺少系统、平台或软件建设" in match["mismatch_reason"]


def test_official_body_system_build_evidence_changes_service_title_to_partial_match() -> None:
    match = classify_procurement(
        "寻找福建省近期有明确 AI 系统建设需求且未截止的政府采购项目",
        "福建省公共资源交易中心政府采购业务人工智能辅助服务项目",
        "采购内容：构建福建省公共资源交易中心政府采购专属AI系统，包含智能支撑底座及智能体模块。",
    )

    assert match["query_category"] == "ai_system"
    assert match["procurement_category"] == "ai_system"
    assert match["match_status"] == "partial"
    assert "来源正文明确描述" in match["match_reason"]


def test_explicit_ai_system_platform_title_matches_system_build_query() -> None:
    match = classify_procurement(
        "寻找福建省近期有明确 AI 系统建设需求且未截止的政府采购项目",
        "福建省政务人工智能系统平台建设采购项目",
        "采购人拟建设人工智能业务系统平台，用于政务事项办理和智能辅助。",
    )

    assert match["query_category"] == "ai_system"
    assert match["procurement_category"] == "ai_system"
    assert match["match_status"] == "matched"


def test_digital_transformation_intent_matches_a_digital_upgrade() -> None:
    match = classify_procurement(
        "企业数字化改造服务采购",
        "制造企业数字化转型升级项目",
        "项目推进企业业务流程数字化改造与系统升级。",
    )

    assert match["query_category"] == "digital_transformation"
    assert match["procurement_category"] == "digital_transformation"
    assert match["match_status"] == "matched"
    assert match["mismatch_reason"] is None


def test_compute_equipment_is_separated_from_ai_software() -> None:
    match = classify_procurement(
        "GPU算力设备采购",
        "AI服务器及GPU集群设备采购项目",
        "采购服务器、计算节点和GPU设备，为本地数据中心提供计算硬件。",
    )

    assert match["query_category"] == "compute_hardware"
    assert match["procurement_category"] == "compute_hardware"
    assert match["match_status"] == "matched"


def test_missing_candidate_text_is_unverified() -> None:
    match = classify_procurement("AI培训课程", "", "")

    assert match["query_category"] == "ai_training"
    assert match["procurement_category"] == "unclear"
    assert match["match_status"] == "unverified"
    assert match["mismatch_reason"]


def test_explicit_non_target_procurement_is_other() -> None:
    match = classify_procurement(
        "办公家具采购",
        "办公家具更新采购项目",
        "采购办公桌椅和会议桌，未涉及数字化、AI服务或算力设备。",
    )

    assert match["query_category"] == "other"
    assert match["procurement_category"] == "other"
    assert match["match_status"] == "matched"


def test_generic_procurement_query_remains_unclear() -> None:
    match = classify_procurement("采购项目", "采购公告", "项目采购公告")

    assert match["query_category"] == "unclear"
    assert match["procurement_category"] == "unclear"
    assert match["match_status"] == "unverified"


def test_candidate_with_only_a_shared_ai_keyword_is_keyword_only() -> None:
    match = classify_procurement(
        "AI培训项目",
        "AI项目采购公告",
        "人工智能相关采购项目，具体采购内容未明确。",
    )

    assert match["query_category"] == "ai_training"
    assert match["procurement_category"] == "unclear"
    assert match["match_status"] == "keyword_only"


def test_ranking_uses_explicit_deadline_and_evidence_quality() -> None:
    aligned = {
        "id": "aligned",
        "title": "人工智能培训课程采购",
        "snippet": "为员工提供人工智能课程和培训服务。",
        "evidence": "为员工提供人工智能课程和培训服务。",
        "match_reason": "source text match",
        "deadline_status": "not_expired",
        "evidence_status": "available",
        "official_source_status": "verified",
    }
    expired = {
        "id": "expired",
        "title": "人工智能培训服务采购",
        "snippet": "采购人工智能培训服务。",
        "evidence": "",
        "deadline_status": "expired",
        "evidence_status": "missing",
        "official_source_status": "insufficient_evidence",
    }

    ranked = rank_and_explain_results("AI培训课程采购", [expired, aligned])

    assert [item["id"] for item in ranked] == ["aligned", "expired"]
    assert ranked[0]["match_status"] == "matched"
    assert ranked[0]["ranking_score"] > ranked[1]["ranking_score"]
    assert "来源证据明确可用" in ranked[0]["ranking_reason"]
    assert "已明确标记为过期" in ranked[1]["ranking_reason"]


def test_absent_deadline_and_evidence_fields_are_not_inferred() -> None:
    candidate = {
        "id": "metadata-absent",
        "title": "人工智能培训课程采购",
        "snippet": "开展人工智能培训课程。",
    }

    ranked = rank_and_explain_results("AI培训课程采购", [candidate])

    assert ranked[0]["match_status"] == "matched"
    assert ranked[0]["match_score"] == ranked[0]["ranking_score"]
    assert ranked[0]["ranking_reason"] == "按采购意图匹配度排序。"
    assert "deadline_status" not in ranked[0]
    assert "evidence_status" not in ranked[0]
