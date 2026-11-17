import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.official_verifier import verify_official_source


class OfficialVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_verifies_project_and_each_available_field_from_official_notice(self):
        claim = "本项目采购内容为知择信息决策平台。"
        notice = (
            "项目名称：知择信息决策平台采购项目\n"
            "采购人信息 / 名称：某市数据资源管理局\n"
            "预算金额：人民币 120 万元\n"
            "信息时间：2026-10-04\n"
            "投标截止时间：2099年10月21日 09:30\n"
            + claim
        )
        calls = []

        async def search(query, domains, limit):
            calls.append((query, domains, limit))
            return [{
                "title": "知择信息决策平台采购公告",
                "url": "https://procurement.example.gov.cn/notices/123",
                "raw_content": notice,
            }]

        result = await verify_official_source(
            project_name="知择信息决策平台采购项目",
            support_claim=claim,
            search=search,
            official_domains=("procurement.example.gov.cn",),
        )

        self.assertEqual(result["official_source_status"], "verified")
        self.assertEqual(
            result["official_source_url"],
            "https://procurement.example.gov.cn/notices/123",
        )
        self.assertEqual(result["purchasing_unit"], "某市数据资源管理局")
        self.assertEqual(result["budget"], "人民币 120 万元")
        self.assertEqual(result["published_at"], "2026-10-04")
        self.assertEqual(result["deadline_at"], "2099年10月21日 09:30")
        self.assertEqual(result["deadline_status"], "not_expired")
        self.assertEqual(
            result["verified_fields"],
            ["project_name", "purchasing_unit", "published_at", "budget", "deadline", "evidence"],
        )
        self.assertEqual(result["unverified_fields"], [])
        self.assertEqual(result["evidence"], claim)
        self.assertIn(result["evidence"], notice)
        self.assertEqual(
            calls[0][1], ("ccgp.gov.cn", "gov.cn", "procurement.example.gov.cn")
        )
        self.assertEqual(calls[0][2], 10)

    async def test_extracts_purchasing_unit_from_entrustment_wording(self):
        notice = (
            "项目名称：目标项目。"
            "公告说明目标项目已启动。"
            "本次采购受某市数据局委托组织实施。"
        )

        async def search(query, domains, limit):
            return [{
                "title": "目标项目政府采购公告",
                "url": "https://agency.gov.cn/notice",
                "content": notice,
            }]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="公告说明目标项目已启动。",
            search=search,
        )
        self.assertEqual(result["purchasing_unit"], "某市数据局")
        self.assertIn("purchasing_unit", result["verified_fields"])

    async def test_single_source_official_title_and_body_verify_project_and_deadline(self):
        project_name = "福建省公共资源交易中心政府采购业务人工智能辅助服务项目"
        record = {
            "title": f"{project_name}单一来源公告 _ 通知公告 _ 省发改委",
            "url": "https://fgw.fujian.gov.cn/notices/one-source",
            "raw_content": (
                "福建省公共资源交易中心参照政府采购单一来源方式组织政府采购业务人工智能辅助服务项目采购活动。"
                "预算金额：200000元。提交首次响应文件截止时间及地点：2025年10月29日9:00(北京时间)。"
            ),
        }

        async def search(query, domains, limit):
            return [record]

        result = await verify_official_source(
            project_name=project_name,
            support_claim="",
            search=search,
        )

        self.assertEqual(result["official_source_status"], "verified")
        self.assertEqual(result["official_source_url"], record["url"])
        self.assertIn("project_name", result["verified_fields"])
        self.assertEqual(result["budget"], "200000元")
        self.assertIn("deadline", result["verified_fields"])
        self.assertEqual(result["deadline_status"], "expired")
        self.assertEqual(result["evidence"], "")
        self.assertIn("evidence", result["unverified_fields"])

    async def test_extracts_official_page_date_and_exact_system_build_evidence(self):
        project_name = "福建省公共资源交易中心政府采购业务人工智能辅助服务"
        notice = (
            "来源：本网 时间：2025-10-23 17:23\n"
            f"项目名称：{project_name}\n"
            "二、采购内容：构建福建省公共资源交易中心政府采购专属AI系统，"
            "包含智能支撑底座、采购文件合规性检查智能体、智能问答智能体。\n"
            "预算金额：200000元。\n"
            "提交首次响应文件截止时间及地点：2025年10月29日9:00(北京时间)。"
        )

        async def search(query, domains, limit):
            return [{
                "title": f"{project_name}单一来源公告",
                "url": "https://fgw.fujian.gov.cn/notices/ai-system",
                "raw_content": notice,
            }]

        result = await verify_official_source(
            project_name=project_name,
            support_claim="AI系统",
            search=search,
        )

        assert result["official_source_status"] == "verified"
        assert result["published_at"] == "2025-10-23 17:23"
        assert "published_at" in result["verified_fields"]
        assert result["budget"] == "200000元"
        assert result["field_evidence"]["region"]["value"] == "福建省"
        assert "region" in result["verified_fields"]
        assert result["deadline_status"] == "expired"
        assert result["evidence"] == (
            "二、采购内容：构建福建省公共资源交易中心政府采购专属AI系统，"
            "包含智能支撑底座、采购文件合规性检查智能体、智能问答智能体。"
        )
        assert result["evidence"] in notice

    async def test_government_domain_alone_does_not_verify_an_unrelated_notice(self):
        async def search(query, domains, limit):
            return [{
                "title": "Official domain, unrelated notice",
                "url": "https://agency.gov.cn/notices/1",
                "raw_content": "采购人：某市政府。预算金额：100万元。",
            }]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="目标项目的采购结论成立。",
            search=search,
        )

        self.assertEqual(result["official_source_status"], "insufficient_evidence")
        self.assertIsNone(result["official_source_url"])
        self.assertEqual(result["verified_fields"], [])
        self.assertEqual(result["evidence"], "")
        self.assertEqual(
            result["unverified_fields"],
            ["project_name", "purchasing_unit", "published_at", "budget", "deadline", "evidence"],
        )

    async def test_portal_homepage_snippet_is_not_treated_as_a_notice(self):
        async def search(query, domains, limit):
            return [{
                "title": "中央政府采购网",
                "url": "https://www.zycg.gov.cn",
                "content": "中央政府采购网最新公告：某管理局2026年AI服务器采购项目招标公告。",
            }]

        result = await verify_official_source(
            project_name="中央政府采购网",
            support_claim="AI服务器采购",
            search=search,
        )

        self.assertEqual(result["official_source_status"], "insufficient_evidence")
        self.assertIsNone(result["official_source_url"])

    async def test_configured_domain_alone_does_not_verify_a_source(self):
        async def search(query, domains, limit):
            return [{
                "url": "https://procurement.example.com/notice",
                "content": "该网站发布了一条采购信息。",
            }]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="目标项目已发布采购公告。",
            search=search,
            official_domains=("procurement.example.com",),
        )
        self.assertEqual(result["official_source_status"], "insufficient_evidence")
        self.assertIsNone(result["official_source_url"])

    async def test_official_domain_and_project_name_without_notice_marker_is_insufficient(self):
        async def search(query, domains, limit):
            return [{
                "title": "目标项目情况介绍",
                "url": "https://agency.gov.cn/about",
                "content": "项目名称：目标项目。本项目采购内容为目标项目。",
            }]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="本项目采购内容为目标项目。",
            search=search,
        )
        self.assertEqual(result["official_source_status"], "insufficient_evidence")
        self.assertIsNone(result["official_source_url"])

    async def test_third_party_is_not_promoted_and_fields_stay_unverified(self):
        async def search(query, domains, limit):
            return [{
                "title": "Project blog",
                "url": "https://news.example.com/project",
                "content": "目标项目的公告称该项目将启动。截止时间为2099-01-01。",
            }]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="目标项目将启动。",
            search=search,
        )

        self.assertEqual(result["official_source_status"], "third_party")
        self.assertEqual(result["deadline_status"], "unverified")
        self.assertIsNone(result["official_source_url"])
        self.assertEqual(result["verified_fields"], [])
        self.assertEqual(
            result["unverified_fields"],
            ["project_name", "purchasing_unit", "published_at", "budget", "deadline", "evidence"],
        )

    async def test_irrelevant_official_record_does_not_hide_relevant_third_party(self):
        async def search(query, domains, limit):
            return [
                {
                    "url": "https://agency.gov.cn/other",
                    "content": "采购项目名称：另一个项目。",
                },
                {
                    "url": "https://news.example.com/target",
                    "content": "目标项目的采购结论成立。",
                },
            ]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="目标项目的采购结论成立。",
            search=search,
        )
        self.assertEqual(result["official_source_status"], "third_party")

    async def test_missing_official_source_does_not_fill_fields_or_deadline_status(self):
        async def search(query, domains, limit):
            return []

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="目标项目将启动。",
            search=search,
        )

        self.assertEqual(result["official_source_status"], "insufficient_evidence")
        self.assertEqual(result["deadline_status"], "unverified")
        self.assertTrue(all(result[key] is None for key in (
            "official_source_url",
            "purchasing_unit",
            "budget",
            "deadline_at",
            "published_at",
        )))
        self.assertEqual(result["evidence"], "")

    async def test_no_deadline_in_verified_announcement_is_not_stated(self):
        async def search(query, domains, limit):
            return [{
                "title": "目标项目政府采购公告",
                "url": "https://agency.gov.cn/notice",
                "content": "项目名称：目标项目。本项目采购内容为目标项目。",
            }]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="本项目采购内容为目标项目。",
            search=search,
        )
        self.assertEqual(result["official_source_status"], "verified")
        self.assertEqual(result["deadline_status"], "not_stated")
        self.assertIn("deadline", result["unverified_fields"])

    async def test_invalid_calendar_deadline_is_unverified(self):
        async def search(query, domains, limit):
            return [{
                "title": "目标项目政府采购公告",
                "url": "https://agency.gov.cn/notice",
                "content": "项目名称：目标项目。本项目采购内容为目标项目。截止时间：2026年13月40日。",
            }]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="本项目采购内容为目标项目。",
            search=search,
        )
        self.assertEqual(result["official_source_status"], "verified")
        self.assertIsNone(result["deadline_at"])
        self.assertEqual(result["deadline_status"], "unverified")
        self.assertIn("deadline", result["unverified_fields"])

    async def test_expired_deadline_uses_current_shanghai_date(self):
        async def search(query, domains, limit):
            return [{
                "title": "目标项目政府采购公告",
                "url": "https://agency.gov.cn/notice",
                "content": "项目名称：目标项目。本项目采购内容为目标项目。截止时间：2000-01-01。",
            }]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="本项目采购内容为目标项目。",
            search=search,
        )
        self.assertEqual(result["deadline_status"], "expired")

    async def test_rejects_lookalike_host_private_ip_userinfo_and_http(self):
        unsafe_records = [
            {
                "url": "https://agency.gov.cn.attacker.example/notice",
                "content": "目标项目的采购结论成立。",
            },
            {
                "url": "https://127.0.0.1/notice",
                "content": "目标项目的采购结论成立。",
            },
            {
                "url": "https://agency.gov.cn@attacker.example/notice",
                "content": "目标项目的采购结论成立。",
            },
            {
                "url": "http://agency.gov.cn/notice",
                "content": "目标项目的采购结论成立。",
            },
        ]

        async def search(query, domains, limit):
            return unsafe_records

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="目标项目的采购结论成立。",
            search=search,
        )

        self.assertEqual(result["official_source_status"], "third_party")
        self.assertIsNone(result["official_source_url"])

    async def test_paraphrased_claim_does_not_block_source_but_has_no_evidence(self):
        notice = "项目名称：目标项目。公告说明目标项目将在本年度启动。"

        async def search(query, domains, limit):
            return [{
                "title": "目标项目政府采购公告",
                "url": "https://agency.gov.cn/notice",
                "raw_content": notice,
            }]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="目标项目将在今年启动。",
            search=search,
        )

        self.assertEqual(result["official_source_status"], "verified")
        self.assertIn("project_name", result["verified_fields"])
        self.assertEqual(result["evidence"], "")
        self.assertIn("evidence", result["unverified_fields"])

    async def test_empty_claim_can_still_verify_notice_fields_without_inventing_evidence(self):
        seen_queries = []

        async def search(query, domains, limit):
            seen_queries.append(query)
            return [{
                "title": "目标项目政府采购公告",
                "url": "https://agency.gov.cn/notice",
                "content": "项目名称：目标项目。预算金额：30万元。",
            }]

        result = await verify_official_source(
            project_name="目标项目",
            support_claim="",
            search=search,
        )

        self.assertEqual(result["official_source_status"], "verified")
        self.assertEqual(result["budget"], "30万元")
        self.assertEqual(result["evidence"], "")
        self.assertIn("evidence", result["unverified_fields"])
        self.assertNotIn('""', seen_queries[0])


if __name__ == "__main__":
    unittest.main()
