import unittest
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pipeline import apply_grounded_annotations, clean_and_ground_results


class PipelineTests(unittest.TestCase):
    def test_deadline_source_evidence_and_official_source_statuses(self):
        yesterday = date.today() - timedelta(days=1)
        tomorrow = date.today() + timedelta(days=1)
        results = clean_and_ground_results(
            "报名",
            [
                {
                    "title": "Official expired notice",
                    "url": "https://service.gov.cn/apply",
                    "content": f"报名截止日期：{yesterday:%Y年%-m月%-d日}。报名说明见本页。",
                },
                {
                    "title": "Configured official source",
                    "url": "https://apply.example.org/program",
                    "content": f"截止日期：{tomorrow:%Y-%m-%d}，报名仍开放。",
                },
                {
                    "title": "Missing evidence",
                    "url": "https://example.net/program",
                    "content": "",
                },
            ],
            limit=3,
            official_source_domains=("example.org",),
        )

        self.assertEqual(results[0].deadline_status, "expired")
        self.assertEqual(results[0].official_source_status, "verified")
        self.assertEqual(results[0].evidence_status, "available")
        self.assertEqual(results[1].deadline_status, "not_expired")
        self.assertEqual(results[1].official_source_status, "verified")
        self.assertEqual(results[2].deadline_status, "not_stated")
        self.assertEqual(results[2].official_source_status, "unverified")
        self.assertEqual(results[2].evidence_status, "missing")

    def test_ambiguous_deadline_is_unverified_and_unrelated_date_is_not_deadline(self):
        today = date.today()
        results = clean_and_ground_results(
            "program",
            [
                {
                    "title": "Unclear deadline",
                    "url": "https://example.org/unclear",
                    "content": "Application deadline information will be announced later.",
                },
                {
                    "title": "Published report",
                    "url": "https://example.org/report",
                    "content": f"Published {today:%Y-%m-%d}. Program information follows.",
                },
            ],
            limit=2,
        )
        self.assertEqual(results[0].deadline_status, "unverified")
        self.assertEqual(results[1].deadline_status, "not_stated")

    def test_evidence_selects_the_provider_passage_with_query_matches(self):
        source_text = (
            "General information about this website.\n\n"
            "The decision process was reviewed by the independent committee."
        )
        results = clean_and_ground_results(
            "decision process",
            [{
                "title": "Review",
                "url": "https://example.org/review",
                "raw_content": source_text,
            }],
            limit=1,
        )
        self.assertEqual(
            results[0].evidence,
            "The decision process was reviewed by the independent committee.",
        )
        self.assertIn(results[0].evidence, source_text)

    def test_only_provider_sources_survive_and_evidence_comes_from_content(self):
        raw = [
            {
                "title": "Decision research",
                "url": "https://example.org/report",
                "content": "Research reports describe the decision process.",
                "raw_content": "Research reports describe the decision process. Full text here.",
                "published_date": "2026-09-01",
            },
            {
                "title": "No body",
                "url": "https://example.org/empty",
                "content": "",
                "raw_content": "",
            },
            {
                "title": "Duplicate URL should be ignored",
                "url": "https://example.org/report",
                "content": "Different record content",
            },
            {
                "title": "Invalid URL must be discarded",
                "url": "javascript:alert(1)",
                "content": "decision",
            },
        ]

        results = clean_and_ground_results("decision process", raw, limit=8)

        self.assertEqual([result.url for result in results], [
            "https://example.org/report",
            "https://example.org/empty",
        ])
        self.assertEqual(
            results[0].evidence,
            "Research reports describe the decision process.",
        )
        self.assertEqual(results[0].verdict, "context")
        self.assertEqual(results[0].evidence_status, "available")
        self.assertEqual(results[1].verdict, "unverified")
        self.assertEqual(results[1].evidence, "")
        self.assertEqual(results[1].evidence_status, "missing")

    def test_annotation_cannot_add_sources_or_unbacked_quotes(self):
        results = clean_and_ground_results(
            "decision",
            [{
                "title": "Report",
                "url": "https://example.org/report",
                "content": "The decision was reviewed.",
            }],
            limit=5,
        )
        result_id = results[0].id

        apply_grounded_annotations(
            results,
            [
                {
                    "id": "invented-id",
                    "verdict": "supports",
                    "confidence": 0.99,
                    "evidence": "invented quote",
                },
                {
                    "id": result_id,
                    "verdict": "supports",
                    "confidence": 0.91,
                    "evidence": "The decision was reviewed.",
                },
            ],
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, result_id)
        self.assertEqual(results[0].verdict, "supports")
        self.assertEqual(results[0].evidence, "The decision was reviewed.")

        apply_grounded_annotations(
            results,
            [{
                "id": result_id,
                "verdict": "supports",
                "confidence": 1,
                "evidence": "This was not in the source.",
            }],
        )
        self.assertEqual(results[0].verdict, "unverified")
        self.assertEqual(results[0].confidence, 0)
        self.assertEqual(results[0].evidence, "")


if __name__ == "__main__":
    unittest.main()
