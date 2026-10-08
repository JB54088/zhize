import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pipeline import apply_grounded_annotations, clean_and_ground_results


class PipelineTests(unittest.TestCase):
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
        self.assertEqual(results[1].verdict, "unverified")
        self.assertEqual(results[1].evidence, "")

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
