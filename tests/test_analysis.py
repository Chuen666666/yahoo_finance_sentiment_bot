import unittest

from yfsent.analysis import analyze_records
from yfsent.domain import ContentRecord, Entity
from yfsent.entities import EntityResolver
from yfsent.text import EvidenceExtractor


class FakeClassifier:
    version = "fake-v1"
    confidence_threshold = 0.60

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        return [
            {"negative": 0.10, "neutral": 0.20, "positive": 0.70}
            for _ in texts
        ]


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entity = Entity(
            "2330",
            "TWSE",
            "台灣積體電路製造股份有限公司",
            "台積電",
            ("台積電",),
        )
        self.resolver = EntityResolver([self.entity])
        self.classifier = FakeClassifier()
        self.extractor = EvidenceExtractor()

    def test_target_only_marks_missing_mention_uncertain(self) -> None:
        records = [
            ContentRecord("1", "rss", "台積電營收成長", "", "https://example.test/1"),
            ContentRecord("2", "rss", "大盤今日上漲", "", "https://example.test/2"),
        ]
        results = analyze_records(
            records,
            resolver=self.resolver,
            classifier=self.classifier,
            extractor=self.extractor,
            target=self.entity,
            target_only=True,
        )

        self.assertEqual([result.label for result in results], ["positive", "uncertain"])
        self.assertTrue(results[0].target_mentioned)
        self.assertFalse(results[1].target_mentioned)
        self.assertTrue(results[1].review_required)

    def test_threshold_can_force_human_review(self) -> None:
        record = ContentRecord(
            "1", "rss", "台積電營收成長", "", "https://example.test/1"
        )
        result = analyze_records(
            [record],
            resolver=self.resolver,
            classifier=self.classifier,
            extractor=self.extractor,
            target=self.entity,
            target_only=True,
            confidence_threshold=0.80,
        )[0]

        self.assertEqual(result.raw_label, "positive")
        self.assertEqual(result.label, "uncertain")
        self.assertTrue(result.review_required)
        self.assertTrue(result.target_mentioned)


if __name__ == "__main__":
    unittest.main()
