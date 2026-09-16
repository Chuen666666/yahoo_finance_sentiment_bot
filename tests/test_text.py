import unittest

from yfsent.domain import Entity
from yfsent.text import EvidenceExtractor, split_spans, target_context

TSMC = Entity(
    code="2330",
    market="TWSE",
    name="台灣積體電路製造股份有限公司",
    short_name="台積電",
    aliases=("台積電", "2330"),
)


class DummyClassifier:
    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        output = []
        for text in texts:
            if "受惠" in text or "成長" in text:
                output.append({"positive": 0.9, "neutral": 0.08, "negative": 0.02})
            else:
                output.append({"positive": 0.1, "neutral": 0.2, "negative": 0.7})
        return output


class TextTests(unittest.TestCase):
    def test_split_spans_preserves_offsets(self) -> None:
        text = "聯電失單，台積電可望受惠。"
        spans = split_spans(text)
        for value, start, end in spans:
            self.assertEqual(text[start:end], value)

    def test_target_context_masks_alias(self) -> None:
        result = target_context("台積電營收成長", TSMC)
        self.assertIn("目標公司營收成長", result)

    def test_extract_exact_evidence(self) -> None:
        text = "聯電失單，台積電可望受惠。"
        evidence = EvidenceExtractor().extract(
            text,
            entity=TSMC,
            label="positive",
            classifier=DummyClassifier(),
        )
        self.assertEqual(evidence[0].text, "受惠")
        self.assertEqual(text[evidence[0].start : evidence[0].end], "受惠")

    def test_neutral_has_no_evidence(self) -> None:
        self.assertEqual(
            EvidenceExtractor().extract("台積電召開董事會", entity=TSMC, label="neutral"), ()
        )


if __name__ == "__main__":
    unittest.main()

