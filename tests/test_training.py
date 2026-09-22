import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from yfsent.sentiment import load_classifier
from yfsent.training import TrainingError, train_and_select


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class TrainingTests(unittest.TestCase):
    def test_tfidf_training_and_loading(self) -> None:
        examples = {
            "positive": ("目標公司營收成長，市場看好後市", "營收成長"),
            "negative": ("目標公司營收下滑，市場擔憂後市", "營收下滑"),
            "neutral": ("目標公司今日召開例行董事會", ""),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.csv"
            auxiliary_path = root / "auxiliary.csv"
            fields = (
                "item_id",
                "entity_name",
                "title",
                "summary",
                "target_text",
                "sentiment",
                "evidence_text",
            )
            with labels_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for label, (text, evidence) in examples.items():
                    for index in range(12):
                        writer.writerow(
                            {
                                "item_id": f"{label}-{index}",
                                "entity_name": "測試公司",
                                "title": text,
                                "summary": "",
                                "target_text": f"目標：測試公司。新聞：{text}",
                                "sentiment": label,
                                "evidence_text": evidence,
                            }
                        )

            with auxiliary_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(30):
                    label = ("positive", "negative", "neutral")[index % 3]
                    writer.writerow(
                        {
                            "item_id": f"aux-{index}",
                            "entity_name": "輔助公司",
                            "title": f"輔助資料 {label} {index}",
                            "summary": "",
                            "target_text": f"目標：輔助公司；內文：{label} {index}",
                            "sentiment": label,
                            "evidence_text": "",
                        }
                    )

            metrics = train_and_select(
                labels_path,
                root / "artifacts",
                benchmark_finbert=False,
                auxiliary_paths=[auxiliary_path],
            )
            self.assertEqual(metrics["selected_model"], "tfidf")
            self.assertEqual(metrics["auxiliary_samples"], 30)
            self.assertIn("threshold", metrics["confidence_policy"])
            self.assertGreater(float(metrics["tfidf"]["macro_f1"]), 0.9)
            classifier = load_classifier(root / "artifacts" / "model.joblib")
            self.assertGreaterEqual(classifier.confidence_threshold, 0.34)
            probabilities = classifier.predict_proba(["目標公司營收大幅成長"])[0]
            self.assertEqual(set(probabilities), {"negative", "neutral", "positive"})


    def test_training_rejects_single_item_group_for_a_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.csv"
            fields = ("item_id", "title", "target_text", "sentiment", "evidence_text")
            with labels_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for label in ("positive", "neutral"):
                    for index in range(15):
                        writer.writerow(
                            {
                                "item_id": f"{label}-{index}",
                                "title": label,
                                "target_text": label,
                                "sentiment": label,
                                "evidence_text": "",
                            }
                        )
                for index in range(5):
                    writer.writerow(
                        {
                            "item_id": "one-negative-article",
                            "title": f"negative {index}",
                            "target_text": f"negative {index}",
                            "sentiment": "negative",
                            "evidence_text": "",
                        }
                    )

            with self.assertRaisesRegex(TrainingError, "negative=1"):
                train_and_select(labels_path, root / "artifacts", benchmark_finbert=False)


if __name__ == "__main__":
    unittest.main()
