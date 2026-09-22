import io
import json
import unittest
import zipfile

from yfsent.external import FINCHINA_TRAIN_MEMBER, parse_finchina_rows


def _archive(documents: list[dict[str, object]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(FINCHINA_TRAIN_MEMBER, json.dumps(documents, ensure_ascii=False))
    return buffer.getvalue()


class ExternalDatasetTests(unittest.TestCase):
    def test_parse_finchina_rows_maps_levels_and_checks_entity(self) -> None:
        payload = _archive(
            [
                {
                    "newscode": 1,
                    "title": "甲公司遭主管機關裁罰",
                    "text": "甲公司因違規被處分。",
                    "institution": [
                        {
                            "ins_name": "甲公司",
                            "sentiment_level": "-2",
                            "label_type": "行政處罰",
                        },
                        {"ins_name": "未出現公司", "sentiment_level": "-1"},
                    ],
                },
                {
                    "newscode": 2,
                    "title": "乙公司發布一般公告",
                    "text": "乙公司今天召開會議。",
                    "institution": [{"ins_name": "乙公司", "sentiment_level": "0"}],
                },
            ]
        )
        rows, stats = parse_finchina_rows(payload, max_per_class=None)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["sentiment"] for row in rows}, {"negative", "neutral"})
        self.assertEqual(stats["skipped"]["entity_not_in_text"], 1)

    def test_parse_finchina_rows_caps_each_class(self) -> None:
        documents = []
        for index in range(5):
            documents.append(
                {
                    "newscode": index,
                    "title": f"公司{index}虧損",
                    "text": f"公司{index}虧損擴大。",
                    "institution": [
                        {"ins_name": f"公司{index}", "sentiment_level": "-1"}
                    ],
                }
            )
        rows, stats = parse_finchina_rows(_archive(documents), max_per_class=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["raw_class_counts"]["negative"], 5)


if __name__ == "__main__":
    unittest.main()
