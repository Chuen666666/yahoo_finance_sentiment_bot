import csv
import tempfile
import unittest
from pathlib import Path

from yfsent.database import Database
from yfsent.domain import ContentRecord, Entity
from yfsent.entities import EntityResolver
from yfsent.labeling import export_label_template


class DatabaseAndLabelingTests(unittest.TestCase):
    def test_round_trip_and_label_export(self) -> None:
        entity = Entity(
            "2330",
            "TWSE",
            "台灣積體電路製造股份有限公司",
            "台積電",
            ("台積電", "2330"),
        )
        record = ContentRecord(
            id="item-1",
            kind="rss",
            title="台積電營收成長",
            text="市場看好後續表現。",
            url="https://example.test/1",
        )
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            output_path = Path(directory) / "labels.csv"
            with Database(db_path) as database:
                database.replace_entities([entity])
                database.upsert_contents([record])
                loaded = database.list_contents()
                self.assertEqual(loaded[0].title, record.title)
                resolver = EntityResolver(database.list_entities())
                count = export_label_template(loaded, resolver, output_path)
            self.assertEqual(count, 1)
            with output_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["entity_code"], "2330")
            self.assertEqual(rows[0]["sentiment"], "")


if __name__ == "__main__":
    unittest.main()
