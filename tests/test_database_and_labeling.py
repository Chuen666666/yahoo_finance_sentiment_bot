import csv
import tempfile
import unittest
from pathlib import Path

from yfsent.database import Database
from yfsent.domain import ContentRecord, Entity, EvidenceSpan, SentimentResult
from yfsent.entities import EntityResolver
from yfsent.labeling import (
    LABEL_FIELDS,
    export_label_template,
    filter_records_by_keywords,
    merge_label_files,
    read_label_pairs,
    write_results,
)


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

    def test_export_can_exclude_existing_item_entity_pair(self) -> None:
        entity = Entity("2330", "TWSE", "台灣積體電路製造股份有限公司", "台積電", ("台積電",))
        record = ContentRecord(
            id="item-1",
            kind="rss",
            title="台積電營運消息",
            text="台積電發布最新消息",
            url="https://example.test/1",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing_path = root / "existing.csv"
            output_path = root / "new.csv"
            with existing_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("item_id", "entity_code"))
                writer.writeheader()
                writer.writerow({"item_id": "item-1", "entity_code": "2330"})

            resolver = EntityResolver([entity])
            count = export_label_template(
                [record],
                resolver,
                output_path,
                excluded_pairs=read_label_pairs([existing_path]),
            )
            self.assertEqual(count, 0)

    def test_filter_records_by_keywords(self) -> None:
        records = [
            ContentRecord("1", "rss", "公司獲利成長", "", "https://example.test/1"),
            ContentRecord("2", "rss", "公司虧損擴大", "", "https://example.test/2"),
        ]
        filtered = filter_records_by_keywords(records, ["虧損", "下修"])
        self.assertEqual([record.id for record in filtered], ["2"])

    def test_merge_label_files_deduplicates_matching_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            output = root / "merged.csv"
            row = {field: "" for field in LABEL_FIELDS}
            row.update(
                {
                    "item_id": "item-1",
                    "entity_code": "2330",
                    "sentiment": "negative",
                }
            )
            for path in (first, second):
                with path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
                    writer.writeheader()
                    writer.writerow(row)
            stats = merge_label_files([first, second], output)
            self.assertEqual(stats, {"rows": 1, "duplicates": 1})
            with output.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_write_results_includes_raw_and_review_fields(self) -> None:
        record = ContentRecord(
            id="item-1",
            kind="rss",
            title="市場上漲",
            text="摘要沒有提到目標公司",
            url="https://example.test/1",
        )
        entity = Entity("2330", "TWSE", "台灣積體電路製造股份有限公司", "台積電")
        result = SentimentResult(
            record_id=record.id,
            entity=entity,
            label="uncertain",
            confidence=0.41,
            model_version="test-v2",
            raw_label="positive",
            review_required=True,
            target_mentioned=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            self.assertEqual(write_results([record], [result], output), 1)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["raw_sentiment"], "positive")
        self.assertEqual(row["final_sentiment"], "uncertain")
        self.assertEqual(row["sentiment"], "uncertain")
        self.assertEqual(row["review_required"], "true")
        self.assertEqual(row["target_mentioned"], "false")

    def test_database_lists_analysis_and_manages_watchlist(self) -> None:
        entity = Entity("2330", "TWSE", "台灣積體電路製造股份有限公司", "台積電")
        record = ContentRecord(
            id="item-1",
            kind="rss",
            title="台積電營收成長",
            text="台積電公布最新營收。",
            url="https://example.test/1",
            published_at="2026-09-22T01:00:00+00:00",
        )
        result = SentimentResult(
            record_id=record.id,
            entity=entity,
            label="positive",
            raw_label="positive",
            confidence=0.81,
            evidence=(EvidenceSpan("成長", 5, 7),),
            model_version="test-v2",
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            Database(Path(directory) / "test.db") as database,
        ):
            database.replace_entities([entity])
            database.upsert_contents([record])
            database.save_results([result])
            rows = database.list_analysis_rows(query="2330", sentiment="positive")
            database.add_watch("台積電", item_limit=20)
            watches = database.list_watchlist()
            database.update_watch_status("台積電", run_at="now", error="暫時失敗")
            updated = database.list_watchlist()
            database.remove_watch("台積電")
            removed = database.list_watchlist()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["raw_label"], "positive")
        self.assertEqual(rows[0]["evidence"][0]["text"], "成長")
        self.assertFalse(rows[0]["review_required"])
        self.assertEqual(watches[0]["item_limit"], 20)
        self.assertEqual(updated[0]["last_error"], "暫時失敗")
        self.assertEqual(removed, [])


if __name__ == "__main__":
    unittest.main()
