import unittest
from pathlib import Path

from yfsent.domain import ContentRecord, Entity
from yfsent.feeds import clean_html, matches_query, parse_feed

FIXTURE = Path(__file__).parent / "fixtures" / "sample_rss.xml"


class FeedTests(unittest.TestCase):
    def test_clean_html(self) -> None:
        self.assertEqual(clean_html("<p>獲利&nbsp;成長</p>"), "獲利 成長")

    def test_parse_rss(self) -> None:
        records = parse_feed(FIXTURE.read_bytes(), feed_url="fixture://rss")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].source, "測試財經網")
        self.assertEqual(records[0].text, "市場預期台積電營收將成長。")
        self.assertEqual(records[0].published_at, "2026-09-16T00:00:00+00:00")
        self.assertEqual(len(records[0].id), 24)

    def test_numeric_query_matches_company_name_not_unrelated_year(self) -> None:
        entity = Entity("2025", "TWSE", "千興不銹鋼股份有限公司", "千興", ("千興", "2025"))
        unrelated = ContentRecord("1", "rss", "2025 年市場展望", "", "https://example.test/1")
        related = ContentRecord("2", "rss", "千興發布財報", "", "https://example.test/2")
        self.assertFalse(matches_query(unrelated, "2025", entity))
        self.assertTrue(matches_query(related, "2025", entity))


if __name__ == "__main__":
    unittest.main()
