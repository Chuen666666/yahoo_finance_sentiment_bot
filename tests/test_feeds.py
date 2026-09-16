import unittest
from pathlib import Path

from yfsent.feeds import clean_html, parse_feed

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


if __name__ == "__main__":
    unittest.main()

