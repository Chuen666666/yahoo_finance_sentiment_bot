import json
import unittest

from yfsent.domain import Entity
from yfsent.entities import EntityResolver, parse_company_rows


class EntityTests(unittest.TestCase):
    def test_parse_twse_and_tpex_shapes(self) -> None:
        twse = json.dumps(
            [
                {
                    "公司代號": "2330",
                    "公司名稱": "台灣積體電路製造股份有限公司",
                    "公司簡稱": "台積電",
                }
            ],
            ensure_ascii=False,
        ).encode()
        tpex = json.dumps(
            [
                {
                    "SecuritiesCompanyCode": "6488",
                    "CompanyName": "環球晶圓股份有限公司",
                    "SecuritiesCompanyName": "環球晶",
                }
            ],
            ensure_ascii=False,
        ).encode()
        self.assertEqual(parse_company_rows(twse, "TWSE")[0].short_name, "台積電")
        self.assertEqual(parse_company_rows(tpex, "TPEX")[0].code, "6488")

    def test_resolver_prefers_longest_and_deduplicates_entity(self) -> None:
        entity = Entity(
            code="2330",
            market="TWSE",
            name="台灣積體電路製造股份有限公司",
            short_name="台積電",
            aliases=("台灣積體電路製造股份有限公司", "台積電", "2330"),
        )
        resolver = EntityResolver([entity])
        mentions = resolver.find_mentions("台積電（2330）今日上漲")
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].surface, "台積電")
        self.assertEqual(resolver.resolve_query("2330"), entity)

    def test_ambiguous_alias_is_ignored(self) -> None:
        first = Entity("1000", "TWSE", "甲股份有限公司", "同名", ("同名", "1000"))
        second = Entity("2000", "TPEX", "乙股份有限公司", "同名", ("同名", "2000"))
        resolver = EntityResolver([first, second])
        self.assertEqual(resolver.find_mentions("同名公司發布新聞"), [])

    def test_repeated_long_alias_still_masks_shorter_company(self) -> None:
        long_entity = Entity("3008", "TWSE", "大立光電股份有限公司", "大立光", ("大立光",))
        short_entity = Entity("4716", "TWSE", "大立高分子工業股份有限公司", "大立", ("大立",))
        resolver = EntityResolver([long_entity, short_entity])
        mentions = resolver.find_mentions("大立光跌停後，大立光股價回穩")
        self.assertEqual([mention.entity.code for mention in mentions], ["3008"])

    def test_bare_numeric_code_is_queryable_but_not_a_mention(self) -> None:
        entity = Entity("2025", "TWSE", "千興不銹鋼股份有限公司", "千興", ("千興", "2025"))
        resolver = EntityResolver([entity])
        self.assertEqual(resolver.resolve_query("2025"), entity)
        self.assertEqual(resolver.find_mentions("截至 2025 年為止"), [])


if __name__ == "__main__":
    unittest.main()
