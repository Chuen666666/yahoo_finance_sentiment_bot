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


if __name__ == "__main__":
    unittest.main()
