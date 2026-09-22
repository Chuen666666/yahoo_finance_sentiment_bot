from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .domain import Entity, EntityMention
from .net import get_bytes

TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
COMPANY_SUFFIXES = ("股份有限公司", "有限公司", "公司")


def _first(row: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip(" －-"):
            return str(value).strip().replace("\u3000", " ")
    return ""


def _strip_company_suffix(name: str) -> str:
    for suffix in COMPANY_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_company_rows(payload: bytes, market: str) -> list[Entity]:
    data = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("公司資料 API 回傳格式不是陣列")

    entities: list[Entity] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        code = _first(
            raw,
            "公司代號",
            "SecuritiesCompanyCode",
            "CompanyCode",
            "Code",
        )
        name = _first(raw, "公司名稱", "CompanyName")
        short_name = _first(
            raw,
            "公司簡稱",
            "SecuritiesCompanyName",
            "CompanyAbbreviation",
            "ShortName",
        )
        if not code or not name:
            continue
        if not short_name:
            short_name = _strip_company_suffix(name)
        aliases = tuple(
            dict.fromkeys(
                alias
                for alias in (name, short_name, _strip_company_suffix(name), code)
                if alias
            )
        )
        entities.append(
            Entity(code=code, market=market, name=name, short_name=short_name, aliases=aliases)
        )
    return entities


def sync_official_entities() -> list[Entity]:
    listed = parse_company_rows(get_bytes(TWSE_URL), "TWSE")
    otc = parse_company_rows(get_bytes(TPEX_URL), "TPEX")
    by_code = {entity.code: entity for entity in (*listed, *otc)}
    return sorted(by_code.values(), key=lambda entity: entity.code)


def load_manual_aliases(path: str | Path | None) -> dict[str, list[str]]:
    if path is None or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("別名檔必須是 {股票代碼: [別名]} 的 JSON 物件")
    return {str(code): [str(value) for value in values] for code, values in data.items()}


class EntityResolver:
    def __init__(
        self,
        entities: Iterable[Entity],
        manual_aliases: dict[str, list[str]] | None = None,
    ) -> None:
        manual_aliases = manual_aliases or {}
        self.entities = {entity.code: entity for entity in entities}
        alias_to_codes: dict[str, set[str]] = defaultdict(set)
        for entity in self.entities.values():
            aliases = (*entity.aliases, *manual_aliases.get(entity.code, []))
            for alias in aliases:
                alias = alias.strip()
                if alias and (len(alias) >= 2 or alias == entity.code):
                    alias_to_codes[alias].add(entity.code)
        self.alias_to_code = {
            alias: next(iter(codes)) for alias, codes in alias_to_codes.items() if len(codes) == 1
        }
        # Bare four-digit codes are indistinguishable from years and financial amounts
        # in general news. Codes remain valid for resolve_query(), while mention
        # detection relies on company names and non-numeric aliases.
        self.aliases = sorted(
            (alias for alias in self.alias_to_code if not alias.isdigit()),
            key=lambda value: (-len(value), value),
        )

    def resolve_query(self, query: str) -> Entity | None:
        normalized = query.strip()
        if normalized in self.entities:
            return self.entities[normalized]
        code = self.alias_to_code.get(normalized)
        return self.entities.get(code) if code else None

    def find_mentions(self, text: str) -> list[EntityMention]:
        candidates: list[EntityMention] = []
        for alias in self.aliases:
            code = self.alias_to_code[alias]
            if alias.isdigit():
                pattern = re.compile(rf"(?<![0-9A-Za-z]){re.escape(alias)}(?![0-9A-Za-z])")
            else:
                pattern = re.compile(re.escape(alias), re.IGNORECASE)
            for match in pattern.finditer(text):
                candidates.append(
                    EntityMention(
                        entity=self.entities[code],
                        surface=match.group(0),
                        start=match.start(),
                        end=match.end(),
                    )
                )

        candidates.sort(key=lambda mention: (mention.start, -(mention.end - mention.start)))
        selected: list[EntityMention] = []
        occupied: list[tuple[int, int]] = []
        seen_entities: set[str] = set()
        for mention in candidates:
            overlaps = any(mention.start < end and mention.end > start for start, end in occupied)
            if overlaps:
                continue
            occupied.append((mention.start, mention.end))
            if mention.entity.code in seen_entities:
                continue
            selected.append(mention)
            seen_entities.add(mention.entity.code)
        return sorted(selected, key=lambda mention: mention.start)
