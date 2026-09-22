from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .domain import ContentRecord, Entity
from .net import get_bytes

CATEGORY_FEEDS = {
    "news": "https://tw.stock.yahoo.com/rss?category=news",
    "tw-market": "https://tw.stock.yahoo.com/rss?category=tw-market",
    "intl-markets": "https://tw.stock.yahoo.com/rss?category=intl-markets",
    "personal-finance": "https://tw.stock.yahoo.com/rss?category=personal-finance",
    "funds-news": "https://tw.stock.yahoo.com/rss?category=funds-news",
    "column": "https://tw.stock.yahoo.com/rss?category=column",
    "research": "https://tw.stock.yahoo.com/rss?category=research",
}

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def clean_html(value: str) -> str:
    value = html.unescape(value or "")
    value = _TAG_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ET.Element, *names: str) -> str:
    wanted = {name.lower() for name in names}
    for child in element:
        if _local_name(child.tag) in wanted:
            return "".join(child.itertext()).strip()
    return ""


def _normalize_date(value: str) -> str:
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                timezone.utc
            ).isoformat()
        except ValueError:
            return value


def parse_feed(payload: bytes, *, feed_url: str = "") -> list[ContentRecord]:
    root = ET.fromstring(payload)
    channel = next((node for node in root.iter() if _local_name(node.tag) == "channel"), root)
    channel_title = clean_html(_child_text(channel, "title")) or "Yahoo股市"
    fetched_at = datetime.now(timezone.utc).isoformat()
    records: list[ContentRecord] = []

    for item in root.iter():
        if _local_name(item.tag) not in {"item", "entry"}:
            continue
        title = clean_html(_child_text(item, "title"))
        summary = clean_html(_child_text(item, "description", "summary", "content", "encoded"))
        link = _child_text(item, "link")
        if not link:
            for child in item:
                if _local_name(child.tag) == "link" and child.attrib.get("href"):
                    link = child.attrib["href"]
                    break
        guid = _child_text(item, "guid", "id")
        published = _normalize_date(_child_text(item, "pubdate", "published", "updated"))
        source = clean_html(_child_text(item, "source")) or channel_title
        if not title:
            continue
        identity = guid or link or f"{title}|{published}|{feed_url}"
        record_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        records.append(
            ContentRecord(
                id=record_id,
                kind="rss",
                title=title,
                text=summary,
                url=link,
                source=source,
                published_at=published,
                fetched_at=fetched_at,
            )
        )
    return records


def matches_query(record: ContentRecord, query: str, target: Entity | None) -> bool:
    haystack = record.combined_text.casefold()
    normalized_query = query.strip().casefold()
    terms = {normalized_query} if normalized_query and not normalized_query.isdigit() else set()
    if target:
        terms.update(
            alias.casefold()
            for alias in target.aliases
            if len(alias) >= 2 and not alias.isdigit()
        )
    return any(term and term in haystack for term in terms)


def fetch_for_query(
    query: str, *, target: Entity | None = None, limit: int = 50
) -> list[ContentRecord]:
    if target or query.strip().isdigit():
        code = target.code if target else query.strip()
        urls = [f"https://tw.stock.yahoo.com/rss/s/{code}"]
        local_filter = False
    else:
        urls = list(CATEGORY_FEEDS.values())
        local_filter = True

    deduplicated: dict[str, ContentRecord] = {}
    for url in urls:
        for record in parse_feed(get_bytes(url), feed_url=url):
            if not local_filter or matches_query(record, query, target):
                deduplicated[record.id] = record
    records = sorted(
        deduplicated.values(),
        key=lambda record: (record.published_at, record.id),
        reverse=True,
    )
    return records[:limit]
