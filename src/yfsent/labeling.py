from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .domain import ContentRecord, Entity, SentimentResult
from .entities import EntityResolver
from .text import target_context

LABEL_FIELDS = (
    "item_id",
    "published_at",
    "source",
    "url",
    "title",
    "summary",
    "entity_code",
    "entity_name",
    "target_text",
    "sentiment",
    "evidence_text",
    "notes",
)


def read_label_pairs(paths: Iterable[str | Path]) -> set[tuple[str, str]]:
    """Return item/entity pairs already present in one or more label CSV files."""
    pairs: set[tuple[str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise ValueError(f"Label CSV does not exist: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                item_id = row.get("item_id", "").strip()
                entity_code = row.get("entity_code", "").strip()
                if item_id and entity_code:
                    pairs.add((item_id, entity_code))
    return pairs


def filter_records_by_keywords(
    records: Iterable[ContentRecord], keywords: Iterable[str]
) -> list[ContentRecord]:
    terms = tuple(term.strip().casefold() for term in keywords if term.strip())
    if not terms:
        return list(records)
    return [
        record
        for record in records
        if any(term in record.combined_text.casefold() for term in terms)
    ]


def merge_label_files(
    input_paths: Iterable[str | Path], output_path: str | Path
) -> dict[str, int]:
    """Merge label CSV files by item/entity pair and reject conflicting labels."""
    merged: dict[tuple[str, str], dict[str, str]] = {}
    duplicate_count = 0
    for raw_path in input_paths:
        path = Path(raw_path)
        if not path.exists():
            raise ValueError(f"Label CSV does not exist: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row_number, raw_row in enumerate(csv.DictReader(handle), start=2):
                row = {field: str(raw_row.get(field) or "") for field in LABEL_FIELDS}
                item_id = row["item_id"].strip()
                entity_code = row["entity_code"].strip()
                sentiment = row["sentiment"].strip().casefold()
                if not item_id or not entity_code:
                    raise ValueError(f"{path}:{row_number} is missing item_id or entity_code")
                if sentiment not in {"negative", "neutral", "positive"}:
                    raise ValueError(f"{path}:{row_number} has invalid sentiment: {sentiment!r}")
                row["sentiment"] = sentiment
                key = (item_id, entity_code)
                previous = merged.get(key)
                if previous is None:
                    merged[key] = row
                    continue
                duplicate_count += 1
                if previous["sentiment"] != sentiment:
                    raise ValueError(
                        f"Conflicting sentiment for item_id={item_id}, entity_code={entity_code}"
                    )
                if not previous["evidence_text"] and row["evidence_text"]:
                    merged[key] = row

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
            writer.writeheader()
            writer.writerows(merged.values())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return {"rows": len(merged), "duplicates": duplicate_count}


def export_label_template(
    records: Iterable[ContentRecord],
    resolver: EntityResolver,
    output_path: str | Path,
    *,
    forced_target: Entity | None = None,
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> int:
    excluded_pairs = excluded_pairs or set()
    rows: list[dict[str, str]] = []
    for record in records:
        mentions = resolver.find_mentions(record.combined_text)
        entities = {mention.entity.code: mention.entity for mention in mentions}
        if forced_target:
            entities[forced_target.code] = forced_target
        for entity in entities.values():
            if (record.id, entity.code) in excluded_pairs:
                continue
            rows.append(
                {
                    "item_id": record.id,
                    "published_at": record.published_at,
                    "source": record.source,
                    "url": record.url,
                    "title": record.title,
                    "summary": record.text,
                    "entity_code": entity.code,
                    "entity_name": entity.short_name,
                    "target_text": target_context(record.combined_text, entity),
                    "sentiment": "",
                    "evidence_text": "",
                    "notes": "",
                }
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_results(
    records: Iterable[ContentRecord],
    results: Iterable[SentimentResult],
    output_path: str | Path,
) -> int:
    record_by_id = {record.id: record for record in records}
    fields = (
        "published_at",
        "source",
        "title",
        "summary",
        "url",
        "entity_code",
        "entity_name",
        "raw_sentiment",
        "final_sentiment",
        "sentiment",
        "confidence",
        "review_required",
        "target_mentioned",
        "evidence_text",
        "evidence_start",
        "evidence_end",
        "model_version",
    )
    rows = []
    for result in results:
        record = record_by_id[result.record_id]
        evidence = result.evidence[0] if result.evidence else None
        rows.append(
            {
                "published_at": record.published_at,
                "source": record.source,
                "title": record.title,
                "summary": record.text,
                "url": record.url,
                "entity_code": result.entity.code,
                "entity_name": result.entity.short_name,
                "raw_sentiment": result.raw_label or result.label,
                "final_sentiment": result.label,
                "sentiment": result.label,
                "confidence": f"{result.confidence:.6f}",
                "review_required": str(result.review_required).lower(),
                "target_mentioned": str(result.target_mentioned).lower(),
                "evidence_text": evidence.text if evidence else "",
                "evidence_start": evidence.start if evidence else "",
                "evidence_end": evidence.end if evidence else "",
                "model_version": result.model_version,
            }
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
