from __future__ import annotations

import csv
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


def export_label_template(
    records: Iterable[ContentRecord],
    resolver: EntityResolver,
    output_path: str | Path,
    *,
    forced_target: Entity | None = None,
) -> int:
    rows: list[dict[str, str]] = []
    for record in records:
        mentions = resolver.find_mentions(record.combined_text)
        entities = {mention.entity.code: mention.entity for mention in mentions}
        if forced_target:
            entities[forced_target.code] = forced_target
        for entity in entities.values():
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
        "sentiment",
        "confidence",
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
                "sentiment": result.label,
                "confidence": f"{result.confidence:.6f}",
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
