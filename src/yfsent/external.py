from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .labeling import LABEL_FIELDS
from .net import get_bytes

FINCHINA_REPOSITORY = "https://github.com/YerayL/FinChina-SA"
FINCHINA_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/YerayL/FinChina-SA/main/FinChina%20SA.zip"
)
FINCHINA_TRAIN_MEMBER = "FinChina SA/train.json"

_LEVEL_TO_LABEL = {
    "-3": "negative",
    "-2": "negative",
    "-1": "negative",
    "0": "neutral",
    "1": "positive",
    "2": "positive",
}


def parse_finchina_rows(
    payload: bytes,
    *,
    max_per_class: int | None = 600,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Convert FinChina-SA's institution annotations to this project's label schema."""
    if max_per_class is not None and max_per_class < 1:
        raise ValueError("max_per_class must be at least 1")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            documents = json.loads(archive.read(FINCHINA_TRAIN_MEMBER))
    except (KeyError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid FinChina-SA archive: {exc}") from exc
    if not isinstance(documents, list):
        raise ValueError("FinChina-SA train.json must contain a JSON list")

    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    skipped = Counter()
    seen: set[tuple[str, str]] = set()

    for document in documents:
        if not isinstance(document, dict):
            skipped["invalid_document"] += 1
            continue
        title = str(document.get("title") or "").strip()
        summary = str(document.get("text") or "").strip()
        combined_text = "\n".join(value for value in (title, summary) if value)
        if not combined_text:
            skipped["empty_text"] += 1
            continue
        raw_item_id = str(document.get("newscode") or "").strip()
        item_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()[:24]
        item_id = f"finchina-train-{raw_item_id or item_hash}"

        institutions = document.get("institution") or []
        if not isinstance(institutions, list):
            skipped["invalid_institutions"] += 1
            continue
        for annotation in institutions:
            if not isinstance(annotation, dict):
                skipped["invalid_annotation"] += 1
                continue
            entity_name = str(annotation.get("ins_name") or "").strip()
            level = str(annotation.get("sentiment_level") or "").strip()
            sentiment = _LEVEL_TO_LABEL.get(level)
            if not entity_name or not sentiment:
                skipped["incomplete_or_unknown_label"] += 1
                continue
            if entity_name not in combined_text:
                skipped["entity_not_in_text"] += 1
                continue

            entity_hash = hashlib.sha256(entity_name.encode("utf-8")).hexdigest()[:12]
            entity_code = f"finchina:{entity_hash}"
            pair = (item_id, entity_code)
            if pair in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(pair)
            label_type = str(annotation.get("label_type") or "").strip()
            row = {
                "item_id": item_id,
                "published_at": "",
                "source": "FinChina-SA/train",
                "url": FINCHINA_REPOSITORY,
                "title": title,
                "summary": summary,
                "entity_code": entity_code,
                "entity_name": entity_name,
                "target_text": f"目標：{entity_name}\n內文：{combined_text}",
                "sentiment": sentiment,
                "evidence_text": "",
                "notes": (
                    "auxiliary_only; split=train; license=Apache-2.0; script=zh-Hans; "
                    f"original_level={level}; label_type={label_type}"
                ),
            }
            candidates[sentiment].append(row)

    raw_counts = {label: len(rows) for label, rows in candidates.items()}
    selected: list[dict[str, str]] = []
    for rows in candidates.values():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{row['item_id']}|{row['entity_code']}".encode()
            ).hexdigest()
        )
        selected.extend(rows if max_per_class is None else rows[:max_per_class])
    selected.sort(key=lambda row: (row["item_id"], row["entity_code"]))

    stats: dict[str, object] = {
        "rows": len(selected),
        "class_counts": dict(Counter(row["sentiment"] for row in selected)),
        "raw_class_counts": raw_counts,
        "max_per_class": max_per_class,
        "skipped": dict(skipped),
        "dataset": FINCHINA_REPOSITORY,
        "license": "Apache-2.0",
    }
    return selected, stats


def download_finchina_labels(
    output_path: str | Path,
    *,
    max_per_class: int | None = 600,
) -> dict[str, object]:
    payload = get_bytes(FINCHINA_ARCHIVE_URL, timeout=90.0)
    rows, stats = parse_finchina_rows(payload, max_per_class=max_per_class)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return stats
