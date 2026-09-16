from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .domain import ContentRecord, Entity, SentimentResult

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS contents (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    published_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    code TEXT PRIMARY KEY,
    market TEXT NOT NULL,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    record_id TEXT NOT NULL,
    entity_code TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_json TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (record_id, entity_code, model_version),
    FOREIGN KEY (record_id) REFERENCES contents(id),
    FOREIGN KEY (entity_code) REFERENCES entities(code)
);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def upsert_contents(self, records: Iterable[ContentRecord]) -> int:
        rows = [
            (
                r.id,
                r.parent_id,
                r.kind,
                r.title,
                r.text,
                r.url,
                r.source,
                r.published_at,
                r.fetched_at,
            )
            for r in records
        ]
        self.connection.executemany(
            """
            INSERT INTO contents
                (id, parent_id, kind, title, text, url, source, published_at, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, text=excluded.text, url=excluded.url,
                source=excluded.source, published_at=excluded.published_at,
                fetched_at=excluded.fetched_at
            """,
            rows,
        )
        self.connection.commit()
        return len(rows)

    def list_contents(self, *, limit: int = 100) -> list[ContentRecord]:
        rows = self.connection.execute(
            """
            SELECT id, parent_id, kind, title, text, url, source, published_at, fetched_at
            FROM contents
            ORDER BY COALESCE(NULLIF(published_at, ''), fetched_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [ContentRecord(**dict(row)) for row in rows]

    def replace_entities(self, entities: Iterable[Entity]) -> int:
        rows = [
            (e.code, e.market, e.name, e.short_name, json.dumps(e.aliases, ensure_ascii=False))
            for e in entities
        ]
        self.connection.executemany(
            """
            INSERT INTO entities (code, market, name, short_name, aliases_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                market=excluded.market, name=excluded.name,
                short_name=excluded.short_name, aliases_json=excluded.aliases_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        self.connection.commit()
        return len(rows)

    def list_entities(self) -> list[Entity]:
        rows = self.connection.execute(
            "SELECT code, market, name, short_name, aliases_json FROM entities"
        ).fetchall()
        return [
            Entity(
                code=row["code"],
                market=row["market"],
                name=row["name"],
                short_name=row["short_name"],
                aliases=tuple(json.loads(row["aliases_json"])),
            )
            for row in rows
        ]

    def save_results(self, results: Iterable[SentimentResult]) -> int:
        rows = []
        for result in results:
            evidence = [
                {"text": span.text, "start": span.start, "end": span.end}
                for span in result.evidence
            ]
            rows.append(
                (
                    result.record_id,
                    result.entity.code,
                    result.label,
                    result.confidence,
                    json.dumps(evidence, ensure_ascii=False),
                    result.model_version,
                )
            )
        self.connection.executemany(
            """
            INSERT INTO predictions
                (record_id, entity_code, label, confidence, evidence_json, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id, entity_code, model_version) DO UPDATE SET
                label=excluded.label, confidence=excluded.confidence,
                evidence_json=excluded.evidence_json, created_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        self.connection.commit()
        return len(rows)

