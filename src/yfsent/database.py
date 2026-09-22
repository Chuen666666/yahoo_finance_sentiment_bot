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
    raw_label TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL,
    review_required INTEGER NOT NULL DEFAULT 0,
    target_mentioned INTEGER NOT NULL DEFAULT 1,
    evidence_json TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (record_id, entity_code, model_version),
    FOREIGN KEY (record_id) REFERENCES contents(id),
    FOREIGN KEY (entity_code) REFERENCES entities(code)
);

CREATE TABLE IF NOT EXISTS watchlist (
    query TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 1,
    item_limit INTEGER NOT NULL DEFAULT 50,
    last_run_at TEXT NOT NULL DEFAULT '',
    last_error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(predictions)").fetchall()
        }
        additions = {
            "raw_label": "TEXT NOT NULL DEFAULT ''",
            "review_required": "INTEGER NOT NULL DEFAULT 0",
            "target_mentioned": "INTEGER NOT NULL DEFAULT 1",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self.connection.execute(
                    f"ALTER TABLE predictions ADD COLUMN {name} {declaration}"
                )
        self.connection.commit()

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
                    result.raw_label or result.label,
                    result.confidence,
                    int(result.review_required),
                    int(result.target_mentioned),
                    json.dumps(evidence, ensure_ascii=False),
                    result.model_version,
                )
            )
        self.connection.executemany(
            """
            INSERT INTO predictions
                (record_id, entity_code, label, raw_label, confidence, review_required,
                 target_mentioned, evidence_json, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id, entity_code, model_version) DO UPDATE SET
                label=excluded.label, raw_label=excluded.raw_label,
                confidence=excluded.confidence,
                review_required=excluded.review_required,
                target_mentioned=excluded.target_mentioned,
                evidence_json=excluded.evidence_json, created_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        self.connection.commit()
        return len(rows)

    def list_analysis_rows(
        self,
        *,
        query: str = "",
        sentiment: str = "",
        limit: int = 100,
    ) -> list[dict[str, object]]:
        where = [
            """p.rowid = (
                SELECT p2.rowid FROM predictions p2
                WHERE p2.record_id = p.record_id AND p2.entity_code = p.entity_code
                ORDER BY p2.created_at DESC, p2.rowid DESC LIMIT 1
            )"""
        ]
        parameters: list[object] = []
        normalized_query = query.strip()
        if normalized_query:
            pattern = f"%{normalized_query}%"
            where.append(
                "(c.title LIKE ? OR c.text LIKE ? OR e.code LIKE ? OR e.short_name LIKE ?)"
            )
            parameters.extend([pattern, pattern, pattern, pattern])
        if sentiment:
            where.append("p.label = ?")
            parameters.append(sentiment)
        parameters.append(limit)
        rows = self.connection.execute(
            f"""
            SELECT c.id, c.title, c.text AS summary, c.url, c.source, c.published_at,
                   e.code AS entity_code, e.short_name AS entity_name,
                   p.label AS sentiment, p.raw_label, p.confidence,
                   p.review_required, p.target_mentioned, p.evidence_json,
                   p.model_version, p.created_at AS analyzed_at
            FROM predictions p
            JOIN contents c ON c.id = p.record_id
            JOIN entities e ON e.code = p.entity_code
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(NULLIF(c.published_at, ''), c.fetched_at) DESC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        output: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            evidence = json.loads(str(item.pop("evidence_json")))
            item["evidence"] = evidence
            item["review_required"] = bool(item["review_required"])
            item["target_mentioned"] = bool(item["target_mentioned"])
            output.append(item)
        return output

    def add_watch(self, query: str, *, item_limit: int = 50) -> None:
        normalized = query.strip()
        if not normalized:
            raise ValueError("追蹤條件不可為空白")
        self.connection.execute(
            """
            INSERT INTO watchlist (query, item_limit) VALUES (?, ?)
            ON CONFLICT(query) DO UPDATE SET enabled=1, item_limit=excluded.item_limit
            """,
            (normalized, item_limit),
        )
        self.connection.commit()

    def remove_watch(self, query: str) -> None:
        self.connection.execute("DELETE FROM watchlist WHERE query = ?", (query.strip(),))
        self.connection.commit()

    def list_watchlist(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT query, enabled, item_limit, last_run_at, last_error, created_at
            FROM watchlist WHERE enabled = 1 ORDER BY created_at, query
            """
        ).fetchall()
        return [
            {**dict(row), "enabled": bool(row["enabled"])}
            for row in rows
        ]

    def update_watch_status(self, query: str, *, run_at: str, error: str = "") -> None:
        self.connection.execute(
            "UPDATE watchlist SET last_run_at = ?, last_error = ? WHERE query = ?",
            (run_at, error, query),
        )
        self.connection.commit()
