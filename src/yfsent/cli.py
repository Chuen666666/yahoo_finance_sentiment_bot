from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .database import Database
from .domain import SentimentResult
from .entities import EntityResolver, load_manual_aliases, sync_official_entities
from .feeds import fetch_for_query
from .labeling import export_label_template, write_results
from .sentiment import ClassifierError, load_classifier
from .text import EvidenceExtractor, target_context
from .training import TrainingError, train_and_select

DEFAULT_DB = Path("data/yfsent.db")
DEFAULT_ALIASES = Path("data/aliases.json")
DEFAULT_MODEL = Path("artifacts/model.joblib")


def _resolver(database: Database, aliases_path: Path) -> EntityResolver:
    entities = database.list_entities()
    if not entities:
        raise RuntimeError("公司資料庫是空的；請先執行 sync-entities")
    return EntityResolver(entities, load_manual_aliases(aliases_path))


def _load_lexicon(model_path: Path) -> dict[str, list[str]]:
    path = model_path.parent / "evidence_lexicon.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [str(value) for value in values] for key, values in data.items()}


def command_sync(args: argparse.Namespace) -> None:
    entities = sync_official_entities()
    with Database(args.db) as database:
        count = database.replace_entities(entities)
    print(f"已同步 {count} 家上市櫃公司")


def command_fetch(args: argparse.Namespace) -> None:
    with Database(args.db) as database:
        resolver = _resolver(database, args.aliases)
        target = resolver.resolve_query(args.query)
        records = fetch_for_query(args.query, target=target, limit=args.limit)
        database.upsert_contents(records)
    print(f"已取得並儲存 {len(records)} 則 RSS 項目")


def command_export_labels(args: argparse.Namespace) -> None:
    with Database(args.db) as database:
        resolver = _resolver(database, args.aliases)
        records = database.list_contents(limit=args.limit)
        target = resolver.resolve_query(args.query) if args.query else None
        count = export_label_template(records, resolver, args.output, forced_target=target)
    print(f"已輸出 {count} 筆待標註資料至 {args.output}")


def command_train(args: argparse.Namespace) -> None:
    metrics = train_and_select(
        args.labels,
        args.artifact_dir,
        benchmark_finbert=not args.skip_finbert,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def command_analyze(args: argparse.Namespace) -> None:
    classifier = load_classifier(args.model)
    extractor = EvidenceExtractor(_load_lexicon(args.model))
    with Database(args.db) as database:
        resolver = _resolver(database, args.aliases)
        target = resolver.resolve_query(args.query)
        if args.offline:
            records = database.list_contents(limit=args.limit)
        else:
            records = fetch_for_query(args.query, target=target, limit=args.limit)
            database.upsert_contents(records)

        pairs = []
        for record in records:
            entities = {
                mention.entity.code: mention.entity
                for mention in resolver.find_mentions(record.combined_text)
            }
            if target:
                entities[target.code] = target
            for entity in entities.values():
                pairs.append((record, entity, target_context(record.combined_text, entity)))

        probabilities = classifier.predict_proba([pair[2] for pair in pairs])
        results: list[SentimentResult] = []
        for (record, entity, _), probability in zip(pairs, probabilities, strict=True):
            label = max(probability, key=probability.get)
            evidence = extractor.extract(
                record.combined_text,
                entity=entity,
                label=label,
                classifier=classifier,
            )
            results.append(
                SentimentResult(
                    record_id=record.id,
                    entity=entity,
                    label=label,
                    confidence=float(probability[label]),
                    evidence=evidence,
                    model_version=classifier.version,
                )
            )
        database.save_results(results)
    count = write_results(records, results, args.output)
    print(f"已分析 {count} 個新聞－公司組合，結果位於 {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yfsent",
        description="Yahoo 股市 RSS 上市櫃公司目標式情緒分析研究工具",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite 資料庫路徑")
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES, help="人工公司別名 JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    sync = commands.add_parser("sync-entities", help="同步 TWSE/TPEx 公司基本資料")
    sync.set_defaults(handler=command_sync)

    fetch = commands.add_parser("fetch", help="依股票代碼或關鍵字取得 RSS")
    fetch.add_argument("--query", required=True)
    fetch.add_argument("--limit", type=int, default=50)
    fetch.set_defaults(handler=command_fetch)

    labels = commands.add_parser("export-labels", help="輸出人工標註 CSV 範本")
    labels.add_argument("--query", help="強制加入指定公司作為情緒目標")
    labels.add_argument("--limit", type=int, default=500)
    labels.add_argument("--output", type=Path, default=Path("data/labels.csv"))
    labels.set_defaults(handler=command_export_labels)

    train = commands.add_parser("train", help="交叉驗證並選擇情緒模型")
    train.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    train.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    train.add_argument(
        "--skip-finbert",
        action="store_true",
        help="略過需要下載模型的 FinBERT 基線，只訓練 TF-IDF",
    )
    train.set_defaults(handler=command_train)

    analyze = commands.add_parser("analyze", help="取得或讀取 RSS 並輸出情緒分析 CSV")
    analyze.add_argument("--query", required=True)
    analyze.add_argument("--limit", type=int, default=50)
    analyze.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    analyze.add_argument("--output", type=Path, default=Path("data/output/results.csv"))
    analyze.add_argument("--offline", action="store_true", help="不連網，分析資料庫現有項目")
    analyze.set_defaults(handler=command_analyze)
    return parser


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ClassifierError, TrainingError, RuntimeError, ValueError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
