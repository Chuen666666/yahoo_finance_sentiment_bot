from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze_records
from .database import Database
from .entities import EntityResolver, load_manual_aliases, sync_official_entities
from .external import download_finchina_labels
from .feeds import fetch_for_query, matches_query
from .labeling import (
    export_label_template,
    filter_records_by_keywords,
    merge_label_files,
    read_label_pairs,
    write_results,
)
from .sentiment import ClassifierError, load_classifier
from .text import EvidenceExtractor
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
        if args.query:
            records = [record for record in records if matches_query(record, args.query, target)]
        records = filter_records_by_keywords(records, args.keyword)
        count = export_label_template(
            records,
            resolver,
            args.output,
            forced_target=target,
            excluded_pairs=read_label_pairs(args.exclude_labels),
        )
    print(f"已輸出 {count} 筆待標註資料至 {args.output}")


def command_import_finchina(args: argparse.Namespace) -> None:
    stats = download_finchina_labels(
        args.output,
        max_per_class=args.max_per_class,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def command_merge_labels(args: argparse.Namespace) -> None:
    stats = merge_label_files(args.input, args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def command_train(args: argparse.Namespace) -> None:
    metrics = train_and_select(
        args.labels,
        args.artifact_dir,
        benchmark_finbert=not args.skip_finbert,
        auxiliary_paths=args.aux_labels,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def command_analyze(args: argparse.Namespace) -> None:
    classifier = load_classifier(args.model)
    confidence_threshold = (
        classifier.confidence_threshold
        if args.confidence_threshold is None
        else args.confidence_threshold
    )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence-threshold 必須介於 0 與 1 之間")
    extractor = EvidenceExtractor(_load_lexicon(args.model))
    with Database(args.db) as database:
        resolver = _resolver(database, args.aliases)
        target = resolver.resolve_query(args.query)
        if args.offline:
            records = [
                record
                for record in database.list_contents(limit=max(10_000, args.limit))
                if matches_query(record, args.query, target)
            ][: args.limit]
        else:
            records = fetch_for_query(args.query, target=target, limit=args.limit)
            database.upsert_contents(records)

        results = analyze_records(
            records,
            resolver=resolver,
            classifier=classifier,
            extractor=extractor,
            target=target,
            target_only=args.target_only,
            confidence_threshold=confidence_threshold,
        )
        database.save_results(results)
    count = write_results(records, results, args.output)
    print(
        f"已分析 {count} 個新聞－公司組合，門檻 {confidence_threshold:.2f}，"
        f"結果位於 {args.output}"
    )


def command_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('網站功能需要額外套件；請執行 pip install -e ".[web,ml]"') from exc

    from .web import create_app

    app = create_app(
        db_path=args.db,
        aliases_path=args.aliases,
        model_path=args.model,
        refresh_minutes=args.refresh_minutes,
    )
    uvicorn.run(app, host=args.host, port=args.port)


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
    labels.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="只輸出內文包含指定關鍵字的項目；可重複使用",
    )
    labels.add_argument(
        "--exclude-labels",
        type=Path,
        action="append",
        default=[],
        help="略過指定 CSV 中已有的 item_id/entity_code；可重複使用",
    )
    labels.set_defaults(handler=command_export_labels)

    finchina = commands.add_parser(
        "import-finchina",
        help="下載 FinChina-SA 機構情緒資料並轉換為輔助訓練 CSV",
    )
    finchina.add_argument(
        "--output",
        type=Path,
        default=Path("data/auxiliary/finchina_train.csv"),
    )
    finchina.add_argument(
        "--max-per-class",
        type=int,
        default=600,
        help="每種情緒最多保留幾筆；預設 600，避免外部資料壓過 Yahoo 標註",
    )
    finchina.set_defaults(handler=command_import_finchina)

    merge = commands.add_parser("merge-labels", help="去重合併已完成的標註 CSV")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, default=Path("data/labels.csv"))
    merge.set_defaults(handler=command_merge_labels)

    train = commands.add_parser("train", help="交叉驗證並選擇情緒模型")
    train.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    train.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    train.add_argument(
        "--aux-labels",
        type=Path,
        action="append",
        default=[],
        help="只加入訓練折、不加入驗證折的輔助標註 CSV；可重複使用",
    )
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
    analyze.add_argument(
        "--target-only",
        action="store_true",
        help="公司 query 僅輸出該公司，不輸出同篇新聞中的其他公司",
    )
    analyze.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="低於此值時設為 uncertain；預設使用訓練時校準的模型門檻",
    )
    analyze.set_defaults(handler=command_analyze)

    serve = commands.add_parser("serve", help="啟動新聞情緒網站與自動更新排程")
    serve.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--refresh-minutes",
        type=float,
        default=30,
        help="自動更新追蹤條件的分鐘間隔；設為 0 可停用",
    )
    serve.set_defaults(handler=command_serve)
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
