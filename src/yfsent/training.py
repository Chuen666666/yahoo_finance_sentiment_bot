from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .domain import SENTIMENTS
from .sentiment import FINBERT_MODEL_ID, FinBertClassifier


class TrainingError(RuntimeError):
    pass


def _read_labels(
    path: str | Path,
) -> tuple[list[str], list[str], list[str], list[dict[str, str]]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    usable = [row for row in rows if row.get("sentiment", "").strip().casefold() in SENTIMENTS]
    if len(usable) < 30:
        raise TrainingError("至少需要 30 筆已標註資料；建議使用 300–500 筆")

    texts: list[str] = []
    labels: list[str] = []
    groups: list[str] = []
    for row in usable:
        label = row["sentiment"].strip().casefold()
        text = row.get("target_text", "").strip()
        if not text:
            text = (
                f"目標：{row.get('entity_name', '')}。"
                f"新聞：{row.get('title', '')}。{row.get('summary', '')}"
            )
        evidence = row.get("evidence_text", "").strip()
        original = f"{row.get('title', '')}。{row.get('summary', '')}"
        if evidence and evidence not in original:
            raise TrainingError(f"item_id={row.get('item_id')} 的 evidence_text 不在原文中")
        texts.append(text)
        labels.append(label)
        groups.append(row.get("item_id", "") or str(len(groups)))
    return texts, labels, groups, usable


def _metric_summary(labels: list[str], predictions: list[str]) -> dict[str, object]:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    return {
        "macro_f1": float(
            f1_score(labels, predictions, labels=list(SENTIMENTS), average="macro")
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(SENTIMENTS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(SENTIMENTS)).tolist(),
        "label_order": list(SENTIMENTS),
    }


def _build_lexicon(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    lexicon: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = row.get("sentiment", "").strip().casefold()
        evidence = row.get("evidence_text", "").strip(" ，,。；;：:")
        if label in {"positive", "negative"} and 1 < len(evidence) <= 40:
            lexicon[label].add(evidence)
    return {
        label: sorted(values, key=lambda value: (-len(value), value))
        for label, values in lexicon.items()
    }


def train_and_select(
    labels_path: str | Path,
    artifact_dir: str | Path,
    *,
    benchmark_finbert: bool = True,
) -> dict[str, object]:
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise TrainingError("訓練需要 ML 套件；請執行 pip install -e \".[ml]\"") from exc

    texts, labels, groups, rows = _read_labels(labels_path)
    counts = Counter(labels)
    missing = [label for label in SENTIMENTS if counts[label] == 0]
    if missing:
        raise TrainingError(f"標註資料缺少類別：{', '.join(missing)}")

    n_splits = min(5, min(counts.values()), len(set(groups)))
    if n_splits < 2:
        raise TrainingError("每個類別至少需要 2 筆，且必須來自不同 RSS 項目")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    tfidf_predictions = [""] * len(labels)
    for train_indices, test_indices in splitter.split(texts, labels, groups):
        pipeline = Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 5),
                        min_df=1,
                        max_features=100_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        train_texts = [texts[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        pipeline.fit(train_texts, train_labels)
        predicted = pipeline.predict([texts[index] for index in test_indices])
        for index, label in zip(test_indices, predicted, strict=True):
            tfidf_predictions[int(index)] = str(label)

    metrics: dict[str, object] = {
        "samples": len(labels),
        "class_counts": dict(counts),
        "folds": n_splits,
        "tfidf": _metric_summary(labels, tfidf_predictions),
    }
    candidates = [(float(metrics["tfidf"]["macro_f1"]), "tfidf")]

    if benchmark_finbert:
        finbert = FinBertClassifier(FINBERT_MODEL_ID)
        probabilities = finbert.predict_proba(texts)
        finbert_predictions = [max(row, key=row.get) for row in probabilities]
        metrics["finbert"] = _metric_summary(labels, finbert_predictions)
        candidates.append((float(metrics["finbert"]["macro_f1"]), "finbert"))

    candidates.sort(key=lambda item: (item[0], item[1] == "tfidf"), reverse=True)
    selected = candidates[0][1]
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    if selected == "tfidf":
        final_pipeline = Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 5),
                        min_df=1,
                        max_features=100_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        final_pipeline.fit(texts, labels)
        bundle = {"kind": "tfidf", "version": "tfidf-char-v1", "pipeline": final_pipeline}
    else:
        bundle = {
            "kind": "finbert",
            "version": f"finbert:{FINBERT_MODEL_ID}",
            "model_id": FINBERT_MODEL_ID,
        }

    metrics["selected_model"] = selected
    joblib.dump(bundle, artifact_path / "model.joblib")
    (artifact_path / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_path / "evidence_lexicon.json").write_text(
        json.dumps(_build_lexicon(rows), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
