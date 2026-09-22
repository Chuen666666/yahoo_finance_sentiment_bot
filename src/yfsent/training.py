from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from .domain import SENTIMENTS, Entity
from .sentiment import FINBERT_MODEL_ID, FinBertClassifier
from .text import target_context


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
        original = f"{row.get('title', '')}。{row.get('summary', '')}"
        entity_name = row.get("entity_name", "").strip()
        entity_code = row.get("entity_code", "").strip()
        entity = Entity(
            code=entity_code,
            market="",
            name=entity_name,
            short_name=entity_name,
            aliases=tuple(value for value in (entity_name, entity_code) if value),
        )
        text = target_context(original, entity, require_mention=True) if entity_name else ""
        if not text:
            text = row.get("target_text", "").strip()
        if not text:
            text = (
                f"目標：{entity_name}。"
                f"新聞：{row.get('title', '')}。{row.get('summary', '')}"
            )
        evidence = row.get("evidence_text", "").strip()
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


def _calibrate_confidence_threshold(
    labels: list[str],
    predictions: list[str],
    confidences: list[float],
    *,
    target_accuracy: float = 0.80,
) -> dict[str, float | int]:
    minimum_samples = max(10, round(len(labels) * 0.10))
    candidates: list[dict[str, float | int]] = []
    for step in range(34, 91):
        threshold = step / 100
        selected = [index for index, value in enumerate(confidences) if value >= threshold]
        if len(selected) < minimum_samples:
            continue
        correct = sum(predictions[index] == labels[index] for index in selected)
        candidates.append(
            {
                "threshold": threshold,
                "selected_samples": len(selected),
                "coverage": len(selected) / len(labels),
                "accuracy": correct / len(selected),
            }
        )

    viable = [row for row in candidates if float(row["accuracy"]) >= target_accuracy]
    if viable:
        return max(viable, key=lambda row: (float(row["coverage"]), -float(row["threshold"])))
    if candidates:
        return max(candidates, key=lambda row: (float(row["accuracy"]), float(row["coverage"])))
    return {
        "threshold": 0.55,
        "selected_samples": 0,
        "coverage": 0.0,
        "accuracy": 0.0,
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
    auxiliary_paths: Iterable[str | Path] = (),
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
    auxiliary_texts: list[str] = []
    auxiliary_labels: list[str] = []
    auxiliary_rows: list[dict[str, str]] = []
    for auxiliary_path in auxiliary_paths:
        aux_texts, aux_labels, _, aux_rows = _read_labels(auxiliary_path)
        auxiliary_texts.extend(aux_texts)
        auxiliary_labels.extend(aux_labels)
        auxiliary_rows.extend(aux_rows)
    counts = Counter(labels)
    missing = [label for label in SENTIMENTS if counts[label] == 0]
    if missing:
        raise TrainingError(f"標註資料缺少類別：{', '.join(missing)}")

    class_group_counts = {
        label: len(
            {
                group
                for group, row_label in zip(groups, labels, strict=True)
                if row_label == label
            }
        )
        for label in SENTIMENTS
    }
    n_splits = min(5, min(class_group_counts.values()))
    if n_splits < 2:
        details = ", ".join(f"{label}={count}" for label, count in class_group_counts.items())
        raise TrainingError(
            f"每類至少需要 2 篇不同 item_id 才能做群組交叉驗證；目前 {details}"
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    tfidf_predictions = [""] * len(labels)
    tfidf_confidences = [0.0] * len(labels)
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
        pipeline.fit(train_texts + auxiliary_texts, train_labels + auxiliary_labels)
        fold_probabilities = pipeline.predict_proba([texts[index] for index in test_indices])
        classes = [str(value) for value in pipeline.classes_]
        for index, probability in zip(test_indices, fold_probabilities, strict=True):
            best_index = int(probability.argmax())
            tfidf_predictions[int(index)] = classes[best_index]
            tfidf_confidences[int(index)] = float(probability[best_index])

    confidence_policy = _calibrate_confidence_threshold(
        labels,
        tfidf_predictions,
        tfidf_confidences,
    )

    metrics: dict[str, object] = {
        "samples": len(labels),
        "class_counts": dict(counts),
        "class_group_counts": class_group_counts,
        "auxiliary_samples": len(auxiliary_labels),
        "auxiliary_class_counts": dict(Counter(auxiliary_labels)),
        "folds": n_splits,
        "context": "target-clause-window-1",
        "confidence_policy": confidence_policy,
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
        final_pipeline.fit(texts + auxiliary_texts, labels + auxiliary_labels)
        bundle = {
            "kind": "tfidf",
            "version": "tfidf-char-v2-target-window",
            "pipeline": final_pipeline,
            "confidence_threshold": confidence_policy["threshold"],
        }
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
        json.dumps(_build_lexicon(rows + auxiliary_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics
