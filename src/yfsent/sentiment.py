from __future__ import annotations

from pathlib import Path

from .domain import SENTIMENTS

FINBERT_MODEL_ID = "yiyanghkust/finbert-tone-chinese"
FINBERT_LABELS = {
    "LABEL_0": "neutral",
    "LABEL_1": "positive",
    "LABEL_2": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "negative": "negative",
}


class ClassifierError(RuntimeError):
    pass


class TfidfClassifier:
    def __init__(
        self,
        pipeline: object,
        *,
        version: str = "tfidf-char-v1",
        confidence_threshold: float = 0.55,
    ) -> None:
        self.pipeline = pipeline
        self.version = version
        self.confidence_threshold = confidence_threshold

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        probabilities = self.pipeline.predict_proba(texts)
        classes = [str(value) for value in self.pipeline.classes_]
        return [
            {
                label: float(row[classes.index(label)]) if label in classes else 0.0
                for label in SENTIMENTS
            }
            for row in probabilities
        ]


class FinBertClassifier:
    def __init__(self, model_id: str = FINBERT_MODEL_ID, *, batch_size: int = 16) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ClassifierError(
                "FinBERT 需要 ML 套件；請執行 pip install -e \".[ml]\""
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        self.model_id = model_id
        self.batch_size = batch_size
        self.version = f"finbert:{model_id}"
        self.confidence_threshold = 0.55

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        output: list[dict[str, float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
                rows = self.torch.softmax(logits, dim=-1).cpu().tolist()
            id2label = self.model.config.id2label
            for row in rows:
                result = {label: 0.0 for label in SENTIMENTS}
                for index, score in enumerate(row):
                    raw_label = str(id2label.get(index, f"LABEL_{index}"))
                    label = FINBERT_LABELS.get(raw_label, raw_label.casefold())
                    if label in result:
                        result[label] = float(score)
                output.append(result)
        return output


def load_classifier(model_path: str | Path) -> TfidfClassifier | FinBertClassifier:
    path = Path(model_path)
    if not path.exists():
        raise ClassifierError(f"找不到模型 {path}；請先執行 train")
    try:
        import joblib
    except ImportError as exc:
        raise ClassifierError("讀取模型需要 joblib；請執行 pip install -e \".[ml]\"") from exc
    bundle = joblib.load(path)
    kind = bundle.get("kind")
    if kind == "tfidf":
        return TfidfClassifier(
            bundle["pipeline"],
            version=bundle.get("version", "tfidf-char-v1"),
            confidence_threshold=float(bundle.get("confidence_threshold", 0.55)),
        )
    if kind == "finbert":
        return FinBertClassifier(bundle.get("model_id", FINBERT_MODEL_ID))
    raise ClassifierError(f"不支援的模型格式：{kind!r}")
