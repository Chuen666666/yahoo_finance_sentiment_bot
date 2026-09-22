from __future__ import annotations

from collections.abc import Iterable

from .domain import ContentRecord, Entity, SentimentResult
from .entities import EntityResolver
from .sentiment import FinBertClassifier, TfidfClassifier
from .text import EvidenceExtractor, target_context

Classifier = TfidfClassifier | FinBertClassifier


def analyze_records(
    records: Iterable[ContentRecord],
    *,
    resolver: EntityResolver,
    classifier: Classifier,
    extractor: EvidenceExtractor,
    target: Entity | None = None,
    target_only: bool = False,
    confidence_threshold: float | None = None,
) -> list[SentimentResult]:
    """Analyze each record/entity pair while keeping target-specific context."""
    threshold = (
        classifier.confidence_threshold
        if confidence_threshold is None
        else confidence_threshold
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence-threshold 必須介於 0 與 1 之間")
    if target_only and target is None:
        raise ValueError("target_only 需要可解析為上市櫃公司的 query")

    pairs: list[tuple[ContentRecord, Entity, str]] = []
    for record in records:
        entities = {
            mention.entity.code: mention.entity
            for mention in resolver.find_mentions(record.combined_text)
        }
        if target_only and target:
            entities = {target.code: target}
        elif target:
            entities[target.code] = target
        for entity in entities.values():
            context = target_context(
                record.combined_text,
                entity,
                require_mention=True,
            )
            pairs.append((record, entity, context))

    contexts = [context for _, _, context in pairs if context]
    probability_rows = iter(classifier.predict_proba(contexts) if contexts else [])
    results: list[SentimentResult] = []
    for record, entity, context in pairs:
        if not context:
            results.append(
                SentimentResult(
                    record_id=record.id,
                    entity=entity,
                    label="uncertain",
                    raw_label="",
                    confidence=0.0,
                    review_required=True,
                    target_mentioned=False,
                    model_version=classifier.version,
                )
            )
            continue

        probability = next(probability_rows)
        raw_label = max(probability, key=probability.get)
        confidence = float(probability[raw_label])
        final_label = raw_label if confidence >= threshold else "uncertain"
        evidence = extractor.extract(
            record.combined_text,
            entity=entity,
            label=raw_label,
            classifier=classifier,
        )
        results.append(
            SentimentResult(
                record_id=record.id,
                entity=entity,
                label=final_label,
                raw_label=raw_label,
                confidence=confidence,
                evidence=evidence,
                model_version=classifier.version,
                review_required=final_label == "uncertain",
                target_mentioned=True,
            )
        )
    return results
