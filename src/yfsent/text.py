from __future__ import annotations

import re
from collections.abc import Iterable

from .domain import Entity, EvidenceSpan

_BOUNDARY_RE = re.compile(r"[^。！？!?；;，,：:\n]+[。！？!?；;，,：:\n]?")
_NEGATORS = ("並未", "沒有", "未能", "未", "不會", "不", "無", "難以", "恐怕", "恐")

SEED_LEXICON = {
    "positive": (
        "受惠",
        "成長",
        "上升",
        "增加",
        "獲利",
        "創新高",
        "優於預期",
        "買超",
        "填息",
        "大增",
        "強勁",
        "突破",
        "上漲",
    ),
    "negative": (
        "失去",
        "下滑",
        "下降",
        "虧損",
        "衰退",
        "遜於預期",
        "賣超",
        "重挫",
        "裁員",
        "違約",
        "減產",
        "下跌",
    ),
}


def split_spans(text: str) -> list[tuple[str, int, int]]:
    spans = []
    for match in _BOUNDARY_RE.finditer(text):
        value = match.group(0).strip()
        if not value:
            continue
        leading = len(match.group(0)) - len(match.group(0).lstrip())
        start = match.start() + leading
        spans.append((value, start, start + len(value)))
    return spans


def _text_aliases(entity: Entity) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                alias
                for alias in (*entity.aliases, entity.name, entity.short_name)
                if alias and not alias.isdigit()
            },
            key=len,
            reverse=True,
        )
    )


def target_context(
    text: str,
    entity: Entity,
    *,
    window: int = 1,
    require_mention: bool = False,
) -> str:
    """Return target clauses plus neighboring clauses, with target aliases masked."""
    aliases = _text_aliases(entity)
    spans = split_spans(text)
    hit_indices = {
        index
        for index, (value, _, _) in enumerate(spans)
        if any(re.search(re.escape(alias), value, re.IGNORECASE) for alias in aliases)
    }
    if hit_indices:
        selected_indices = {
            neighbor
            for index in hit_indices
            for neighbor in range(max(0, index - window), min(len(spans), index + window + 1))
        }
        result = "".join(spans[index][0] for index in sorted(selected_indices))
    elif require_mention:
        return ""
    else:
        result = text

    for alias in aliases:
        result = re.sub(re.escape(alias), "目標公司", result, flags=re.IGNORECASE)
    return f"目標：{entity.short_name}。新聞：{result}"


class EvidenceExtractor:
    def __init__(self, learned_lexicon: dict[str, Iterable[str]] | None = None) -> None:
        merged = {label: list(values) for label, values in SEED_LEXICON.items()}
        if learned_lexicon:
            for label, values in learned_lexicon.items():
                if label in merged:
                    merged[label].extend(str(value) for value in values if value)
        self.lexicon = {
            label: tuple(sorted(set(values), key=lambda value: (-len(value), value)))
            for label, values in merged.items()
        }

    def extract(
        self,
        text: str,
        *,
        entity: Entity,
        label: str,
        classifier: object | None = None,
    ) -> tuple[EvidenceSpan, ...]:
        if label == "neutral":
            return ()
        clauses = split_spans(text)
        entity_aliases = _text_aliases(entity)
        relevant = [
            span
            for span in clauses
            if any(
                re.search(re.escape(alias), span[0], re.IGNORECASE)
                for alias in entity_aliases
            )
        ]
        candidates = relevant or clauses

        best = candidates[0] if candidates else (text, 0, len(text))
        if classifier is not None and candidates:
            inputs = [target_context(clause, entity) for clause, _, _ in candidates]
            probabilities = classifier.predict_proba(inputs)
            scored = [
                (float(probability.get(label, 0.0)), candidate)
                for probability, candidate in zip(probabilities, candidates, strict=True)
            ]
            best = max(scored, key=lambda item: item[0])[1]

        clause, clause_start, clause_end = best
        for term in self.lexicon.get(label, ()):
            offset = clause.find(term)
            if offset < 0:
                continue
            start = clause_start + offset
            end = start + len(term)
            prefix_start = max(clause_start, start - 4)
            prefix = text[prefix_start:start]
            for negator in _NEGATORS:
                if prefix.endswith(negator):
                    start -= len(negator)
                    break
            return (EvidenceSpan(text=text[start:end], start=start, end=end),)
        return (
            EvidenceSpan(text=text[clause_start:clause_end], start=clause_start, end=clause_end),
        )
