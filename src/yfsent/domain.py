from __future__ import annotations

from dataclasses import dataclass, field

SENTIMENTS = ("negative", "neutral", "positive")


@dataclass(frozen=True, slots=True)
class ContentRecord:
    id: str
    kind: str
    title: str
    text: str
    url: str
    source: str = "Yahoo股市"
    published_at: str = ""
    fetched_at: str = ""
    parent_id: str | None = None

    @property
    def combined_text(self) -> str:
        if self.text and self.text != self.title:
            return f"{self.title}。{self.text}"
        return self.title


@dataclass(frozen=True, slots=True)
class Entity:
    code: str
    market: str
    name: str
    short_name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EntityMention:
    entity: Entity
    surface: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SentimentResult:
    record_id: str
    entity: Entity
    label: str
    confidence: float
    evidence: tuple[EvidenceSpan, ...] = field(default_factory=tuple)
    model_version: str = ""

