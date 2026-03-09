from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EntityType = Literal["Asset", "Sector", "MacroIndicator", "Signal", "Position", "Thesis", "Evaluation", "Catalyst"]
ParserSource = Literal["structured", "llm", "deterministic_fallback"]


@dataclass(slots=True)
class OntologyNode:
    id: str
    type: EntityType
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OntologyEdge:
    source_id: str
    target_id: str
    relation_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InterpretedQuery:
    intent: str
    source: ParserSource
    filters: dict[str, Any] = field(default_factory=dict)
    entity: str | None = None
    original_query: str | None = None
