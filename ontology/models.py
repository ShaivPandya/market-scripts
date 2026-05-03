from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EntityType = Literal["Asset", "Sector", "MacroIndicator", "Signal", "Position", "Thesis", "Evaluation", "Catalyst"]
RelationType = Literal[
    "references_asset",
    "belongs_to_sector",
    "has_thesis",
    "evaluated_by",
    "has_catalyst",
    "emits_signal",
    "affected_by",
    "exposed_to_signal",
]
ParserSource = Literal["structured", "llm", "deterministic_fallback"]


@dataclass(slots=True)
class OntologyNode:
    id: str
    type: EntityType
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    schema_name: str = "legacy"
    schema_version: int = 0


@dataclass(slots=True)
class OntologyEdge:
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    schema_name: str = "legacy"
    schema_version: int = 0
    relation_schema_name: str = "legacy"
    relation_schema_version: int = 0


@dataclass(slots=True)
class InterpretedQuery:
    intent: str
    source: ParserSource
    filters: dict[str, Any] = field(default_factory=dict)
    entity: str | None = None
    original_query: str | None = None
