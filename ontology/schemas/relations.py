from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from ontology.models import EntityType, RelationType
from ontology.schemas.base import NonBlankStr, OntologySchemaBase, clean_optional_text, clean_text
from ontology.schemas.objects import SignalDirection

REFERENCES_ASSET: RelationType = "references_asset"
BELONGS_TO_SECTOR: RelationType = "belongs_to_sector"
HAS_THESIS: RelationType = "has_thesis"
EVALUATED_BY: RelationType = "evaluated_by"
HAS_CATALYST: RelationType = "has_catalyst"
EMITS_SIGNAL: RelationType = "emits_signal"
AFFECTED_BY: RelationType = "affected_by"
EXPOSED_TO_SIGNAL: RelationType = "exposed_to_signal"


class RelationCardinality(StrEnum):
    MANY_TO_MANY = "many_to_many"
    SOURCE_UNIQUE = "source_unique"
    TARGET_UNIQUE = "target_unique"
    SOURCE_AND_TARGET_UNIQUE = "source_and_target_unique"


@dataclass(frozen=True, slots=True)
class RelationDefinition:
    name: RelationType
    source_type: EntityType
    target_type: EntityType
    cardinality: RelationCardinality
    required_properties: frozenset[str]
    optional: bool = False


RELATION_REGISTRY: dict[RelationType, RelationDefinition] = {
    REFERENCES_ASSET: RelationDefinition(
        name=REFERENCES_ASSET,
        source_type="Position",
        target_type="Asset",
        cardinality=RelationCardinality.SOURCE_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
    ),
    BELONGS_TO_SECTOR: RelationDefinition(
        name=BELONGS_TO_SECTOR,
        source_type="Asset",
        target_type="Sector",
        cardinality=RelationCardinality.SOURCE_UNIQUE,
        required_properties=frozenset({"ontology_run_id", "source"}),
    ),
    HAS_THESIS: RelationDefinition(
        name=HAS_THESIS,
        source_type="Position",
        target_type="Thesis",
        cardinality=RelationCardinality.SOURCE_AND_TARGET_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EVALUATED_BY: RelationDefinition(
        name=EVALUATED_BY,
        source_type="Thesis",
        target_type="Evaluation",
        cardinality=RelationCardinality.TARGET_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    HAS_CATALYST: RelationDefinition(
        name=HAS_CATALYST,
        source_type="Thesis",
        target_type="Catalyst",
        cardinality=RelationCardinality.TARGET_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EMITS_SIGNAL: RelationDefinition(
        name=EMITS_SIGNAL,
        source_type="MacroIndicator",
        target_type="Signal",
        cardinality=RelationCardinality.TARGET_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
    ),
    AFFECTED_BY: RelationDefinition(
        name=AFFECTED_BY,
        source_type="Sector",
        target_type="MacroIndicator",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
    ),
    EXPOSED_TO_SIGNAL: RelationDefinition(
        name=EXPOSED_TO_SIGNAL,
        source_type="Position",
        target_type="Signal",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset(
            {"component", "source", "name", "threshold", "direction", "contribution", "ontology_run_id"}
        ),
    ),
}

ALLOWED_RELATIONS: dict[RelationType, tuple[EntityType, EntityType]] = {
    name: (definition.source_type, definition.target_type) for name, definition in RELATION_REGISTRY.items()
}
OPTIONAL_RELATIONS = {name for name, definition in RELATION_REGISTRY.items() if definition.optional}
RELATION_TYPE_SQL_VALUES = ", ".join(f"'{relation_type}'" for relation_type in RELATION_REGISTRY)


class RelationPropertiesV1(OntologySchemaBase):
    ontology_run_id: NonBlankStr
    source: str | None = None

    @field_validator("ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("source", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class PositionSignalExposureV1(OntologySchemaBase):
    component: NonBlankStr
    source: NonBlankStr
    name: NonBlankStr
    value: float | int | str | bool | None = None
    threshold: NonBlankStr
    direction: SignalDirection
    contribution: float = Field(ge=0.0, le=1.0)
    ontology_run_id: NonBlankStr

    @field_validator("component", "source", "name", "threshold", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("direction", mode="before")
    @classmethod
    def _direction(cls, value: object) -> str:
        text = str(value or "").strip().lower()
        if text in {"deteriorating", "stable", "improving", "neutral", "unknown"}:
            return text
        return "unknown"


EdgePropertiesV1 = RelationPropertiesV1 | PositionSignalExposureV1


def get_relation_definition(relation_type: str) -> RelationDefinition:
    try:
        return RELATION_REGISTRY[relation_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported relation type: {relation_type}") from exc


def edge_schema_name(relation_type: str) -> str:
    get_relation_definition(relation_type)
    if relation_type == EXPOSED_TO_SIGNAL:
        return "PositionSignalExposure"
    return "Relation"


def edge_schema_for_relation(relation_type: str):
    get_relation_definition(relation_type)
    if relation_type == EXPOSED_TO_SIGNAL:
        return PositionSignalExposureV1
    return RelationPropertiesV1


def dump_edge_properties(model: EdgePropertiesV1) -> dict[str, Any]:
    return model.model_dump(mode="json")
