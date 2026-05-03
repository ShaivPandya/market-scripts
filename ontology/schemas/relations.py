from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from ontology.schemas.base import NonBlankStr, OntologySchemaBase, clean_optional_text, clean_text
from ontology.schemas.objects import SignalDirection

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

ALLOWED_RELATIONS: dict[str, tuple[str, str]] = {
    "references_asset": ("Position", "Asset"),
    "belongs_to_sector": ("Asset", "Sector"),
    "has_thesis": ("Position", "Thesis"),
    "evaluated_by": ("Thesis", "Evaluation"),
    "has_catalyst": ("Thesis", "Catalyst"),
    "emits_signal": ("MacroIndicator", "Signal"),
    "affected_by": ("Sector", "MacroIndicator"),
    "exposed_to_signal": ("Position", "Signal"),
}

OPTIONAL_RELATIONS = {"has_thesis", "evaluated_by", "has_catalyst"}


class RelationPropertiesV1(OntologySchemaBase):
    ontology_run_id: str | None = None
    source: str | None = None

    @field_validator("ontology_run_id", "source", mode="before")
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


def edge_schema_name(relation_type: str) -> str:
    if relation_type == "exposed_to_signal":
        return "PositionSignalExposure"
    return "Relation"


def edge_schema_for_relation(relation_type: str):
    if relation_type == "exposed_to_signal":
        return PositionSignalExposureV1
    return RelationPropertiesV1


def dump_edge_properties(model: EdgePropertiesV1) -> dict[str, Any]:
    return model.model_dump(mode="json")
