from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ontology.schemas.base import (
    NonBlankStr,
    OntologySchemaBase,
    Score,
    clean_lower_text,
    clean_optional_text,
    clean_text,
    expected_risk_level,
)
from ontology.schemas.identity import canonical_ticker, slug

RiskLevel = Literal["low", "medium", "high"]
SignalDirection = Literal["deteriorating", "stable", "improving", "neutral", "unknown"]
ThesisStatus = Literal["active", "under_review", "invalidated"]


class PositionV1(OntologySchemaBase):
    ticker: NonBlankStr
    asset: NonBlankStr
    direction: NonBlankStr
    timeframe: NonBlankStr
    latest_price: float | None = None
    as_of: str | None = None
    risk_score: Score
    risk_level: RiskLevel
    volatility_cluster: Score
    breadth_stress: Score
    sector_stress: Score
    macro_regime: Score
    ontology_run_id: NonBlankStr

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("asset", "direction", mode="before")
    @classmethod
    def _lower_text(cls, value: object) -> str:
        return clean_lower_text(value)

    @field_validator("timeframe", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("as_of", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)

    @model_validator(mode="after")
    def _risk_level_matches_score(self) -> PositionV1:
        expected = expected_risk_level(float(self.risk_score))
        if self.risk_level != expected:
            raise ValueError(f"risk_level must be {expected!r} for risk_score={self.risk_score}")
        return self


class AssetV1(OntologySchemaBase):
    ticker: NonBlankStr
    asset: NonBlankStr
    name: str | None = None
    currency: str | None = None
    exchange: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("asset", mode="before")
    @classmethod
    def _asset(cls, value: object) -> str:
        return clean_lower_text(value)

    @field_validator("name", "currency", "exchange", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class SectorV1(OntologySchemaBase):
    name: NonBlankStr
    sector_source: NonBlankStr

    @field_validator("name", "sector_source", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)


class MacroIndicatorV1(OntologySchemaBase):
    indicator_key: NonBlankStr
    name: NonBlankStr
    source: NonBlankStr
    as_of: NonBlankStr
    ontology_run_id: NonBlankStr

    @field_validator("indicator_key", mode="before")
    @classmethod
    def _indicator_key(cls, value: object) -> str:
        return slug(value)

    @field_validator("name", "source", "as_of", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)


class SignalV1(OntologySchemaBase):
    signal_key: NonBlankStr
    name: NonBlankStr
    source: NonBlankStr
    value: float | int | str | bool | None = None
    threshold: NonBlankStr
    direction: SignalDirection
    raw_signal: str | int | float | bool | None = None
    component: str | None = None
    sector: str | None = None
    ontology_run_id: NonBlankStr

    @field_validator("signal_key", mode="before")
    @classmethod
    def _signal_key(cls, value: object) -> str:
        return slug(value)

    @field_validator("name", "source", "threshold", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("component", "sector", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)

    @field_validator("direction", mode="before")
    @classmethod
    def _direction(cls, value: object) -> str:
        text = str(value or "").strip().lower()
        if text in {"deteriorating", "stable", "improving", "neutral", "unknown"}:
            return text
        return "unknown"


class ThesisV1(OntologySchemaBase):
    ticker: NonBlankStr
    status: ThesisStatus
    created_at: NonBlankStr
    updated_at: NonBlankStr
    ontology_run_id: NonBlankStr

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("created_at", "updated_at", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)


class EvaluationV1(OntologySchemaBase):
    ticker: NonBlankStr
    evaluated_at: NonBlankStr
    thesis_status: NonBlankStr
    technical_read: NonBlankStr
    fundamental_read: NonBlankStr
    action: NonBlankStr
    confidence: NonBlankStr
    risk_flag: str | None = None
    key_developments: list[str] = Field(default_factory=list)
    ontology_run_id: NonBlankStr

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator(
        "evaluated_at",
        "thesis_status",
        "technical_read",
        "fundamental_read",
        "action",
        "confidence",
        "ontology_run_id",
        mode="before",
    )
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("risk_flag", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)

    @field_validator("key_developments", mode="before")
    @classmethod
    def _key_developments(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class CatalystV1(OntologySchemaBase):
    ticker: NonBlankStr
    name: NonBlankStr
    description: NonBlankStr
    source: NonBlankStr
    category: str | None = None
    target_date: str | None = None
    status: str | None = None
    ontology_run_id: NonBlankStr

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("name", "description", "source", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("category", "target_date", "status", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


OntologyObjectV1 = PositionV1 | AssetV1 | SectorV1 | MacroIndicatorV1 | SignalV1 | ThesisV1 | EvaluationV1 | CatalystV1
JsonObject = dict[str, Any]
