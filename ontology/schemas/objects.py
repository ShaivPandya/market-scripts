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
    timeframe: NonBlankStr = "operational"
    latest_price: float | None = None
    as_of: str | None = None
    risk_score: Score = 0.0
    risk_level: RiskLevel = "low"
    volatility_cluster: Score = 0.0
    breadth_stress: Score = 0.0
    sector_stress: Score = 0.0
    macro_regime: Score = 0.0
    ontology_run_id: NonBlankStr = "operational"
    contrarian: bool = False
    conviction: int | None = Field(default=None, ge=1, le=5)
    cost_basis: float | None = None
    shares: float | None = None
    role: str = "position"

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


class InvestorV1(OntologySchemaBase):
    investor_id: NonBlankStr
    name: NonBlankStr
    suitability_profile: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("investor_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("name", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("suitability_profile", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class AccountV1(OntologySchemaBase):
    account_id: NonBlankStr
    investor_id: NonBlankStr
    account_type: str | None = None
    tax_status: NonBlankStr = "unknown"
    tax_lot_data_available: bool | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("account_id", "investor_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("tax_status", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("account_type", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class PortfolioV1(OntologySchemaBase):
    portfolio_id: NonBlankStr
    account_id: NonBlankStr
    base_currency: NonBlankStr = "USD"
    benchmark: str | None = None
    cash: float | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("portfolio_id", "account_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("base_currency", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("benchmark", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class MandateV1(OntologySchemaBase):
    mandate_id: NonBlankStr
    benchmark: str | None = None
    permitted_asset_classes: list[str] = Field(default_factory=list)
    permitted_actions: list[str] = Field(default_factory=list)
    liquidity_needs: str | None = None
    time_horizon_days_min: int | None = None
    time_horizon_days_max: int | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("mandate_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("benchmark", "liquidity_needs", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)

    @field_validator("ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)


class InvestmentPolicyV1(OntologySchemaBase):
    policy_id: NonBlankStr
    account_id: NonBlankStr
    mandate_id: NonBlankStr
    constraints: dict[str, Any] = Field(default_factory=dict)
    disclosures: list[str] = Field(default_factory=list)
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("policy_id", "account_id", "mandate_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)


class RiskLimitV1(OntologySchemaBase):
    limit_id: NonBlankStr
    policy_id: NonBlankStr
    metric: NonBlankStr
    comparator: NonBlankStr
    threshold: float | str
    severity: NonBlankStr = "fail"
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("limit_id", "policy_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("metric", "comparator", "severity", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)


class RiskMetricV1(OntologySchemaBase):
    metric_id: NonBlankStr
    metric: NonBlankStr
    value: float | int | str | bool | None = None
    as_of: str | None = None
    source: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("metric_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("metric", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("as_of", "source", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ScenarioV1(OntologySchemaBase):
    scenario_id: NonBlankStr
    name: NonBlankStr
    result: dict[str, Any] = Field(default_factory=dict)
    loss_pct: float | None = None
    status: NonBlankStr = "unknown"
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("scenario_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("name", "status", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)


class PolicyGateResultV1(OntologySchemaBase):
    gate_result_id: NonBlankStr
    decision: NonBlankStr
    review_required: bool = False
    failure_reasons: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    account_id: str | None = None
    portfolio_id: str | None = None
    policy_id: str | None = None
    evaluated_at: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("gate_result_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("decision", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("account_id", "portfolio_id", "policy_id", "evaluated_at", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class TradeProposalV1(OntologySchemaBase):
    proposal_id: NonBlankStr
    recommendation_id: str | None = None
    account_id: str | None = None
    action: NonBlankStr
    instrument: NonBlankStr
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    approval_id: int | None = None
    status: NonBlankStr = "staged"
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("proposal_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("action", "instrument", "status", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("recommendation_id", "account_id", mode="before")
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
    earnings_note: str | None = None
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

    @field_validator("earnings_note", "risk_flag", mode="before")
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
    evidence: str | None = None
    ontology_run_id: NonBlankStr

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("name", "description", "source", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("category", "target_date", "status", "evidence", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class HedgePositionV1(OntologySchemaBase):
    ticker: NonBlankStr
    direction: NonBlankStr
    asset: NonBlankStr = "equity"
    cost_basis: float | None = None
    shares: float | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("asset", "direction", mode="before")
    @classmethod
    def _lower_text(cls, value: object) -> str:
        return clean_lower_text(value)


class KillConditionV1(OntologySchemaBase):
    ticker: NonBlankStr
    condition: NonBlankStr
    legacy_id: int | None = None
    metric: str | None = None
    threshold: str | None = None
    status: NonBlankStr = "active"
    triggered_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    created_by: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("condition", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("metric", "threshold", "triggered_at", "created_at", "updated_at", "created_by", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ThesisClaimV1(OntologySchemaBase):
    ticker: NonBlankStr
    claim: NonBlankStr
    legacy_id: int | None = None
    expected_evidence: str | None = None
    disconfirming_evidence: str | None = None
    source_requirements: list[Any] = Field(default_factory=list)
    cadence: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: NonBlankStr = "active"
    linked_catalyst_ids: list[int] = Field(default_factory=list)
    linked_kill_condition_ids: list[int] = Field(default_factory=list)
    source_type: str | None = None
    source_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("claim", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "expected_evidence",
        "disconfirming_evidence",
        "cadence",
        "source_type",
        "source_id",
        "created_at",
        "updated_at",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ActionItemV1(OntologySchemaBase):
    description: NonBlankStr
    action_type: NonBlankStr = "review"
    legacy_id: int | None = None
    ticker: str | None = None
    urgency: NonBlankStr = "normal"
    status: NonBlankStr = "open"
    source_type: str | None = None
    source_id: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    resolution_note: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("description", "action_type", "urgency", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("source_type", "source_id", "created_at", "completed_at", "resolution_note", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class WatchTriggerV1(OntologySchemaBase):
    condition: NonBlankStr
    trigger_type: NonBlankStr = "custom"
    legacy_id: int | None = None
    ticker: str | None = None
    status: NonBlankStr = "active"
    source_type: str | None = None
    source_id: str | None = None
    created_at: str | None = None
    fired_at: str | None = None
    expires_at: str | None = None
    definition: dict[str, Any] | None = None
    last_checked_at: str | None = None
    last_result: dict[str, Any] | None = None
    last_evidence: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("condition", "trigger_type", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "source_type",
        "source_id",
        "created_at",
        "fired_at",
        "expires_at",
        "last_checked_at",
        "last_evidence",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ResearchNoteV1(OntologySchemaBase):
    title: NonBlankStr
    content: NonBlankStr
    legacy_id: int | None = None
    ticker: str | None = None
    note_type: NonBlankStr = "general"
    source_type: str | None = None
    source_id: str | None = None
    created_at: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("title", "content", "note_type", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("source_type", "source_id", "created_at", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ApprovalV1(OntologySchemaBase):
    legacy_id: int | None = None
    entity_type: NonBlankStr
    entity_id: int | None = None
    ticker: str | None = None
    action_id: str | None = None
    action_schema_name: str | None = None
    action_schema_version: int | None = None
    action_input_hash: str | None = None
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    status: NonBlankStr = "pending"
    application_status: str | None = None
    created_at: str | None = None
    resolved_at: str | None = None
    resolved_note: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("entity_type", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "action_id",
        "action_schema_name",
        "action_input_hash",
        "reason",
        "source_type",
        "source_id",
        "application_status",
        "created_at",
        "resolved_at",
        "resolved_note",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ActionRunV1(OntologySchemaBase):
    legacy_id: int | None = None
    action_id: NonBlankStr
    action_schema_name: str | None = None
    action_schema_version: int = 1
    actor_type: NonBlankStr
    actor_id: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    approval_id: int | None = None
    parent_action_run_id: int | None = None
    input_hash: str | None = None
    status: NonBlankStr = "running"
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    provenance_event_id: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("action_id", "actor_type", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "action_schema_name",
        "actor_id",
        "source_type",
        "source_id",
        "input_hash",
        "error",
        "started_at",
        "completed_at",
        "provenance_event_id",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ActionEventV1(OntologySchemaBase):
    legacy_id: int | None = None
    action_run_id: int
    event_type: NonBlankStr
    message: str | None = None
    payload: dict[str, Any] | None = None
    created_at: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("event_type", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("message", "created_at", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class WorkflowRunV1(OntologySchemaBase):
    run_id: NonBlankStr
    workflow_name: NonBlankStr
    ticker: str | None = None
    status: NonBlankStr = "running"
    started_at: str | None = None
    completed_at: str | None = None
    synthesis: str | None = None
    artifacts: dict[str, Any] | None = None
    tool_sections: list[Any] | None = None
    provenance_event_id: str | None = None
    error: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("run_id", "workflow_name", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("started_at", "completed_at", "synthesis", "provenance_event_id", "error", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class WorkflowArtifactV1(OntologySchemaBase):
    artifact_id: NonBlankStr
    workflow_run_id: str | None = None
    artifact_key: NonBlankStr
    artifact_index: int | None = None
    artifact_value: dict[str, Any] | list[Any] | str | None = None
    approval_id: int | None = None
    provenance_event_id: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("artifact_id", "artifact_key", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("workflow_run_id", "provenance_event_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class RecommendationV1(OntologySchemaBase):
    legacy_id: int | None = None
    report_type: str | None = None
    as_of: str | None = None
    action: NonBlankStr
    ticker: str | None = None
    instrument: str | None = None
    status: str | None = None
    approval_id: int | None = None
    approval_status: str | None = None
    outcome_status: str | None = None
    account_id: str | None = None
    portfolio_id: str | None = None
    policy_id: str | None = None
    policy_gate_result_id: int | None = None
    policy_gate_decision: str | None = None
    policy_gate_review_required: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("action", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "report_type",
        "as_of",
        "instrument",
        "status",
        "approval_status",
        "outcome_status",
        "account_id",
        "portfolio_id",
        "policy_id",
        "policy_gate_decision",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ReportRunV1(OntologySchemaBase):
    report_id: NonBlankStr
    report_type: NonBlankStr
    as_of: NonBlankStr
    source: str | None = None
    source_run_id: str | None = None
    source_url: str | None = None
    status: NonBlankStr = "completed"
    report_hash: str | None = None
    input_hash: str | None = None
    summary: dict[str, Any] | None = None
    artifact_paths: dict[str, Any] | None = None
    issue_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    synced_at: str | None = None
    error: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("report_id", "report_type", "as_of", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "source",
        "source_run_id",
        "source_url",
        "report_hash",
        "input_hash",
        "issue_url",
        "created_at",
        "updated_at",
        "synced_at",
        "error",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class DocumentArtifactV1(OntologySchemaBase):
    document_type: NonBlankStr
    document_id: NonBlankStr
    title: str | None = None
    ticker: str | None = None
    content_hash: str | None = None
    artifact_uri: str | None = None
    status: NonBlankStr = "active"
    source_type: str | None = None
    source_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("document_type", "document_id", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "title", "content_hash", "artifact_uri", "source_type", "source_id", "created_at", "updated_at", mode="before"
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


OntologyObjectV1 = (
    PositionV1
    | AssetV1
    | InvestorV1
    | AccountV1
    | PortfolioV1
    | MandateV1
    | InvestmentPolicyV1
    | RiskLimitV1
    | RiskMetricV1
    | ScenarioV1
    | PolicyGateResultV1
    | TradeProposalV1
    | SectorV1
    | MacroIndicatorV1
    | SignalV1
    | ThesisV1
    | EvaluationV1
    | CatalystV1
    | HedgePositionV1
    | KillConditionV1
    | ThesisClaimV1
    | ActionItemV1
    | WatchTriggerV1
    | ResearchNoteV1
    | ApprovalV1
    | ActionRunV1
    | ActionEventV1
    | WorkflowRunV1
    | WorkflowArtifactV1
    | RecommendationV1
    | ReportRunV1
    | DocumentArtifactV1
)
JsonObject = dict[str, Any]
