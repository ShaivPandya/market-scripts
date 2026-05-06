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
RecommendationDecisionState = Literal[
    "generated",
    "proposed",
    "under_review",
    "approved",
    "rejected",
    "acted",
    "closed",
    "superseded",
]
TradeProposalDecisionState = Literal[
    "staged",
    "policy_checked",
    "pending_approval",
    "approved",
    "rejected",
    "executed_action_recorded",
    "expired",
    "superseded",
]
ApprovalResolutionState = Literal["pending", "approved", "rejected", "expired"]
ApprovalApplicationState = Literal["pending", "applying", "applied", "failed", "not_applicable"]
ActionRunState = Literal["running", "succeeded", "failed", "rolled_back", "denied"]
WorkflowArtifactState = Literal["extracted", "ignored", "auto_recorded", "proposed", "approved", "rejected", "failed"]


class PositionV1(OntologySchemaBase):
    ticker: NonBlankStr
    asset: NonBlankStr
    direction: NonBlankStr
    account_id: str | None = None
    portfolio_id: str | None = None
    instrument_id: str | None = None
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
    quantity: float | None = None
    instrument_type: str = "security"
    price_symbol: str | None = None
    contract_multiplier: float = 1.0
    role: str = "position"

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("asset", "direction", mode="before")
    @classmethod
    def _lower_text(cls, value: object) -> str:
        return clean_lower_text(value)

    @field_validator("timeframe", "ontology_run_id", "instrument_type", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("account_id", "portfolio_id", "instrument_id", "as_of", "price_symbol", mode="before")
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


class InstrumentV1(OntologySchemaBase):
    instrument_id: NonBlankStr
    ticker: str | None = None
    name: str | None = None
    asset_class: NonBlankStr = "security"
    instrument_type: NonBlankStr = "security"
    issuer_id: str | None = None
    currency: str | None = None
    exchange: str | None = None
    country: str | None = None
    sector: str | None = None
    industry: str | None = None
    price_symbol: str | None = None
    status: NonBlankStr = "active"
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("instrument_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("asset_class", "instrument_type", "status", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "name",
        "issuer_id",
        "currency",
        "exchange",
        "country",
        "sector",
        "industry",
        "price_symbol",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class IssuerV1(OntologySchemaBase):
    issuer_id: NonBlankStr
    name: NonBlankStr
    ticker: str | None = None
    country: str | None = None
    sector: str | None = None
    industry: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("issuer_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("name", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("country", "sector", "industry", mode="before")
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
    ontology_run_id: NonBlankStr = "operational"

    @model_validator(mode="before")
    @classmethod
    def _drop_deprecated_fields(cls, value: object) -> object:
        if isinstance(value, dict) and "tax_lot_data_available" in value:
            value = dict(value)
            value.pop("tax_lot_data_available", None)
        return value

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
    policy_version: int = 1
    owner_account_id: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    status: NonBlankStr = "active"
    constraints: dict[str, Any] = Field(default_factory=dict)
    disclosures: list[str] = Field(default_factory=list)
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("policy_id", "account_id", "mandate_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("owner_account_id", "effective_from", "effective_to", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)

    @field_validator("status", "ontology_run_id", mode="before")
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
    scope_type: str | None = None
    scope_id: str | None = None
    value: float | int | str | bool | None = None
    unit: str | None = None
    method: str | None = None
    window: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    source_record_ids: list[str] = Field(default_factory=list)
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

    @field_validator("scope_type", "scope_id", "unit", "method", "window", "as_of", "source", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ScenarioV1(OntologySchemaBase):
    scenario_id: NonBlankStr
    name: NonBlankStr
    scenario_type: NonBlankStr = "stress"
    scope_type: str | None = None
    scope_id: str | None = None
    assumptions_hash: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    result_metrics: dict[str, Any] = Field(default_factory=dict)
    loss_pct: float | None = None
    generated_by_source: str | None = None
    generated_by_action: str | None = None
    generated_by_run_id: str | None = None
    as_of: str | None = None
    status: NonBlankStr = "unknown"
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("scenario_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("name", "scenario_type", "status", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "scope_type",
        "scope_id",
        "assumptions_hash",
        "generated_by_source",
        "generated_by_action",
        "generated_by_run_id",
        "as_of",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


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
    portfolio_id: str | None = None
    action: NonBlankStr
    instrument: NonBlankStr
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    sizing_summary: dict[str, Any] = Field(default_factory=dict)
    risk_summary: dict[str, Any] = Field(default_factory=dict)
    policy_gate_result_id: str | None = None
    approval_id: str | None = None
    decision_state: TradeProposalDecisionState = "staged"
    expires_at: str | None = None
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

    @field_validator(
        "recommendation_id",
        "account_id",
        "portfolio_id",
        "policy_gate_result_id",
        "approval_id",
        "expires_at",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class SourceRecordV1(OntologySchemaBase):
    source_record_id: NonBlankStr
    vendor: NonBlankStr
    source_name: NonBlankStr
    source_version: NonBlankStr = "unknown"
    dataset: NonBlankStr
    record_kind: NonBlankStr
    record_key_hash: NonBlankStr
    payload_hash: NonBlankStr
    status: NonBlankStr = "ok"
    quality: NonBlankStr = "ok"
    as_of: str | None = None
    load_time: str | None = None
    artifact_uri: str | None = None
    provenance_event_id: str | None = None
    metadata: dict[str, Any] | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator(
        "source_record_id",
        "vendor",
        "source_name",
        "source_version",
        "dataset",
        "record_kind",
        "record_key_hash",
        "payload_hash",
        "status",
        "quality",
        "ontology_run_id",
        mode="before",
    )
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("as_of", "load_time", "artifact_uri", "provenance_event_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ObjectVersionRefV1(OntologySchemaBase):
    ref_id: NonBlankStr
    object_uid: NonBlankStr
    object_type: str | None = None
    version_id: NonBlankStr
    valid_from: str | None = None
    tx_from: str | None = None
    temporal_confidence: str | None = None
    source_record_id: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ref_id", "object_uid", "version_id", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("object_type", "valid_from", "tx_from", "temporal_confidence", "source_record_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ExecutedDecisionRecordV1(OntologySchemaBase):
    decision_record_id: NonBlankStr
    approval_id: str | None = None
    action_run_id: str | None = None
    action_id: NonBlankStr
    target_object_uid: str | None = None
    target_object_type: str | None = None
    applied_object_versions: list[dict[str, Any]] = Field(default_factory=list)
    applied_at: str | None = None
    status: NonBlankStr = "recorded"
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("decision_record_id", "action_id", "status", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "approval_id",
        "action_run_id",
        "target_object_uid",
        "target_object_type",
        "applied_at",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ExecutedActionV1(OntologySchemaBase):
    executed_action_id: NonBlankStr
    action_id: NonBlankStr
    approval_id: str | None = None
    action_run_id: str | None = None
    execution_mode: NonBlankStr = "approval_required"
    produced_object_versions: list[dict[str, Any]] = Field(default_factory=list)
    mutated_object_versions: list[dict[str, Any]] = Field(default_factory=list)
    applied_at: str | None = None
    status: NonBlankStr = "applied"
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("executed_action_id", "action_id", "execution_mode", "status", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("approval_id", "action_run_id", "applied_at", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class AuditEventV1(OntologySchemaBase):
    event_id: NonBlankStr
    occurred_at: str | None = None
    actor_type: NonBlankStr = "system"
    actor_id: str | None = None
    action_name: NonBlankStr
    action_category: NonBlankStr
    status: NonBlankStr
    object_refs: list[dict[str, Any]] = Field(default_factory=list)
    before_summary: dict[str, Any] | None = None
    after_summary: dict[str, Any] | None = None
    source_lineage: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    lineage_root_id: str | None = None
    retention_class: NonBlankStr = "audit_365d"
    ontology_run_id: NonBlankStr = "operational"

    @field_validator(
        "event_id",
        "actor_type",
        "action_name",
        "action_category",
        "status",
        "retention_class",
        "ontology_run_id",
        mode="before",
    )
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("occurred_at", "actor_id", "lineage_root_id", mode="before")
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
    instrument_id: str | None = None
    ontology_run_id: NonBlankStr

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("created_at", "updated_at", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("instrument_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


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
    quantity: float | None = None
    instrument_type: str = "security"
    price_symbol: str | None = None
    contract_multiplier: float = 1.0
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("asset", "direction", mode="before")
    @classmethod
    def _lower_text(cls, value: object) -> str:
        return clean_lower_text(value)

    @field_validator("instrument_type", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("price_symbol", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


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


class EvidenceV1(OntologySchemaBase):
    evidence_id: NonBlankStr
    evidence_type: NonBlankStr = "source_excerpt"
    title: str | None = None
    summary: str | None = None
    source_record_id: str | None = None
    document_artifact_id: str | None = None
    object_uid: str | None = None
    object_version_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    observed_at: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("evidence_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("evidence_type", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "title",
        "summary",
        "source_record_id",
        "document_artifact_id",
        "object_uid",
        "object_version_id",
        "observed_at",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class CitationV1(OntologySchemaBase):
    citation_id: NonBlankStr
    source_record_id: str | None = None
    document_artifact_id: str | None = None
    title: str | None = None
    url: str | None = None
    source_path: str | None = None
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    quote_hash: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("citation_id", mode="before")
    @classmethod
    def _id(cls, value: object) -> str:
        return slug(value)

    @field_validator("ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "source_record_id", "document_artifact_id", "title", "url", "source_path", "quote_hash", mode="before"
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
    entity_id: str | None = None
    ticker: str | None = None
    target_object_uid: str | None = None
    target_object_type: str | None = None
    action_id: str | None = None
    action_schema_name: str | None = None
    action_schema_version: int | None = None
    action_input_hash: str | None = None
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    status: NonBlankStr = "pending"
    resolution_state: ApprovalResolutionState = "pending"
    application_state: ApprovalApplicationState = "pending"
    application_status: str | None = None
    risk_class: str | None = None
    policy_gate_result: dict[str, Any] | None = None
    policy_gate_result_id: str | None = None
    policy_gate_decision: str | None = None
    base_state_hash: str | None = None
    supersedes_approval_id: str | None = None
    requested_by_actor_id: str | None = None
    resolved_by_actor_id: str | None = None
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
        "entity_id",
        "target_object_uid",
        "target_object_type",
        "reason",
        "source_type",
        "source_id",
        "application_status",
        "risk_class",
        "policy_gate_result_id",
        "policy_gate_decision",
        "base_state_hash",
        "supersedes_approval_id",
        "requested_by_actor_id",
        "resolved_by_actor_id",
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
    approval_id: str | None = None
    parent_action_run_id: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    status: NonBlankStr = "running"
    execution_state: ActionRunState | None = None
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
        "approval_id",
        "parent_action_run_id",
        "input_hash",
        "output_hash",
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
    action_run_id: str
    event_type: NonBlankStr
    message: str | None = None
    payload: dict[str, Any] | None = None
    created_at: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("action_run_id", "event_type", mode="before")
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
    artifact_hash: str | None = None
    state: WorkflowArtifactState = "extracted"
    action_id: str | None = None
    approval_id: str | None = None
    provenance_event_id: str | None = None
    metadata: dict[str, Any] | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("artifact_id", "artifact_key", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "workflow_run_id", "artifact_hash", "action_id", "approval_id", "provenance_event_id", mode="before"
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class RecommendationV1(OntologySchemaBase):
    recommendation_id: str | None = None
    legacy_id: int | None = None
    idempotency_key: str | None = None
    source_kind: NonBlankStr = "report"
    report_type: str | None = None
    as_of: str | None = None
    action: NonBlankStr
    ticker: str | None = None
    instrument: str | None = None
    decision_state: RecommendationDecisionState = "generated"
    status: str | None = None
    approval_id: str | None = None
    approval_required: bool = False
    approval_status: str | None = None
    outcome_status: str | None = None
    supersedes_recommendation_id: str | None = None
    account_id: str | None = None
    portfolio_id: str | None = None
    policy_id: str | None = None
    policy_gate_result_id: str | None = None
    policy_gate_decision: str | None = None
    policy_gate_review_required: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
    horizon: str | None = None
    rationale_summary: str | None = None
    rationale_hash: str | None = None
    source_quality: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("action", "source_kind", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "recommendation_id",
        "idempotency_key",
        "report_type",
        "as_of",
        "instrument",
        "status",
        "approval_id",
        "approval_status",
        "outcome_status",
        "supersedes_recommendation_id",
        "account_id",
        "portfolio_id",
        "policy_id",
        "policy_gate_result_id",
        "policy_gate_decision",
        "horizon",
        "rationale_summary",
        "rationale_hash",
        "source_quality",
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


class ProvenanceEventV1(OntologySchemaBase):
    event_id: NonBlankStr
    id: str | None = None
    event_type: NonBlankStr
    event_name: NonBlankStr
    status: Literal["started", "succeeded", "failed"] = "started"
    started_at: str | None = None
    completed_at: str | None = None
    actor_type: str | None = None
    actor_id: str | None = None
    parent_actor_id: str | None = None
    request_id: str | None = None
    parent_event_id: str | None = None
    workflow_run_id: str | None = None
    ontology_run_id: str | None = None
    agent_session_id: str | None = None
    action_run_id: str | int | None = None
    approval_id: str | int | None = None
    audit_event_id: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    summary: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    metadata: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    error: str | None = None
    criticality: str | None = None
    lineage_root_id: str | None = None
    idempotency_key: str | None = None
    producer_name: str | None = None
    producer_version: str | None = None
    redaction_policy: NonBlankStr
    retention_class: NonBlankStr

    @field_validator("event_id", "event_type", "event_name", "retention_class", "redaction_policy", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "id",
        "started_at",
        "completed_at",
        "actor_type",
        "actor_id",
        "parent_actor_id",
        "request_id",
        "parent_event_id",
        "workflow_run_id",
        "ontology_run_id",
        "agent_session_id",
        "audit_event_id",
        "input_hash",
        "output_hash",
        "error",
        "criticality",
        "lineage_root_id",
        "idempotency_key",
        "producer_name",
        "producer_version",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)

    @model_validator(mode="after")
    def _has_context_anchor(self) -> ProvenanceEventV1:
        if any(
            (
                self.actor_type,
                self.actor_id,
                self.request_id,
                self.parent_event_id,
                self.workflow_run_id,
                self.ontology_run_id,
                self.agent_session_id,
                self.action_run_id,
                self.approval_id,
                self.audit_event_id,
                self.lineage_root_id,
            )
        ):
            return self
        raise ValueError(
            "ProvenanceEvent requires at least one actor, request, session, workflow, action, approval, ontology run, parent event, audit event, or lineage root anchor"
        )


class RelationVersionRefV1(OntologySchemaBase):
    ref_id: NonBlankStr
    relation_uid: str | None = None
    relation_type: str | None = None
    version_id: NonBlankStr
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ref_id", "version_id", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("relation_uid", "relation_type", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class SchemaDefinitionRefV1(OntologySchemaBase):
    ref_id: NonBlankStr
    schema_kind: NonBlankStr
    schema_name: NonBlankStr
    schema_version_value: int = Field(alias="schema_version_value")
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("ref_id", "schema_kind", "schema_name", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)


class OntologyRunRefV1(OntologySchemaBase):
    run_id: NonBlankStr
    status: str | None = None
    as_of: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("run_id", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("status", "as_of", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class AgentSessionRefV1(OntologySchemaBase):
    session_id: NonBlankStr
    actor_id: str | None = None
    started_at: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("session_id", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("actor_id", "started_at", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ModelCallRefV1(OntologySchemaBase):
    call_id: NonBlankStr
    model: str | None = None
    provider: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("call_id", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("model", "provider", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ToolCallRefV1(OntologySchemaBase):
    call_id: NonBlankStr
    tool_name: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("call_id", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("tool_name", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ComputedSnapshotRefV1(OntologySchemaBase):
    snapshot_key: NonBlankStr
    snapshot_id: str | None = None
    status: str | None = None
    ontology_run_id: NonBlankStr = "operational"

    @field_validator("snapshot_key", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("snapshot_id", "status", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class InvestmentIdeaV1(OntologySchemaBase):
    idea_id: NonBlankStr
    id: str | int | None = None
    legacy_id: int | None = None
    ticker: NonBlankStr
    company_name: str | None = None
    status: NonBlankStr = "watching"
    user_notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    tags_json: list[str] | str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    archived_at: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    latest_evaluation_id: str | int | None = None
    latest_job_id: str | None = None
    accepted_recommendation_id: str | int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ontology_run_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("idea_id", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "company_name",
        "user_notes",
        "created_at",
        "updated_at",
        "archived_at",
        "source_type",
        "source_id",
        "latest_job_id",
        "ontology_run_id",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class IdeaEvaluationV1(OntologySchemaBase):
    evaluation_id: NonBlankStr
    id: str | int | None = None
    legacy_id: int | None = None
    idea_id: NonBlankStr
    ticker: NonBlankStr
    job_id: str | None = None
    evaluated_at: NonBlankStr
    action: NonBlankStr
    recommendation_status: NonBlankStr = "clear"
    score: float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    thesis_statement: str | None = None
    rationale: str | None = None
    factor_scores: dict[str, Any] = Field(default_factory=dict)
    missing_information: list[dict[str, Any]] = Field(default_factory=list)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    disconfirming_evidence: list[dict[str, Any]] = Field(default_factory=list)
    catalyst: str | None = None
    invalidation: str | None = None
    portfolio_fit: dict[str, Any] = Field(default_factory=dict)
    recommendation_record: dict[str, Any] | None = None
    recommendation_id: str | int | None = None
    approval_id: str | int | None = None
    recommendation_approval_id: str | int | None = None
    action_approval_id: str | int | None = None
    accepted: bool = False
    accepted_at: str | None = None
    accepted_by: str | None = None
    created_at: str | None = None
    ontology_run_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator("evaluation_id", "idea_id", "evaluated_at", "action", "recommendation_status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "job_id",
        "thesis_statement",
        "rationale",
        "catalyst",
        "invalidation",
        "accepted_at",
        "accepted_by",
        "created_at",
        "ontology_run_id",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class IdeaComparisonRunV1(OntologySchemaBase):
    comparison_run_id: NonBlankStr
    id: str | int | None = None
    legacy_id: int | None = None
    run_id: NonBlankStr
    job_id: str | None = None
    scope_statuses: list[str] = Field(default_factory=list)
    summary: str | None = None
    ranking_count: int | None = None
    rankings: list[dict[str, Any]] = Field(default_factory=list)
    raw_result: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    ontology_run_id: str | None = None

    @field_validator("comparison_run_id", "run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("job_id", "summary", "created_at", "updated_at", "ontology_run_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class IdeaComparisonRankingV1(OntologySchemaBase):
    ranking_id: NonBlankStr
    id: str | int | None = None
    legacy_id: int | None = None
    comparison_run_id: NonBlankStr
    run_id: str | None = None
    idea_id: NonBlankStr
    evaluation_id: NonBlankStr
    ticker: NonBlankStr
    rank: int = Field(ge=1)
    action: NonBlankStr
    score: float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_level: NonBlankStr = "low"
    rationale: str | None = None
    created_at: str | None = None
    ontology_run_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return canonical_ticker(value)

    @field_validator(
        "ranking_id",
        "comparison_run_id",
        "idea_id",
        "evaluation_id",
        "action",
        "confidence_level",
        mode="before",
    )
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("run_id", "rationale", "created_at", "ontology_run_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class FactorScoreV1(OntologySchemaBase):
    factor_score_id: NonBlankStr
    id: str | int | None = None
    legacy_id: int | None = None
    parent_uid: NonBlankStr
    parent_type: NonBlankStr
    factor_name: NonBlankStr
    score: float | None = Field(default=None, ge=0, le=100)
    status: str | None = None
    rationale: str | None = None
    missing: list[str] = Field(default_factory=list)
    weight: float | None = None
    created_at: str | None = None
    ontology_run_id: str | None = None

    @field_validator("factor_score_id", "parent_uid", "parent_type", "factor_name", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("status", "rationale", "created_at", "ontology_run_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class MissingInformationRequirementV1(OntologySchemaBase):
    requirement_id: NonBlankStr
    id: str | int | None = None
    legacy_id: int | None = None
    parent_uid: NonBlankStr
    parent_type: NonBlankStr
    field: NonBlankStr
    severity: NonBlankStr = "medium"
    reason: str | None = None
    status: NonBlankStr = "open"
    created_at: str | None = None
    ontology_run_id: str | None = None

    @field_validator("requirement_id", "parent_uid", "parent_type", "field", "severity", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("reason", "created_at", "ontology_run_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class OptimizationMissionV1(OntologySchemaBase):
    mission_id: NonBlankStr
    id: str | int | None = None
    legacy_id: int | None = None
    name: NonBlankStr
    status: NonBlankStr = "active"
    schedule_label: str | None = None
    scenario: dict[str, Any] = Field(default_factory=dict)
    source_config: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    ontology_run_id: str | None = None

    @field_validator("mission_id", "name", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("schedule_label", "created_at", "updated_at", "ontology_run_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class OptimizationRunV1(OntologySchemaBase):
    run_id: NonBlankStr
    id: str | int | None = None
    legacy_id: int | None = None
    mission_id: NonBlankStr
    mission_name: str | None = None
    status: NonBlankStr = "running"
    started_at: str | None = None
    completed_at: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    source_freshness: dict[str, Any] = Field(default_factory=dict)
    input_hash: str | None = None
    output_hash: str | None = None
    error: str | None = None
    snapshots: list[dict[str, Any]] | None = None
    updated_at: str | None = None
    ontology_run_id: str | None = None

    @field_validator("run_id", "mission_id", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "mission_name",
        "started_at",
        "completed_at",
        "input_hash",
        "output_hash",
        "error",
        "updated_at",
        "ontology_run_id",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class OptimizationActionSnapshotV1(OntologySchemaBase):
    snapshot_id: NonBlankStr
    id: str | int | None = None
    legacy_id: int | None = None
    run_id: NonBlankStr
    mission_id: NonBlankStr
    ticker: str | None = None
    asset: str | None = None
    direction: str | None = None
    action: NonBlankStr
    conviction_band: str | None = None
    priority_score: float | None = None
    scenario_score: float | None = None
    score_delta: float | None = None
    confidence: float | None = None
    gate_status: str | None = None
    severity: str | None = None
    risk: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    source_links: dict[str, Any] = Field(default_factory=dict)
    state_hash: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    ontology_run_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("snapshot_id", "run_id", "mission_id", "action", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "asset",
        "direction",
        "conviction_band",
        "gate_status",
        "severity",
        "state_hash",
        "created_at",
        "updated_at",
        "ontology_run_id",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class OptimizationAlertV1(OntologySchemaBase):
    alert_id: NonBlankStr
    id: str | int | None = None
    legacy_id: int | None = None
    mission_id: NonBlankStr
    run_id: NonBlankStr
    ticker: str | None = None
    alert_type: NonBlankStr
    severity: NonBlankStr = "normal"
    status: NonBlankStr = "open"
    change_summary: NonBlankStr
    previous_snapshot_id: str | int | None = None
    current_snapshot_id: str | int | None = None
    approval_id: str | int | None = None
    recommendation_id: str | int | None = None
    action_item_approval_id: str | int | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    previous_snapshot: dict[str, Any] | None = None
    current_snapshot: dict[str, Any] | None = None
    dismissal_note: str | None = None
    dismissed_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    ontology_run_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator(
        "alert_id", "mission_id", "run_id", "alert_type", "severity", "status", "change_summary", mode="before"
    )
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("dismissal_note", "dismissed_at", "created_at", "updated_at", "ontology_run_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class SourceFreshnessV1(OntologySchemaBase):
    freshness_id: NonBlankStr
    id: str | int | None = None
    parent_uid: str | None = None
    parent_type: str | None = None
    source_name: NonBlankStr
    status: NonBlankStr = "unknown"
    checked_at: str | None = None
    as_of: str | None = None
    freshness_category: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ontology_run_id: str | None = None

    @field_validator("freshness_id", "source_name", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "parent_uid",
        "parent_type",
        "checked_at",
        "as_of",
        "freshness_category",
        "error",
        "ontology_run_id",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ManagementQualityAssessmentV1(OntologySchemaBase):
    assessment_id: NonBlankStr
    id: str | int | None = None
    issuer_id: NonBlankStr
    ticker: str | None = None
    status: NonBlankStr = "active"
    overall_rating: str | None = None
    bottom_line: str | None = None
    owner_mindset_rating: str | None = None
    owner_mindset_text: str | None = None
    business_value_understanding_rating: str | None = None
    business_value_understanding_text: str | None = None
    follow_through_rating: str | None = None
    follow_through_text: str | None = None
    content_hash: str | None = None
    document_id: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    ontology_run_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("assessment_id", "issuer_id", "status", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "overall_rating",
        "bottom_line",
        "owner_mindset_rating",
        "owner_mindset_text",
        "business_value_understanding_rating",
        "business_value_understanding_text",
        "follow_through_rating",
        "follow_through_text",
        "content_hash",
        "document_id",
        "source_type",
        "source_id",
        "created_at",
        "updated_at",
        "ontology_run_id",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ManagementQualityScorecardRowV1(OntologySchemaBase):
    row_id: NonBlankStr
    id: str | int | None = None
    assessment_id: NonBlankStr
    issuer_id: NonBlankStr
    ticker: str | None = None
    question: NonBlankStr
    rating: NonBlankStr
    evidence: str | None = None
    ordinal: int = Field(default=0, ge=0)
    ontology_run_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("row_id", "assessment_id", "issuer_id", "question", "rating", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("evidence", "ontology_run_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ManagementQualityAccomplishmentV1(OntologySchemaBase):
    accomplishment_id: NonBlankStr
    id: str | int | None = None
    assessment_id: NonBlankStr
    issuer_id: NonBlankStr
    ticker: str | None = None
    title: str | None = None
    text: NonBlankStr
    period: str | None = None
    ordinal: int = Field(default=0, ge=0)
    ontology_run_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("accomplishment_id", "assessment_id", "issuer_id", "text", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("title", "period", "ontology_run_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class ManagementQualitySetbackV1(OntologySchemaBase):
    setback_id: NonBlankStr
    id: str | int | None = None
    assessment_id: NonBlankStr
    issuer_id: NonBlankStr
    ticker: str | None = None
    title: str | None = None
    text: NonBlankStr
    response_rating: str | None = None
    response_text: str | None = None
    ordinal: int = Field(default=0, ge=0)
    ontology_run_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _optional_ticker(cls, value: object) -> str | None:
        return canonical_ticker(value) if clean_optional_text(value) else None

    @field_validator("setback_id", "assessment_id", "issuer_id", "text", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("title", "response_rating", "response_text", "ontology_run_id", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


OntologyObjectV1 = (
    PositionV1
    | AssetV1
    | InstrumentV1
    | IssuerV1
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
    | SourceRecordV1
    | ObjectVersionRefV1
    | ExecutedActionV1
    | ExecutedDecisionRecordV1
    | AuditEventV1
    | SectorV1
    | MacroIndicatorV1
    | SignalV1
    | ThesisV1
    | EvaluationV1
    | CatalystV1
    | HedgePositionV1
    | KillConditionV1
    | ThesisClaimV1
    | EvidenceV1
    | CitationV1
    | ActionItemV1
    | WatchTriggerV1
    | ResearchNoteV1
    | ApprovalV1
    | ActionRunV1
    | ActionEventV1
    | ProvenanceEventV1
    | RelationVersionRefV1
    | SchemaDefinitionRefV1
    | OntologyRunRefV1
    | AgentSessionRefV1
    | ModelCallRefV1
    | ToolCallRefV1
    | ComputedSnapshotRefV1
    | WorkflowRunV1
    | WorkflowArtifactV1
    | RecommendationV1
    | ReportRunV1
    | DocumentArtifactV1
    | InvestmentIdeaV1
    | IdeaEvaluationV1
    | IdeaComparisonRunV1
    | IdeaComparisonRankingV1
    | FactorScoreV1
    | MissingInformationRequirementV1
    | OptimizationMissionV1
    | OptimizationRunV1
    | OptimizationActionSnapshotV1
    | OptimizationAlertV1
    | SourceFreshnessV1
    | ManagementQualityAssessmentV1
    | ManagementQualityScorecardRowV1
    | ManagementQualityAccomplishmentV1
    | ManagementQualitySetbackV1
)
JsonObject = dict[str, Any]
