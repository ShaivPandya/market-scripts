"""Pydantic models for the structured decision-quality object."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from decision_quality.actions import CanonicalAction


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


OpportunityType = Literal[
    "undervalued_asset",
    "regime_shift",
    "reflexive_process",
    "unsustainable_process",
    "forced_liquidation",
    "policy_inflection",
    "quality_compounder",
    "cyclical_upturn",
    "unclear",
]

ActionabilityStatus = Literal["actionable", "missing_inputs", "blocked_by_policy", "watch_only", "do_nothing"]
ExpressionDirectness = Literal["direct", "proxy", "mixed", "not_applicable"]
SizingDeltaDirection = Literal["increase", "decrease", "hold", "exit", "not_applicable"]
SizingDeltaUnit = Literal[
    "portfolio_weight",
    "bps",
    "shares",
    "contracts",
    "notional",
    "fraction_of_position",
    "not_applicable",
]
SizingDeltaBasis = Literal[
    "target_weight",
    "current_position",
    "risk_budget",
    "gross_exposure",
    "not_applicable",
]


class EvidenceItem(StrictModel):
    claim: str
    support: str
    source_refs: list[str] = Field(default_factory=list)


class Mispricing(StrictModel):
    consensus_view: str
    variant_view: str
    pricing_evidence: str
    why_consensus_is_wrong: str


class CatalystOrReasonNow(StrictModel):
    event_or_condition: str
    expected_timeframe: str
    why_now: str
    source_evidence: list[str] = Field(default_factory=list)


class Invalidation(StrictModel):
    observable: str
    metric_or_event: str
    threshold: str
    timeframe: str
    implication: str


class PriceActionRead(StrictModel):
    observed_behavior: str
    interpretation: str
    confirms_thesis: bool | None
    data_needed: list[str] = Field(default_factory=list)


class Actionability(StrictModel):
    status: ActionabilityStatus
    reason: str
    missing_inputs: list[str] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _legacy_statuses(cls, value: object) -> object:
        normalized = str(value or "").strip().lower()
        if normalized in {"watch", "research"}:
            return "watch_only" if normalized == "watch" else "missing_inputs"
        return value


class Expression(StrictModel):
    primary: str
    instrument_type: str
    directness: ExpressionDirectness
    alternatives: list[str] = Field(default_factory=list)
    follow_on: str


class Conviction(StrictModel):
    level: int | None = Field(default=None, ge=1, le=5)
    max_level: Literal[5] = 5
    raw_target_weight: float | None = None
    upgrade_condition: str


class SizingDelta(StrictModel):
    direction: SizingDeltaDirection = "not_applicable"
    amount: float | None = None
    unit: SizingDeltaUnit = "not_applicable"
    basis: SizingDeltaBasis = "not_applicable"
    condition: str = ""


class SizingContext(StrictModel):
    starting_size: str
    add_conditions: str
    liquidity_constraints: str
    portfolio_constraints: str
    sizing_delta: SizingDelta = Field(default_factory=SizingDelta)


class TradeAfterTrade(StrictModel):
    if_right: str
    if_wrong: str
    next_review_trigger: str


class DecisionQuality(StrictModel):
    simple_thesis: str
    opportunity_type: OpportunityType
    embedded_macro_exposure: str = ""
    mispricing: Mispricing
    catalyst_or_reason_now: CatalystOrReasonNow
    invalidation: Invalidation
    evidence_for: list[EvidenceItem] = Field(default_factory=list)
    evidence_against: list[EvidenceItem] = Field(default_factory=list)
    price_action_read: PriceActionRead
    actionability: Actionability
    recommended_action: CanonicalAction
    expression: Expression
    conviction: Conviction
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_reason: str
    sizing_context: SizingContext
    trade_after_trade: TradeAfterTrade


class DecisionQualityGateReason(StrictModel):
    code: str
    severity: Literal["info", "warning", "blocker"]
    message: str


class DecisionQualityGate(StrictModel):
    status: Literal["pass", "downgraded", "blocked", "invalid"]
    original_action: str
    final_action: str
    original_recommendation_status: str
    final_recommendation_status: str
    confidence_cap: float | None = Field(default=None, ge=0, le=1)
    reasons: list[DecisionQualityGateReason] = Field(default_factory=list)


def decision_quality_schema() -> dict[str, Any]:
    return DecisionQuality.model_json_schema()


def parse_decision_quality(value: Any) -> tuple[DecisionQuality | None, list[str]]:
    if value is None:
        return None, ["decision_quality is missing"]
    try:
        return DecisionQuality.model_validate(value), []
    except ValidationError as exc:
        return None, [error["msg"] for error in exc.errors()]
