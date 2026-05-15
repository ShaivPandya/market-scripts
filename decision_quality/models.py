"""Pydantic models for the structured decision-quality object."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from decision_quality.actions import CANONICAL_ACTIONS, CanonicalAction


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
    "crowded_narrative_avoid",
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


_OPPORTUNITY_TYPES = {
    "undervalued_asset",
    "regime_shift",
    "reflexive_process",
    "unsustainable_process",
    "forced_liquidation",
    "policy_inflection",
    "quality_compounder",
    "cyclical_upturn",
    "crowded_narrative_avoid",
    "unclear",
}


def _dq_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return str(value)


def _dq_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dq_first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return {}


def _dq_pick(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _dq_strings(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return [text for item in value if (text := _dq_text(item))]
    return [text] if (text := _dq_text(value)) else []


def _dq_bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = _dq_text(value).lower()
    if normalized.startswith(("yes", "true", "confirmed", "supports")):
        return True
    if normalized.startswith(("no", "false", "not confirmed", "does not", "against")):
        return False
    return None


def _dq_evidence_items(value: Any) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else ([value] if value not in (None, "", {}, []) else [])
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            claim = _dq_text(_dq_pick(item, "claim", "summary", "title", "point", "finding", "evidence"))
            support = _dq_text(_dq_pick(item, "support", "detail", "rationale", "summary", "evidence", "claim"))
            sources = _dq_strings(_dq_pick(item, "source_refs", "source_ref", "sources", "source", "url", "citation"))
        else:
            claim = _dq_text(item)
            support = claim
            sources = []
        if claim or support:
            normalized.append({"claim": claim or support, "support": support or claim, "source_refs": sources})
    return normalized


def _dq_choice(value: Any, allowed: set[str], fallback: str) -> str:
    normalized = _dq_text(value).lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in allowed else fallback


def _dq_directness(value: Any) -> str:
    return _dq_choice(value, {"direct", "proxy", "mixed", "not_applicable"}, "not_applicable")


def _dq_action(value: Any) -> str:
    action = _dq_text(value).lower()
    return action if action in CANONICAL_ACTIONS else "watch"


def _dq_actionability(value: Any, recommended_action: Any = None) -> dict[str, Any]:
    data = _dq_dict(value)
    status = _dq_text(_dq_pick(data, "status", "actionability_status")).lower().replace("-", "_").replace(" ", "_")
    if status in {"watch", "watch_only"}:
        status = "watch_only"
    elif status in {"research", "missing", "missing_input", "missing_inputs", "needs_research"}:
        status = "missing_inputs"
    elif status in {"blocked", "blocked_by_policy"}:
        status = "blocked_by_policy"
    elif status in {"do_nothing", "nothing"}:
        status = "do_nothing"
    elif status != "actionable":
        action = _dq_text(recommended_action).lower()
        status = (
            "actionable"
            if action in {"buy", "add", "short", "sell", "trim", "reduce", "exit", "hedge", "rebalance"}
            else "watch_only"
        )
    return {
        "status": status,
        "reason": _dq_text(_dq_pick(data, "reason", "rationale", "why", "summary")),
        "missing_inputs": _dq_strings(_dq_pick(data, "missing_inputs", "missing", "needs")),
    }


def _dq_sizing_delta(value: Any) -> dict[str, Any]:
    data = _dq_dict(value)
    direction_aliases = {
        "add": "increase",
        "increase": "increase",
        "up": "increase",
        "trim": "decrease",
        "reduce": "decrease",
        "decrease": "decrease",
        "down": "decrease",
        "hold": "hold",
        "same": "hold",
        "flat": "hold",
        "no_change": "hold",
        "exit": "exit",
        "sell_all": "exit",
        "n/a": "not_applicable",
        "na": "not_applicable",
        "none": "not_applicable",
        "not_applicable": "not_applicable",
    }
    unit_aliases = {
        "%": "portfolio_weight",
        "percent": "portfolio_weight",
        "weight": "portfolio_weight",
        "portfolio_weight": "portfolio_weight",
        "bps": "bps",
        "basis_points": "bps",
        "shares": "shares",
        "contracts": "contracts",
        "notional": "notional",
        "fraction": "fraction_of_position",
        "fraction_of_position": "fraction_of_position",
        "percent_of_position": "fraction_of_position",
        "n/a": "not_applicable",
        "na": "not_applicable",
        "none": "not_applicable",
        "not_applicable": "not_applicable",
    }
    basis_aliases = {
        "target": "target_weight",
        "target_weight": "target_weight",
        "current": "current_position",
        "current_position": "current_position",
        "current_weight": "current_position",
        "position": "current_position",
        "risk": "risk_budget",
        "risk_budget": "risk_budget",
        "gross": "gross_exposure",
        "gross_exposure": "gross_exposure",
        "n/a": "not_applicable",
        "na": "not_applicable",
        "none": "not_applicable",
        "not_applicable": "not_applicable",
    }
    direction = direction_aliases.get(_dq_text(data.get("direction")).lower().replace(" ", "_"), "not_applicable")
    unit = unit_aliases.get(_dq_text(data.get("unit")).lower().replace(" ", "_"), "not_applicable")
    basis = basis_aliases.get(_dq_text(data.get("basis")).lower().replace(" ", "_"), "not_applicable")
    amount = data.get("amount")
    try:
        amount = None if amount in (None, "") else float(amount)
    except (TypeError, ValueError):
        amount = None
    return {
        "direction": direction,
        "amount": amount,
        "unit": unit,
        "basis": basis,
        "condition": _dq_text(_dq_pick(data, "condition", "when", "trigger", "rationale")),
    }


def _dq_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if 1 < confidence <= 100:
        confidence = confidence / 100
    return max(0, min(1, confidence))


def _dq_float_or_none(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def coerce_decision_quality_input(value: Any) -> Any:
    """Normalize common LLM schema variants before strict validation."""
    if not isinstance(value, dict):
        return value

    recommended_action = _dq_pick(value, "recommended_action", "action")
    catalyst = _dq_dict(_dq_pick(value, "catalyst_or_reason_now", "catalyst", "reason_now"))
    invalidation = _dq_first_dict(_dq_pick(value, "invalidation", "kill_condition", "invalidation_condition"))
    price_action = _dq_dict(_dq_pick(value, "price_action_read", "price_action", "technical_confirmation"))
    conviction = _dq_pick(value, "conviction", "conviction_context")
    conviction_data = {"level": conviction} if isinstance(conviction, int) else _dq_dict(conviction)
    sizing_context = _dq_pick(value, "sizing_context", "sizing")
    sizing_data = {"starting_size": sizing_context} if isinstance(sizing_context, str) else _dq_dict(sizing_context)
    expression = _dq_dict(_dq_pick(value, "expression", "instrument", "trade_expression"))
    trade_after_trade = _dq_pick(value, "trade_after_trade", "next_trade")
    trade_after_trade_data = (
        {"if_right": trade_after_trade, "if_wrong": "", "next_review_trigger": ""}
        if isinstance(trade_after_trade, str)
        else _dq_dict(trade_after_trade)
    )

    return {
        "simple_thesis": _dq_text(_dq_pick(value, "simple_thesis", "thesis", "thesis_statement", "summary")),
        "opportunity_type": _dq_choice(value.get("opportunity_type"), _OPPORTUNITY_TYPES, "unclear"),
        "embedded_macro_exposure": _dq_text(_dq_pick(value, "embedded_macro_exposure", "macro_exposure")),
        "mispricing": {
            "consensus_view": _dq_text(
                _dq_pick(_dq_dict(value.get("mispricing")), "consensus_view", "consensus", "market_view")
            ),
            "variant_view": _dq_text(
                _dq_pick(_dq_dict(value.get("mispricing")), "variant_view", "variant", "differentiated_view")
            ),
            "pricing_evidence": _dq_text(
                _dq_pick(
                    _dq_dict(value.get("mispricing")),
                    "pricing_evidence",
                    "what_is_priced",
                    "pricing",
                    "valuation",
                    "evidence",
                )
            ),
            "why_consensus_is_wrong": _dq_text(
                _dq_pick(
                    _dq_dict(value.get("mispricing")),
                    "why_consensus_is_wrong",
                    "why_consensus_may_be_wrong",
                    "why_wrong",
                    "edge",
                )
            ),
        },
        "catalyst_or_reason_now": {
            "event_or_condition": _dq_text(
                _dq_pick(catalyst, "event_or_condition", "event", "primary", "condition", "reason")
            ),
            "expected_timeframe": _dq_text(
                _dq_pick(catalyst, "expected_timeframe", "timeframe", "when", "expected_date", "status")
            ),
            "why_now": _dq_text(_dq_pick(catalyst, "why_now", "why", "reason_now", "primary")),
            "source_evidence": _dq_strings(_dq_pick(catalyst, "source_evidence", "evidence", "sources", "source_refs")),
        },
        "invalidation": {
            "observable": _dq_text(
                _dq_pick(invalidation, "observable", "metric", "metric_or_event", "event", "condition")
            ),
            "metric_or_event": _dq_text(_dq_pick(invalidation, "metric_or_event", "metric", "event", "condition")),
            "threshold": _dq_text(_dq_pick(invalidation, "threshold", "level", "trigger")),
            "timeframe": _dq_text(_dq_pick(invalidation, "timeframe", "when", "period")),
            "implication": _dq_text(_dq_pick(invalidation, "implication", "why_it_matters", "then", "consequence")),
        },
        "evidence_for": _dq_evidence_items(_dq_pick(value, "evidence_for", "supporting_evidence", "evidence")),
        "evidence_against": _dq_evidence_items(_dq_pick(value, "evidence_against", "disconfirming_evidence", "risks")),
        "price_action_read": {
            "observed_behavior": _dq_text(
                _dq_pick(price_action, "observed_behavior", "what_price_did", "observed", "behavior", "price_action")
            ),
            "interpretation": _dq_text(_dq_pick(price_action, "interpretation", "what_it_implies", "read", "meaning")),
            "confirms_thesis": _dq_bool_or_none(
                _dq_pick(price_action, "confirms_thesis", "confirms", "supports_thesis")
            ),
            "data_needed": _dq_strings(_dq_pick(price_action, "data_needed", "missing_data", "missing", "needed")),
        },
        "actionability": _dq_actionability(
            _dq_pick(value, "actionability", "actionability_status"), recommended_action
        ),
        "recommended_action": _dq_action(recommended_action or "watch"),
        "expression": {
            "primary": _dq_text(_dq_pick(expression, "primary", "instrument", "ticker", "asset")),
            "instrument_type": _dq_text(_dq_pick(expression, "instrument_type", "type", "asset_class")),
            "directness": _dq_directness(_dq_pick(expression, "directness", "direct_or_proxy")),
            "alternatives": _dq_strings(_dq_pick(expression, "alternatives", "alternative_expressions")),
            "follow_on": _dq_text(_dq_pick(expression, "follow_on", "follow_on_trade", "next_trade")),
        },
        "conviction": {
            "level": conviction_data.get("level"),
            "max_level": conviction_data.get("max_level") or 5,
            "raw_target_weight": _dq_float_or_none(
                conviction_data.get("raw_target_weight") or conviction_data.get("target_weight")
            ),
            "upgrade_condition": _dq_text(_dq_pick(conviction_data, "upgrade_condition", "upgrade", "path_to_5")),
        },
        "confidence": _dq_confidence(value.get("confidence")),
        "confidence_reason": _dq_text(
            _dq_pick(value, "confidence_reason", "confidence_rationale", "confidence_explanation")
        ),
        "sizing_context": {
            "starting_size": _dq_text(_dq_pick(sizing_data, "starting_size", "start", "initial_size")),
            "add_conditions": _dq_text(_dq_pick(sizing_data, "add_conditions", "add_condition", "upgrade_conditions")),
            "liquidity_constraints": _dq_text(_dq_pick(sizing_data, "liquidity_constraints", "liquidity")),
            "portfolio_constraints": _dq_text(_dq_pick(sizing_data, "portfolio_constraints", "portfolio")),
            "sizing_delta": _dq_sizing_delta(_dq_pick(sizing_data, "sizing_delta", "delta")),
        },
        "trade_after_trade": {
            "if_right": _dq_text(_dq_pick(trade_after_trade_data, "if_right", "right", "upside", "next_if_right")),
            "if_wrong": _dq_text(_dq_pick(trade_after_trade_data, "if_wrong", "wrong", "downside", "next_if_wrong")),
            "next_review_trigger": _dq_text(
                _dq_pick(trade_after_trade_data, "next_review_trigger", "review_trigger", "trigger")
            ),
        },
    }


def parse_decision_quality(value: Any) -> tuple[DecisionQuality | None, list[str]]:
    if value is None:
        return None, ["decision_quality is missing"]
    try:
        return DecisionQuality.model_validate(coerce_decision_quality_input(value)), []
    except ValidationError as exc:
        errors: list[str] = []
        for error in exc.errors():
            loc = ".".join(str(item) for item in error.get("loc", ()))
            message = str(error.get("msg") or "validation error")
            errors.append(f"{loc}: {message}" if loc else message)
        return None, errors
