"""Pydantic models for the OpportunityCandidate pre-decision triage object."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator

from decision_quality.actions import ACTIONABLE_ACTIONS
from decision_quality.models import OpportunityType, StrictModel

CandidateNextAction = Literal[
    "watch",
    "research",
    "avoid",
    "do_nothing",
    "graduate_to_decision_quality",
]

CANDIDATE_NEXT_ACTIONS: tuple[str, ...] = (
    "watch",
    "research",
    "avoid",
    "do_nothing",
    "graduate_to_decision_quality",
)

GRADUATE_ACTION = "graduate_to_decision_quality"

CandidateSourceKind = Literal[
    "agent_chat",
    "monitor_hit",
    "idea_watchlist",
    "workflow",
    "manual",
    "other",
]

_SOURCE_KINDS = {
    "agent_chat",
    "monitor_hit",
    "idea_watchlist",
    "workflow",
    "manual",
    "other",
}

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


class CandidateSourceRef(StrictModel):
    source_record_id: str | None = None
    document_artifact_id: str | None = None
    url: str | None = None
    source_path: str | None = None
    label: str | None = None


class OpportunityCandidate(StrictModel):
    ticker: str | None = None
    source: CandidateSourceKind = "agent_chat"
    trigger: str
    opportunity_type: OpportunityType = "unclear"
    consensus: str
    variant_view: str
    why_now: str
    price_confirmation: str
    crowding: str = ""
    payoff_asymmetry: str = ""
    missing_inputs: list[str] = Field(default_factory=list)
    source_refs: list[CandidateSourceRef] = Field(default_factory=list)
    next_action: CandidateNextAction = "research"
    summary: str = ""

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: object) -> str:
        text = str(value or "agent_chat").strip().lower()
        return text if text in _SOURCE_KINDS else "other"

    @field_validator("opportunity_type", mode="before")
    @classmethod
    def _normalize_opportunity_type(cls, value: object) -> str:
        text = str(value or "unclear").strip().lower()
        return text if text in _OPPORTUNITY_TYPES else "unclear"

    @field_validator("next_action", mode="before")
    @classmethod
    def _normalize_next_action(cls, value: object) -> str:
        return _oc_next_action(value)

    @field_validator("ticker", mode="before")
    @classmethod
    def _normalize_ticker(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip().upper()
        return text or None

    @field_validator("missing_inputs", mode="before")
    @classmethod
    def _normalize_missing_inputs(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


class OpportunityCandidateGateReason(StrictModel):
    code: str
    severity: Literal["info", "warning", "blocker"]
    message: str


class OpportunityCandidateGate(StrictModel):
    status: Literal["pass", "downgraded", "blocked", "invalid"]
    original_action: str
    final_action: str
    should_graduate: bool
    reasons: list[OpportunityCandidateGateReason] = Field(default_factory=list)


def opportunity_candidate_schema() -> dict[str, Any]:
    return OpportunityCandidate.model_json_schema()


def _oc_text(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _oc_pick(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _oc_source_refs(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    items: list[Any]
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    refs: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            refs.append(
                {
                    "source_record_id": _oc_text(item.get("source_record_id")) or None,
                    "document_artifact_id": _oc_text(item.get("document_artifact_id")) or None,
                    "url": _oc_text(item.get("url")) or None,
                    "source_path": _oc_text(item.get("source_path")) or None,
                    "label": _oc_text(item.get("label")) or None,
                }
            )
        elif isinstance(item, str) and item.strip():
            refs.append({"label": item.strip(), "source_path": item.strip()})
    return refs


def _oc_next_action(value: object) -> str:
    action = _oc_text(value, default="research").lower()
    aliases = {
        "graduate": "graduate_to_decision_quality",
        "graduate_to_dq": "graduate_to_decision_quality",
        "pressure_test": "graduate_to_decision_quality",
        "full_dq": "graduate_to_decision_quality",
        "reject": "avoid",
        "pass": "do_nothing",
        "no_action": "do_nothing",
    }
    action = aliases.get(action, action)
    if action in ACTIONABLE_ACTIONS:
        return GRADUATE_ACTION
    return action if action in CANDIDATE_NEXT_ACTIONS else "research"


def coerce_opportunity_candidate_input(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    price_confirmation = _oc_pick(value, "price_confirmation", "price_action", "technical_confirmation")
    if isinstance(price_confirmation, dict):
        price_confirmation = _oc_text(
            _oc_pick(price_confirmation, "observed_behavior", "interpretation", "summary")
        )

    return {
        "ticker": _oc_pick(value, "ticker", "symbol"),
        "source": _oc_text(_oc_pick(value, "source", "source_kind"), default="agent_chat").lower(),
        "trigger": _oc_text(_oc_pick(value, "trigger", "source_trigger", "attention_trigger")),
        "opportunity_type": _oc_text(_oc_pick(value, "opportunity_type"), default="unclear"),
        "consensus": _oc_text(_oc_pick(value, "consensus", "consensus_view", "market_view")),
        "variant_view": _oc_text(_oc_pick(value, "variant_view", "variant", "differentiated_view")),
        "why_now": _oc_text(_oc_pick(value, "why_now", "reason_now", "catalyst")),
        "price_confirmation": _oc_text(price_confirmation),
        "crowding": _oc_text(_oc_pick(value, "crowding", "crowding_risk")),
        "payoff_asymmetry": _oc_text(_oc_pick(value, "payoff_asymmetry", "asymmetry", "payoff")),
        "missing_inputs": _oc_pick(value, "missing_inputs", "missing_information", "gaps") or [],
        "source_refs": _oc_source_refs(_oc_pick(value, "source_refs", "evidence_refs", "sources")),
        "next_action": _oc_next_action(_oc_pick(value, "next_action", "recommended_next_step", "triage_action")),
        "summary": _oc_text(_oc_pick(value, "summary", "simple_thesis", "headline")),
    }


def parse_opportunity_candidate(value: Any) -> tuple[OpportunityCandidate | None, list[str]]:
    if value is None:
        return None, ["opportunity_candidate is missing"]
    try:
        return OpportunityCandidate.model_validate(coerce_opportunity_candidate_input(value)), []
    except ValidationError as exc:
        errors: list[str] = []
        for error in exc.errors():
            loc = ".".join(str(item) for item in error.get("loc", ()))
            message = str(error.get("msg") or "validation error")
            errors.append(f"{loc}: {message}" if loc else message)
        return None, errors


def opportunity_candidate_to_json(candidate: OpportunityCandidate) -> str:
    return json.dumps(candidate.model_dump(mode="json"), ensure_ascii=True, sort_keys=True)
