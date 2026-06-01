"""Scout/skeptic/sizer hard gate for high-stakes proactive alert action items."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from decision_quality.actions import ACTIONABLE_ACTIONS, normalize_action
from decision_quality.candidate_gates import apply_opportunity_candidate_gates
from decision_quality.gates import apply_decision_quality_gates
from decision_quality.models import StrictModel, parse_decision_quality
from decision_quality.opportunity_candidate import (
    OpportunityCandidate,
    parse_opportunity_candidate,
)

PassStatus = Literal["pass", "fail", "missing", "skipped"]
GateStatus = Literal["pass", "downgraded", "blocked", "not_applicable", "disabled"]

HIGH_STAKES_ACTION_TYPES = frozenset({"resize", "exit"})
WORKFLOW_SOURCE_TYPES = frozenset({"workflow", "system"})
MONITOR_DOWNGRADE_ACTION_TYPES = frozenset({"research", "review", "watch"})

_FALSE_VALUES = {"0", "false", "no", "off"}


class PassArtifact(StrictModel):
    ran: bool = False
    status: PassStatus = "missing"
    summary: str = ""
    reason_codes: list[str] = Field(default_factory=list)


class ScoutPassArtifact(PassArtifact):
    trigger: str | None = None
    opportunity_type: str | None = None
    variant_view: str | None = None
    why_now: str | None = None
    next_action: str | None = None


class SkepticPassArtifact(PassArtifact):
    gate_status: str | None = None
    should_graduate: bool | None = None
    final_action: str | None = None


class SizerPassArtifact(PassArtifact):
    gate_status: str | None = None
    final_action: str | None = None
    max_sizing_delta_bps: int | None = None


class ProactiveAlertGateResult(StrictModel):
    enabled: bool = True
    applied: bool = False
    gate_status: GateStatus = "not_applicable"
    action_allowed: bool = True
    original_action_type: str | None = None
    final_action_type: str | None = None
    scout: ScoutPassArtifact = Field(default_factory=ScoutPassArtifact)
    skeptic: SkepticPassArtifact = Field(default_factory=SkepticPassArtifact)
    sizer: SizerPassArtifact = Field(default_factory=SizerPassArtifact)
    summary: str = ""

    def to_trace_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def proactive_alert_gate_enabled() -> bool:
    raw = os.environ.get("PROACTIVE_ALERT_DQ_GATE_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSE_VALUES


def is_high_stakes_action_item(payload: Mapping[str, Any]) -> bool:
    action_type = str(payload.get("action_type") or "").strip().lower()
    return action_type in HIGH_STAKES_ACTION_TYPES


def should_apply_proactive_alert_gate(action_id: str, source_type: str) -> bool:
    if not proactive_alert_gate_enabled():
        return False
    if str(action_id or "").strip() != "create_action_item":
        return False
    return str(source_type or "").strip().lower() in WORKFLOW_SOURCE_TYPES


def _nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_nonempty(item) for item in value)
    return True


def _action_type_to_canonical(action_type: str) -> str:
    mapping = {
        "resize": "trim",
        "exit": "exit",
        "research": "research",
        "review": "research",
        "watch": "watch",
    }
    return normalize_action(mapping.get(action_type, action_type))


def _downgrade_action_type(original_action_type: str, *, final_action: str | None = None) -> str:
    normalized = str(final_action or "").strip().lower()
    if normalized in {"watch", "research", "avoid", "do_nothing"}:
        return "research" if normalized != "watch" else "research"
    if original_action_type == "exit":
        return "review"
    return "research"


def _extract_alert_context(
    payload: Mapping[str, Any],
    alert_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    context = dict(alert_context or {})
    description = str(payload.get("description") or "").strip()
    if description and not context.get("change_summary"):
        context.setdefault("change_summary", description.split("\n", 1)[0].strip())
    if payload.get("ticker") and not context.get("ticker"):
        context["ticker"] = payload.get("ticker")
    return context


def _resolve_opportunity_candidate(
    payload: Mapping[str, Any],
    alert_context: Mapping[str, Any] | None,
) -> tuple[OpportunityCandidate | None, list[str]]:
    embedded = payload.get("opportunity_candidate")
    if embedded is not None:
        return parse_opportunity_candidate(embedded)

    context = _extract_alert_context(payload, alert_context)
    trigger = str(context.get("change_summary") or context.get("trigger") or "").strip()
    description = str(payload.get("description") or "").strip()
    if not trigger and description:
        trigger = description.split("\n", 1)[0].strip()
    if not trigger:
        return None, ["opportunity_candidate is missing"]

    ticker = str(payload.get("ticker") or context.get("ticker") or "").strip().upper() or None
    source = str(context.get("source") or "workflow").strip().lower()
    if source not in {"monitor_hit", "workflow", "idea_watchlist", "manual", "other", "agent_chat"}:
        source = "workflow"

    candidate = OpportunityCandidate(
        ticker=ticker,
        source=source,  # type: ignore[arg-type]
        trigger=trigger,
        opportunity_type=str(context.get("opportunity_type") or "unclear"),  # type: ignore[arg-type]
        consensus=str(context.get("consensus") or "Consensus not established in automated alert."),
        variant_view=str(context.get("variant_view") or description or trigger),
        why_now=str(context.get("why_now") or trigger),
        price_confirmation=str(context.get("price_confirmation") or "Not verified in automated alert path."),
        crowding=str(context.get("crowding") or ""),
        payoff_asymmetry=str(context.get("payoff_asymmetry") or ""),
        missing_inputs=[str(item) for item in context.get("missing_inputs") or []]
        or ["decision_quality pressure-test", "sizing context"],
        next_action="graduate_to_decision_quality",
        summary=str(context.get("summary") or trigger),
    )
    return candidate, []


def _evaluate_scout_pass(
    payload: Mapping[str, Any],
    alert_context: Mapping[str, Any] | None,
) -> ScoutPassArtifact:
    candidate, parse_errors = _resolve_opportunity_candidate(payload, alert_context)
    if candidate is None:
        return ScoutPassArtifact(
            ran=True,
            status="fail",
            summary="Scout pass could not build an OpportunityCandidate.",
            reason_codes=[code for code in parse_errors if code] or ["MISSING_OPPORTUNITY_CANDIDATE"],
        )

    missing_fields = [
        name
        for name, value in (
            ("trigger", candidate.trigger),
            ("variant_view", candidate.variant_view),
            ("why_now", candidate.why_now),
        )
        if not _nonempty(value)
    ]
    if missing_fields:
        return ScoutPassArtifact(
            ran=True,
            status="fail",
            summary=f"Scout pass missing required fields: {', '.join(missing_fields)}.",
            reason_codes=[f"MISSING_{field.upper()}" for field in missing_fields],
            trigger=candidate.trigger,
            opportunity_type=str(candidate.opportunity_type),
            variant_view=candidate.variant_view,
            why_now=candidate.why_now,
            next_action=candidate.next_action,
        )

    return ScoutPassArtifact(
        ran=True,
        status="pass",
        summary="Scout pass identified trigger, variant view, and why-now from the alert.",
        trigger=candidate.trigger,
        opportunity_type=str(candidate.opportunity_type),
        variant_view=candidate.variant_view,
        why_now=candidate.why_now,
        next_action=candidate.next_action,
    )


def _evaluate_skeptic_pass(
    payload: Mapping[str, Any],
    alert_context: Mapping[str, Any] | None,
) -> SkepticPassArtifact:
    candidate, parse_errors = _resolve_opportunity_candidate(payload, alert_context)
    context_pack = payload.get("context_pack") if isinstance(payload.get("context_pack"), dict) else None
    data_quality = payload.get("data_quality") if isinstance(payload.get("data_quality"), dict) else None
    gate = apply_opportunity_candidate_gates(
        candidate,
        parse_errors=parse_errors,
        context_pack=context_pack,
        data_quality=data_quality,
    )
    reason_codes = [reason.code for reason in gate.reasons if reason.severity in {"blocker", "warning"}]

    decision_quality, dq_parse_errors = parse_decision_quality(payload.get("decision_quality"))
    if decision_quality is None:
        reason_codes.append("MISSING_DECISION_QUALITY")
        return SkepticPassArtifact(
            ran=True,
            status="fail",
            summary="Skeptic pass blocked: decision quality artifact is required for high-stakes alerts.",
            reason_codes=reason_codes,
            gate_status=gate.status,
            should_graduate=gate.should_graduate,
            final_action=gate.final_action,
        )

    dq_gate = apply_decision_quality_gates(
        decision_quality,
        current_action=normalize_action(decision_quality.recommended_action),
        recommendation_status="clear",
        data_quality=data_quality,
        parse_errors=dq_parse_errors,
    )
    dq_reason_codes = [reason.code for reason in dq_gate.reasons if reason.severity == "blocker"]
    reason_codes.extend(dq_reason_codes)

    skeptic_pass = gate.should_graduate and dq_gate.status == "pass"
    if skeptic_pass:
        return SkepticPassArtifact(
            ran=True,
            status="pass",
            summary="Skeptic pass cleared graduation and decision-quality blockers.",
            reason_codes=reason_codes,
            gate_status=dq_gate.status,
            should_graduate=gate.should_graduate,
            final_action=dq_gate.final_action,
        )

    summary = "Skeptic pass blocked high-stakes actionability."
    if not gate.should_graduate:
        summary = "Skeptic pass blocked graduation before decision quality."
    elif dq_gate.status != "pass":
        summary = f"Skeptic pass blocked decision quality ({dq_gate.status})."

    return SkepticPassArtifact(
        ran=True,
        status="fail",
        summary=summary,
        reason_codes=reason_codes or ["SKEPTIC_BLOCKED"],
        gate_status=dq_gate.status,
        should_graduate=gate.should_graduate,
        final_action=dq_gate.final_action,
    )


def _evaluate_sizer_pass(
    payload: Mapping[str, Any],
    *,
    original_action_type: str,
    skeptic: SkepticPassArtifact,
) -> SizerPassArtifact:
    decision_quality, dq_parse_errors = parse_decision_quality(payload.get("decision_quality"))
    data_quality = payload.get("data_quality") if isinstance(payload.get("data_quality"), dict) else None
    canonical_action = _action_type_to_canonical(original_action_type)

    if decision_quality is None:
        return SizerPassArtifact(
            ran=True,
            status="fail",
            summary="Sizer pass blocked: sizing context requires decision quality.",
            reason_codes=["MISSING_DECISION_QUALITY"],
        )

    gate = apply_decision_quality_gates(
        decision_quality,
        current_action=canonical_action,
        recommendation_status="clear",
        data_quality=data_quality,
        parse_errors=dq_parse_errors,
    )
    reason_codes = [reason.code for reason in gate.reasons if reason.severity == "blocker"]
    final_action = normalize_action(gate.final_action or skeptic.final_action or "watch")

    sizing_delta = decision_quality.sizing_context.sizing_delta
    max_bps: int | None = None
    if sizing_delta is not None and str(sizing_delta.unit or "").strip().lower() == "bps":
        try:
            max_bps = int(float(sizing_delta.amount))
        except (TypeError, ValueError):
            max_bps = None

    if gate.status == "pass" and final_action in ACTIONABLE_ACTIONS:
        return SizerPassArtifact(
            ran=True,
            status="pass",
            summary="Sizer pass cleared actionable sizing within gate limits.",
            reason_codes=reason_codes,
            gate_status=gate.status,
            final_action=final_action,
            max_sizing_delta_bps=max_bps,
        )

    if final_action in MONITOR_DOWNGRADE_ACTION_TYPES or gate.status in {"downgraded", "blocked"}:
        return SizerPassArtifact(
            ran=True,
            status="pass",
            summary=f"Sizer pass limited action to {final_action}.",
            reason_codes=reason_codes or ["SIZING_LIMITED"],
            gate_status=gate.status,
            final_action=final_action,
            max_sizing_delta_bps=max_bps,
        )

    return SizerPassArtifact(
        ran=True,
        status="fail",
        summary="Sizer pass could not establish risk-aware actionability.",
        reason_codes=reason_codes or ["SIZING_BLOCKED"],
        gate_status=gate.status,
        final_action=final_action,
        max_sizing_delta_bps=max_bps,
    )


def evaluate_proactive_alert_gate(
    payload: Mapping[str, Any],
    *,
    alert_context: Mapping[str, Any] | None = None,
) -> ProactiveAlertGateResult:
    original_action_type = str(payload.get("action_type") or "").strip().lower()
    scout = _evaluate_scout_pass(payload, alert_context)
    skeptic = _evaluate_skeptic_pass(payload, alert_context)
    sizer = _evaluate_sizer_pass(payload, original_action_type=original_action_type, skeptic=skeptic)

    action_allowed = (
        scout.status == "pass"
        and skeptic.status == "pass"
        and sizer.status == "pass"
        and normalize_action(sizer.final_action or "") in ACTIONABLE_ACTIONS
    )

    if action_allowed:
        return ProactiveAlertGateResult(
            enabled=True,
            applied=True,
            gate_status="pass",
            action_allowed=True,
            original_action_type=original_action_type,
            final_action_type=original_action_type,
            scout=scout,
            skeptic=skeptic,
            sizer=sizer,
            summary="Scout, skeptic, and sizer passes cleared high-stakes actionability.",
        )

    final_action_type = _downgrade_action_type(original_action_type, final_action=sizer.final_action)
    summary_parts = [part for part in (scout.summary, skeptic.summary, sizer.summary) if part]
    gate_status: GateStatus = "downgraded"
    if scout.status == "fail":
        gate_status = "blocked"

    return ProactiveAlertGateResult(
        enabled=True,
        applied=True,
        gate_status=gate_status,
        action_allowed=False,
        original_action_type=original_action_type,
        final_action_type=final_action_type,
        scout=scout,
        skeptic=skeptic,
        sizer=sizer,
        summary=" ".join(summary_parts) or "High-stakes alert downgraded to monitor-only action.",
    )


def apply_proactive_alert_gate(
    action_id: str,
    payload: Mapping[str, Any],
    *,
    source_type: str,
    alert_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ProactiveAlertGateResult]:
    payload_dict = dict(payload)
    if not should_apply_proactive_alert_gate(action_id, source_type):
        return payload_dict, ProactiveAlertGateResult(enabled=False, applied=False, gate_status="disabled")
    if not is_high_stakes_action_item(payload_dict):
        return payload_dict, ProactiveAlertGateResult(enabled=True, applied=False, gate_status="not_applicable")

    gate_result = evaluate_proactive_alert_gate(payload_dict, alert_context=alert_context)
    payload_dict["scout_skeptic_sizer_gate"] = gate_result.to_trace_dict()
    if gate_result.action_allowed:
        return payload_dict, gate_result

    payload_dict["action_type"] = gate_result.final_action_type or _downgrade_action_type(
        str(payload_dict.get("action_type") or "")
    )
    original_description = str(payload_dict.get("description") or "").strip()
    downgrade_note = f"[Scout/skeptic/sizer gate downgrade: {gate_result.summary}]"
    payload_dict["description"] = (
        f"{original_description}\n\n{downgrade_note}".strip() if original_description else downgrade_note
    )
    return payload_dict, gate_result
