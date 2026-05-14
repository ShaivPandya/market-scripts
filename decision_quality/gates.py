"""Hard gates for structured decision quality."""

from __future__ import annotations

from typing import Any

from decision_quality.actions import ACTIONABLE_ACTIONS, normalize_action
from decision_quality.models import DecisionQuality, DecisionQualityGate, DecisionQualityGateReason

SIZING_DELTA_REQUIRED_ACTIONS = {"add", "trim", "reduce", "rebalance"}
CATALYST_REQUIRED_ACTIONS = {"buy", "add", "short", "sell", "hedge", "rebalance"}
BLOCKING_DATA_QUALITY = {"stale", "failed"}


def _reason(code: str, severity: str, message: str) -> DecisionQualityGateReason:
    return DecisionQualityGateReason(code=code, severity=severity, message=message)  # type: ignore[arg-type]


def _nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_nonempty(item) for item in value)
    return True


def _critical_quality(data_quality: dict[str, Any] | None) -> bool:
    quality = data_quality or {}
    values = {
        str(quality.get("critical_data_quality") or "").strip().lower(),
        str(quality.get("overall_status") or "").strip().lower(),
        str(quality.get("quality") or "").strip().lower(),
    }
    return bool(values & BLOCKING_DATA_QUALITY)


def _fallback_action(original_action: str) -> str:
    if original_action in {"short", "sell", "avoid"}:
        return "avoid"
    if original_action == "do_nothing":
        return "do_nothing"
    return "watch"


def _gate_status(reasons: list[DecisionQualityGateReason], original_action: str, final_action: str) -> str:
    if any(reason.severity == "blocker" for reason in reasons):
        return "blocked" if final_action == original_action else "downgraded"
    if reasons:
        return "downgraded" if final_action != original_action else "pass"
    return "pass"


def apply_decision_quality_gates(
    decision_quality: DecisionQuality | None,
    *,
    current_action: str,
    recommendation_status: str,
    data_quality: dict[str, Any] | None = None,
    parse_errors: list[str] | None = None,
) -> DecisionQualityGate:
    original_action = normalize_action(current_action)
    final_action = original_action
    original_status = str(recommendation_status or "clear").strip().lower() or "clear"
    final_status = original_status if original_status in {"clear", "review_required", "blocked", "error"} else "clear"
    reasons: list[DecisionQualityGateReason] = []
    confidence_cap: float | None = None

    if _critical_quality(data_quality):
        reasons.append(
            _reason(
                "CRITICAL_DATA_QUALITY",
                "blocker",
                "Critical data quality is stale or failed, so actionable decisions are blocked.",
            )
        )

    if decision_quality is None:
        code = (
            "MISSING_DECISION_QUALITY"
            if not parse_errors or parse_errors == ["decision_quality is missing"]
            else "INVALID_DECISION_QUALITY"
        )
        message = "; ".join(parse_errors or []) or "Decision quality object is missing."
        reasons.append(_reason(code, "blocker", message))
    else:
        actionability_status = decision_quality.actionability.status
        if actionability_status != "actionable" and original_action in ACTIONABLE_ACTIONS:
            reasons.append(
                _reason(
                    "NON_ACTIONABLE_STATUS",
                    "blocker",
                    f"Actionability status is {actionability_status}, not actionable.",
                )
            )

        catalyst = decision_quality.catalyst_or_reason_now
        if original_action in CATALYST_REQUIRED_ACTIONS and not (
            _nonempty(catalyst.event_or_condition)
            and _nonempty(catalyst.expected_timeframe)
            and _nonempty(catalyst.why_now)
            and _nonempty(catalyst.source_evidence)
        ):
            reasons.append(
                _reason(
                    "MISSING_CATALYST",
                    "blocker",
                    "Actionable decisions require a catalyst or reason-now with timing and evidence.",
                )
            )

        invalidation = decision_quality.invalidation
        if original_action in ACTIONABLE_ACTIONS and not (
            _nonempty(invalidation.observable)
            and _nonempty(invalidation.metric_or_event)
            and _nonempty(invalidation.threshold)
            and _nonempty(invalidation.timeframe)
            and _nonempty(invalidation.implication)
        ):
            reasons.append(
                _reason(
                    "MISSING_INVALIDATION",
                    "blocker",
                    "Actionable decisions require observable, thresholded, time-bounded invalidation.",
                )
            )

        if original_action in ACTIONABLE_ACTIONS and not decision_quality.evidence_against:
            reasons.append(
                _reason(
                    "MISSING_EVIDENCE_AGAINST",
                    "warning",
                    "No disconfirming evidence was supplied; confidence is capped pending review.",
                )
            )
            confidence_cap = 0.6 if confidence_cap is None else min(confidence_cap, 0.6)

        if original_action in ACTIONABLE_ACTIONS and decision_quality.conviction.level is None:
            reasons.append(
                _reason(
                    "INVALID_CONVICTION",
                    "blocker",
                    "Actionable decisions require conviction on the 1-5 scale.",
                )
            )

        sizing_delta = decision_quality.sizing_context.sizing_delta
        if original_action in SIZING_DELTA_REQUIRED_ACTIONS and (
            sizing_delta.direction == "not_applicable"
            or sizing_delta.unit == "not_applicable"
            or sizing_delta.basis == "not_applicable"
        ):
            reasons.append(
                _reason(
                    "MISSING_SIZING_DELTA",
                    "warning",
                    "Add, trim, reduce, and rebalance decisions require a structured sizing_delta.",
                )
            )

        price_action = decision_quality.price_action_read
        if original_action in {"buy", "add", "short", "sell"} and not (
            _nonempty(price_action.observed_behavior) and _nonempty(price_action.interpretation)
        ):
            reasons.append(
                _reason(
                    "MISSING_PRICE_ACTION",
                    "warning",
                    "Directional decisions should include a price-action read or explain missing data.",
                )
            )

    if any(reason.severity == "blocker" for reason in reasons) and original_action in ACTIONABLE_ACTIONS:
        final_action = _fallback_action(original_action)
        final_status = "review_required"
    elif any(reason.severity == "warning" for reason in reasons) and final_status == "clear":
        final_status = "review_required"

    if final_status in {"blocked", "error"} and final_action in ACTIONABLE_ACTIONS:
        final_action = _fallback_action(final_action)

    return DecisionQualityGate(
        status=_gate_status(reasons, original_action, final_action),  # type: ignore[arg-type]
        original_action=original_action,
        final_action=final_action,
        original_recommendation_status=original_status,
        final_recommendation_status=final_status,
        confidence_cap=confidence_cap,
        reasons=reasons,
    )
