"""Triage gates for OpportunityCandidate pre-decision objects."""

from __future__ import annotations

from decision_quality.actions import ACTIONABLE_ACTIONS, NON_ACTIONABLE_ACTIONS
from decision_quality.opportunity_candidate import (
    CANDIDATE_NEXT_ACTIONS,
    OpportunityCandidate,
    OpportunityCandidateGate,
    OpportunityCandidateGateReason,
)

GRADUATE_ACTION = "graduate_to_decision_quality"
ALLOWED_TRIAGE_ACTIONS = set(CANDIDATE_NEXT_ACTIONS)


def _reason(code: str, severity: str, message: str) -> OpportunityCandidateGateReason:
    return OpportunityCandidateGateReason(code=code, severity=severity, message=message)  # type: ignore[arg-type]


def _nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_nonempty(item) for item in value)
    return True


def _fallback_action(original_action: str) -> str:
    if original_action in {"avoid", "do_nothing"}:
        return original_action
    return "research"


def _gate_status(
    reasons: list[OpportunityCandidateGateReason],
    original_action: str,
    final_action: str,
) -> str:
    if any(reason.severity == "blocker" for reason in reasons):
        return "blocked" if final_action == original_action else "downgraded"
    if reasons:
        return "downgraded" if final_action != original_action else "pass"
    return "pass"


def apply_opportunity_candidate_gates(
    candidate: OpportunityCandidate | None,
    *,
    parse_errors: list[str] | None = None,
) -> OpportunityCandidateGate:
    original_action = "research"
    final_action = original_action
    reasons: list[OpportunityCandidateGateReason] = []

    if candidate is None:
        code = (
            "MISSING_OPPORTUNITY_CANDIDATE"
            if not parse_errors or parse_errors == ["opportunity_candidate is missing"]
            else "INVALID_OPPORTUNITY_CANDIDATE"
        )
        message = "; ".join(parse_errors or []) or "OpportunityCandidate object is missing."
        reasons.append(_reason(code, "blocker", message))
        return OpportunityCandidateGate(
            status="invalid",
            original_action=original_action,
            final_action="research",
            should_graduate=False,
            reasons=reasons,
        )

    original_action = str(candidate.next_action or "research").strip().lower()
    final_action = original_action

    if original_action in ACTIONABLE_ACTIONS:
        reasons.append(
            _reason(
                "ACTIONABLE_NEXT_ACTION_BLOCKED",
                "blocker",
                "OpportunityCandidate cannot emit actionable next_action values; use graduate_to_decision_quality instead.",
            )
        )
        final_action = GRADUATE_ACTION
    elif original_action not in ALLOWED_TRIAGE_ACTIONS:
        reasons.append(
            _reason(
                "INVALID_NEXT_ACTION",
                "blocker",
                f"next_action {original_action!r} is not an allowed triage action.",
            )
        )
        final_action = _fallback_action(original_action)

    if original_action in ACTIONABLE_ACTIONS:
        pass
    elif original_action not in NON_ACTIONABLE_ACTIONS and original_action != GRADUATE_ACTION:
        if final_action == original_action:
            final_action = _fallback_action(original_action)

    if not _nonempty(candidate.trigger):
        reasons.append(_reason("MISSING_TRIGGER", "blocker", "trigger is required for triage."))
        if final_action == GRADUATE_ACTION:
            final_action = "research"

    if not _nonempty(candidate.why_now):
        reasons.append(_reason("MISSING_WHY_NOW", "warning", "why_now is empty; keep the candidate in research/watch."))
        if final_action == GRADUATE_ACTION:
            final_action = "research"

    if final_action == GRADUATE_ACTION and original_action == GRADUATE_ACTION and candidate.missing_inputs:
        critical_gaps = [item for item in candidate.missing_inputs if str(item).strip()]
        if len(critical_gaps) >= 3:
            reasons.append(
                _reason(
                    "TOO_MANY_MISSING_INPUTS",
                    "warning",
                    "Too many missing inputs remain for graduation; downgrade to research.",
                )
            )
            final_action = "research"

    should_graduate = final_action == GRADUATE_ACTION
    status = _gate_status(reasons, original_action, final_action)

    return OpportunityCandidateGate(
        status=status,  # type: ignore[arg-type]
        original_action=original_action,
        final_action=final_action,
        should_graduate=should_graduate,
        reasons=reasons,
    )
