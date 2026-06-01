"""Triage gates for OpportunityCandidate pre-decision objects."""

from __future__ import annotations

from typing import Any

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


def _context_pack_blocks_graduation(
    context_pack: dict[str, Any] | None,
    *,
    data_quality: dict[str, Any] | None = None,
) -> tuple[bool, list[OpportunityCandidateGateReason], list[str]]:
    if not isinstance(context_pack, dict):
        return False, [], []

    reasons: list[OpportunityCandidateGateReason] = []
    missing_inputs = [str(item).strip() for item in context_pack.get("missing_inputs") or [] if str(item).strip()]
    missing_tools = [str(item) for item in context_pack.get("missing_tools") or [] if str(item).strip()]
    is_complete = bool(context_pack.get("is_complete", True))

    if missing_tools:
        reasons.append(
            _reason(
                "CONTEXT_PACK_MISSING_TOOLS",
                "warning",
                f"Context pack {context_pack.get('pack_id')!r} is missing required tools: {', '.join(missing_tools)}.",
            )
        )

    dq = data_quality if isinstance(data_quality, dict) else {}
    if dq.get("critical_data_quality") in {"stale", "failed"}:
        reasons.append(
            _reason(
                "CONTEXT_PACK_DATA_QUALITY",
                "warning",
                "Required context-pack sources failed freshness or reliability checks.",
            )
        )
        is_complete = False

    blocking_codes = [str(item) for item in context_pack.get("blocking_reason_codes") or dq.get("blocking_reason_codes") or []]
    if blocking_codes:
        reasons.append(
            _reason(
                "CONTEXT_PACK_BLOCKERS",
                "warning",
                f"Context pack blockers remain: {', '.join(blocking_codes)}.",
            )
        )
        is_complete = False

    if not is_complete and not missing_inputs:
        missing_inputs.append("required context-pack inputs")

    return not is_complete, reasons, missing_inputs


def apply_opportunity_candidate_gates(
    candidate: OpportunityCandidate | None,
    *,
    parse_errors: list[str] | None = None,
    context_pack: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
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

    pack_blocks, pack_reasons, pack_missing_inputs = _context_pack_blocks_graduation(
        context_pack,
        data_quality=data_quality,
    )
    reasons.extend(pack_reasons)
    if pack_blocks and final_action == GRADUATE_ACTION:
        reasons.append(
            _reason(
                "CONTEXT_PACK_INCOMPLETE",
                "warning",
                "Required context pack is incomplete; keep the candidate in research until pack inputs are filled.",
            )
        )
        final_action = "research"

    combined_missing_inputs = list(
        dict.fromkeys([*(candidate.missing_inputs or []), *pack_missing_inputs])
    )
    if final_action == GRADUATE_ACTION and original_action == GRADUATE_ACTION and len(combined_missing_inputs) >= 3:
        if not any(reason.code == "TOO_MANY_MISSING_INPUTS" for reason in reasons):
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
