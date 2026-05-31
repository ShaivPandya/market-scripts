"""Normalize governance and decision-support state for API responses."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from ontology.approval_workflow import (
    approval_requirement_progress,
    normalize_approval_decisions,
    normalize_approval_requirements,
)

ACTIONABLE_RECOMMENDATION_ACTIONS = {
    "buy",
    "add",
    "short",
    "sell",
    "trim",
    "reduce",
    "exit",
    "hedge",
    "rebalance",
}
ACTIONABLE_COURSE_OF_ACTIONS = {
    "buy",
    "add",
    "short",
    "sell",
    "trim",
    "exit",
    "rebalance",
}
STALE_APPROVAL_MESSAGE = (
    "This proposal is stale because the underlying state changed. Reject and restage it to review the current state."
)
_HEDGE_SCOPE_ACTION_IDS = {"update_hedge_positions"}
_HEDGE_SCOPE_ENTITY_TYPES = {"hedge_positions"}
_OBSOLETE_POLICY_REASON_FRAGMENTS = (
    "missing investor/account constraint",
    "investor/account constraint",
    "missing investor, account, tax, or policy constraints",
    ".".join(("investor", "suitability_profile")),
    ".".join(("account", "account_type")),
    ".".join(("account", "tax_status")),
    ".".join(("policy", "min_cash_reserve_pct")),
    ".".join(("policy", "taxable_account_rules")),
    "suitability_profile",
    "min_cash_reserve_pct",
    "taxable_account_rules",
    "_".join(("tax", "lot", "data", "available")),
    ".".join(("tax", "_".join(("tax", "lots")))),
    " ".join(("tax", "lots")),
    "-".join(("tax", "lot")),
    "_".join(("time", "horizon")),
    " ".join(("time", "horizon")),
    "_".join(("horizon", "mismatch")),
    ".".join(("horizon", "minimum")),
    ".".join(("horizon", "maximum")),
    "recommendation horizon is shorter than " + "".join(("man", "date")) + " minimum",
    "recommendation horizon is longer than " + "".join(("man", "date")) + " maximum",
)
_RETIRED_POLICY_SCOPE = "".join(("man", "date"))
_OBSOLETE_POLICY_KEY_FRAGMENTS = (
    _RETIRED_POLICY_SCOPE,
    "suitability_profile",
    "account_type",
    "tax_status",
    "min_cash_reserve_pct",
    "taxable_account_rules",
)
_DROP_VALUE = object()


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _approval_base_state(record: dict[str, Any]) -> dict[str, Any]:
    action_id = str(record.get("action_id") or "").strip()
    stored_hash = str(record.get("base_state_hash") or "").strip()
    if not action_id or not stored_hash:
        return {
            "base_state_status": "untracked",
            "base_state_valid": None,
            "base_state_message": None,
        }

    proposed_change = _as_dict(record.get("proposed_change"))
    if proposed_change is None:
        return {
            "base_state_status": "unknown",
            "base_state_valid": None,
            "base_state_message": "Unable to evaluate approval base state from the proposed change.",
        }

    try:
        from ontology.action_registry import compute_action_base_state_hash

        current_hash = compute_action_base_state_hash(action_id, proposed_change)
        if current_hash and current_hash != stored_hash:
            return {
                "base_state_status": "stale",
                "base_state_valid": False,
                "base_state_message": STALE_APPROVAL_MESSAGE,
            }
    except Exception:
        return {
            "base_state_status": "unknown",
            "base_state_valid": None,
            "base_state_message": "Unable to evaluate approval base state against current state.",
        }

    return {
        "base_state_status": "valid",
        "base_state_valid": True,
        "base_state_message": None,
    }


def _nested_policy_gate(record: dict[str, Any]) -> dict[str, Any] | None:
    direct = _as_dict(record.get("policy_gate_result"))
    if direct:
        return direct
    proposed = _as_dict(record.get("proposed_change"))
    if proposed:
        nested = _as_dict(proposed.get("policy_gate_result"))
        if nested:
            return nested
        rec = _as_dict(proposed.get("record"))
        if rec:
            nested = _as_dict(rec.get("policy_gate_result"))
            if nested:
                return nested
    return None


def _policy_state(gate: dict[str, Any] | None) -> str:
    if not gate:
        return "missing"
    decision = str(gate.get("decision") or "").strip().lower()
    if decision in {"pass", "warn", "review_required", "blocked", "error"}:
        return decision
    return "missing"


def _policy_state_from_fields(record: dict[str, Any], gate: dict[str, Any] | None) -> str:
    state = _policy_state(gate)
    if state != "missing":
        return state
    decision = str(record.get("policy_gate_decision") or "").strip().lower()
    if decision in {"pass", "warn", "review_required", "blocked", "error"}:
        return decision
    return "missing"


def _is_obsolete_policy_reason(reason: Any) -> bool:
    if not isinstance(reason, dict):
        return False
    text = " ".join(str(reason.get(key) or "") for key in ("code", "check", "message")).lower()
    return _RETIRED_POLICY_SCOPE in text or any(fragment in text for fragment in _OBSOLETE_POLICY_REASON_FRAGMENTS)


def _is_hedge_scope(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    action_id = str(record.get("action_id") or "").strip()
    entity_type = str(record.get("entity_type") or "").strip()
    if action_id in _HEDGE_SCOPE_ACTION_IDS or entity_type in _HEDGE_SCOPE_ENTITY_TYPES:
        return True
    proposed = _as_dict(record.get("proposed_change"))
    positions = proposed.get("positions") if proposed else None
    if not isinstance(positions, list) or not positions:
        return False
    for row in positions:
        if not isinstance(row, dict):
            return False
        role = str(row.get("role") or "").strip().lower()
        position_type = str(row.get("type") or row.get("position_type") or "").strip().lower()
        if role != "hedge" and position_type != "hedge":
            return False
    return True


def _is_hedge_concentration_reason(reason: Any, record: dict[str, Any] | None) -> bool:
    if not _is_hedge_scope(record) or not isinstance(reason, dict):
        return False
    code = str(reason.get("code") or "").strip().lower()
    check = str(reason.get("check") or "").strip().lower()
    message = str(reason.get("message") or "").strip().lower()
    return code == "concentration_limit" and (
        check == "concentration.position" or "max position concentration" in message
    )


def _filter_obsolete_policy_reasons(value: Any, *, policy_context: dict[str, Any] | None = None) -> list[Any]:
    rows = value if isinstance(value, list) else []
    return [
        row
        for row in rows
        if not _is_obsolete_policy_reason(row) and not _is_hedge_concentration_reason(row, policy_context)
    ]


def _scrub_retired_policy_scope(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(fragment in key_text for fragment in _OBSOLETE_POLICY_KEY_FRAGMENTS):
                continue
            cleaned = _scrub_retired_policy_scope(item)
            if cleaned is not _DROP_VALUE:
                out[key] = cleaned
        return out
    if isinstance(value, list):
        scrubbed: list[Any] = []
        for item in value:
            cleaned = _scrub_retired_policy_scope(item)
            if cleaned is not _DROP_VALUE:
                scrubbed.append(cleaned)
        return scrubbed
    if isinstance(value, str):
        text = value.lower()
        if _RETIRED_POLICY_SCOPE in text or any(fragment in text for fragment in _OBSOLETE_POLICY_REASON_FRAGMENTS):
            return _DROP_VALUE
    return value


def _decision_from_gate_reasons(gate: dict[str, Any], *, changed: bool) -> str:
    checks = gate.get("check_results")
    if isinstance(checks, list) and checks:
        if any(c.get("severity") == "block" and c.get("status") == "fail" for c in checks if isinstance(c, dict)):
            return "blocked"
        if any(c.get("status") == "fail" for c in checks if isinstance(c, dict)):
            return "review_required"
        if any(c.get("status") == "warn" for c in checks if isinstance(c, dict)):
            return "warn"
        return "pass"
    if gate.get("failure_reasons"):
        return "blocked" if str(gate.get("decision") or "").lower() == "blocked" else "review_required"
    if gate.get("warnings"):
        return "warn"
    decision = str(gate.get("decision") or "").strip().lower()
    if changed and decision in {"warn", "review_required", "blocked"}:
        return "pass"
    return decision if decision in {"pass", "warn", "review_required", "blocked", "error"} else "missing"


def _filter_policy_gate(
    gate: dict[str, Any] | None, *, policy_context: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    if not gate:
        return None
    filtered = deepcopy(gate)
    reasons_changed = False
    for key in ("failure_reasons", "warnings", "check_results"):
        original = filtered.get(key)
        if isinstance(original, list):
            next_rows = _filter_obsolete_policy_reasons(original, policy_context=policy_context)
            row_changed = len(next_rows) != len(original)
            reasons_changed = reasons_changed or row_changed
            filtered[key] = next_rows
    filtered = _scrub_retired_policy_scope(filtered)
    if not isinstance(filtered, dict):
        return None
    if reasons_changed:
        filtered["decision"] = _decision_from_gate_reasons(filtered, changed=True)
        filtered["review_required"] = filtered["decision"] == "review_required"
        uncertainty = filtered.get("uncertainty")
        if isinstance(uncertainty, dict):
            next_uncertainty = dict(uncertainty)
            next_uncertainty.pop("missing_constraint_count", None)
            next_uncertainty["level"] = "medium"
            next_uncertainty["notes"] = []
            filtered["uncertainty"] = next_uncertainty
    return filtered


def _replace_nested_policy_gate(record: dict[str, Any], gate: dict[str, Any] | None) -> dict[str, Any]:
    if gate is None:
        return record
    out = deepcopy(record)
    if isinstance(out.get("policy_gate_result"), dict):
        out["policy_gate_result"] = gate
    nested_record = _as_dict(out.get("record"))
    if nested_record is not None and isinstance(nested_record.get("policy_gate_result"), dict):
        nested_record["policy_gate_result"] = gate
        nested_record["policy_gate_status"] = gate.get("decision")
        nested_record["policy_gate_decision"] = gate.get("decision")
        nested_record["policy_gate_review_required"] = bool(gate.get("review_required"))
        nested_record["policy_gate_failures"] = gate.get("failure_reasons", [])
        nested_record["policy_gate_warnings"] = gate.get("warnings", [])
        nested_record["policy_gate_disclosures"] = gate.get("disclosures", [])
        out["record"] = nested_record
    return out


def _filter_top_level_policy_fields(record: dict[str, Any]) -> None:
    changed = False
    for key in ("policy_gate_failures_json", "policy_gate_warnings_json"):
        original = record.get(key)
        if isinstance(original, list):
            next_rows = _filter_obsolete_policy_reasons(original, policy_context=record)
            changed = changed or len(next_rows) != len(original)
            record[key] = next_rows
    if changed and not record.get("policy_gate_failures_json") and not record.get("policy_gate_warnings_json"):
        decision = str(record.get("policy_gate_decision") or "").strip().lower()
        if decision in {"warn", "review_required", "blocked"}:
            record["policy_gate_decision"] = "pass"
            record["policy_gate_review_required"] = False


def _quality_state(value: Any) -> str:
    state = str(value or "").strip().lower()
    if state in {"ok", "degraded", "stale", "failed", "missing"}:
        return state
    return "missing"


def _lineage_state(value: Any) -> str:
    state = str(value or "").strip().lower()
    if state in {"complete", "partial", "retry_pending", "missing"}:
        return state
    if state:
        return state
    return "missing"


def _approval_decision_state(status: Any, application_status: Any) -> str:
    status_value = str(status or "pending").strip().lower()
    application_value = str(application_status or "pending").strip().lower()
    if status_value == "rejected":
        return "rejected"
    if application_value == "failed":
        return "failed"
    if status_value == "approved" and application_value == "applied":
        return "applied"
    if status_value == "approved":
        return "approved"
    if status_value == "pending":
        return "pending_approval"
    if status_value == "expired":
        return "rejected"
    return "proposal"


def _recommendation_decision_state(record: dict[str, Any]) -> str:
    approval_status = str(record.get("approval_status") or "none").strip().lower()
    status = str(record.get("status") or "").strip().lower()
    recommendation_status = str(record.get("recommendation_status") or "").strip().lower()
    quality = str(record.get("critical_data_quality") or "").strip().lower()
    if approval_status == "pending":
        return "pending_approval"
    if approval_status == "approved":
        return "approved"
    if approval_status == "rejected":
        return "rejected"
    if status == "error" or recommendation_status == "error" or quality == "failed":
        return "failed"
    return "recommendation"


def _course_of_action_decision_state(record: dict[str, Any]) -> str:
    stored = str(record.get("decision_state") or "").strip().lower()
    if stored in {"under_review", "proposed"}:
        return "pending_approval"
    if stored == "applied":
        return "applied"
    if stored in {"approved", "rejected", "draft", "generated", "closed", "superseded"}:
        return "recommendation" if stored == "generated" else stored
    approval_status = str(record.get("approval_status") or "none").strip().lower()
    if approval_status == "pending":
        return "pending_approval"
    if approval_status == "approved":
        return "approved"
    if approval_status == "rejected":
        return "rejected"
    return "recommendation"


def normalize_approval(
    record: dict[str, Any] | None,
    *,
    source_health_review: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return an approval with additive normalized decision fields."""

    if record is None:
        return None
    out = deepcopy(record)
    review = _as_dict(source_health_review) or _as_dict(out.get("source_health_review"))
    if review is not None:
        out["source_health_review"] = deepcopy(review)
    _filter_top_level_policy_fields(out)
    status = str(out.get("status") or "pending").strip().lower()
    application_status = str(out.get("application_status") or "pending").strip().lower()
    gate = _filter_policy_gate(_nested_policy_gate(out), policy_context=out)
    proposed_change = _replace_nested_policy_gate(_as_dict(out.get("proposed_change")) or {}, gate)
    approval_requirements = normalize_approval_requirements(out.get("approval_requirements"))
    approval_decisions = normalize_approval_decisions(out.get("approval_decisions"))
    approval_progress = approval_requirement_progress(approval_requirements, approval_decisions)
    out["proposed_change"] = proposed_change
    out["approval_requirements"] = approval_progress["requirements"]
    out["approval_decisions"] = approval_decisions
    out["approval_progress"] = approval_progress
    out["remaining_approval_requirements"] = approval_progress["remaining_requirements"]
    out["decision_state"] = _approval_decision_state(status, application_status)
    out["decision_kind"] = "proposal"
    out["effect_scope"] = "internal_state"
    out["execution_capability"] = "none"
    out["policy_gate"] = gate
    out["policy_state"] = _policy_state_from_fields(out, gate)
    proposed_record = _as_dict(proposed_change.get("record")) or {}
    out["quality_state"] = _quality_state(
        out.get("critical_data_quality")
        or proposed_change.get("critical_data_quality")
        or proposed_record.get("critical_data_quality")
    )
    out["lineage_state"] = _lineage_state(out.get("lineage_completeness"))
    base_state = _approval_base_state(out)
    out.update(base_state)
    is_stale = base_state["base_state_status"] == "stale"
    source_health_blocked = str((review or {}).get("status") or "").strip().lower() == "blocked"
    out["can_approve"] = (
        status == "pending"
        and application_status in {"pending", "failed"}
        and not is_stale
        and not source_health_blocked
        and (not approval_progress["completed"] or application_status == "failed")
    )
    out["can_reject"] = status == "pending"
    out["can_retry_apply"] = status == "pending" and application_status == "failed" and not is_stale
    out["can_restage"] = status == "pending" and bool(out.get("action_id")) and is_stale
    out["review_route"] = f"/workspace?approval_id={out.get('id')}" if out.get("id") is not None else None
    return out


def normalize_recommendation(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a recommendation with additive normalized decision fields."""

    if record is None:
        return None
    out = deepcopy(record)
    _filter_top_level_policy_fields(out)
    gate = _filter_policy_gate(_nested_policy_gate(out), policy_context=out)
    action = str(out.get("action") or "").strip().lower()
    approval_required = bool(out.get("approval_required")) or action in ACTIONABLE_RECOMMENDATION_ACTIONS
    out["decision_state"] = _recommendation_decision_state(out)
    out["decision_kind"] = "recommendation"
    out["effect_scope"] = "internal_state" if approval_required else "read_only"
    out["execution_capability"] = "none"
    out["approval_state"] = out.get("approval_status") or "none"
    out["outcome_state"] = out.get("outcome_status") or "pending"
    out["policy_gate"] = gate
    out["policy_state"] = _policy_state_from_fields(out, gate)
    out["quality_state"] = _quality_state(out.get("critical_data_quality"))
    out["lineage_state"] = _lineage_state(out.get("lineage_completeness"))
    raw_payload = out.get("payload")
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    raw_outcome = payload.get("outcome")
    outcome = raw_outcome if isinstance(raw_outcome, dict) else {}
    if outcome:
        out["draft_postmortem"] = outcome.get("draft_postmortem")
        out["final_postmortem"] = outcome.get("final_postmortem")
        out["final_label_status"] = outcome.get("final_label_status")
        out["process_label"] = outcome.get("process_label")
        out["lessons_learned"] = outcome.get("lessons_learned")
    return out


def normalize_course_of_action(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a CourseOfAction with additive normalized decision fields."""

    if record is None:
        return None
    out = deepcopy(record)
    _filter_top_level_policy_fields(out)
    gate = _filter_policy_gate(_nested_policy_gate(out), policy_context=out)
    action = str(out.get("action") or "").strip().lower()
    approval_required = bool(out.get("approval_required")) or action in ACTIONABLE_COURSE_OF_ACTIONS
    out["decision_state"] = _course_of_action_decision_state(out)
    out["decision_kind"] = "course_of_action"
    out["effect_scope"] = "internal_state" if approval_required else "read_only"
    out["execution_capability"] = "none"
    out["approval_state"] = out.get("approval_status") or "none"
    out["outcome_state"] = out.get("outcome_status") or "pending"
    out["policy_gate"] = gate
    out["policy_state"] = _policy_state_from_fields(out, gate)
    out["quality_state"] = _quality_state(out.get("source_quality"))
    out["lineage_state"] = _lineage_state(out.get("lineage_completeness"))
    raw_payload = out.get("payload")
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    raw_outcome = payload.get("outcome")
    outcome = raw_outcome if isinstance(raw_outcome, dict) else {}
    if outcome:
        out["draft_postmortem"] = outcome.get("draft_postmortem")
        out["final_postmortem"] = outcome.get("final_postmortem")
        out["final_label_status"] = outcome.get("final_label_status")
        out["process_label"] = outcome.get("process_label")
        out["lessons_learned"] = outcome.get("lessons_learned")
    return out


def normalize_decision_outcome(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a DecisionOutcome with additive normalized decision fields."""

    if record is None:
        return None
    out = deepcopy(record)
    final_status = str(out.get("final_label_status") or "draft").strip().lower()
    out["decision_kind"] = "decision_outcome"
    out["decision_state"] = "draft" if final_status == "draft" else "finalized"
    out["effect_scope"] = "read_only"
    out["execution_capability"] = "none"
    out["outcome_state"] = out.get("outcome_status") or "pending"
    out["learning_state"] = final_status
    out["requires_review"] = final_status == "draft" and out.get("outcome_status") == "evaluated"
    return out


def normalize_action_item(record: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(record)
    status = str(out.get("status") or "open").strip().lower()
    out["decision_kind"] = "internal_state_change"
    out["decision_state"] = "applied" if status in {"completed", "dismissed"} else status
    out["effect_scope"] = "internal_state"
    out["execution_capability"] = "none"
    return out


def analysis_metadata(*, as_of: str | None = None, quality_state: str = "ok") -> dict[str, Any]:
    return {
        "decision_state": "analysis",
        "decision_kind": "analysis",
        "effect_scope": "read_only",
        "execution_capability": "none",
        "quality_state": _quality_state(quality_state),
        "as_of": as_of or datetime.now(UTC).isoformat(),
    }


def normalize_staged_response(response: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(response)
    status = str(out.get("status") or "").strip().lower()
    application_status = str(out.get("application_status") or "pending").strip().lower()
    if status == "failed" or application_status == "failed":
        out["decision_state"] = "failed"
    elif status == "applied" or application_status == "applied":
        out["decision_state"] = "applied"
    else:
        out["decision_state"] = "pending_approval"
    out["decision_kind"] = "internal_state_change" if out["decision_state"] == "applied" else "proposal"
    out["effect_scope"] = "internal_state"
    out["execution_capability"] = "none"
    if out.get("approval_id") is not None:
        out["review_route"] = f"/workspace?approval_id={out.get('approval_id')}"
    return out
