"""Normalize governance and decision-support state for API responses."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

ACTIONABLE_RECOMMENDATION_ACTIONS = {"buy", "sell", "reduce", "exit", "rebalance", "hedge"}
STALE_APPROVAL_MESSAGE = (
    "This proposal is stale because the underlying state changed. Reject and restage it to review the current state."
)


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
        from portfolio.action_registry import compute_action_base_state_hash

        current_hash = compute_action_base_state_hash(action_id, proposed_change)
    except Exception:
        return {
            "base_state_status": "unknown",
            "base_state_valid": None,
            "base_state_message": "Unable to evaluate approval base state right now.",
        }

    if not current_hash:
        return {
            "base_state_status": "unknown",
            "base_state_valid": None,
            "base_state_message": "Approval base state could not be recomputed.",
        }
    if current_hash != stored_hash:
        return {
            "base_state_status": "stale",
            "base_state_valid": False,
            "base_state_message": STALE_APPROVAL_MESSAGE,
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


def _quality_state(value: Any) -> str:
    state = str(value or "").strip().lower()
    if state in {"ok", "degraded", "stale", "failed", "missing"}:
        return state
    return "missing"


def _lineage_state(value: Any) -> str:
    state = str(value or "").strip().lower()
    if state in {"complete", "partial", "legacy_partial", "retry_pending", "missing"}:
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


def normalize_approval(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return an approval with additive normalized decision fields."""

    if record is None:
        return None
    out = deepcopy(record)
    status = str(out.get("status") or "pending").strip().lower()
    application_status = str(out.get("application_status") or "pending").strip().lower()
    gate = _nested_policy_gate(out)
    out["decision_state"] = _approval_decision_state(status, application_status)
    out["decision_kind"] = "proposal"
    out["effect_scope"] = "internal_state"
    out["execution_capability"] = "none"
    out["policy_gate"] = gate
    out["policy_state"] = _policy_state(gate)
    proposed_change = _as_dict(out.get("proposed_change")) or {}
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
    out["can_approve"] = status == "pending" and application_status in {"pending", "failed"} and not is_stale
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
    gate = _nested_policy_gate(out)
    action = str(out.get("action") or "").strip().lower()
    approval_required = bool(out.get("approval_required")) or action in ACTIONABLE_RECOMMENDATION_ACTIONS
    out["decision_state"] = _recommendation_decision_state(out)
    out["decision_kind"] = "recommendation"
    out["effect_scope"] = "internal_state" if approval_required else "read_only"
    out["execution_capability"] = "none"
    out["approval_state"] = out.get("approval_status") or "none"
    out["outcome_state"] = out.get("outcome_status") or "pending"
    out["policy_gate"] = gate
    out["policy_state"] = _policy_state(gate)
    out["quality_state"] = _quality_state(out.get("critical_data_quality"))
    out["lineage_state"] = _lineage_state(out.get("lineage_completeness"))
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
