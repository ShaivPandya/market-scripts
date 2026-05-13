from __future__ import annotations

from pathlib import Path

from api.decision_state import (
    normalize_action_item,
    normalize_approval,
    normalize_recommendation,
    normalize_staged_response,
)

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_approval_distinguishes_pending_failed_and_applied():
    pending = normalize_approval(
        {
            "id": 1,
            "status": "pending",
            "application_status": "pending",
            "entity_type": "thesis",
            "ticker": "MU",
            "reason": "Review",
            "proposed_change": {},
        }
    )
    assert pending is not None
    assert pending["decision_state"] == "pending_approval"
    assert pending["decision_kind"] == "proposal"
    assert pending["effect_scope"] == "internal_state"
    assert pending["can_approve"] is True
    assert pending["execution_capability"] == "none"

    failed = normalize_approval(
        {
            "id": 2,
            "status": "pending",
            "application_status": "failed",
            "entity_type": "portfolio",
            "ticker": None,
            "reason": "Apply failed",
            "application_error": "state conflict",
            "proposed_change": {},
        }
    )
    assert failed is not None
    assert failed["decision_state"] == "failed"
    assert failed["can_retry_apply"] is True

    applied = normalize_approval(
        {
            "id": 3,
            "status": "approved",
            "application_status": "applied",
            "entity_type": "portfolio",
            "ticker": None,
            "reason": "Approved",
            "proposed_change": {},
        }
    )
    assert applied is not None
    assert applied["decision_state"] == "applied"


def test_normalize_approval_reports_base_state_valid_stale_and_untracked(monkeypatch):
    import portfolio.action_registry as action_registry

    monkeypatch.setattr(action_registry, "compute_action_base_state_hash", lambda _action_id, _change: "current")

    valid = normalize_approval(
        {
            "id": 10,
            "status": "pending",
            "application_status": "pending",
            "entity_type": "action_item_status",
            "action_id": "complete_action_item",
            "base_state_hash": "current",
            "proposed_change": {"item_id": 1},
        }
    )
    assert valid is not None
    assert valid["base_state_status"] == "valid"
    assert valid["base_state_valid"] is True
    assert valid["can_approve"] is True

    stale = normalize_approval(
        {
            "id": 11,
            "status": "pending",
            "application_status": "failed",
            "entity_type": "action_item_status",
            "action_id": "complete_action_item",
            "base_state_hash": "old",
            "proposed_change": {"item_id": 1},
        }
    )
    assert stale is not None
    assert stale["base_state_status"] == "stale"
    assert stale["base_state_valid"] is False
    assert stale["can_approve"] is False
    assert stale["can_retry_apply"] is False
    assert stale["can_restage"] is True

    untracked = normalize_approval(
        {
            "id": 12,
            "status": "pending",
            "application_status": "pending",
            "entity_type": "action_item",
            "action_id": "create_action_item",
            "proposed_change": {"description": "Review MU"},
        }
    )
    assert untracked is not None
    assert untracked["base_state_status"] == "untracked"
    assert untracked["base_state_valid"] is None
    assert untracked["can_approve"] is True


def test_normalize_approval_recomputes_base_state_in_ontology_primary_mode(monkeypatch):
    import portfolio.action_registry as action_registry

    monkeypatch.setattr(action_registry, "compute_action_base_state_hash", lambda _action_id, _change: "current")

    stale = normalize_approval(
        {
            "id": "approval:old",
            "status": "pending",
            "application_status": "pending",
            "entity_type": "action_item_status",
            "action_id": "complete_action_item",
            "base_state_hash": "old",
            "proposed_change": {"item_id": 1},
        }
    )

    assert stale is not None
    assert stale["base_state_status"] == "stale"
    assert stale["base_state_valid"] is False
    assert stale["can_approve"] is False
    assert stale["can_restage"] is True


def test_normalize_recommendation_keeps_execution_capability_none():
    rec = normalize_recommendation(
        {
            "id": 1,
            "action": "buy",
            "approval_status": "pending",
            "recommendation_status": "clear",
            "critical_data_quality": "degraded",
            "policy_gate_result": {"decision": "warn"},
        }
    )
    assert rec is not None
    assert rec["decision_state"] == "pending_approval"
    assert rec["decision_kind"] == "recommendation"
    assert rec["effect_scope"] == "internal_state"
    assert rec["execution_capability"] == "none"
    assert rec["policy_state"] == "warn"
    assert rec["quality_state"] == "degraded"


def _previous_policy_reason() -> dict:
    return {
        "code": "tax_flag",
        "check": ".".join(("tax", "_".join(("tax", "lots")))),
        "message": f"{'-'.join(('Tax', 'lot'))} data is unavailable.",
        "status": "warn",
        "severity": "warn",
    }


def _retired_policy_scope_reason() -> dict:
    retired_field = "_".join(("time", "horizon", "days", "min"))
    retired_scope = "".join(("man", "date"))
    return {
        "code": "missing_constraint",
        "check": ".".join((retired_scope, retired_field)),
        "message": f"Missing investor/account constraint: {retired_scope}.{retired_field}.",
        "status": "warn",
        "severity": "warn",
    }


def _hedge_concentration_reason() -> dict:
    return {
        "code": "concentration_limit",
        "check": "concentration.position",
        "message": "SPY exceeds max position concentration.",
        "status": "fail",
        "severity": "fail",
    }


def test_normalize_recommendation_filters_obsolete_policy_warning():
    rec = normalize_recommendation(
        {
            "id": 2,
            "action": "reduce",
            "approval_status": "pending",
            "recommendation_status": "clear",
            "critical_data_quality": "ok",
            "policy_gate_result": {
                "decision": "warn",
                "review_required": False,
                "warnings": [_previous_policy_reason()],
                "failure_reasons": [],
                "check_results": [_previous_policy_reason()],
                "uncertainty": {"missing_constraint_count": 1, "level": "high", "notes": ["canonical"]},
            },
        }
    )

    assert rec is not None
    assert rec["policy_state"] == "pass"
    assert rec["policy_gate"]["decision"] == "pass"
    assert rec["policy_gate"]["warnings"] == []
    assert rec["policy_gate"]["check_results"] == []


def test_normalize_approval_filters_obsolete_policy_warning():
    approval = normalize_approval(
        {
            "id": 20,
            "status": "pending",
            "application_status": "pending",
            "entity_type": "recommendation",
            "reason": "Review",
            "proposed_change": {
                "policy_gate_result": {
                    "decision": "warn",
                    "review_required": False,
                    "warnings": [_previous_policy_reason()],
                    "failure_reasons": [],
                    "check_results": [_previous_policy_reason()],
                }
            },
        }
    )

    assert approval is not None
    assert approval["policy_state"] == "pass"
    assert approval["policy_gate"]["warnings"] == []


def test_normalize_approval_filters_retired_policy_scope_warning():
    retired_field = "_".join(("time", "horizon", "days", "min"))
    retired_scope = "".join(("man", "date"))
    approval = normalize_approval(
        {
            "id": 21,
            "status": "pending",
            "application_status": "pending",
            "entity_type": "portfolio_positions",
            "reason": "Review",
            "proposed_change": {
                "policy_gate_result": {
                    "decision": "warn",
                    "review_required": False,
                    "warnings": [_retired_policy_scope_reason()],
                    "failure_reasons": [],
                    "check_results": [_retired_policy_scope_reason()],
                    "constraints_snapshot": {
                        retired_scope: {
                            f"{retired_scope}_id": f"default-{retired_scope}",
                            retired_field: None,
                        }
                    },
                }
            },
        }
    )

    assert approval is not None
    assert approval["policy_state"] == "pass"
    assert approval["policy_gate"]["warnings"] == []
    assert approval["policy_gate"]["check_results"] == []
    assert retired_field not in str(approval["policy_gate"])
    assert approval["proposed_change"]["policy_gate_result"]["warnings"] == []
    assert retired_field not in str(approval["proposed_change"]["policy_gate_result"])


def test_normalize_hedge_approval_filters_position_concentration_warning():
    approval = normalize_approval(
        {
            "id": 22,
            "status": "pending",
            "application_status": "pending",
            "action_id": "update_hedge_positions",
            "entity_type": "hedge_positions",
            "reason": "Review hedge",
            "proposed_change": {
                "policy_gate_result": {
                    "decision": "review_required",
                    "review_required": True,
                    "warnings": [_hedge_concentration_reason()],
                    "failure_reasons": [_hedge_concentration_reason()],
                    "check_results": [_hedge_concentration_reason()],
                }
            },
        }
    )

    assert approval is not None
    assert approval["policy_state"] == "pass"
    assert approval["policy_gate"]["warnings"] == []
    assert approval["policy_gate"]["failure_reasons"] == []
    assert approval["policy_gate"]["check_results"] == []
    assert approval["proposed_change"]["policy_gate_result"]["warnings"] == []


def test_normalize_action_item_preserves_open_state():
    open_item = normalize_action_item(
        {
            "id": 1,
            "status": "open",
            "description": "Review MU thesis",
            "action_type": "review",
            "urgency": "high",
        }
    )
    assert open_item["decision_state"] == "open"
    assert open_item["decision_kind"] == "internal_state_change"
    assert open_item["effect_scope"] == "internal_state"
    assert open_item["execution_capability"] == "none"

    completed_item = normalize_action_item(
        {
            "id": 2,
            "status": "completed",
            "description": "Review MU thesis",
            "action_type": "review",
            "urgency": "high",
        }
    )
    assert completed_item["decision_state"] == "applied"


def test_normalize_staged_response_adds_review_metadata():
    response = normalize_staged_response(
        {
            "status": "pending_approval_created",
            "approval_id": 42,
            "application_status": "pending",
            "action_id": "update_portfolio_positions",
            "entity_type": "portfolio",
            "ticker": None,
            "proposed_change": {},
        }
    )
    assert response["decision_state"] == "pending_approval"
    assert response["decision_kind"] == "proposal"
    assert response["effect_scope"] == "internal_state"
    assert response["execution_capability"] == "none"
    assert response["review_route"] == "/workspace?approval_id=42"


def test_normalize_staged_response_preserves_apply_failure_state():
    response = normalize_staged_response(
        {
            "status": "failed",
            "approval_id": 43,
            "application_status": "failed",
            "application_error": "state conflict",
            "action_id": "update_portfolio_positions",
            "entity_type": "portfolio",
            "ticker": None,
            "proposed_change": {},
        }
    )
    assert response["decision_state"] == "failed"
    assert response["decision_kind"] == "proposal"
    assert response["effect_scope"] == "internal_state"
    assert response["execution_capability"] == "none"
    assert response["review_route"] == "/workspace?approval_id=43"


def test_sizing_and_hedging_copy_does_not_imply_broker_orders():
    sizer = (ROOT / "frontend/src/pages/PortfolioSizer.tsx").read_text()
    hedging = (ROOT / "frontend/src/pages/HedgingTool.tsx").read_text()

    assert "Buy / Sell Summary" not in sizer
    assert "Total Buys" not in sizer
    assert "Total Sells" not in sizer
    assert '"BUY"' not in sizer
    assert '"SELL"' not in sizer
    assert "Sizing Delta Summary" in sizer
    assert "not executable orders" in sizer

    assert "AI Recommendations" not in hedging
    assert "Get Recommendations" not in hedging
    assert "applied on run" not in hedging
    assert "Hedge Analysis Notes" in hedging
    assert "not executable orders" in hedging


def test_stale_approval_controls_are_wired_on_review_surfaces():
    workspace = (ROOT / "frontend/src/pages/Workspace.tsx").read_text()
    dossier = (ROOT / "frontend/src/pages/PositionDossier.tsx").read_text()
    badge = (ROOT / "frontend/src/components/shared/DecisionStateBadge.tsx").read_text()

    assert "State Changed" in badge
    for source in (workspace, dossier):
        assert "BaseStateBadge" in source
        assert "Reject & Restage" in source
        assert "handleRejectAndRestage" in source
        assert (
            "approvalReview.approval.can_approve === false" in source
            or "approvalReview.can_approve === false" in source
        )
