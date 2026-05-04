from __future__ import annotations

from pathlib import Path

from api.decision_state import normalize_approval, normalize_recommendation, normalize_staged_response

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
