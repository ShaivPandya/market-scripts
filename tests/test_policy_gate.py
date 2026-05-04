from __future__ import annotations

import portfolio.core_db as core_db
from auto_report.recommendations import persist_recommendations
from portfolio.policy_gate import evaluate_policy_gate


def _buy_payload() -> dict:
    return {
        "report_type": "daily",
        "as_of": "2026-05-02",
        "stance": "Neutral / Watchful",
        "recommendation_status": "clear",
        "critical_data_quality": "ok",
        "blocked_reasons": [],
        "do_nothing_rationale": "",
        "what_changed": ["Breadth improved."],
        "recommended_actions": [
            {
                "action": "buy",
                "ticker": "MU",
                "instrument": "MU",
                "horizon": "1 trading day",
                "target_change": "start one-third size",
                "rationale": "Validated setup with bounded risk.",
                "evidence": ["price action confirms"],
                "disconfirming_evidence": ["liquidity is mixed"],
                "catalyst": "earnings",
                "invalidation": "breaks support",
                "expected_onset_window": "1 week",
                "confidence": 0.64,
                "source_quality": "ok",
                "approval_required": True,
            }
        ],
        "alternatives": [],
        "opportunity_cost": [],
    }


def test_missing_constraints_warn_without_blocking_actionable_recommendation():
    gate = evaluate_policy_gate(
        "create_recommendation",
        {"record": _buy_payload()["recommended_actions"][0] | {"critical_data_quality": "ok"}},
    )

    assert gate["decision"] == "warn"
    assert gate["review_required"] is False
    assert any(reason["code"] == "missing_constraint" for reason in gate["warnings"])
    assert any("Decision support only" in disclosure for disclosure in gate["disclosures"])


def test_concentration_failure_requires_review_but_is_reviewable():
    gate = evaluate_policy_gate(
        "update_portfolio_positions",
        {
            "positions": [
                {
                    "ticker": "MU",
                    "asset": "equity",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 4,
                    "cost_basis": 100,
                    "shares": 10,
                }
            ]
        },
    )

    assert gate["decision"] == "review_required"
    assert gate["review_required"] is True
    assert any(reason["code"] == "concentration_limit" for reason in gate["failure_reasons"])


def test_persisted_recommendation_stores_policy_gate_result(tmp_path, monkeypatch):
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "policy_gate.db")
    monkeypatch.setattr(core_db, "_conn", None)
    try:
        rows = persist_recommendations(
            _buy_payload(),
            source_report_path="/tmp/recommendations.md",
            source_json_path="/tmp/recommendations.json",
            prompt_metadata={"model": "test", "prompt_hash": "p", "input_hash": "i", "validation_status": "ok"},
        )

        approval = core_db.get_pending_approval(rows[0]["approval_id"])
        assert approval is not None
        record = approval["proposed_change"]["record"]
        assert record["policy_gate_decision"] == "warn"
        assert record["policy_gate_warnings"]

        core_db.resolve_approval(approval["id"], "approved", "Reviewed policy gate warnings")
        recommendation = core_db.get_recommendations(report_type="daily")[0]
        assert recommendation["policy_gate_result_id"] is not None
        assert recommendation["policy_gate_decision"] == "warn"
        assert recommendation["policy_gate_warnings_json"]

        stored_gate = core_db.get_policy_gate_result(int(recommendation["policy_gate_result_id"]))
        assert stored_gate is not None
        assert stored_gate["decision"] == "warn"
        assert stored_gate["result_json"]["decision"] == "warn"
    finally:
        if core_db._conn:
            core_db._conn.close()
        monkeypatch.setattr(core_db, "_conn", None)


def test_policy_gate_evaluate_api(auth_client):
    action = _buy_payload()["recommended_actions"][0] | {"critical_data_quality": "ok"}
    resp = auth_client.post(
        "/api/v1/policy-gate/evaluate",
        json={"action_id": "create_recommendation", "payload": {"record": action}},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "warn"
    assert body["account_id"] == "default-account"


def test_policy_gate_blocks_actionable_recommendation_without_risk_score(monkeypatch):
    monkeypatch.setenv("RISK_RECOMMENDATION_GATE_ENABLED", "1")
    action = _buy_payload()["recommended_actions"][0] | {"critical_data_quality": "ok"}

    gate = evaluate_policy_gate("create_recommendation", {"record": action})

    assert gate["decision"] == "blocked"
    assert any(reason["check"] == "risk.first_class_snapshot" for reason in gate["failure_reasons"])


def test_policy_gate_requires_review_for_degraded_risk_with_score(monkeypatch):
    monkeypatch.setenv("RISK_RECOMMENDATION_GATE_ENABLED", "1")
    action = _buy_payload()["recommended_actions"][0] | {
        "critical_data_quality": "degraded",
        "risk_snapshot_id": "position-risk:MU:degraded",
        "portfolio_risk_snapshot_id": "portfolio-risk:degraded",
        "risk_quality": "degraded",
        "risk_score": 0.71,
        "risk_bindings": {"risk_score": 0.71},
    }

    gate = evaluate_policy_gate("create_recommendation", {"record": action})

    assert gate["decision"] == "review_required"
    assert any(reason["check"] == "risk.first_class_snapshot" for reason in gate["failure_reasons"])
