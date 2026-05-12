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


def _obsolete_policy_fragments() -> tuple[str, ...]:
    return (
        "missing investor/account constraint",
        "investor/account constraint",
        "suitability_profile",
        "account_type",
        "tax_status",
        "min_cash_reserve_pct",
        "taxable_account_rules",
        "_".join(("tax", "lot", "data", "available")),
        ".".join(("tax", "_".join(("tax", "lots")))),
        " ".join(("tax", "lots")),
        "-".join(("tax", "lot")),
        "_".join(("time", "horizon")),
        " ".join(("time", "horizon")),
        "_".join(("horizon", "mismatch")),
    )


def _gate_text(gate: dict) -> str:
    return str(gate).lower()


def test_actionable_recommendation_does_not_generate_missing_constraint_warnings():
    gate = evaluate_policy_gate(
        "create_recommendation",
        {"record": _buy_payload()["recommended_actions"][0] | {"critical_data_quality": "ok"}},
    )

    assert gate["decision"] == "pass"
    assert gate["review_required"] is False
    assert not gate["warnings"]
    assert "missing_constraint_count" not in gate["uncertainty"]
    assert "min_cash_reserve_pct" not in _gate_text(gate)
    assert "taxable_account_rules" not in _gate_text(gate)
    assert any("Decision support only" in disclosure for disclosure in gate["disclosures"])
    assert not any(fragment in _gate_text(gate) for fragment in _obsolete_policy_fragments())


def test_reduce_recommendation_does_not_invent_tax_status_warning():
    action = _buy_payload()["recommended_actions"][0] | {
        "action": "reduce",
        "critical_data_quality": "ok",
    }

    gate = evaluate_policy_gate("create_recommendation", {"record": action})

    assert gate["decision"] == "pass"
    assert not gate["warnings"]
    assert "tax_status" not in _gate_text(gate)
    assert "tax_flag" not in _gate_text(gate)
    assert not any(fragment in _gate_text(gate) for fragment in _obsolete_policy_fragments())


def test_concentration_failure_requires_review_but_is_reviewable():
    gate = evaluate_policy_gate(
        "update_portfolio_positions",
        {
            "book_size": 100_000,
            "positions": [
                {
                    "ticker": "MU",
                    "asset": "equity",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 4,
                    "cost_basis": 100,
                    "shares": 250,
                }
            ],
        },
    )

    assert gate["decision"] == "review_required"
    assert gate["review_required"] is True
    assert any(reason["code"] == "concentration_limit" for reason in gate["failure_reasons"])


def test_concentration_uses_book_size_not_position_total_share():
    gate = evaluate_policy_gate(
        "update_portfolio_positions",
        {
            "book_size": 100_000,
            "positions": [
                {
                    "ticker": "NVDA",
                    "asset": "equity",
                    "direction": "long",
                    "cost_basis": 178.50,
                    "shares": 100,
                    "notional_base": 17_850,
                    "valuation_status": "ok",
                }
            ],
        },
    )

    concentration_failures = [reason for reason in gate["failure_reasons"] if reason["code"] == "concentration_limit"]
    assert gate["decision"] == "pass"
    assert concentration_failures == []


def test_max_position_concentration_does_not_apply_to_hedge_updates():
    gate = evaluate_policy_gate(
        "update_hedge_positions",
        {
            "book_size": 100_000,
            "positions": [
                {
                    "ticker": "SPY",
                    "asset": "equity",
                    "direction": "short",
                    "cost_basis": 500,
                    "shares": 50,
                    "notional_base": 25_000,
                    "valuation_status": "ok",
                }
            ],
        },
    )

    position_failures = [
        reason for reason in gate["failure_reasons"] if reason["check"] == "concentration.position"
    ]
    assert gate["decision"] == "pass"
    assert position_failures == []


def test_max_position_concentration_does_not_apply_to_marked_hedge_rows():
    gate = evaluate_policy_gate(
        "update_portfolio_positions",
        {
            "book_size": 100_000,
            "positions": [
                {
                    "ticker": "SH",
                    "asset": "equity",
                    "direction": "long",
                    "role": "hedge",
                    "cost_basis": 25,
                    "shares": 1000,
                    "notional_base": 25_000,
                    "valuation_status": "ok",
                }
            ],
        },
    )

    position_failures = [
        reason for reason in gate["failure_reasons"] if reason["check"] == "concentration.position"
    ]
    assert gate["decision"] == "pass"
    assert position_failures == []


def test_policy_gate_uses_base_currency_notional_for_foreign_position():
    positions = [
        {
            "ticker": "SPY",
            "asset": "equity",
            "direction": "long",
            "cost_basis": 100,
            "shares": 1000,
            "notional_base": 100_000,
            "valuation_status": "ok",
        },
        {
            "ticker": "GLD",
            "asset": "commodity",
            "direction": "long",
            "cost_basis": 100,
            "shares": 250,
            "notional_base": 25_000,
            "valuation_status": "ok",
        },
        {
            "ticker": "8001.T",
            "asset": "equity",
            "direction": "long",
            "cost_basis": 8001,
            "shares": 100,
            "currency": "JPY",
            "base_currency": "USD",
            "fx_rate_to_base": 1 / 155,
            "cost_basis_base": 8001 / 155,
            "notional_base": 8001 * 100 / 155,
            "valuation_status": "ok",
        },
    ]

    gate = evaluate_policy_gate("update_portfolio_positions", {"positions": positions})

    concentration_failures = [reason for reason in gate["failure_reasons"] if reason["code"] == "concentration_limit"]
    assert not any(reason["message"].startswith("8001.T ") for reason in concentration_failures)


def test_policy_gate_requires_review_when_foreign_fx_valuation_missing():
    gate = evaluate_policy_gate(
        "update_portfolio_positions",
        {
            "positions": [
                {
                    "ticker": "8001.T",
                    "asset": "equity",
                    "direction": "long",
                    "cost_basis": 8001,
                    "shares": 100,
                    "currency": "JPY",
                    "base_currency": "USD",
                    "valuation_status": "missing_fx_rate",
                },
                {
                    "ticker": "GLD",
                    "asset": "commodity",
                    "direction": "long",
                    "cost_basis": 100,
                    "shares": 100,
                    "notional_base": 10_000,
                    "valuation_status": "ok",
                },
            ]
        },
    )

    assert gate["decision"] == "review_required"
    assert any(reason["check"] == "portfolio.valuation.fx_missing" for reason in gate["failure_reasons"])
    assert not any(
        "8001.T exceeds max position concentration" in reason["message"] for reason in gate["failure_reasons"]
    )


def test_policy_gate_falls_back_for_legacy_usd_rows_without_base_notional():
    gate = evaluate_policy_gate(
        "update_portfolio_positions",
        {
            "book_size": 100_000,
            "positions": [
                {
                    "ticker": "SPY",
                    "asset": "equity",
                    "direction": "long",
                    "cost_basis": 100,
                    "shares": 250,
                    "base_currency": "USD",
                    "valuation_status": "missing_position_inputs",
                }
            ],
        },
    )

    assert any(reason["code"] == "concentration_limit" for reason in gate["failure_reasons"])
    assert not any(reason["check"] == "portfolio.valuation.inputs_missing" for reason in gate["warnings"])


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
        assert record["policy_gate_decision"] == "pass"
        assert record["policy_gate_warnings"] == []

        core_db.resolve_approval(approval["id"], "approved", "Reviewed policy gate")
        recommendation = core_db.get_recommendations(report_type="daily")[0]
        assert recommendation["policy_gate_result_id"] is not None
        assert recommendation["policy_gate_decision"] == "pass"
        assert recommendation["policy_gate_warnings_json"] == []

        stored_gate = core_db.get_policy_gate_result(int(recommendation["policy_gate_result_id"]))
        assert stored_gate is not None
        assert stored_gate["decision"] == "pass"
        assert stored_gate["result_json"]["decision"] == "pass"
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
    assert body["decision"] == "pass"
    assert body["account_id"] == "default-account"
    assert not any(fragment in _gate_text(body) for fragment in _obsolete_policy_fragments())


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
