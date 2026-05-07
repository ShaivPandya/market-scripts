from __future__ import annotations

from uuid import uuid4

import portfolio.core_db as core_db


def _reset_core_db(tmp_path, monkeypatch):
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "continuous_optimizer.db")
    monkeypatch.setattr(core_db, "_conn", None)


def _action(
    ticker: str,
    action: str,
    *,
    band: str = "small",
    gate: str = "pass",
    confidence: float = 0.72,
    priority: float = 1.2,
    score: float = 1.1,
) -> dict:
    return {
        "ticker": ticker,
        "asset": "equity",
        "direction": "long",
        "action": action,
        "conviction_band": band,
        "priority_score": priority,
        "scenario_score": score,
        "score_delta": 0.2,
        "baseline_score": score - 0.2,
        "confidence": confidence,
        "gate_status": gate,
        "gate_reasons": [],
        "deterministic_rationale": f"{ticker} should {action.lower()}.",
        "warnings": [],
        "data_coverage": {"ratio": 1.0, "available": 4, "applicable": 4},
        "factor_breakdown": [],
        "sizing_implication": {"implication": "increase exposure", "conviction_band": band},
    }


def _patch_analyzer(monkeypatch, actions: list[dict], *, calls: list[dict] | None = None):
    from portfolio.portfolio_optimizer import portfolio_analyzer

    source_token = {"test_token": uuid4().hex}

    def fake_get_data(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return {
            "course_of_action": {
                "summary": {"as_of": "2026-05-04T14:15:00+00:00", "mission": "balanced"},
                "action_queue": actions,
            }
        }

    monkeypatch.setattr(portfolio_analyzer, "analyzer_source_cache_token", lambda: source_token)
    monkeypatch.setattr(portfolio_analyzer, "get_data", fake_get_data)


def _patch_context(monkeypatch, risk_level: str | None = None):
    import api.continuous_optimizer as optimizer

    def fake_context(tickers):
        risk = {
            ticker: {"risk_level": risk_level, "risk_score": 0.2, "computed_at": "2026-05-04T14:00:00+00:00"}
            for ticker in tickers
        }
        return {"position_risk": risk}, {"position_risk": {"status": "ok"}}

    monkeypatch.setattr(optimizer, "_collect_context", fake_context)


def test_continuous_optimizer_snapshots_initial_run_without_alerts(tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    _patch_context(monkeypatch)
    _patch_analyzer(monkeypatch, [_action("TSM", "Increase Long", band="medium")])

    from api.continuous_optimizer import run_continuous_optimizer

    result = run_continuous_optimizer({"source": "test"})

    assert result["status"] == "completed"
    assert result["snapshots_created"] == 1
    assert result["alerts_created"] == 0
    assert core_db.get_optimization_alerts(status="open") == []


def test_continuous_optimizer_suppresses_unchanged_repeated_runs(tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    _patch_context(monkeypatch)
    _patch_analyzer(monkeypatch, [_action("TSM", "Increase Long", band="medium")])

    from api.continuous_optimizer import run_continuous_optimizer

    run_continuous_optimizer({"source": "test"})
    result = run_continuous_optimizer({"source": "test"})

    assert result["alerts_created"] == 0
    assert len(core_db.get_optimization_runs()) == 2
    assert len(core_db.get_optimization_snapshots(ticker="TSM")) == 2


def test_continuous_optimizer_reuses_cached_analyzer_result(tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    _patch_context(monkeypatch)
    calls: list[dict] = []
    _patch_analyzer(monkeypatch, [_action("TSM", "Increase Long", band="medium")], calls=calls)

    from api.continuous_optimizer import run_continuous_optimizer

    run_continuous_optimizer({"source": "test"})
    run_continuous_optimizer({"source": "test"})

    assert len(calls) == 1


def test_continuous_optimizer_alerts_and_stages_on_material_action_change(tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    _patch_context(monkeypatch)

    from api.continuous_optimizer import run_continuous_optimizer

    _patch_analyzer(monkeypatch, [_action("UGL", "Hold Long", band="none", confidence=0.5, priority=0.4, score=-0.2)])
    run_continuous_optimizer({"source": "test"})

    _patch_analyzer(
        monkeypatch, [_action("UGL", "Trim Long", band="medium", confidence=0.78, priority=2.0, score=-1.4)]
    )
    result = run_continuous_optimizer({"source": "test"})

    assert result["alerts_created"] == 1
    alert = core_db.get_optimization_alerts(status="open")[0]
    assert alert["ticker"] == "UGL"
    assert alert["alert_type"] == "action_changed"
    assert alert["action_item_approval_id"] is not None
    approvals = core_db.get_pending_approvals(status="pending", ticker="UGL")
    assert approvals[0]["action_id"] == "create_action_item"
    assert "Continuous optimizer" in approvals[0]["proposed_change"]["description"]


def test_continuous_optimizer_risk_gate_deterioration_increases_alert_severity(tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)

    from api.continuous_optimizer import run_continuous_optimizer

    _patch_context(monkeypatch, risk_level="low")
    _patch_analyzer(monkeypatch, [_action("TSM", "Hold Long", band="small", confidence=0.66, priority=1.0)])
    run_continuous_optimizer({"source": "test"})

    _patch_context(monkeypatch, risk_level="critical")
    _patch_analyzer(monkeypatch, [_action("TSM", "Hold Long", band="small", confidence=0.66, priority=1.0)])
    result = run_continuous_optimizer({"source": "test"})

    assert result["alerts_created"] == 1
    alert = core_db.get_optimization_alerts(status="open")[0]
    assert alert["alert_type"] == "risk_gate_changed"
    assert alert["severity"] == "urgent"


def test_continuous_optimizer_degraded_sources_record_alert_without_staging(tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    import api.continuous_optimizer as optimizer

    monkeypatch.setattr(
        optimizer,
        "_collect_context",
        lambda tickers: ({"position_risk": {}}, {"position_risk": {"status": "degraded", "error": "stale"}}),
    )

    from api.continuous_optimizer import run_continuous_optimizer

    _patch_analyzer(monkeypatch, [_action("TSM", "Hold Long", band="none", confidence=0.5, priority=0.4)])
    run_continuous_optimizer({"source": "test"})

    _patch_analyzer(monkeypatch, [_action("TSM", "Increase Long", band="medium", confidence=0.74, priority=1.8)])
    result = run_continuous_optimizer({"source": "test"})

    assert result["summary"]["source_quality"] == "degraded"
    alert = core_db.get_optimization_alerts(status="open")[0]
    assert alert["action_item_approval_id"] is None
    assert alert["evidence"]["staging_blocked"]["sources"] == ["position_risk"]
    assert core_db.get_pending_approvals(status="pending", ticker="TSM") == []
