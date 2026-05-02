from __future__ import annotations

import portfolio.core_db as core_db


def _reset_core_db(tmp_path, monkeypatch):
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "api_recommendations.db")
    monkeypatch.setattr(core_db, "_conn", None)


def _create_sample_recommendation():
    return core_db.create_recommendation(
        {
            "report_type": "daily",
            "as_of": "2026-05-02",
            "stance": "Neutral / Watchful",
            "recommendation_status": "blocked",
            "critical_data_quality": "failed",
            "blocked_reasons": ["liquidity: timeout"],
            "what_changed": [],
            "do_nothing_rationale": "Data failed.",
            "action": "do_nothing",
            "instrument": "portfolio",
            "horizon": "1 trading day",
            "target_change": "none",
            "rationale": "Do not act on failed data.",
            "confidence": 1.0,
            "source_quality": "failed",
            "evidence": [],
            "disconfirming_evidence": [],
        }
    )


def test_recommendations_api_lists_latest(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    rec = _create_sample_recommendation()

    resp = auth_client.get("/api/v1/recommendations/latest")

    assert resp.status_code == 200
    assert resp.json()["daily"]["id"] == rec["id"]
    assert resp.json()["daily"]["recommendation_status"] == "blocked"


def test_workspace_includes_recommendation_summary(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    _create_sample_recommendation()
    import api.agent_tools as agent_tools

    def fake_execute_tool(name, _args):
        if name == "get_portfolio":
            return {"positions": [], "total_pnl": None, "total_pnl_pct": None}
        return {}

    monkeypatch.setattr(agent_tools, "execute_tool", fake_execute_tool)
    monkeypatch.setattr(
        "api.signal_snapshot.get_signal_aggregator_snapshot_or_module_response",
        lambda **kwargs: {
            "regime": {"label": "neutral", "score": 0, "confidence": 1.0},
            "_meta": {"snapshot": {"as_of": "2026-05-02", "stale": False}},
        },
    )

    resp = auth_client.get("/api/v1/workspace")

    assert resp.status_code == 200
    data = resp.json()
    assert data["recommendations"]["latest_daily"]["recommendation_status"] == "blocked"
    assert data["recommendations"]["blocked_warnings"][0]["critical_data_quality"] == "failed"
