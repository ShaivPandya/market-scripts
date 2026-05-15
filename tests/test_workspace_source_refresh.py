from __future__ import annotations

import pytest


def test_workspace_source_refresh_runs_persisted_backend_refreshes(monkeypatch):
    import api.macro_snapshots as macro_snapshots
    import api.maintenance_jobs as maintenance_jobs
    import api.market_snapshots as market_snapshots
    import api.position_risk as position_risk

    monkeypatch.setattr(market_snapshots, "refresh_market_snapshots", lambda: {"snapshots": ["market"]})
    monkeypatch.setattr(macro_snapshots, "refresh_macro_snapshots", lambda: {"snapshots": ["macro"]})
    monkeypatch.setattr(position_risk, "refresh_portfolio_risk", lambda: {"result_id": "portfolio-risk:1"})

    result = maintenance_jobs.refresh_workspace_sources()

    assert [step["step"] for step in result["steps"]] == [
        "market_snapshot_refresh",
        "macro_snapshot_refresh",
        "portfolio_risk_refresh",
    ]
    assert all(step["status"] == "ok" for step in result["steps"])


def test_workspace_source_refresh_fails_if_a_backend_refresh_fails(monkeypatch):
    import api.macro_snapshots as macro_snapshots
    import api.maintenance_jobs as maintenance_jobs
    import api.market_snapshots as market_snapshots
    import api.position_risk as position_risk

    def fail_market():
        raise RuntimeError("market unavailable")

    monkeypatch.setattr(market_snapshots, "refresh_market_snapshots", fail_market)
    monkeypatch.setattr(macro_snapshots, "refresh_macro_snapshots", lambda: {"snapshots": ["macro"]})
    monkeypatch.setattr(position_risk, "refresh_portfolio_risk", lambda: {"result_id": "portfolio-risk:1"})

    with pytest.raises(RuntimeError, match="market unavailable"):
        maintenance_jobs.refresh_workspace_sources()
