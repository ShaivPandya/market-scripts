from __future__ import annotations

from typing import Any


def test_regime_thresholds():
    from api.signal_aggregator import _regime_label

    assert _regime_label(39.99) == "risk-on"
    assert _regime_label(40.0) == "transitional"
    assert _regime_label(64.99) == "transitional"
    assert _regime_label(65.0) == "risk-off"


def test_vix_score_formula():
    from api.signal_aggregator import _score_vix

    score, highlights = _score_vix({"latest_df": [{"Ratio": 0.9, "VIX": 30.0}]})
    assert score is not None
    # ratio comp: clamp((1 - 0.9)/0.2) = 0.5  -> 35
    # vix comp:   clamp((30 - 18)/12) = 1.0   -> 30
    # total = 65
    assert round(score, 2) == 65.0
    assert highlights["ratio"] == 0.9
    assert highlights["vix"] == 30.0


def test_build_signal_aggregator_degraded_reweights(monkeypatch):
    from api import signal_aggregator as sa

    def fake_fetch(_: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        raw = {
            "vix_term_structure": {"latest_df": [{"Ratio": 0.92, "VIX": 24.0, "Date": "2026-03-07"}]},
            "market_breadth": None,
            "top50_breadth": None,
            "liquidity": None,
            "positioning": None,
            "sector_metrics": None,
            "momentum": {"results": [{"avg10_rel_roc": -0.2, "rel_roc42": -0.3}]},
        }
        status = {
            "vix_term_structure": {"status": "ok"},
            "market_breadth": {"status": "error", "detail": "failed"},
            "top50_breadth": {"status": "error", "detail": "failed"},
            "liquidity": {"status": "error", "detail": "failed"},
            "positioning": {"status": "error", "detail": "failed"},
            "sector_metrics": {"status": "error", "detail": "failed"},
            "momentum": {"status": "ok"},
        }
        return raw, status

    def fake_history(*args, **kwargs):
        return {
            "frequency": "weekly",
            "lookback_weeks": 156,
            "coverage": {
                "included_factors": ["vix"],
                "missing_factors": ["liquidity", "positioning", "breadth", "sector", "momentum"],
                "module_status": {"vix": "ok", "liquidity": "error", "positioning": "error"},
            },
            "series": [{"date": "2026-03-07", "score": 50.0, "label": "transitional", "factors": {"vix": 50.0}}],
            "episodes": [],
            "scores": [50.0],
        }

    monkeypatch.setattr(sa, "_fetch_current_modules", fake_fetch)
    monkeypatch.setattr(sa, "_build_history", fake_history)

    result = sa.build_signal_aggregator(lookback_weeks=156, positioning_instruments="SP500,EUR")
    assert result["status"] == "degraded"
    assert "breadth" in result["failed_modules"]
    assert "liquidity" in result["failed_modules"]
    assert "positioning" in result["failed_modules"]

    factors = {f["key"]: f for f in result["factors"]}
    # Only vix (20%) + momentum (10%) available -> effective weights 2/3 and 1/3.
    assert round(float(factors["vix"]["weight"]), 4) == round(2 / 3, 4)
    assert round(float(factors["momentum"]["weight"]), 4) == round(1 / 3, 4)
    assert result["regime"]["label"] in {"risk-on", "transitional", "risk-off"}


def test_signal_aggregator_endpoint_uses_query_params(auth_client, monkeypatch):
    import api.routers.signal_aggregator as signal_router

    calls: list[tuple[int, str, bool]] = []

    def fake_build(lookback_weeks: int, positioning_instruments: str, include_raw_modules: bool):
        calls.append((lookback_weeks, positioning_instruments, include_raw_modules))
        return {
            "status": "ok",
            "as_of": "2026-03-07",
            "regime": {"label": "transitional", "score": 52.0, "confidence": 1.0, "history_percentile": 61.0},
            "weights": {"configured": {}, "effective": {}},
            "factors": [],
            "module_status": {},
            "failed_modules": [],
            "history": {"frequency": "weekly", "lookback_weeks": lookback_weeks, "coverage": {}, "series": [], "episodes": []},
        }

    monkeypatch.setattr(signal_router, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(signal_router, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(signal_router, "build_signal_aggregator", fake_build)

    resp = auth_client.get(
        "/api/v1/signal-aggregator",
        params={
            "lookback_weeks": 104,
            "positioning_instruments": "SP500,EUR",
            "include_raw_modules": "true",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["history"]["lookback_weeks"] == 104
    assert calls == [(104, "SP500,EUR", True)]


def test_signal_aggregator_endpoint_degraded_payload(auth_client, monkeypatch):
    import api.routers.signal_aggregator as signal_router

    def fake_build(*args, **kwargs):
        return {
            "status": "degraded",
            "as_of": "2026-03-07",
            "regime": {"label": "risk-off", "score": 70.0, "confidence": 0.8, "history_percentile": 92.0},
            "weights": {"configured": {}, "effective": {}},
            "factors": [{"key": "vix", "status": "ok", "score": 80.0, "weight": 1.0, "contribution": 80.0, "highlights": {}}],
            "module_status": {"liquidity": {"status": "error", "detail": "timeout"}},
            "failed_modules": ["liquidity"],
            "history": {"frequency": "weekly", "lookback_weeks": 156, "coverage": {}, "series": [], "episodes": []},
        }

    monkeypatch.setattr(signal_router, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(signal_router, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(signal_router, "build_signal_aggregator", fake_build)

    resp = auth_client.get("/api/v1/signal-aggregator", params={"lookback_weeks": 157})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert "liquidity" in data["failed_modules"]
