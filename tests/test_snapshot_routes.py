from __future__ import annotations


def test_market_breadth_route_uses_snapshot(auth_client, monkeypatch):
    import api.routers.market_technicals as router

    monkeypatch.setattr(router, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(router, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        router,
        "get_snapshot_response",
        lambda key: {"total_analyzed": 503, "pct_above_200dma": 55.0, "_meta": {"snapshot": {"key": key}}},
    )

    resp = auth_client.get("/api/v1/market-breadth")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_analyzed"] == 503
    assert body["_meta"]["snapshot"]["key"] == router.SNAPSHOT_MARKET_BREADTH


def test_signal_aggregator_route_uses_snapshot(auth_client, monkeypatch):
    import api.routers.signal_aggregator as router

    monkeypatch.setattr(router, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(router, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        router,
        "get_signal_aggregator_snapshot_response",
        lambda **kwargs: {
            "status": "ok",
            "as_of": "2026-05-01",
            "regime": {"label": "risk-on", "score": 20.0, "confidence": 1.0},
            "history": {"lookback_weeks": kwargs["lookback_weeks"], "series": [], "episodes": []},
            "_meta": {"snapshot": {"key": "signal_aggregator:current:v1"}},
        },
    )
    monkeypatch.setattr(
        router,
        "build_signal_aggregator",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live compute should not run")),
    )

    resp = auth_client.get("/api/v1/signal-aggregator", params={"lookback_weeks": 104})
    assert resp.status_code == 200
    body = resp.json()
    assert body["regime"]["label"] == "risk-on"
    assert body["history"]["lookback_weeks"] == 104
