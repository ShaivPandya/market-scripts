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


def test_sector_metrics_route_uses_snapshot(auth_client, monkeypatch):
    import api.routers.sector_metrics as router

    monkeypatch.setattr(router, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(router, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        router,
        "get_snapshot_response",
        lambda key: {"weights_df": [{"Sector": "Technology", "Weight_Now": 30.0}], "_meta": {"snapshot": {"key": key}}},
    )

    resp = auth_client.get("/api/v1/sector-metrics")

    assert resp.status_code == 200
    body = resp.json()
    assert body["weights_df"][0]["Sector"] == "Technology"
    assert body["_meta"]["snapshot"]["key"] == router.SNAPSHOT_SECTOR_METRICS


def test_sector_metrics_route_repairs_legacy_snapshot_without_sector(auth_client, monkeypatch):
    import api.routers.sector_metrics as router

    monkeypatch.setattr(router, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(router, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        router,
        "get_snapshot_response",
        lambda key: {"weights_df": [{"Weight_Now": 17.8}], "_meta": {"snapshot": {"key": key}}},
    )

    resp = auth_client.get("/api/v1/sector-metrics")

    assert resp.status_code == 200
    body = resp.json()
    assert body["weights_df"][0]["Sector"] == "Communication Services"


def test_sector_metrics_route_fails_fast_when_snapshot_required(auth_client, monkeypatch):
    import api.routers.sector_metrics as router

    monkeypatch.setattr(router, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(router, "get_snapshot_response", lambda _key: None)
    monkeypatch.setattr(router, "snapshots_required", lambda: True)
    monkeypatch.setattr(
        "equities.sector_metrics.sector_metrics.get_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live compute should not run")),
    )

    resp = auth_client.get("/api/v1/sector-metrics")

    assert resp.status_code == 503
    assert resp.json()["type"] == "SnapshotUnavailableError"


def test_signal_aggregator_route_uses_snapshot(auth_client, monkeypatch):
    import api.routers.signal_aggregator as router

    monkeypatch.setattr(router, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(router, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        router,
        "get_signal_aggregator_snapshot_or_module_response",
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


def test_signal_aggregator_route_falls_back_to_module_snapshots(auth_client, monkeypatch):
    import api.routers.signal_aggregator as router
    import api.signal_snapshot as signal_snapshot

    monkeypatch.setattr(router, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(router, "set_cached", lambda *args, **kwargs: None)

    def with_meta(payload, key):
        return {
            **payload,
            "_meta": {
                "snapshot": {
                    "key": key,
                    "as_of": "2026-05-01",
                    "fetched_at": "2026-05-01T22:00:00",
                    "data_age_seconds": 60,
                    "stale": False,
                    "refresh_status": "ok",
                    "error": None,
                    "version": 1,
                }
            },
        }

    payloads = {
        signal_snapshot.SNAPSHOT_VIX_TERM_STRUCTURE: with_meta(
            {"latest_df": [{"Date": "2026-05-01", "Ratio": 1.1, "VIX": 15.0}]},
            signal_snapshot.SNAPSHOT_VIX_TERM_STRUCTURE,
        ),
        signal_snapshot.SNAPSHOT_MARKET_BREADTH: with_meta(
            {"pct_above_200dma": 60.0, "pct_above_20dma": 55.0, "pct_at_20day_low": 10.0},
            signal_snapshot.SNAPSHOT_MARKET_BREADTH,
        ),
        signal_snapshot.SNAPSHOT_TOP50_BREADTH: with_meta(
            {"pct_below_50dma": 20.0, "pct_3plus_dist": 10.0, "pct_broke_20low": 5.0},
            signal_snapshot.SNAPSHOT_TOP50_BREADTH,
        ),
        signal_snapshot.SNAPSHOT_LIQUIDITY: with_meta(
            {"latest_date": "2026-05-01", "composite_score": 0.0, "regime": "normal"},
            signal_snapshot.SNAPSHOT_LIQUIDITY,
        ),
        signal_snapshot.SNAPSHOT_SECTOR_METRICS: with_meta(
            {
                "timestamp": "2026-05-01T22:00:00",
                "weights_df": [{"RelPerf_3M_pp": 1.0, "Chg_3M_pp": 1.0, "Pct_Above_200DMA": 5.0, "Weight_Now": 100.0}],
            },
            signal_snapshot.SNAPSHOT_SECTOR_METRICS,
        ),
        signal_snapshot.SNAPSHOT_MOMENTUM: with_meta(
            {"results": [{"avg10_rel_roc": 0.2, "rel_roc42": 0.3}]},
            signal_snapshot.SNAPSHOT_MOMENTUM,
        ),
    }

    monkeypatch.setattr(signal_snapshot, "get_snapshot_response", lambda key: payloads.get(key))

    resp = auth_client.get("/api/v1/signal-aggregator", params={"lookback_weeks": 104})

    assert resp.status_code == 200
    body = resp.json()
    assert body["regime"]["label"] == "risk-on"
    assert body["_meta"]["snapshot"]["source"] == "module_snapshots"
    assert body["history"]["lookback_weeks"] == 104
