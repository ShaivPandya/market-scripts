from __future__ import annotations

import pandas as pd
import pytest


def test_market_breadth_route_uses_snapshot(auth_client, monkeypatch):
    import api.routers.market_technicals as router

    monkeypatch.setattr(router, "get_or_set_cached", lambda _cache, _key, loader, **_kwargs: loader())
    monkeypatch.setattr(
        router,
        "get_snapshot_response",
        lambda key: {"total_analyzed": 503, "pct_above_200dma": 55.0, "_meta": {"snapshot": {"key": key}}},
    )

    resp = auth_client.get("/api/market-breadth")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_analyzed"] == 503
    assert body["_meta"]["snapshot"]["key"] == router.SNAPSHOT_MARKET_BREADTH


def test_sector_metrics_route_uses_snapshot(auth_client, monkeypatch):
    import api.routers.sector_metrics as router

    monkeypatch.setattr(router, "get_or_set_cached", lambda _cache, _key, loader, **_kwargs: loader())
    monkeypatch.setattr(
        router,
        "get_snapshot_response",
        lambda key: {"weights_df": [{"Sector": "Technology", "Weight_Now": 30.0}], "_meta": {"snapshot": {"key": key}}},
    )

    resp = auth_client.get("/api/sector-metrics")

    assert resp.status_code == 200
    body = resp.json()
    assert body["weights_df"][0]["Sector"] == "Technology"
    assert body["_meta"]["snapshot"]["key"] == router.SNAPSHOT_SECTOR_METRICS


def test_sector_metrics_route_repairs_current_snapshot_without_sector(auth_client, monkeypatch):
    import api.routers.sector_metrics as router

    monkeypatch.setattr(router, "get_or_set_cached", lambda _cache, _key, loader, **_kwargs: loader())
    monkeypatch.setattr(
        router,
        "get_snapshot_response",
        lambda key: {"weights_df": [{"Weight_Now": 17.8}], "_meta": {"snapshot": {"key": key}}},
    )

    resp = auth_client.get("/api/sector-metrics")

    assert resp.status_code == 200
    body = resp.json()
    assert body["weights_df"][0]["Sector"] == "Communication Services"


def test_sector_metrics_route_fails_fast_when_snapshot_required(auth_client, monkeypatch):
    import api.routers.sector_metrics as router

    monkeypatch.setattr(router, "get_or_set_cached", lambda _cache, _key, loader, **_kwargs: loader())
    monkeypatch.setattr(router, "get_snapshot_response", lambda _key: None)
    monkeypatch.setattr(router, "snapshots_required", lambda: True)
    monkeypatch.setattr(
        "equities.sector_metrics.sector_metrics.get_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live compute should not run")),
    )

    resp = auth_client.get("/api/sector-metrics")

    assert resp.status_code == 503
    assert resp.json()["type"] == "SnapshotUnavailableError"


def test_sector_metrics_series_route_serializes_prices_and_relative_ratios(auth_client, monkeypatch):
    import api.routers.sector_metrics as router
    from equities.sector_metrics import sector_metrics as sector_metrics_mod

    monkeypatch.setattr(router, "get_or_set_cached", lambda _cache, _key, loader, **_kwargs: loader())
    monkeypatch.setattr(
        sector_metrics_mod,
        "SECTOR_ETFS",
        {"Information Technology": "XLK", "Energy": "XLE"},
    )
    monkeypatch.setattr(sector_metrics_mod, "BENCHMARK_ETF", "SPY")

    calls = []

    def fake_fetch_etf_prices(sector_etfs, benchmark, period="2y", interval="1d"):
        calls.append({"sector_etfs": sector_etfs, "benchmark": benchmark, "period": period, "interval": interval})
        return pd.DataFrame(
            {
                "XLK": [100.0, 110.0],
                "XLE": [50.0, 45.0],
                "SPY": [200.0, 220.0],
            },
            index=pd.to_datetime(["2026-05-14", "2026-05-15"]),
        )

    monkeypatch.setattr(sector_metrics_mod, "fetch_etf_prices", fake_fetch_etf_prices)

    resp = auth_client.get("/api/sector-metrics/series", params={"timeframe": "Invalid"})

    assert resp.status_code == 200
    body = resp.json()
    assert calls == [
        {
            "sector_etfs": {"Information Technology": "XLK", "Energy": "XLE"},
            "benchmark": "SPY",
            "period": "90d",
            "interval": "1d",
        }
    ]
    assert body["timeframe"] == "Daily"
    assert body["benchmark"] == "SPY"
    assert body["sector_order"] == ["Information Technology", "Energy"]
    assert body["sector_prices"]["Information Technology"][0] == {
        "date": "2026-05-14T00:00:00",
        "value": 100.0,
    }
    assert body["sector_relative_prices"]["Information Technology"][1]["value"] == pytest.approx(110.0 / 220.0)
    assert body["sector_relative_prices"]["Energy"][1]["value"] == pytest.approx(45.0 / 220.0)


def test_liquidity_route_preserves_payload_and_attaches_quality_warnings(auth_client, monkeypatch):
    import api.routers.liquidity as router

    def fake_response(**_kwargs):
        return {
            "latest_date": "2026-05-13",
            "composite_score": -0.03,
            "regime": "normal",
            "component_as_of": {"net_liquidity": "2026-05-13"},
            "data_quality": {"status": "ok", "warnings": []},
            "_meta": {
                "snapshot": {
                    "key": router.SNAPSHOT_LIQUIDITY,
                    "stale": True,
                    "refresh_status": "error",
                    "error": "FRED down",
                }
            },
        }

    monkeypatch.setattr(router, "get_snapshot_backed_response", fake_response)

    resp = auth_client.get("/api/liquidity")

    assert resp.status_code == 200
    body = resp.json()
    assert body["latest_date"] == "2026-05-13"
    assert body["component_as_of"] == {"net_liquidity": "2026-05-13"}
    assert body["data_quality"]["status"] == "degraded"
    assert any("Snapshot is stale" in warning for warning in body["data_quality"]["warnings"])
    assert any("FRED down" in warning for warning in body["data_quality"]["warnings"])


def test_signal_aggregator_route_uses_snapshot(auth_client, monkeypatch):
    import api.routers.signal_aggregator as router

    monkeypatch.setattr(router, "get_or_set_cached", lambda _cache, _key, loader, **_kwargs: loader())
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

    resp = auth_client.get("/api/signal-aggregator", params={"lookback_weeks": 104})
    assert resp.status_code == 200
    body = resp.json()
    assert body["regime"]["label"] == "risk-on"
    assert body["history"]["lookback_weeks"] == 104


def test_signal_aggregator_route_falls_back_to_module_snapshots(auth_client, monkeypatch):
    import api.routers.signal_aggregator as router
    import api.signal_snapshot as signal_snapshot

    monkeypatch.setattr(router, "get_or_set_cached", lambda _cache, _key, loader, **_kwargs: loader())

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

    resp = auth_client.get("/api/signal-aggregator", params={"lookback_weeks": 104})

    assert resp.status_code == 200
    body = resp.json()
    assert body["regime"]["label"] == "risk-on"
    assert body["_meta"]["snapshot"]["source"] == "module_snapshots"
    assert body["history"]["lookback_weeks"] == 104
