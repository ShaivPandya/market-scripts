from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import pandas as pd


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

    def fake_fetch(**kwargs: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        raw = {
            "vix_term_structure": {"latest_df": [{"Ratio": 0.92, "VIX": 24.0, "Date": "2026-03-07"}]},
            "market_breadth": None,
            "top50_breadth": None,
            "liquidity": None,
            "sector_metrics": None,
            "momentum": {"results": [{"avg10_rel_roc": -0.2, "rel_roc42": -0.3}]},
        }
        status = {
            "vix_term_structure": {"status": "ok"},
            "market_breadth": {"status": "error", "detail": "failed"},
            "top50_breadth": {"status": "error", "detail": "failed"},
            "liquidity": {"status": "error", "detail": "failed"},
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
                "missing_factors": ["liquidity", "breadth", "sector", "momentum"],
                "module_status": {"vix": "ok", "liquidity": "error"},
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
    factors = {f["key"]: f for f in result["factors"]}
    # Only vix (20%) + momentum (10%) available -> effective weights 2/3 and 1/3.
    assert round(float(factors["vix"]["weight"]), 4) == round(2 / 3, 4)
    assert round(float(factors["momentum"]["weight"]), 4) == round(1 / 3, 4)
    assert result["regime"]["label"] in {"risk-on", "transitional", "risk-off"}


def test_build_signal_aggregator_live_fills_missing_liquidity_snapshot(monkeypatch):
    from api import signal_aggregator as sa

    raw = {
        "vix_term_structure": {"latest_df": [{"Ratio": 1.1, "VIX": 15.0, "Date": "2026-05-01"}]},
        "market_breadth": {"pct_above_200dma": 60.0, "pct_above_20dma": 55.0, "pct_at_20day_low": 10.0},
        "top50_breadth": {"pct_below_50dma": 20.0, "pct_3plus_dist": 10.0, "pct_broke_20low": 5.0},
        "liquidity": None,
        "sector_metrics": {
            "timestamp": "2026-05-01T22:00:00",
            "weights_df": [{"RelPerf_3M_pp": 1.0, "Chg_3M_pp": 1.0, "Pct_Above_200DMA": 5.0, "Weight_Now": 100.0}],
        },
        "momentum": {"results": [{"avg10_rel_roc": 0.2, "rel_roc42": 0.3}]},
    }
    status = {
        "vix_term_structure": {"status": "ok"},
        "market_breadth": {"status": "ok"},
        "top50_breadth": {"status": "ok"},
        "liquidity": {"status": "error", "detail": "Snapshot unavailable: liquidity:current:v1"},
        "sector_metrics": {"status": "ok"},
        "momentum": {"status": "ok"},
    }

    monkeypatch.setattr(
        "macro.liquidity.liquidity.get_snapshot",
        lambda: {"latest_date": "2026-05-01", "composite_score": -0.5, "regime": "tight"},
    )

    result = sa.build_signal_aggregator_from_payloads(raw, status, include_history=False)
    factors = {f["key"]: f for f in result["factors"]}

    assert result["module_status"]["liquidity"]["status"] == "ok"
    assert factors["liquidity"]["status"] == "ok"
    assert "liquidity" not in result["failed_modules"]


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
            "history": {
                "frequency": "weekly",
                "lookback_weeks": lookback_weeks,
                "coverage": {},
                "series": [],
                "episodes": [],
            },
        }

    monkeypatch.setattr(signal_router, "get_or_set_cached", lambda _cache, _key, loader, **_kwargs: loader())
    monkeypatch.setattr(signal_router, "get_signal_aggregator_snapshot_or_module_response", lambda **kwargs: None)
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


def test_signal_aggregator_endpoint_force_refresh_bypasses_snapshot(auth_client, monkeypatch):
    import api.routers.signal_aggregator as signal_router

    calls: list[str] = []

    def fake_build(lookback_weeks: int, positioning_instruments: str, include_raw_modules: bool):
        calls.append(positioning_instruments)
        return {
            "status": "ok",
            "as_of": "2026-03-07",
            "regime": {"label": "risk-on", "score": 20.0, "confidence": 1.0, "history_percentile": None},
            "weights": {"configured": {}, "effective": {}},
            "factors": [],
            "module_status": {},
            "failed_modules": [],
            "history": {
                "frequency": "weekly",
                "lookback_weeks": lookback_weeks,
                "coverage": {},
                "series": [],
                "episodes": [],
            },
        }

    def fail_cache_read(*args, **kwargs):
        raise AssertionError("cache should not be read")

    import api.cache as cache_mod

    monkeypatch.setattr(cache_mod, "get_cached", fail_cache_read)
    monkeypatch.setattr(cache_mod, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        signal_router,
        "get_signal_aggregator_snapshot_or_module_response",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("snapshot should not be read")),
    )
    monkeypatch.setattr(signal_router, "build_signal_aggregator", fake_build)

    resp = auth_client.get(
        "/api/v1/signal-aggregator",
        params={
            "lookback_weeks": 104,
            "positioning_instruments": "sp500, eur",
            "force_refresh": "true",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert calls == ["SP500,EUR"]


def test_signal_aggregator_force_refresh_coalesces(monkeypatch):
    import api.cache as cache_mod
    import api.routers.signal_aggregator as signal_router

    monkeypatch.setattr(cache_mod, "_DISK_CACHE_ENABLED", False)
    monkeypatch.setattr(
        signal_router,
        "get_signal_aggregator_snapshot_or_module_response",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("snapshot should not be read")),
    )

    calls = 0
    calls_lock = threading.Lock()

    def fake_build(lookback_weeks: int, positioning_instruments: str, include_raw_modules: bool):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return {
            "status": "ok",
            "as_of": "2026-03-07",
            "regime": {"label": "risk-on", "score": 20.0, "confidence": 1.0, "history_percentile": None},
            "weights": {"configured": {}, "effective": {}},
            "factors": [],
            "module_status": {},
            "failed_modules": [],
            "history": {
                "frequency": "weekly",
                "lookback_weeks": lookback_weeks,
                "coverage": {},
                "series": [],
                "episodes": [],
            },
        }

    monkeypatch.setattr(signal_router, "build_signal_aggregator", fake_build)

    def request():
        return signal_router.get_signal_aggregator(
            lookback_weeks=111,
            positioning_instruments="SP500,EUR",
            include_raw_modules=False,
            force_refresh=True,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _idx: request(), range(4)))

    assert calls == 1
    assert [result["history"]["lookback_weeks"] for result in results] == [111, 111, 111, 111]


def test_signal_aggregator_endpoint_cache_key_includes_positioning(auth_client, monkeypatch):
    import api.routers.signal_aggregator as signal_router

    cache_keys: list[str] = []

    def fake_get_or_set_cached(_cache, key: str, loader, **_kwargs):
        cache_keys.append(key)
        return loader()

    def fake_build(lookback_weeks: int, positioning_instruments: str, include_raw_modules: bool):
        return {
            "status": "ok",
            "as_of": "2026-03-07",
            "regime": {"label": "risk-on", "score": 20.0, "confidence": 1.0, "history_percentile": None},
            "weights": {"configured": {}, "effective": {}},
            "factors": [],
            "module_status": {},
            "failed_modules": [],
            "history": {
                "frequency": "weekly",
                "lookback_weeks": lookback_weeks,
                "coverage": {},
                "series": [],
                "episodes": [],
            },
        }

    monkeypatch.setattr(signal_router, "get_or_set_cached", fake_get_or_set_cached)
    monkeypatch.setattr(signal_router, "get_signal_aggregator_snapshot_or_module_response", lambda **kwargs: None)
    monkeypatch.setattr(signal_router, "build_signal_aggregator", fake_build)

    auth_client.get("/api/v1/signal-aggregator", params={"positioning_instruments": "SP500,EUR"})
    auth_client.get("/api/v1/signal-aggregator", params={"positioning_instruments": "NASDAQ,US10Y"})

    assert len(cache_keys) == 2
    assert cache_keys[0] != cache_keys[1]
    assert "positioning=SP500,EUR" in cache_keys[0]
    assert "positioning=NASDAQ,US10Y" in cache_keys[1]


def test_signal_aggregator_endpoint_degraded_payload(auth_client, monkeypatch):
    import api.routers.signal_aggregator as signal_router

    def fake_build(*args, **kwargs):
        return {
            "status": "degraded",
            "as_of": "2026-03-07",
            "regime": {"label": "risk-off", "score": 70.0, "confidence": 0.8, "history_percentile": 92.0},
            "weights": {"configured": {}, "effective": {}},
            "factors": [
                {"key": "vix", "status": "ok", "score": 80.0, "weight": 1.0, "contribution": 80.0, "highlights": {}}
            ],
            "module_status": {"liquidity": {"status": "error", "detail": "timeout"}},
            "failed_modules": ["liquidity"],
            "history": {"frequency": "weekly", "lookback_weeks": 156, "coverage": {}, "series": [], "episodes": []},
        }

    monkeypatch.setattr(signal_router, "get_or_set_cached", lambda _cache, _key, loader, **_kwargs: loader())
    monkeypatch.setattr(signal_router, "get_signal_aggregator_snapshot_or_module_response", lambda **kwargs: None)
    monkeypatch.setattr(signal_router, "build_signal_aggregator", fake_build)

    resp = auth_client.get("/api/v1/signal-aggregator", params={"lookback_weeks": 157})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert "liquidity" in data["failed_modules"]


def test_sp500_price_cache_refreshes_when_close_changed_under_ttl(monkeypatch, tmp_path):
    from api import signal_aggregator as sa

    cache_dir = tmp_path / "signal_aggregator"
    monkeypatch.setattr(sa, "_SP500_CACHE_DIR", cache_dir)
    monkeypatch.setattr(sa, "_SP500_CACHE_DATA", cache_dir / "sp500_prices.pkl")
    monkeypatch.setattr(sa, "_SP500_CACHE_META", cache_dir / "sp500_prices_meta.json")

    stale_df = pd.DataFrame({"AAPL": [1.0]}, index=pd.to_datetime(["2000-01-01"]))
    sa._save_sp500_cache(stale_df, "2000-01-01")
    meta_path = cache_dir / "sp500_prices_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = (datetime.now() - timedelta(hours=2)).isoformat()
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    fresh_df = pd.DataFrame({"AAPL": [2.0]}, index=pd.to_datetime(["2099-01-01"]))
    monkeypatch.setattr(sa, "_latest_market_close_date", lambda: "2099-01-01")
    monkeypatch.setattr(sa, "_download_sp500_prices_uncached", lambda: fresh_df)

    out, market_cache = sa._download_sp500_prices_with_meta()

    assert out.iloc[-1]["AAPL"] == 2.0
    assert market_cache["status"] == "refresh"
    assert market_cache["stale"] is False


def test_sp500_price_cache_returns_stale_fallback_when_probe_fails(monkeypatch, tmp_path):
    from api import signal_aggregator as sa

    cache_dir = tmp_path / "signal_aggregator"
    monkeypatch.setattr(sa, "_SP500_CACHE_DIR", cache_dir)
    monkeypatch.setattr(sa, "_SP500_CACHE_DATA", cache_dir / "sp500_prices.pkl")
    monkeypatch.setattr(sa, "_SP500_CACHE_META", cache_dir / "sp500_prices_meta.json")

    stale_df = pd.DataFrame({"AAPL": [1.0]}, index=pd.to_datetime(["2000-01-01"]))
    sa._save_sp500_cache(stale_df, "2000-01-01")
    meta_path = cache_dir / "sp500_prices_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = (datetime.now() - timedelta(hours=2)).isoformat()
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(sa, "_latest_market_close_date", lambda: None)
    monkeypatch.setattr(
        sa,
        "_download_sp500_prices_uncached",
        lambda: (_ for _ in ()).throw(AssertionError("should not refresh after probe failure fallback")),
    )

    out, market_cache = sa._download_sp500_prices_with_meta()

    assert out.iloc[-1]["AAPL"] == 1.0
    assert market_cache["status"] == "stale_fallback"
    assert market_cache["stale"] is True
