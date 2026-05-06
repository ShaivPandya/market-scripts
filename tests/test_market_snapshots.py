from __future__ import annotations

import pandas as pd


def test_refresh_market_snapshots_writes_module_and_signal_payloads(monkeypatch):
    from api import market_snapshots as ms

    writes: list[tuple[str, dict, str | None]] = []
    failures: list[tuple[str, str]] = []

    def fake_build_signal_aggregator(**kwargs):
        assert kwargs["lookback_weeks"] == 520
        assert kwargs["include_raw_modules"] is True
        return {
            "status": "ok",
            "as_of": "2026-05-01",
            "regime": {"label": "risk-on", "score": 20.0, "confidence": 1.0},
            "module_status": {
                "market_breadth": {"status": "ok"},
                "top50_breadth": {"status": "ok"},
                "sector_metrics": {"status": "ok"},
                "liquidity": {"status": "ok"},
                "vix_term_structure": {"status": "ok"},
                "momentum": {"status": "ok"},
            },
            "history": {"series": [], "episodes": []},
            "raw_modules": {
                "market_breadth": {"as_of_date": "2026-05-01", "total_analyzed": 503},
                "top50_breadth": {"universe_size": 50},
                "sector_metrics": {"timestamp": "2026-05-01T23:00:00", "weights_df": [{"Weight_Now": 17.8}]},
                "liquidity": {"latest_date": "2026-05-01"},
                "vix_term_structure": {"latest_df": [{"Date": "2026-05-01"}]},
                "momentum": {"date": "2026-05-01"},
            },
        }

    class Record:
        def __init__(self, as_of_date):
            self.as_of_date = as_of_date

    monkeypatch.setattr("api.signal_aggregator.build_signal_aggregator", fake_build_signal_aggregator)
    monkeypatch.setattr(
        ms,
        "write_snapshot_success",
        lambda key, payload, **kwargs: (
            writes.append((key, payload, kwargs.get("as_of_date"))) or Record(kwargs.get("as_of_date"))
        ),
    )
    monkeypatch.setattr(
        ms,
        "write_snapshot_failure",
        lambda key, error, **kwargs: failures.append((key, error)) or None,
    )

    result = ms.refresh_market_snapshots()
    assert failures == []
    written_keys = {key for key, _payload, _as_of in writes}
    assert ms.SNAPSHOT_MARKET_BREADTH in written_keys
    assert ms.SNAPSHOT_TOP50_BREADTH in written_keys
    assert ms.SNAPSHOT_SIGNAL_AGGREGATOR in written_keys
    sector_payload = next(payload for key, payload, _as_of in writes if key == ms.SNAPSHOT_SECTOR_METRICS)
    assert sector_payload["weights_df"][0]["Sector"] == "Communication Services"
    assert result["snapshots"][-1]["snapshot_key"] == ms.SNAPSHOT_SIGNAL_AGGREGATOR


def test_signal_current_modules_share_one_sp500_price_frame(monkeypatch):
    from api import signal_aggregator as sa
    from equities.market_technicals import market_breadth, top50_breadth, vix_term_structure
    from equities.sector_metrics import sector_metrics
    from macro.liquidity import liquidity
    from portfolio.momentum.price_momentum import momentum

    dates = pd.bdate_range("2026-04-01", periods=3)
    close = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]}, index=dates)
    prices = pd.concat(
        {
            "Close": close,
            "High": close,
            "Low": close,
            "Volume": close,
        },
        axis=1,
    )
    download_calls = 0
    shared_calls: list[str] = []

    market_cache_meta = {
        "status": "hit",
        "stale": False,
        "reason": "test metadata",
        "cache_ttl_seconds": 86400,
    }

    def fake_download_with_meta():
        nonlocal download_calls
        download_calls += 1
        return prices, market_cache_meta

    def assert_shared(name):
        def _inner(*args, **kwargs):
            assert kwargs.get("prices_df") is prices
            shared_calls.append(name)
            return {"ok": name}

        return _inner

    vix_df = pd.DataFrame(
        [{"VIX": 20.0, "VIX3M": 22.0, "Ratio": 1.1, "Signal": "Neutral"}],
        index=pd.to_datetime(["2026-05-01"]),
    )

    monkeypatch.setattr(sa, "_download_sp500_prices_with_meta", fake_download_with_meta)
    monkeypatch.setattr(market_breadth, "get_data", assert_shared("market_breadth"))
    monkeypatch.setattr(top50_breadth, "get_data", assert_shared("top50_breadth"))
    monkeypatch.setattr(sector_metrics, "get_data", assert_shared("sector_metrics"))
    monkeypatch.setattr(vix_term_structure, "load_term_structure", lambda start: (vix_df, "VIX3M"))
    monkeypatch.setattr(vix_term_structure, "add_signals", lambda data, low, high: data)
    monkeypatch.setattr(liquidity, "get_snapshot", lambda: {"composite_score": 0.0, "regime": "normal"})
    monkeypatch.setattr(momentum, "get_data", lambda: {"results": []})

    raw, status = sa._fetch_current_modules(lookback_weeks=26)

    assert download_calls == 1
    assert set(shared_calls) == {"market_breadth", "top50_breadth", "sector_metrics"}
    assert status["market_breadth"]["status"] == "ok"
    assert status["market_breadth"]["market_cache"] == market_cache_meta
    assert status["top50_breadth"]["market_cache"] == market_cache_meta
    assert status["sector_metrics"]["market_cache"] == market_cache_meta
    assert raw["market_breadth"] == {"ok": "market_breadth"}
