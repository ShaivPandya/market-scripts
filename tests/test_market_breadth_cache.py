from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from equities.market_technicals import market_breadth as mb


def _write_cache(path: Path, payload: dict, fetched_at: datetime, as_of_date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": mb._CACHE_VERSION,
                "fetched_at": fetched_at.isoformat(),
                "as_of_date": as_of_date,
                "universe": "sp500",
                "period": "1y",
                "payload": payload,
            }
        ),
        encoding="utf-8",
    )


def test_get_data_uses_fresh_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "market_breadth_sp500_1y.json"
    cached = {"pct_above_200dma": 54.2, "as_of_date": "2026-03-06", "tickers": ["AAPL", "MSFT"]}
    _write_cache(cache_path, cached, datetime.now() - timedelta(hours=2), "2026-03-06")

    monkeypatch.setattr(mb, "_breadth_cache_path", lambda *_: cache_path)
    monkeypatch.setattr(mb, "get_tickers", lambda *_: (_ for _ in ()).throw(AssertionError("should not fetch tickers")))
    monkeypatch.setattr(
        mb,
        "calculate_breadth_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not recalculate")),
    )

    out = mb.get_data(universe="sp500", period="1y")
    assert out == cached


def test_get_data_renews_stale_cache_when_close_unchanged(monkeypatch, tmp_path):
    cache_path = tmp_path / "market_breadth_sp500_1y.json"
    old_fetch = datetime.now() - timedelta(hours=30)
    cached = {"pct_above_200dma": 50.0, "as_of_date": "2026-03-06", "tickers": ["AAPL"]}
    _write_cache(cache_path, cached, old_fetch, "2026-03-06")

    monkeypatch.setattr(mb, "_breadth_cache_path", lambda *_: cache_path)
    monkeypatch.setattr(mb, "_latest_market_close_date", lambda: "2026-03-06")
    monkeypatch.setattr(mb, "get_tickers", lambda *_: (_ for _ in ()).throw(AssertionError("should not fetch tickers")))
    monkeypatch.setattr(
        mb,
        "calculate_breadth_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not recalculate")),
    )

    out = mb.get_data(universe="sp500", period="1y")
    assert out == cached

    stored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(stored["fetched_at"]) > old_fetch


def test_get_data_refreshes_when_close_changed(monkeypatch, tmp_path):
    cache_path = tmp_path / "market_breadth_sp500_1y.json"
    cached = {"pct_above_200dma": 40.0, "as_of_date": "2026-03-06", "tickers": ["AAPL"]}
    _write_cache(cache_path, cached, datetime.now() - timedelta(hours=30), "2026-03-06")

    fresh_metrics = {"pct_above_200dma": 61.0, "as_of_date": "2026-03-07", "failed_tickers": []}
    monkeypatch.setattr(mb, "_breadth_cache_path", lambda *_: cache_path)
    monkeypatch.setattr(mb, "_latest_market_close_date", lambda: "2026-03-07")
    monkeypatch.setattr(mb, "get_tickers", lambda *_: ["AAPL", "MSFT"])
    monkeypatch.setattr(mb, "calculate_breadth_metrics", lambda *_args, **_kwargs: fresh_metrics.copy())

    out = mb.get_data(universe="sp500", period="1y")
    assert out["pct_above_200dma"] == 61.0
    assert out["as_of_date"] == "2026-03-07"
    assert out["tickers"] == ["AAPL", "MSFT"]

    stored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert stored["as_of_date"] == "2026-03-07"
    assert stored["payload"]["pct_above_200dma"] == 61.0


def test_get_data_returns_stale_cache_when_refresh_fails(monkeypatch, tmp_path):
    cache_path = tmp_path / "market_breadth_sp500_1y.json"
    cached = {"pct_above_200dma": 49.0, "as_of_date": "2026-03-06", "tickers": ["AAPL"]}
    _write_cache(cache_path, cached, datetime.now() - timedelta(hours=30), "2026-03-06")

    monkeypatch.setattr(mb, "_breadth_cache_path", lambda *_: cache_path)
    monkeypatch.setattr(mb, "_latest_market_close_date", lambda: "2026-03-07")
    monkeypatch.setattr(mb, "get_tickers", lambda *_: ["AAPL"])
    monkeypatch.setattr(
        mb,
        "calculate_breadth_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )

    out = mb.get_data(universe="sp500", period="1y")
    assert out == cached


def test_get_data_raises_when_no_cache_and_refresh_fails(monkeypatch, tmp_path):
    cache_path = tmp_path / "market_breadth_sp500_1y.json"
    monkeypatch.setattr(mb, "_breadth_cache_path", lambda *_: cache_path)
    monkeypatch.setattr(mb, "get_tickers", lambda *_: ["AAPL"])
    monkeypatch.setattr(
        mb,
        "calculate_breadth_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("download failed")),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        mb.get_data(universe="sp500", period="1y")
