from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from portfolio.portfolio_optimizer import anchor_signal_cache as anchor_cache
from portfolio.portfolio_optimizer import composite_signal


def _raw_price_frame(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "avg20_roc63": [float(i + 1) for i in range(len(tickers))],
            "rel_roc42": [float(i + 2) for i in range(len(tickers))],
            "avg10_rel_roc": [float(i + 3) for i in range(len(tickers))],
        },
        index=tickers,
    )


def _eps_frame(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eps_yoy_change": [float(i + 1) for i in range(len(tickers))],
            "eps_cagr": [float(i + 2) for i in range(len(tickers))],
            "eps_growth_acceleration": [float(i + 3) for i in range(len(tickers))],
        },
        index=tickers,
    )


def _revenue_frame(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "revenue_yoy_change": [float(i + 1) for i in range(len(tickers))],
            "revenue_cagr": [float(i + 2) for i in range(len(tickers))],
            "revenue_growth_acceleration": [float(i + 3) for i in range(len(tickers))],
        },
        index=tickers,
    )


def _quality_frame(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gpoa": [float(i + 1) for i in range(len(tickers))],
            "roe": [float(i + 2) for i in range(len(tickers))],
            "roa": [float(i + 3) for i in range(len(tickers))],
        },
        index=tickers,
    )


def test_anchor_price_cache_uses_one_trading_day_stale_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_ANALYZER_ANCHOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(anchor_cache, "latest_market_close_date", lambda _benchmark: "2026-05-06")

    def price_loader(_tickers, years):
        return pd.DataFrame({"AAA": [1.0], "SPY": [1.0]})

    def price_momentum_fetcher(tickers, _benchmark_map, _prices):
        return _raw_price_frame(list(tickers))

    first, first_meta = anchor_cache.get_anchor_price_raw(
        anchor_universe=["AAA"],
        benchmark="SPY",
        years=5,
        price_loader=price_loader,
        price_momentum_fetcher=price_momentum_fetcher,
    )
    assert first_meta["cache_status"] == "refresh"
    assert first.index.tolist() == ["AAA"]

    monkeypatch.setattr(anchor_cache, "latest_market_close_date", lambda _benchmark: "2026-05-07")

    def fail_price_loader(_tickers, years):
        raise RuntimeError("yahoo unavailable")

    stale, stale_meta = anchor_cache.get_anchor_price_raw(
        anchor_universe=["AAA"],
        benchmark="SPY",
        years=5,
        price_loader=fail_price_loader,
        price_momentum_fetcher=price_momentum_fetcher,
    )
    assert stale_meta["cache_status"] == "stale_fallback"
    assert stale_meta["stale"] is True
    assert stale.index.tolist() == ["AAA"]

    monkeypatch.setattr(anchor_cache, "latest_market_close_date", lambda _benchmark: "2026-05-08")
    with pytest.raises(RuntimeError, match="yahoo unavailable"):
        anchor_cache.get_anchor_price_raw(
            anchor_universe=["AAA"],
            benchmark="SPY",
            years=5,
            price_loader=fail_price_loader,
            price_momentum_fetcher=price_momentum_fetcher,
        )


def test_spdr_anchor_universe_uses_weekly_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_ANALYZER_ANCHOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(anchor_cache, "_today", lambda: date(2026, 5, 6))
    calls = {"holdings": 0}

    def holdings_fetcher(etfs, top_n):
        calls["holdings"] += 1
        return {etfs[0]: pd.Series({"AAA": 0.6, "BBB": 0.4})}

    first, first_meta = anchor_cache.get_spdr_anchor_universe(
        top_n=2,
        min_unique=1,
        sector_etfs=["XLA"],
        holdings_fetcher=holdings_fetcher,
    )
    second, second_meta = anchor_cache.get_spdr_anchor_universe(
        top_n=2,
        min_unique=1,
        sector_etfs=["XLA"],
        holdings_fetcher=holdings_fetcher,
    )

    assert first == second == ["AAA", "BBB"]
    assert calls["holdings"] == 1
    assert first_meta["cache_status"] == "refresh"
    assert second_meta["cache_status"] == "hit"


def test_anchor_signal_generation_reuses_cached_anchor_fundamentals(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_ANALYZER_ANCHOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(anchor_cache, "_today", lambda: date(2026, 5, 6))
    monkeypatch.setattr(anchor_cache, "latest_market_close_date", lambda _benchmark: "2026-05-06")
    monkeypatch.setattr(composite_signal, "SPDR_SECTOR_ETFS", ["XLA"])

    calls = {"holdings": 0, "prices": 0, "quality": 0, "eps": 0, "revenue": 0}

    def holdings_fetcher(etfs, top_n):
        calls["holdings"] += 1
        return {etfs[0]: pd.Series({"AAA": 0.6, "BBB": 0.4})}

    def price_loader(tickers, years):
        calls["prices"] += 1
        return pd.DataFrame({ticker: [1.0, 2.0] for ticker in tickers})

    def price_momentum_fetcher(tickers, _benchmark_map, _prices):
        return _raw_price_frame(list(tickers))

    def quality_fetcher(tickers, **_kwargs):
        calls["quality"] += 1
        return _quality_frame(list(tickers))

    def eps_fetcher(tickers, **_kwargs):
        calls["eps"] += 1
        return _eps_frame(list(tickers))

    def revenue_fetcher(tickers, **_kwargs):
        calls["revenue"] += 1
        return _revenue_frame(list(tickers))

    monkeypatch.setattr(composite_signal, "fetch_etf_top_holdings_batch", holdings_fetcher)
    monkeypatch.setattr(composite_signal, "fetch_prices", price_loader)
    monkeypatch.setattr(composite_signal, "fetch_price_momentum_batch", price_momentum_fetcher)
    monkeypatch.setattr(composite_signal, "fetch_quality_batch", quality_fetcher)
    monkeypatch.setattr(composite_signal, "fetch_eps_momentum_batch", eps_fetcher)
    monkeypatch.setattr(composite_signal, "fetch_revenue_momentum_batch", revenue_fetcher)

    first, first_meta = composite_signal.generate_anchor_normalized_long_equity_signals(
        ["AAA"],
        anchor_top_n=2,
        anchor_min_unique=1,
    )
    second, second_meta = composite_signal.generate_anchor_normalized_long_equity_signals(
        ["AAA"],
        anchor_top_n=2,
        anchor_min_unique=1,
    )

    assert "AAA" in first.index
    assert "AAA" in second.index
    assert first_meta["signal_anchor_cache_status"] == "refresh"
    assert second_meta["signal_anchor_cache_status"] == "hit"
    assert calls == {"holdings": 1, "prices": 1, "quality": 1, "eps": 1, "revenue": 1}


def test_extra_portfolio_tickers_are_not_written_to_anchor_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_ANALYZER_ANCHOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(anchor_cache, "_today", lambda: date(2026, 5, 6))
    monkeypatch.setattr(anchor_cache, "latest_market_close_date", lambda _benchmark: "2026-05-06")
    monkeypatch.setattr(composite_signal, "SPDR_SECTOR_ETFS", ["XLA"])

    def holdings_fetcher(etfs, top_n):
        return {etfs[0]: pd.Series({"AAA": 0.6, "BBB": 0.4})}

    def price_loader(tickers, years):
        return pd.DataFrame({ticker: [1.0, 2.0] for ticker in tickers})

    def price_momentum_fetcher(tickers, _benchmark_map, _prices):
        return _raw_price_frame(list(tickers))

    monkeypatch.setattr(composite_signal, "fetch_etf_top_holdings_batch", holdings_fetcher)
    monkeypatch.setattr(composite_signal, "fetch_prices", price_loader)
    monkeypatch.setattr(composite_signal, "fetch_price_momentum_batch", price_momentum_fetcher)
    monkeypatch.setattr(
        composite_signal, "fetch_quality_batch", lambda tickers, **_kwargs: _quality_frame(list(tickers))
    )
    monkeypatch.setattr(
        composite_signal, "fetch_eps_momentum_batch", lambda tickers, **_kwargs: _eps_frame(list(tickers))
    )
    monkeypatch.setattr(
        composite_signal,
        "fetch_revenue_momentum_batch",
        lambda tickers, **_kwargs: _revenue_frame(list(tickers)),
    )

    output, meta = composite_signal.generate_anchor_normalized_long_equity_signals(
        ["AAA", "EXTRA"],
        anchor_top_n=2,
        anchor_min_unique=1,
    )

    assert output.index.tolist() == ["AAA", "EXTRA"]
    assert meta["signal_anchor_cache_status"] == "refresh"
    cached_text = "\n".join(path.read_text(encoding="utf-8") for path in Path(tmp_path).rglob("*.json"))
    assert "AAA" in cached_text
    assert "BBB" in cached_text
    assert "EXTRA" not in cached_text
