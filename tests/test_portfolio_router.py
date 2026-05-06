from datetime import datetime

import pandas as pd
from fastapi import Response


def test_portfolio_all_timeframes_route_returns_supported_timeframes(monkeypatch):
    from api import cache
    from api.cache import short_cache
    from api.routers import portfolio as portfolio_router
    from portfolio import portfolio_dashboard

    short_cache.clear()
    monkeypatch.setattr(cache, "_DISK_CACHE_ENABLED", False)

    dates = pd.date_range("2026-05-01", periods=2, freq="D")

    def fake_get_data(timeframe: str = "Daily", all_timeframes: bool = False):
        assert all_timeframes is True
        return {
            "timeframes": {
                tf: {
                    "positions": {"MU": pd.Series([100.0, 101.0], index=dates)},
                    "metadata": {"MU": {"asset": "equity", "direction": "long"}},
                    "timeframe": tf,
                    "timestamp": datetime(2026, 5, 6, 12, 0, 0),
                    "position_order": ["MU"],
                }
                for tf in portfolio_router.VALID_TIMEFRAMES
            },
            "timestamp": datetime(2026, 5, 6, 12, 0, 0),
            "analytics": {"portfolio": {"position_count": 1}},
        }

    monkeypatch.setattr(portfolio_dashboard, "get_data", fake_get_data)
    monkeypatch.setattr(
        portfolio_router.OntologyRuntimeReadService,
        "positions",
        lambda self, include_hedges=True: [{"ticker": "MU", "asset": "equity"}],
    )

    payload = portfolio_router.get_portfolio(Response(), all_timeframes=True)

    assert set(payload["timeframes"]) == portfolio_router.VALID_TIMEFRAMES
    assert payload["timeframes"]["Daily"]["positions"]["MU"] == [
        {"date": "2026-05-01T00:00:00", "value": 100.0},
        {"date": "2026-05-02T00:00:00", "value": 101.0},
    ]
    assert payload["timeframes"]["Daily"]["position_order"] == ["MU"]
    assert payload["holdings"] == [{"ticker": "MU", "asset": "equity"}]


def test_portfolio_cache_key_changes_when_holdings_change(monkeypatch):
    from api import cache
    from api.cache import short_cache
    from api.routers import portfolio as portfolio_router
    from portfolio import portfolio_dashboard

    short_cache.clear()
    monkeypatch.setattr(cache, "_DISK_CACHE_ENABLED", False)

    current_holdings = [{"ticker": "NUAI", "asset": "equity", "role": "position"}]
    calls: list[str] = []
    dates = pd.date_range("2026-05-01", periods=2, freq="D")

    def fake_get_data(timeframe: str = "Daily", all_timeframes: bool = False):
        del timeframe
        assert all_timeframes is True
        ticker = current_holdings[0]["ticker"]
        calls.append(ticker)
        return {
            "timeframes": {
                tf: {
                    "positions": {ticker: pd.Series([100.0, 101.0], index=dates)},
                    "metadata": {ticker: {"asset": "equity", "direction": "long"}},
                    "timeframe": tf,
                    "timestamp": datetime(2026, 5, 6, 12, 0, 0),
                    "position_order": [ticker],
                }
                for tf in portfolio_router.VALID_TIMEFRAMES
            },
            "timestamp": datetime(2026, 5, 6, 12, 0, 0),
            "analytics": {"portfolio": {"position_count": 1}},
        }

    monkeypatch.setattr(portfolio_dashboard, "get_data", fake_get_data)
    monkeypatch.setattr(
        portfolio_router.OntologyRuntimeReadService,
        "positions",
        lambda self, include_hedges=True: list(current_holdings),
    )

    first = portfolio_router.get_portfolio(Response(), all_timeframes=True)
    current_holdings[:] = [{"ticker": "NVDA", "asset": "equity", "role": "position"}]
    second = portfolio_router.get_portfolio(Response(), all_timeframes=True)

    assert calls == ["NUAI", "NVDA"]
    assert first["timeframes"]["Daily"]["position_order"] == ["NUAI"]
    assert second["timeframes"]["Daily"]["position_order"] == ["NVDA"]
    assert second["holdings"] == [{"ticker": "NVDA", "asset": "equity", "role": "position"}]
