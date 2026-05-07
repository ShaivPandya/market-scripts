import pandas as pd

from portfolio.portfolio_analytics import compute_analytics


def test_current_notional_and_weights_use_latest_price_with_fx_and_multiplier():
    dates = pd.to_datetime(["2026-04-27", "2026-05-05"])
    prices = {
        "AAA": pd.Series([100.0, 200.0], index=dates),
        "BBB": pd.Series([100.0, 100.0], index=dates),
    }
    positions = [
        {
            "ticker": "AAA",
            "direction": "long",
            "cost_basis": 100.0,
            "quantity": 10.0,
            "contract_multiplier": 2.0,
            "fx_rate_to_base": 0.5,
            "notional_base": 1000.0,
        },
        {
            "ticker": "BBB",
            "direction": "long",
            "cost_basis": 100.0,
            "quantity": 10.0,
            "contract_multiplier": 2.0,
            "fx_rate_to_base": 0.5,
            "notional_base": 1000.0,
        },
    ]

    analytics = compute_analytics(prices, positions)

    aaa = analytics["per_position"]["AAA"]
    bbb = analytics["per_position"]["BBB"]
    assert aaa["current_notional"] == 2000.0
    assert bbb["current_notional"] == 1000.0
    assert aaa["cost_notional"] == 1000.0
    assert aaa["notional_base"] == 1000.0
    assert aaa["weight"] == 0.6667
    assert bbb["weight"] == 0.3333


def test_spot_fx_analytics_use_base_units_and_pair_direction():
    dates = pd.to_datetime(["2026-04-27", "2026-05-05"])
    prices = {
        "EURUSD=X": pd.Series([1.08, 1.10], index=dates),
        "USDJPY=X": pd.Series([155.0, 160.0], index=dates),
    }
    positions = [
        {
            "ticker": "EURUSD=X",
            "direction": "long",
            "instrument_type": "spot_fx",
            "cost_basis": 1.08,
            "quantity": 100_000.0,
            "fx_base_currency": "EUR",
            "fx_quote_currency": "USD",
            "currency": "USD",
            "base_currency": "USD",
            "fx_rate_to_base": 1.0,
            "notional_base": 108_000.0,
        },
        {
            "ticker": "USDJPY=X",
            "direction": "short",
            "instrument_type": "spot_fx",
            "cost_basis": 155.0,
            "quantity": 100_000.0,
            "fx_base_currency": "USD",
            "fx_quote_currency": "JPY",
            "currency": "JPY",
            "base_currency": "USD",
            "fx_rate_to_base": 1 / 155,
            "notional_base": 100_000.0,
        },
    ]

    analytics = compute_analytics(prices, positions)

    eurusd = analytics["per_position"]["EURUSD=X"]
    usdjpy = analytics["per_position"]["USDJPY=X"]
    assert eurusd["current_notional"] == 110_000.0
    assert eurusd["unrealized_pnl_dollar"] == 2000.0
    assert usdjpy["current_notional"] == 100_000.0
    assert usdjpy["unrealized_pnl_pct"] == -3.23
    assert usdjpy["unrealized_pnl_dollar"] == -3125.0
