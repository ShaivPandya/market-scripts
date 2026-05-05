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
