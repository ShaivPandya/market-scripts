import pandas as pd

from portfolio.portfolio_analytics import compute_analytics


def test_option_notional_uses_contract_multiplier(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    positions = {"META": pd.Series([500.0, 520.0], index=dates)}
    holdings = [
        {
            "ticker": "META",
            "asset": "equity",
            "direction": "long",
            "cost_basis": 10.0,
            "quantity": 3.0,
            "shares": 3.0,
            "instrument_type": "option",
            "underlying_ticker": "META",
            "option_contract_symbol": "META260116C00500000",
            "option_expiration": "2026-01-16",
            "option_strike": 500.0,
            "option_type": "call",
            "price_symbol": "META260116C00500000",
            "contract_multiplier": 100.0,
            "position_id": "META260116C00500000",
        }
    ]

    monkeypatch.setattr(
        "portfolio.portfolio_analytics._option_current_price",
        lambda pos: 12.0,
    )

    analytics = compute_analytics(positions, holdings)
    leg = analytics["per_position"]["META260116C00500000"]

    assert leg["current_price"] == 12.0
    assert leg["current_notional"] == 3600.0
    assert leg["unrealized_pnl_dollar"] == 600.0
