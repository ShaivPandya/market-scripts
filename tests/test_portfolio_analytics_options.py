import pandas as pd

from portfolio.portfolio_analytics import compute_analytics


def _opt(underlying, option_type, strike, *, qty, cost, direction, contract):
    return {
        "ticker": underlying,
        "asset": "equity",
        "direction": direction,
        "cost_basis": cost,
        "quantity": float(qty),
        "shares": float(qty),
        "instrument_type": "option",
        "underlying_ticker": underlying,
        "option_contract_symbol": contract,
        "option_expiration": "2026-01-16",
        "option_strike": strike,
        "option_type": option_type,
        "price_symbol": contract,
        "contract_multiplier": 100.0,
        "position_id": contract,
    }


def _patch_option_prices(monkeypatch, price_by_contract):
    monkeypatch.setattr(
        "portfolio.portfolio_analytics._option_current_price",
        lambda pos: price_by_contract.get(pos.get("option_contract_symbol")),
    )


def test_long_call_spread_nets_long_with_abs_net_weight(monkeypatch):
    long_call = _opt("META", "call", 500.0, qty=1, cost=10.0, direction="long", contract="META260116C00500000")
    short_call = _opt("META", "call", 520.0, qty=1, cost=4.0, direction="short", contract="META260116C00520000")
    _patch_option_prices(monkeypatch, {"META260116C00500000": 12.0, "META260116C00520000": 5.0})

    legs = compute_analytics({}, [long_call, short_call])["per_position"]
    long_leg = legs["META260116C00500000"]
    short_leg = legs["META260116C00520000"]

    # Gross per-leg detail preserved.
    assert long_leg["current_notional"] == 1200.0
    assert short_leg["current_notional"] == 500.0
    # Signed and netted.
    assert long_leg["signed_current_notional"] == 1200.0
    assert short_leg["signed_current_notional"] == -500.0
    assert long_leg["net_current_notional"] == 700.0
    assert long_leg["net_direction"] == "long"
    assert long_leg["near_zero_net"] is False
    # Option leg weights sum to the underlying's abs-net share of the book.
    assert round(long_leg["weight"] + short_leg["weight"], 4) == 1.0


def test_short_call_spread_nets_short(monkeypatch):
    short_call = _opt("META", "call", 500.0, qty=1, cost=10.0, direction="short", contract="META260116C00500000")
    long_call = _opt("META", "call", 520.0, qty=1, cost=4.0, direction="long", contract="META260116C00520000")
    _patch_option_prices(monkeypatch, {"META260116C00500000": 12.0, "META260116C00520000": 5.0})

    legs = compute_analytics({}, [short_call, long_call])["per_position"]
    leg = legs["META260116C00500000"]

    assert leg["signed_current_notional"] == -1200.0
    assert leg["net_current_notional"] == -700.0
    assert leg["net_direction"] == "short"
    assert leg["near_zero_net"] is False


def test_naked_short_put_counts_long(monkeypatch):
    short_put = _opt("META", "put", 480.0, qty=2, cost=6.0, direction="short", contract="META260116P00480000")
    _patch_option_prices(monkeypatch, {"META260116P00480000": 7.0})

    leg = compute_analytics({}, [short_put])["per_position"]["META260116P00480000"]

    assert leg["current_notional"] == 1400.0
    assert leg["signed_current_notional"] == 1400.0
    assert leg["net_direction"] == "long"


def test_long_put_counts_short(monkeypatch):
    long_put = _opt("META", "put", 480.0, qty=2, cost=6.0, direction="long", contract="META260116P00480000")
    _patch_option_prices(monkeypatch, {"META260116P00480000": 7.0})

    leg = compute_analytics({}, [long_put])["per_position"]["META260116P00480000"]

    assert leg["signed_current_notional"] == -1400.0
    assert leg["net_direction"] == "short"


def test_offsetting_straddle_flags_near_zero_and_does_not_dilute_equity(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    equity = {
        "ticker": "AAPL",
        "asset": "equity",
        "direction": "long",
        "cost_basis": 100.0,
        "quantity": 10.0,
        "shares": 10.0,
        "instrument_type": "security",
        "position_id": "AAPL",
    }
    long_call = _opt("META", "call", 500.0, qty=1, cost=10.0, direction="long", contract="META260116C00500000")
    long_put = _opt("META", "put", 500.0, qty=1, cost=10.0, direction="long", contract="META260116P00500000")
    _patch_option_prices(monkeypatch, {"META260116C00500000": 10.0, "META260116P00500000": 10.0})

    analytics = compute_analytics({"AAPL": pd.Series([150.0, 160.0], index=dates)}, [equity, long_call, long_put])
    legs = analytics["per_position"]

    call_leg = legs["META260116C00500000"]
    assert call_leg["near_zero_net"] is True
    assert call_leg["net_direction"] == "neutral"
    # Net option size is ~0, so the straddle does not enter the weight denominator
    # and the equity carries the full book weight.
    assert legs["AAPL"]["weight"] == 1.0
    assert call_leg["weight"] == 0.0
    assert legs["META260116P00500000"]["weight"] == 0.0


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
