from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.instruments import infer_underlying_direction
from portfolio.portfolio_optimizer import portfolio_analyzer, portfolio_sizer


def test_equity_beta_inputs_ignore_non_equity_returns(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_compute_betas(rets: pd.DataFrame, benchmark: str):
        captured["columns"] = list(rets.columns)
        captured.setdefault("benchmarks", []).append(benchmark)
        return pd.Series(
            {
                "SPY": {"EQ": 1.25, "SPY": 1.0, "IWM": 0.65, "QQQ": 0.95},
                "IWM": {"EQ": 0.85, "SPY": 1.15, "IWM": 1.0, "QQQ": 0.75},
                "QQQ": {"EQ": 1.55, "SPY": 1.05, "IWM": 0.80, "QQQ": 1.0},
            }[benchmark]
        )

    monkeypatch.setattr(portfolio_sizer, "compute_betas", fake_compute_betas)

    rets = pd.DataFrame(
        {
            "EQ": [0.01, 0.02],
            "GLD": [0.03, 0.04],
            "EURUSD=X": [0.001, 0.002],
            "SPY": [0.01, 0.015],
            "IWM": [0.012, 0.013],
            "QQQ": [0.011, 0.017],
        }
    )

    beta_by_benchmark, beta_display_by_benchmark, betas_all_by_benchmark, equity_tickers = (
        portfolio_sizer._compute_equity_beta_inputs(
            rets=rets,
            tickers=["EQ", "GLD", "EURUSD=X"],
            market_tickers=["SPY", "IWM", "QQQ"],
            eq_mask=np.array([True, False, False]),
        )
    )

    assert captured == {"columns": ["EQ", "SPY", "IWM", "QQQ"], "benchmarks": ["SPY", "IWM", "QQQ"]}
    assert equity_tickers == ["EQ"]
    assert beta_by_benchmark["SPY"].to_dict() == {"EQ": 1.25, "GLD": 0.0, "EURUSD=X": 0.0}
    assert beta_by_benchmark["IWM"].to_dict() == {"EQ": 0.85, "GLD": 0.0, "EURUSD=X": 0.0}
    assert beta_by_benchmark["QQQ"].to_dict() == {"EQ": 1.55, "GLD": 0.0, "EURUSD=X": 0.0}
    assert beta_display_by_benchmark["SPY"].loc["EQ"] == 1.25
    assert pd.isna(beta_display_by_benchmark["SPY"].loc["GLD"])
    assert pd.isna(beta_display_by_benchmark["IWM"].loc["EURUSD=X"])
    assert betas_all_by_benchmark["QQQ"].loc["QQQ"] == 1.0


def test_sizer_universe_filters_non_equity_positions():
    meta = pd.DataFrame(
        {
            "asset": ["equity", "commodity", "fx", "bond", "equity"],
            "instrument_type": ["security", "future", "spot_fx", "future", "future"],
        },
        index=["NVDA", "BZ=F", "EURUSD=X", "ZN=F", "ES=F"],
    )

    included, excluded = portfolio_sizer._filter_equity_sizing_universe(
        meta,
        ["NVDA", "BZ=F", "EURUSD=X", "ZN=F", "ES=F"],
    )

    assert included == ["NVDA", "ES=F"]
    assert excluded == ["BZ=F", "EURUSD=X", "ZN=F"]


def test_spot_fx_prices_are_not_double_converted(monkeypatch):
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        portfolio_analyzer,
        "fetch_currencies",
        lambda tickers: {ticker: ("JPY" if ticker == "USDJPY=X" else "USD") for ticker in tickers},
    )

    def fake_download_prices(tickers: list[str], fx_tickers: list[str]):
        captured["tickers"] = tickers
        captured["fx_tickers"] = fx_tickers
        return pd.DataFrame({"USDJPY=X": [155.0, 160.0], "SPY": [500.0, 501.0]})

    monkeypatch.setattr(portfolio_analyzer, "download_prices", fake_download_prices)

    meta = pd.DataFrame(
        {
            "instrument_type": ["spot_fx"],
            "price_symbol": ["USDJPY=X"],
            "asset": ["fx"],
        },
        index=["USDJPY=X"],
    )

    _, ticker_currencies, _ = portfolio_analyzer.fetch_prices_for_portfolio_symbols(meta, ["USDJPY=X"], ["SPY"])

    assert ticker_currencies["USDJPY=X"] == "USD"
    assert captured["fx_tickers"] == []


def test_prepare_instrument_metadata_allows_duplicate_ticker_index():
    meta = pd.DataFrame(
        {
            "price_symbol": ["AAPL", "AAPL260116P00150000"],
            "instrument_type": ["security", "option"],
            "asset": ["equity", "equity"],
        },
        index=["AAPL", "AAPL"],
    )

    out = portfolio_analyzer.prepare_instrument_metadata(meta)

    assert out["price_symbol"].tolist() == ["AAPL", "AAPL260116P00150000"]
    assert out["instrument_type"].tolist() == ["security", "option"]


def test_leg_price_map_allows_duplicate_raw_ticker_legs():
    meta = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "asset": "equity",
                "instrument_type": "security",
                "price_symbol": "AAPL",
            },
            {
                "ticker": "AAPL",
                "asset": "equity",
                "instrument_type": "option",
                "price_symbol": "AAPL260116P00150000",
            },
        ]
    )
    prices = pd.DataFrame({"AAPL": [100.0], "AAPL260116P00150000": [5.0]})

    price_map = portfolio_sizer._leg_price_map(meta, prices)

    assert price_map == {"AAPL": 100.0, "AAPL260116P00150000": 5.0}


def test_spot_fx_unit_notional_uses_base_units():
    rows = pd.DataFrame(
        {
            "instrument_type": ["spot_fx", "spot_fx", "future"],
            "price": [1.10, 160.0, 5000.0],
            "contract_multiplier": [1.0, 1.0, 50.0],
            "fx_base_currency": ["EUR", "USD", ""],
            "fx_quote_currency": ["USD", "JPY", ""],
        }
    )

    unit = portfolio_sizer.unit_notional_in_base(rows)

    assert unit.tolist() == [1.10, 1.0, 250_000.0]


def test_spy_only_beta_hedge_uses_no_iwm_leg():
    weights = pd.Series({"AAA": 0.40, "BBB": -0.10})
    beta_by_benchmark = {
        "SPY": pd.Series({"AAA": 1.0, "BBB": 1.0}),
        "IWM": pd.Series({"AAA": 1.5, "BBB": 1.0}),
        "QQQ": pd.Series({"AAA": 1.2, "BBB": 0.9}),
    }
    betas_all_by_benchmark = {
        "SPY": pd.Series({"SPY": 1.0, "IWM": 0.7, "QQQ": 0.9}),
        "IWM": pd.Series({"SPY": 0.8, "IWM": 1.0, "QQQ": 0.75}),
        "QQQ": pd.Series({"SPY": 1.1, "IWM": 0.8, "QQQ": 1.0}),
    }

    hedged_weights, summary = portfolio_sizer._apply_beta_hedges_with_gross_cap(
        weights,
        beta_by_benchmark,
        betas_all_by_benchmark,
        long_mask=np.array([True, False]),
        short_mask=np.array([False, True]),
        eq_mask=np.array([True, True]),
        selected_hedges=["SPY"],
    )

    assert hedged_weights.equals(weights)
    assert summary["selected_hedges"] == ["SPY"]
    assert set(summary["hedge_weights"]) == {"SPY"}
    assert abs(summary["post_hedge_beta_spy"]) < 1e-6
    assert summary["hedge_gross"] == abs(summary["hedge_spy_weight"])
    assert np.isclose(summary["gross_with_hedges"], float(np.abs(weights).sum() + abs(summary["hedge_spy_weight"])))


def test_spy_qqq_beta_hedge_uses_selected_legs_only():
    weights = pd.Series({"AAA": 0.40, "BBB": -0.10})
    beta_by_benchmark = {
        "SPY": pd.Series({"AAA": 1.0, "BBB": 1.0}),
        "IWM": pd.Series({"AAA": 1.5, "BBB": 1.0}),
        "QQQ": pd.Series({"AAA": 1.4, "BBB": 0.8}),
    }
    betas_all_by_benchmark = {
        "SPY": pd.Series({"SPY": 1.0, "IWM": 0.7, "QQQ": 0.85}),
        "IWM": pd.Series({"SPY": 0.8, "IWM": 1.0, "QQQ": 0.7}),
        "QQQ": pd.Series({"SPY": 1.05, "IWM": 0.75, "QQQ": 1.0}),
    }

    _hedged_weights, summary = portfolio_sizer._apply_beta_hedges_with_gross_cap(
        weights,
        beta_by_benchmark,
        betas_all_by_benchmark,
        long_mask=np.array([True, False]),
        short_mask=np.array([False, True]),
        eq_mask=np.array([True, True]),
        selected_hedges=["SPY", "QQQ"],
    )

    assert summary["selected_hedges"] == ["SPY", "QQQ"]
    assert set(summary["hedge_weights"]) == {"SPY", "QQQ"}
    assert abs(summary["post_hedge_beta_spy"]) < 1e-4
    assert abs(summary["post_hedge_beta_qqq"]) < 1e-4


def test_all_three_beta_hedge_uses_all_selected_legs():
    weights = pd.Series({"AAA": 0.40, "BBB": -0.10})
    beta_by_benchmark = {
        "SPY": pd.Series({"AAA": 1.0, "BBB": 1.0}),
        "IWM": pd.Series({"AAA": 1.5, "BBB": 1.0}),
        "QQQ": pd.Series({"AAA": 1.4, "BBB": 0.8}),
    }
    betas_all_by_benchmark = {
        "SPY": pd.Series({"SPY": 1.0, "IWM": 0.0, "QQQ": 0.0}),
        "IWM": pd.Series({"SPY": 0.0, "IWM": 1.0, "QQQ": 0.0}),
        "QQQ": pd.Series({"SPY": 0.0, "IWM": 0.0, "QQQ": 1.0}),
    }

    _hedged_weights, summary = portfolio_sizer._apply_beta_hedges_with_gross_cap(
        weights,
        beta_by_benchmark,
        betas_all_by_benchmark,
        long_mask=np.array([True, False]),
        short_mask=np.array([False, True]),
        eq_mask=np.array([True, True]),
        selected_hedges=["SPY", "IWM", "QQQ"],
    )

    assert summary["selected_hedges"] == ["SPY", "IWM", "QQQ"]
    assert set(summary["hedge_weights"]) == {"SPY", "IWM", "QQQ"}
    assert summary["hedge_spy_weight"] != 0.0
    assert summary["hedge_iwm_weight"] != 0.0
    assert summary["hedge_qqq_weight"] != 0.0
    assert abs(summary["post_hedge_beta_spy"]) < 1e-5
    assert abs(summary["post_hedge_beta_iwm"]) < 1e-5
    assert abs(summary["post_hedge_beta_qqq"]) < 1e-5


def test_custom_beta_hedge_uses_dynamic_keys():
    weights = pd.Series({"AAA": 0.30, "BBB": -0.05})
    beta_by_benchmark = {
        "SMH": pd.Series({"AAA": 1.6, "BBB": 0.8}),
    }
    betas_all_by_benchmark = {
        "SMH": pd.Series({"SMH": 1.0}),
    }

    _hedged_weights, summary = portfolio_sizer._apply_beta_hedges_with_gross_cap(
        weights,
        beta_by_benchmark,
        betas_all_by_benchmark,
        long_mask=np.array([True, False]),
        short_mask=np.array([False, True]),
        eq_mask=np.array([True, True]),
        selected_hedges=["SMH"],
    )

    assert summary["selected_hedges"] == ["SMH"]
    assert set(summary["hedge_weights"]) == {"SMH"}
    assert summary["hedge_smh_weight"] != 0.0
    assert abs(summary["post_hedge_beta_smh"]) < 1e-6


def test_grouped_convictions_size_group_then_split_members():
    meta = pd.DataFrame(
        {"direction": ["long", "long", "long"]},
        index=["SKH", "MU", "SSNLF"],
    )
    positions = portfolio_sizer._parse_positions(
        [
            {"ticker": "SKH", "conviction": 4, "group_name": "Memory", "group_conviction": 5},
            {"ticker": "MU", "conviction": 3, "group_name": " memory ", "group_conviction": 5},
            {"ticker": "SSNLF", "conviction": 3, "group_name": "Memory", "group_conviction": 5},
        ]
    )

    weights = portfolio_sizer._build_conviction_weights(meta, positions)

    group_target = portfolio_sizer.LONG_MAX
    assert weights["SKH"] == pytest.approx(group_target * 4 / 10)
    assert weights["MU"] == pytest.approx(group_target * 3 / 10)
    assert weights["SSNLF"] == pytest.approx(group_target * 3 / 10)
    assert weights.sum() == pytest.approx(group_target)


def test_placeholder_group_name_does_not_group_position():
    meta = pd.DataFrame(
        {"direction": ["long", "long"]},
        index=["META", "MU"],
    )
    positions = portfolio_sizer._parse_positions(
        [
            {"ticker": "META", "conviction": 3, "group_name": "N/A", "group_conviction": 5},
            {"ticker": "MU", "conviction": 3, "group_name": "Memory", "group_conviction": 5},
        ]
    )

    weights = portfolio_sizer._build_conviction_weights(meta, positions)

    assert positions["META"]["group_key"] is None
    assert weights["META"] == pytest.approx(portfolio_sizer.LONG_MAX * 3 / portfolio_sizer.CONVICTION_MAX)
    assert weights["MU"] == pytest.approx(portfolio_sizer.LONG_MAX)


def test_grouped_convictions_reject_mixed_direction():
    meta = pd.DataFrame(
        {"direction": ["long", "short"]},
        index=["MU", "SSNLF"],
    )
    positions = portfolio_sizer._parse_positions(
        [
            {"ticker": "MU", "conviction": 4, "group_name": "Memory", "group_conviction": 5},
            {"ticker": "SSNLF", "conviction": 3, "group_name": "memory", "group_conviction": 5},
        ]
    )

    with pytest.raises(ValueError, match="cannot mix long and short"):
        portfolio_sizer._build_conviction_weights(meta, positions)


def _opt_leg(option_type, direction, *, qty, cost):
    return {
        "instrument_type": "option",
        "option_type": option_type,
        "direction": direction,
        "quantity": qty,
        "cost_basis": cost,
        "contract_multiplier": 100,
    }


def test_infer_direction_short_put_is_long():
    assert infer_underlying_direction([_opt_leg("put", "short", qty=2, cost=6.0)]) == ("long", False)


def test_infer_direction_long_put_is_short():
    assert infer_underlying_direction([_opt_leg("put", "long", qty=2, cost=6.0)]) == ("short", False)


def test_infer_direction_call_spread_is_long():
    legs = [_opt_leg("call", "long", qty=1, cost=12.0), _opt_leg("call", "short", qty=1, cost=5.0)]
    assert infer_underlying_direction(legs) == ("long", False)


def test_infer_direction_balanced_straddle_is_neutral():
    legs = [_opt_leg("call", "long", qty=1, cost=10.0), _opt_leg("put", "long", qty=1, cost=10.0)]
    assert infer_underlying_direction(legs) == ("neutral", True)


def test_infer_direction_share_leg_wins():
    legs = [{"instrument_type": "security", "direction": "long"}, _opt_leg("put", "long", qty=1, cost=10.0)]
    assert infer_underlying_direction(legs) == ("long", False)


def test_collapse_options_only_synthesizes_equity_row():
    df = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "asset": "equity",
                "direction": "short",
                "instrument_type": "option",
                "option_type": "put",
                "underlying_ticker": "NVDA",
                "price_symbol": "NVDA260116P00100000",
                "quantity": 2,
                "shares": 2,
                "cost_basis": 6.0,
                "contract_multiplier": 100,
            }
        ]
    )

    collapsed, skipped = portfolio_sizer._collapse_positions_to_underlyings(df)

    assert skipped == []
    assert list(collapsed["ticker"]) == ["NVDA"]
    row = collapsed.iloc[0]
    assert row["instrument_type"] == "security"
    assert row["asset"] == "equity"
    assert row["price_symbol"] == "NVDA"
    assert float(row["contract_multiplier"]) == 1.0
    assert row["direction"] == "long"  # short put -> long equity exposure


def test_collapse_equity_plus_option_dedupes_to_equity_row():
    df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "asset": "equity",
                "direction": "long",
                "instrument_type": "security",
                "price_symbol": "AAPL",
                "quantity": 10,
                "shares": 10,
                "cost_basis": 100.0,
                "contract_multiplier": 1,
                "underlying_ticker": None,
                "option_type": None,
            },
            {
                "ticker": "AAPL",
                "asset": "equity",
                "direction": "long",
                "instrument_type": "option",
                "option_type": "put",
                "underlying_ticker": "AAPL",
                "price_symbol": "AAPL260116P00150000",
                "quantity": 1,
                "shares": 1,
                "cost_basis": 5.0,
                "contract_multiplier": 100,
            },
        ]
    )

    collapsed, skipped = portfolio_sizer._collapse_positions_to_underlyings(df)

    assert skipped == []
    assert list(collapsed["ticker"]) == ["AAPL"]
    row = collapsed.iloc[0]
    assert row["instrument_type"] == "security"
    assert row["price_symbol"] == "AAPL"
    assert row["direction"] == "long"


def test_collapse_skips_offsetting_option_underlying():
    df = pd.DataFrame(
        [
            {
                "ticker": "TSLA",
                "asset": "equity",
                "direction": "long",
                "instrument_type": "option",
                "option_type": "call",
                "underlying_ticker": "TSLA",
                "price_symbol": "c",
                "quantity": 1,
                "shares": 1,
                "cost_basis": 10.0,
                "contract_multiplier": 100,
            },
            {
                "ticker": "TSLA",
                "asset": "equity",
                "direction": "long",
                "instrument_type": "option",
                "option_type": "put",
                "underlying_ticker": "TSLA",
                "price_symbol": "p",
                "quantity": 1,
                "shares": 1,
                "cost_basis": 10.0,
                "contract_multiplier": 100,
            },
        ]
    )

    collapsed, skipped = portfolio_sizer._collapse_positions_to_underlyings(df)

    assert skipped == ["TSLA"]
    assert collapsed.empty


def test_compute_current_exposure_short_put_uses_premium_notional():
    meta = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "asset": "equity",
                "direction": "short",
                "instrument_type": "option",
                "option_type": "put",
                "underlying_ticker": "NVDA",
                "quantity": 2,
                "cost_basis": 6.0,
                "contract_multiplier": 100,
            }
        ]
    )
    exposure = portfolio_sizer._compute_current_underlying_dollar_exposure(meta, {"NVDA": 500.0})
    assert exposure["NVDA"] == pytest.approx(2 * 6.0 * 100.0)


def test_compute_current_exposure_sums_equity_and_option_legs():
    meta = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "asset": "equity",
                "direction": "long",
                "instrument_type": "security",
                "quantity": 10,
                "shares": 10,
            },
            {
                "ticker": "AAPL",
                "asset": "equity",
                "direction": "long",
                "instrument_type": "option",
                "option_type": "call",
                "underlying_ticker": "AAPL",
                "quantity": 1,
                "cost_basis": 5.0,
                "contract_multiplier": 100,
            },
        ]
    )
    exposure = portfolio_sizer._compute_current_underlying_dollar_exposure(meta, {"AAPL": 100.0})
    assert exposure["AAPL"] == pytest.approx(10 * 100.0 + 1 * 5.0 * 100.0)


def test_attach_sizing_delta_columns_reduces_buy_when_option_exposure_exists():
    weights_df = pd.DataFrame(
        {
            "ticker": ["NVDA"],
            "instrument_type": ["security"],
            "price": [500.0],
            "contract_multiplier": [1.0],
            "dollar_weight": [50_000.0],
            "target_quantity": [100],
        }
    )
    current_dollar = {"NVDA": 12_000.0}
    out = portfolio_sizer._attach_sizing_delta_columns(weights_df, current_dollar)

    assert out["current_dollar_weight"].iloc[0] == pytest.approx(12_000.0)
    assert out["target_dollar_weight"].iloc[0] == pytest.approx(50_000.0)
    assert out["delta_dollar_weight"].iloc[0] == pytest.approx(38_000.0)
    assert int(out["current_quantity"].iloc[0]) == 24
    assert int(out["delta_quantity"].iloc[0]) == 76


def test_offsetting_options_contribute_near_zero_current_exposure():
    meta = pd.DataFrame(
        [
            {
                "ticker": "TSLA",
                "asset": "equity",
                "direction": "long",
                "instrument_type": "option",
                "option_type": "call",
                "underlying_ticker": "TSLA",
                "quantity": 1,
                "cost_basis": 10.0,
                "contract_multiplier": 100,
            },
            {
                "ticker": "TSLA",
                "asset": "equity",
                "direction": "long",
                "instrument_type": "option",
                "option_type": "put",
                "underlying_ticker": "TSLA",
                "quantity": 1,
                "cost_basis": 10.0,
                "contract_multiplier": 100,
            },
        ]
    )
    exposure = portfolio_sizer._compute_current_underlying_dollar_exposure(meta, {"TSLA": 200.0})
    assert abs(exposure.get("TSLA", 0.0)) < 1e-6
