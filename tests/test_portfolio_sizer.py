from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
        beta_hedge_mode="spy",
    )

    assert hedged_weights.equals(weights)
    assert summary["beta_hedge_mode"] == "spy"
    assert summary["hedge_iwm_weight"] == 0.0
    assert summary["hedge_qqq_weight"] == 0.0
    assert abs(summary["post_hedge_beta_spy"]) < 1e-6
    assert abs(summary["post_hedge_beta_iwm"]) > 0.1
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
        beta_hedge_mode="spy_qqq",
    )

    assert summary["selected_hedges"] == ["SPY", "QQQ"]
    assert summary["hedge_iwm_weight"] == 0.0
    assert summary["hedge_weights"]["IWM"] == 0.0
    assert abs(summary["post_hedge_beta_spy"]) < 1e-4
    assert abs(summary["post_hedge_beta_qqq"]) < 1e-4
    assert abs(summary["post_hedge_beta_iwm"]) > 0.01


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
        beta_hedge_mode="spy_iwm_qqq",
    )

    assert summary["selected_hedges"] == ["SPY", "IWM", "QQQ"]
    assert set(summary["hedge_weights"]) == {"SPY", "IWM", "QQQ"}
    assert summary["hedge_spy_weight"] != 0.0
    assert summary["hedge_iwm_weight"] != 0.0
    assert summary["hedge_qqq_weight"] != 0.0
    assert abs(summary["post_hedge_beta_spy"]) < 1e-5
    assert abs(summary["post_hedge_beta_iwm"]) < 1e-5
    assert abs(summary["post_hedge_beta_qqq"]) < 1e-5


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
