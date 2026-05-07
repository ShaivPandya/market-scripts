from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.portfolio_optimizer import portfolio_analyzer, portfolio_sizer


def test_equity_beta_inputs_ignore_non_equity_returns(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_compute_beta_frame(rets: pd.DataFrame, tickers: list[str]):
        captured["columns"] = list(rets.columns)
        captured["tickers"] = list(tickers)
        beta_frame = pd.DataFrame(
            {
                "beta_spy": [1.25],
                "beta_iwm": [0.85],
            },
            index=tickers,
        )
        betas_all_spy = pd.Series({"EQ": 1.25, "SPY": 1.0, "IWM": 0.65})
        betas_all_iwm = pd.Series({"EQ": 0.85, "SPY": 1.15, "IWM": 1.0})
        return beta_frame, betas_all_spy, betas_all_iwm

    monkeypatch.setattr(portfolio_sizer, "compute_beta_frame", fake_compute_beta_frame)

    rets = pd.DataFrame(
        {
            "EQ": [0.01, 0.02],
            "GLD": [0.03, 0.04],
            "EURUSD=X": [0.001, 0.002],
            "SPY": [0.01, 0.015],
            "IWM": [0.012, 0.013],
        }
    )

    betas_spy, betas_iwm, display_spy, display_iwm, *_ = portfolio_sizer._compute_equity_beta_inputs(
        rets=rets,
        tickers=["EQ", "GLD", "EURUSD=X"],
        market_tickers=["SPY", "IWM"],
        eq_mask=np.array([True, False, False]),
    )

    assert captured == {"columns": ["EQ", "SPY", "IWM"], "tickers": ["EQ"]}
    assert betas_spy.to_dict() == {"EQ": 1.25, "GLD": 0.0, "EURUSD=X": 0.0}
    assert betas_iwm.to_dict() == {"EQ": 0.85, "GLD": 0.0, "EURUSD=X": 0.0}
    assert display_spy.loc["EQ"] == 1.25
    assert pd.isna(display_spy.loc["GLD"])
    assert pd.isna(display_iwm.loc["EURUSD=X"])


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
    betas_spy = pd.Series({"AAA": 1.0, "BBB": 1.0})
    betas_iwm = pd.Series({"AAA": 1.5, "BBB": 1.0})
    betas_all_spy = pd.Series({"SPY": 1.0, "IWM": 0.7})
    betas_all_iwm = pd.Series({"SPY": 0.8, "IWM": 1.0})

    hedged_weights, summary = portfolio_sizer._apply_beta_hedges_with_gross_cap(
        weights,
        betas_spy,
        betas_iwm,
        betas_all_spy,
        betas_all_iwm,
        long_mask=np.array([True, False]),
        short_mask=np.array([False, True]),
        eq_mask=np.array([True, True]),
        beta_hedge_mode="spy",
    )

    assert hedged_weights.equals(weights)
    assert summary["beta_hedge_mode"] == "spy"
    assert summary["hedge_iwm_weight"] == 0.0
    assert abs(summary["post_hedge_beta_spy"]) < 1e-6
    assert abs(summary["post_hedge_beta_iwm"]) > 0.1
    assert summary["hedge_gross"] == abs(summary["hedge_spy_weight"])
    assert np.isclose(summary["gross_with_hedges"], float(np.abs(weights).sum() + abs(summary["hedge_spy_weight"])))
