from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.portfolio_optimizer import portfolio_sizer


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
