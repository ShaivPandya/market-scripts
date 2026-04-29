from __future__ import annotations

import pandas as pd


def test_max_drawdown_from_returns():
    from commodities.aluminum.metrics import max_drawdown

    returns = pd.Series([0.10, -0.20, 0.05])

    assert round(max_drawdown(returns), 6) == -0.20


def test_hit_rate_ignores_flat_predictions_by_default():
    from commodities.aluminum.metrics import hit_rate

    preds = pd.Series([0.10, -0.10, 0.0, 0.20])
    actuals = pd.Series([0.05, -0.02, 0.10, -0.01])

    assert hit_rate(preds, actuals) == 2 / 3


def test_rmse_and_mae():
    from commodities.aluminum.metrics import mae, rmse

    preds = pd.Series([0.0, 0.1, 0.2])
    actuals = pd.Series([0.0, 0.0, 0.4])

    assert round(rmse(preds, actuals), 6) == round(((0.0**2 + 0.1**2 + (-0.2) ** 2) / 3) ** 0.5, 6)
    assert round(mae(preds, actuals), 6) == 0.1
