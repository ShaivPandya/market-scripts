from __future__ import annotations

import numpy as np
import pandas as pd


def _feature_frame(periods: int = 18) -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    target = pd.Series(np.where(np.arange(periods) % 2 == 0, 0.02, -0.01), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "aluminum_price_usd_tonne": 2000.0 + np.arange(periods),
            "aluminum_return_1m": np.linspace(-0.02, 0.02, periods),
            "aluminum_return_3m": np.linspace(-0.01, 0.03, periods),
            "aluminum_return_6m": np.linspace(0.00, 0.04, periods),
            "aluminum_momentum_12m": np.linspace(-0.05, 0.05, periods),
            "aluminum_volatility_6m": 0.10,
            "inventory_change_1m": np.nan,
            "inventory_change_3m": np.nan,
            "power_proxy_change_1m": np.nan,
            "month": dates.month,
            "quarter": dates.quarter,
            "target_return_1m_forward": target,
            "has_world_bank_price": True,
            "has_eia_power_proxy": False,
            "has_shfe_inventory": False,
            "has_lme_price": False,
            "has_lme_stock": False,
        }
    )


def test_walk_forward_split_uses_rows_strictly_before_forecast_date():
    from commodities.aluminum.backtest import walk_forward_backtest
    from commodities.aluminum.config import AluminumBacktestConfig

    features = _feature_frame(periods=10)
    trades = walk_forward_backtest(
        features,
        AluminumBacktestConfig(min_train_months=4, model_type="zero", forecast_threshold=0.005),
    )

    assert trades.iloc[0]["date"] == features.iloc[4]["date"]
    assert trades.iloc[0]["train_observations"] == 4


def test_transaction_cost_logic_with_position_changes(monkeypatch):
    import commodities.aluminum.backtest as bt
    from commodities.aluminum.config import AluminumBacktestConfig

    class FakeModel:
        def fit(self, X_train, y_train):
            return self

        def predict(self, X_test):
            month = int(X_test["month"].iloc[0])
            return np.array([0.02 if month % 2 else -0.02])

    monkeypatch.setattr(bt, "make_model", lambda *args, **kwargs: FakeModel())

    trades = bt.walk_forward_backtest(
        _feature_frame(periods=8),
        AluminumBacktestConfig(
            min_train_months=3,
            model_type="ridge",
            forecast_threshold=0.005,
            transaction_cost_bps=10.0,
        ),
    )

    assert abs(trades.iloc[0]["position_change"]) == 1
    assert trades.iloc[0]["transaction_cost"] == 0.001
    assert abs(trades.iloc[1]["position_change"]) == 2
    assert trades.iloc[1]["transaction_cost"] == 0.002


def test_factor_diagnostics_classifies_unavailable_optional_features():
    from commodities.aluminum.backtest import compute_factor_diagnostics

    diagnostics = compute_factor_diagnostics(_feature_frame(periods=18)).set_index("feature")

    assert diagnostics.loc["inventory_change_1m", "classification"] == "unavailable"
    assert diagnostics.loc["power_proxy_change_1m", "classification"] == "unavailable"
    assert diagnostics.loc["aluminum_return_1m", "observations"] > 0


def test_backtest_metrics_include_validation_gate():
    from commodities.aluminum.backtest import build_backtest_metrics, compute_factor_diagnostics, walk_forward_backtest
    from commodities.aluminum.config import AluminumBacktestConfig

    features = _feature_frame(periods=24)
    config = AluminumBacktestConfig(min_train_months=6, model_type="zero")
    trades = walk_forward_backtest(features, config)
    diagnostics = compute_factor_diagnostics(features)
    metrics = build_backtest_metrics(trades, features, diagnostics, config)

    assert "production_validation_passed" in metrics.columns
    assert not bool(metrics.loc[metrics["strategy"] == "model_strategy", "production_validation_passed"].iloc[0])
