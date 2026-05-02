"""Backtest metrics for aluminum strategy validation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return float("nan")
    wealth = float((1.0 + r).prod())
    if wealth <= 0:
        return -1.0
    return float(wealth ** (periods_per_year / len(r)) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    vol = annualized_volatility(returns, periods_per_year=periods_per_year)
    if not math.isfinite(vol) or vol <= 0:
        return float("nan")
    return annualized_return(returns, periods_per_year=periods_per_year) / vol


def equity_curve(returns: pd.Series) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    return (1.0 + r).cumprod()


def max_drawdown(returns_or_equity: pd.Series, *, input_is_equity: bool = False) -> float:
    eq = pd.to_numeric(returns_or_equity, errors="coerce").dropna()
    if eq.empty:
        return float("nan")
    if not input_is_equity:
        eq = equity_curve(eq)
    peak = eq.cummax()
    drawdown = eq / peak - 1.0
    return float(drawdown.min())


def hit_rate(predictions: pd.Series, actuals: pd.Series, *, threshold: float = 0.0, drop_flat: bool = True) -> float:
    pred = pd.to_numeric(predictions, errors="coerce")
    actual = pd.to_numeric(actuals, errors="coerce")
    aligned = pd.concat([pred.rename("pred"), actual.rename("actual")], axis=1).dropna()
    if aligned.empty:
        return float("nan")

    signal = np.where(aligned["pred"] > threshold, 1, np.where(aligned["pred"] < -threshold, -1, 0))
    actual_sign = np.sign(aligned["actual"].to_numpy(dtype=float))
    if drop_flat:
        mask = signal != 0
        signal = signal[mask]
        actual_sign = actual_sign[mask]
    if len(signal) == 0:
        return float("nan")
    return float((signal == actual_sign).mean())


def rmse(predictions: pd.Series, actuals: pd.Series) -> float:
    aligned = pd.concat(
        [pd.to_numeric(predictions, errors="coerce"), pd.to_numeric(actuals, errors="coerce")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return float("nan")
    err = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(np.sqrt(np.mean(np.square(err))))


def mae(predictions: pd.Series, actuals: pd.Series) -> float:
    aligned = pd.concat(
        [pd.to_numeric(predictions, errors="coerce"), pd.to_numeric(actuals, errors="coerce")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return float("nan")
    err = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(np.mean(np.abs(err)))


def return_metrics(
    returns: pd.Series,
    *,
    label: str,
    predictions: pd.Series | None = None,
    actuals: pd.Series | None = None,
    position_changes: pd.Series | None = None,
) -> dict[str, Any]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    trades = (
        int((pd.to_numeric(position_changes, errors="coerce").fillna(0.0) != 0).sum())
        if position_changes is not None
        else 0
    )
    turnover = (
        float(pd.to_numeric(position_changes, errors="coerce").abs().mean()) if position_changes is not None else 0.0
    )

    out: dict[str, Any] = {
        "strategy": label,
        "observations": int(len(r)),
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe_ratio": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "number_of_trades": trades,
        "average_turnover": turnover,
    }

    if predictions is not None and actuals is not None:
        out["hit_rate"] = hit_rate(predictions, actuals)
        out["rmse"] = rmse(predictions, actuals)
        out["mae"] = mae(predictions, actuals)
    else:
        out["hit_rate"] = float("nan")
        out["rmse"] = float("nan")
        out["mae"] = float("nan")
    return out


def source_coverage(features: pd.DataFrame) -> dict[str, float]:
    coverage_cols = {
        "world_bank": "has_world_bank_price",
        "eia": "has_eia_power_proxy",
        "shfe": "has_shfe_inventory",
        "lme_price": "has_lme_price",
        "lme_stock": "has_lme_stock",
    }
    out: dict[str, float] = {}
    total = max(len(features), 1)
    for key, col in coverage_cols.items():
        if col not in features.columns:
            out[f"coverage_{key}"] = 0.0
        else:
            out[f"coverage_{key}"] = float(features[col].fillna(False).astype(bool).sum() / total)
    return out
