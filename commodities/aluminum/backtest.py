"""Aluminum data pipeline, walk-forward backtest, diagnostics, and outputs."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from commodities.aluminum.config import (
    DRAWDOWN_PNG,
    EQUITY_CURVE_CSV,
    EQUITY_CURVE_PNG,
    FACTOR_DIAGNOSTICS_CSV,
    FEATURES_CSV,
    LME_PRICES_PROCESSED_CSV,
    LME_STOCKS_PROCESSED_CSV,
    METRICS_CSV,
    PROCESSED_DIR,
    RESULTS_DIR,
    SHFE_PROCESSED_CSV,
    TRADES_CSV,
    VALIDATION_MAX_SINGLE_YEAR_PNL_SHARE,
    VALIDATION_MIN_FORECASTS,
    VALIDATION_MIN_FUNDAMENTAL_FEATURE_IC,
    VALIDATION_MIN_NET_SHARPE,
    VALIDATION_MIN_OPTIONAL_SOURCE_MONTHS,
    VALIDATION_MIN_POSITIVE_YEAR_RATIO,
    VALIDATION_MIN_PREDICTION_SPEARMAN_IC,
    VALIDATION_MIN_SHARPE_EDGE_VS_BUY_HOLD,
    VALIDATION_MIN_TRAIN_MONTHS,
    WORLD_BANK_PROCESSED_CSV,
    AluminumBacktestConfig,
    ensure_directories,
)
from commodities.aluminum.features import FEATURE_COLUMNS, TARGET_COLUMN, build_monthly_features, feature_metadata
from commodities.aluminum.metrics import equity_curve, max_drawdown, return_metrics, rmse, source_coverage
from commodities.aluminum.models import make_model
from commodities.aluminum.sources.eia import empty_eia_frame, fetch_eia_power_proxy
from commodities.aluminum.sources.lme import empty_lme_prices_frame, empty_lme_stocks_frame, load_lme_xml_data
from commodities.aluminum.sources.shfe import empty_shfe_frame, fetch_shfe_aluminum_inventory
from commodities.aluminum.sources.world_bank import fetch_world_bank_aluminum_prices

log = logging.getLogger(__name__)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _read_csv(path: Path, *, empty: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return empty.copy()
    return pd.read_csv(path, parse_dates=["date"])


def fetch_and_cache_sources(
    *,
    refresh: bool = False,
    lme_xml_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch/cache normalized aluminum source data.

    World Bank is the required target series and raises on failure. Optional
    sources return empty frames with warnings when unavailable.
    """
    ensure_directories()

    world_bank = fetch_world_bank_aluminum_prices(refresh=refresh)
    _write_csv(world_bank, WORLD_BANK_PROCESSED_CSV)

    eia = fetch_eia_power_proxy()
    _write_csv(eia, PROCESSED_DIR / "eia_power_proxy.csv")

    shfe = fetch_shfe_aluminum_inventory(refresh=refresh)
    _write_csv(shfe, SHFE_PROCESSED_CSV)

    lme_prices, lme_stocks = load_lme_xml_data(xml_dir=lme_xml_dir)
    _write_csv(lme_prices, LME_PRICES_PROCESSED_CSV)
    _write_csv(lme_stocks, LME_STOCKS_PROCESSED_CSV)

    return {
        "world_bank": world_bank,
        "eia": eia,
        "shfe": shfe,
        "lme_prices": lme_prices,
        "lme_stocks": lme_stocks,
    }


def load_processed_sources() -> dict[str, pd.DataFrame]:
    return {
        "world_bank": _read_csv(
            WORLD_BANK_PROCESSED_CSV, empty=pd.DataFrame(columns=["date", "aluminum_price_usd_tonne", "source"])
        ),
        "eia": _read_csv(PROCESSED_DIR / "eia_power_proxy.csv", empty=empty_eia_frame()),
        "shfe": _read_csv(SHFE_PROCESSED_CSV, empty=empty_shfe_frame()),
        "lme_prices": _read_csv(LME_PRICES_PROCESSED_CSV, empty=empty_lme_prices_frame()),
        "lme_stocks": _read_csv(LME_STOCKS_PROCESSED_CSV, empty=empty_lme_stocks_frame()),
    }


def build_and_cache_features(
    *,
    refresh: bool = False,
    lme_xml_dir: str | Path | None = None,
) -> pd.DataFrame:
    ensure_directories()
    sources = (
        fetch_and_cache_sources(refresh=refresh, lme_xml_dir=lme_xml_dir)
        if refresh or not WORLD_BANK_PROCESSED_CSV.exists()
        else load_processed_sources()
    )
    if sources["world_bank"].empty:
        sources = fetch_and_cache_sources(refresh=refresh, lme_xml_dir=lme_xml_dir)

    features = build_monthly_features(
        world_bank_prices=sources["world_bank"],
        eia_power_proxy=sources["eia"],
        shfe_inventory=sources["shfe"],
        lme_prices=sources["lme_prices"],
        lme_stocks=sources["lme_stocks"],
    )
    _write_csv(features, FEATURES_CSV)
    return features


def load_or_build_features(config: AluminumBacktestConfig) -> pd.DataFrame:
    if FEATURES_CSV.exists() and not config.refresh:
        return pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    return build_and_cache_features(refresh=config.refresh, lme_xml_dir=config.lme_xml_dir)


def available_feature_columns(features: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in FEATURE_COLUMNS:
        if col not in features.columns:
            continue
        if features[col].notna().any():
            cols.append(col)
    return cols


def _window(features: pd.DataFrame, config: AluminumBacktestConfig) -> pd.DataFrame:
    out = features.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if config.start_date:
        out = out[out["date"] >= pd.Timestamp(config.start_date)]
    if config.end_date:
        out = out[out["date"] <= pd.Timestamp(config.end_date)]
    return out.reset_index(drop=True)


def _training_columns(train: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [col for col in feature_cols if train[col].notna().any()]


def walk_forward_backtest(features: pd.DataFrame, config: AluminumBacktestConfig) -> pd.DataFrame:
    """Run expanding-window monthly validation."""
    full = features.copy()
    full["date"] = pd.to_datetime(full["date"], errors="coerce")
    full = full.dropna(subset=["date", TARGET_COLUMN]).sort_values("date").reset_index(drop=True)
    if full.empty:
        raise RuntimeError("No aluminum feature rows with target returns are available")

    feature_cols = available_feature_columns(full)
    if not feature_cols and config.model_type != "zero":
        log.warning("No feature columns available; model forecasts will fall back to zero")

    test_rows = _window(full, config)
    records: list[dict[str, Any]] = []
    previous_position = 0

    for _, row in test_rows.iterrows():
        forecast_date = pd.Timestamp(row["date"])
        train = full[full["date"] < forecast_date].dropna(subset=[TARGET_COLUMN])
        if len(train) < config.min_train_months:
            continue

        cols = _training_columns(train, feature_cols)
        if config.model_type == "zero" or not cols:
            forecast = 0.0
        else:
            model = make_model(config.model_type, random_seed=config.random_seed)
            X_train = train[cols]
            y_train = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
            valid_train = y_train.notna()
            model.fit(X_train.loc[valid_train], y_train.loc[valid_train])
            forecast = float(model.predict(pd.DataFrame([row[cols].to_dict()]))[0])

        actual = float(row[TARGET_COLUMN])
        if forecast > config.forecast_threshold:
            position = 1
        elif forecast < -config.forecast_threshold:
            position = -1
        else:
            position = 0

        position_change = position - previous_position
        transaction_cost = abs(position_change) * config.transaction_cost_bps / 10000.0
        gross_strategy_return = position * actual
        strategy_return = gross_strategy_return - transaction_cost

        records.append(
            {
                "date": forecast_date,
                "forecast_return": forecast,
                "actual_forward_return": actual,
                "signal": position,
                "previous_signal": previous_position,
                "position_change": position_change,
                "transaction_cost": transaction_cost,
                "gross_strategy_return": gross_strategy_return,
                "strategy_return": strategy_return,
                "buy_hold_return": actual,
                "zero_return_forecast": 0.0,
                "train_observations": int(len(train)),
                "test_observations": 1,
                "feature_count": int(len(cols)),
            }
        )
        previous_position = position

    if not records:
        raise RuntimeError(
            "No walk-forward forecasts were produced. Lower min_train_months or expand the selected date range."
        )
    return pd.DataFrame(records)


def compute_factor_diagnostics(features: pd.DataFrame) -> pd.DataFrame:
    meta = feature_metadata(features).set_index("feature")
    rows: list[dict[str, Any]] = []

    for feature in FEATURE_COLUMNS:
        if feature not in features.columns:
            rows.append({"feature": feature, "classification": "unavailable", "observations": 0})
            continue

        subset = features[[feature, TARGET_COLUMN]].copy()
        subset[feature] = pd.to_numeric(subset[feature], errors="coerce")
        subset[TARGET_COLUMN] = pd.to_numeric(subset[TARGET_COLUMN], errors="coerce")
        subset = subset.dropna()
        obs = len(subset)

        pearson = float(subset[feature].corr(subset[TARGET_COLUMN], method="pearson")) if obs >= 3 else float("nan")
        spearman = float(subset[feature].corr(subset[TARGET_COLUMN], method="spearman")) if obs >= 3 else float("nan")

        low_mean = high_mean = spread = float("nan")
        if obs >= 12 and subset[feature].nunique() >= 3:
            low = subset[feature].quantile(1.0 / 3.0)
            high = subset[feature].quantile(2.0 / 3.0)
            low_mean = float(subset.loc[subset[feature] <= low, TARGET_COLUMN].mean())
            high_mean = float(subset.loc[subset[feature] >= high, TARGET_COLUMN].mean())
            spread = high_mean - low_mean

        if obs < 24 or not bool(meta.loc[feature, "available"]):
            classification = "unavailable"
        elif math.isfinite(spearman) and abs(spearman) >= 0.05 and math.isfinite(spread) and abs(spread) >= 0.002:
            classification = "useful" if spearman * spread >= 0 else "unstable"
        else:
            classification = "weak"

        rows.append(
            {
                "feature": feature,
                "category": meta.loc[feature, "category"],
                "source": meta.loc[feature, "source"],
                "is_lagged": bool(meta.loc[feature, "is_lagged"]),
                "observations": obs,
                "pearson_correlation": pearson,
                "spearman_rank_ic": spearman,
                "low_tercile_forward_return": low_mean,
                "high_tercile_forward_return": high_mean,
                "high_minus_low_tercile_spread": spread,
                "classification": classification,
            }
        )

    return pd.DataFrame(rows)


def _positive_year_ratio(trades: pd.DataFrame) -> float:
    by_year = trades.assign(year=pd.to_datetime(trades["date"]).dt.year).groupby("year")["strategy_return"].sum()
    if by_year.empty:
        return float("nan")
    return float((by_year > 0).mean())


def _max_single_year_pnl_share(trades: pd.DataFrame) -> float:
    by_year = trades.assign(year=pd.to_datetime(trades["date"]).dt.year).groupby("year")["strategy_return"].sum()
    positive_total = float(by_year[by_year > 0].sum())
    if positive_total <= 0:
        return float("inf")
    return float(by_year.max() / positive_total)


def _prediction_ic(trades: pd.DataFrame) -> float:
    if len(trades) < 3:
        return float("nan")
    return float(trades["forecast_return"].corr(trades["actual_forward_return"], method="spearman"))


def evaluate_validation_bar(
    *,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    trades: pd.DataFrame,
    features: pd.DataFrame,
    config: AluminumBacktestConfig,
) -> dict[str, Any]:
    strategy = metrics.set_index("strategy")
    model = strategy.loc["model_strategy"]
    buy_hold = strategy.loc["buy_and_hold_aluminum"]
    zero = strategy.loc["zero_return_forecast"]

    pred_ic = _prediction_ic(trades)
    rmse_edge = 1.0 - (float(model["rmse"]) / float(zero["rmse"])) if float(zero["rmse"]) > 0 else float("nan")
    hit = float(model["hit_rate"])
    non_price = diagnostics[diagnostics["category"].isin(["fundamental_inventory", "cost_proxy"])]
    best_non_price_ic = (
        float(non_price["spearman_rank_ic"].abs().max())
        if not non_price.empty and non_price["spearman_rank_ic"].notna().any()
        else 0.0
    )
    optional_months = int(
        features[["has_eia_power_proxy", "has_shfe_inventory", "has_lme_stock"]]
        .fillna(False)
        .astype(bool)
        .any(axis=1)
        .sum()
    )
    positive_year_ratio = _positive_year_ratio(trades)
    max_year_share = _max_single_year_pnl_share(trades)

    checks = {
        "min_forecasts": len(trades) >= VALIDATION_MIN_FORECASTS,
        "min_train_months": config.min_train_months >= VALIDATION_MIN_TRAIN_MONTHS,
        "net_sharpe": float(model["sharpe_ratio"]) >= VALIDATION_MIN_NET_SHARPE,
        "sharpe_edge": (float(model["sharpe_ratio"]) - float(buy_hold["sharpe_ratio"]))
        >= VALIDATION_MIN_SHARPE_EDGE_VS_BUY_HOLD,
        "prediction_ic": math.isfinite(pred_ic) and pred_ic >= VALIDATION_MIN_PREDICTION_SPEARMAN_IC,
        "forecast_error_or_hit_rate": (math.isfinite(rmse_edge) and rmse_edge >= 0.05)
        or (math.isfinite(hit) and hit >= 0.53),
        "non_price_feature_ic": best_non_price_ic >= VALIDATION_MIN_FUNDAMENTAL_FEATURE_IC,
        "positive_year_ratio": math.isfinite(positive_year_ratio)
        and positive_year_ratio >= VALIDATION_MIN_POSITIVE_YEAR_RATIO,
        "single_year_concentration": math.isfinite(max_year_share)
        and max_year_share <= VALIDATION_MAX_SINGLE_YEAR_PNL_SHARE,
        "optional_source_months": optional_months >= VALIDATION_MIN_OPTIONAL_SOURCE_MONTHS,
    }

    return {
        "production_validation_passed": all(checks.values()),
        "failed_validation_checks": ",".join(key for key, passed in checks.items() if not passed),
        "prediction_spearman_ic": pred_ic,
        "rmse_improvement_vs_zero": rmse_edge,
        "best_non_price_feature_abs_ic": best_non_price_ic,
        "optional_source_months": optional_months,
        "positive_year_ratio": positive_year_ratio,
        "max_single_year_pnl_share": max_year_share,
    }


def build_backtest_metrics(
    trades: pd.DataFrame,
    features: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: AluminumBacktestConfig,
) -> pd.DataFrame:
    model_metrics = return_metrics(
        trades["strategy_return"],
        label="model_strategy",
        predictions=trades["forecast_return"],
        actuals=trades["actual_forward_return"],
        position_changes=trades["position_change"],
    )
    buy_hold_metrics = return_metrics(trades["buy_hold_return"], label="buy_and_hold_aluminum")
    zero_metrics = return_metrics(
        pd.Series(np.zeros(len(trades)), index=trades.index),
        label="zero_return_forecast",
        predictions=trades["zero_return_forecast"],
        actuals=trades["actual_forward_return"],
        position_changes=pd.Series(np.zeros(len(trades)), index=trades.index),
    )
    rows = [model_metrics, buy_hold_metrics, zero_metrics]
    coverage = source_coverage(features)
    metrics = pd.DataFrame([{**row, **coverage} for row in rows])
    validation = evaluate_validation_bar(
        metrics=metrics,
        diagnostics=diagnostics,
        trades=trades,
        features=features,
        config=config,
    )
    for key, value in validation.items():
        metrics[key] = value
    metrics["model_type"] = config.model_type
    metrics["forecast_threshold"] = config.forecast_threshold
    metrics["transaction_cost_bps"] = config.transaction_cost_bps
    metrics["min_train_months"] = config.min_train_months
    return metrics


def build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades[["date"]].copy()
    out["strategy_equity"] = equity_curve(trades["strategy_return"]).to_numpy()
    out["buy_hold_equity"] = equity_curve(trades["buy_hold_return"]).to_numpy()
    out["zero_forecast_equity"] = 1.0
    out["strategy_drawdown"] = out["strategy_equity"] / out["strategy_equity"].cummax() - 1.0
    out["buy_hold_drawdown"] = out["buy_hold_equity"] / out["buy_hold_equity"].cummax() - 1.0
    return out


def plot_results(equity: pd.DataFrame | None = None) -> None:
    ensure_directories()
    if equity is None:
        if not EQUITY_CURVE_CSV.exists():
            raise RuntimeError("No equity curve CSV exists; run the backtest first")
        equity = pd.read_csv(EQUITY_CURVE_CSV, parse_dates=["date"])

    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "market_scripts_matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "market_scripts_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity["date"], equity["strategy_equity"], label="Model strategy")
    ax.plot(equity["date"], equity["buy_hold_equity"], label="Buy and hold aluminum")
    ax.plot(equity["date"], equity["zero_forecast_equity"], label="Zero forecast")
    ax.set_title("Aluminum Backtest Equity Curve")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EQUITY_CURVE_PNG, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity["date"], equity["strategy_drawdown"], label="Model strategy")
    ax.plot(equity["date"], equity["buy_hold_drawdown"], label="Buy and hold aluminum")
    ax.set_title("Aluminum Backtest Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DRAWDOWN_PNG, dpi=150)
    plt.close(fig)


def run_aluminum_backtest(config: AluminumBacktestConfig) -> dict[str, pd.DataFrame]:
    ensure_directories()
    features = load_or_build_features(config)
    trades = walk_forward_backtest(features, config)
    diagnostics = compute_factor_diagnostics(features)
    metrics = build_backtest_metrics(trades, features, diagnostics, config)
    equity = build_equity_curve(trades)

    _write_csv(features, FEATURES_CSV)
    _write_csv(trades, TRADES_CSV)
    _write_csv(metrics, METRICS_CSV)
    _write_csv(equity, EQUITY_CURVE_CSV)
    _write_csv(diagnostics, FACTOR_DIAGNOSTICS_CSV)
    plot_results(equity)

    return {
        "features": features,
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity,
        "factor_diagnostics": diagnostics,
    }


def print_backtest_summary(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    model = metrics[metrics["strategy"] == "model_strategy"].iloc[0]
    print("\nALUMINUM BACKTEST SUMMARY")
    print("=" * 70)
    print(f"Model strategy Sharpe: {model['sharpe_ratio']:.3f}")
    print(f"Net annualized return: {model['annualized_return']:.2%}")
    print(f"Max drawdown:          {model['max_drawdown']:.2%}")
    print(
        f"Hit rate:              {model['hit_rate']:.2%}"
        if pd.notna(model["hit_rate"])
        else "Hit rate:              n/a"
    )
    print(f"Prediction rank IC:    {model['prediction_spearman_ic']:.3f}")
    print(f"Production gate pass:  {bool(model['production_validation_passed'])}")
    if model["failed_validation_checks"]:
        print(f"Failed checks:         {model['failed_validation_checks']}")

    print("\nFACTOR DIAGNOSTICS")
    print("-" * 70)
    for _, row in diagnostics.iterrows():
        ic = row.get("spearman_rank_ic")
        ic_text = "n/a" if pd.isna(ic) else f"{ic:.3f}"
        print(f"{row['feature']:<28} IC={ic_text:>7}  {row['classification']}")

    print(f"\nOutputs written under {RESULTS_DIR}")
