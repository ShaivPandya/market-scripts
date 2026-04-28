"""Monthly aluminum feature engineering.

All non-calendar predictor columns are shifted by one month after calculation.
This means a row dated month-end ``t`` uses source data observable no later than
month-end ``t-1`` to predict ``target_return_1m_forward`` from ``t`` to ``t+1``.
The conservative shift is intentional and prevents target-period price changes
from leaking into model inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PRICE_FEATURES = [
    "aluminum_return_1m",
    "aluminum_return_3m",
    "aluminum_return_6m",
    "aluminum_momentum_12m",
    "aluminum_volatility_6m",
]
FUNDAMENTAL_FEATURES = ["inventory_change_1m", "inventory_change_3m"]
COST_PROXY_FEATURES = ["power_proxy_change_1m"]
CALENDAR_FEATURES = ["month", "quarter"]
FEATURE_COLUMNS = PRICE_FEATURES + FUNDAMENTAL_FEATURES + COST_PROXY_FEATURES + CALENDAR_FEATURES
TARGET_COLUMN = "target_return_1m_forward"


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


def _month_end_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return _empty_frame()
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce") + pd.offsets.MonthEnd(0)
    out = out.dropna(subset=[date_col])
    if out.empty:
        return _empty_frame()
    return out.set_index(date_col).sort_index()


def _monthly_last(df: pd.DataFrame, value_col: str) -> pd.Series:
    indexed = _month_end_index(df)
    if indexed.empty or value_col not in indexed.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(indexed[value_col], errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.resample("ME").last().dropna()


def _monthly_sum_then_last(df: pd.DataFrame, value_col: str) -> pd.Series:
    indexed = _month_end_index(df)
    if indexed.empty or value_col not in indexed.columns:
        return pd.Series(dtype=float)
    daily = pd.to_numeric(indexed[value_col], errors="coerce").groupby(level=0).sum(min_count=1)
    if daily.empty:
        return pd.Series(dtype=float)
    return daily.resample("ME").last().dropna()


def _source_flag(index: pd.DatetimeIndex, series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(False, index=index)
    return series.reindex(index).notna()


def build_monthly_features(
    *,
    world_bank_prices: pd.DataFrame,
    eia_power_proxy: pd.DataFrame | None = None,
    shfe_inventory: pd.DataFrame | None = None,
    lme_prices: pd.DataFrame | None = None,
    lme_stocks: pd.DataFrame | None = None,
    feature_lag_months: int = 1,
    drop_missing_target: bool = True,
) -> pd.DataFrame:
    """Build monthly lagged features and next-month aluminum return target."""
    price = _monthly_last(world_bank_prices, "aluminum_price_usd_tonne")
    if price.empty:
        raise RuntimeError("World Bank aluminum price series is required to build features")

    monthly = pd.DataFrame({"aluminum_price_usd_tonne": price})
    monthly.index = pd.DatetimeIndex(monthly.index)

    if lme_prices is not None and not lme_prices.empty:
        monthly["lme_aluminum_cash"] = _monthly_last(lme_prices, "lme_aluminum_cash").reindex(monthly.index)
        monthly["lme_aluminum_3m"] = _monthly_last(lme_prices, "lme_aluminum_3m").reindex(monthly.index)

    power_proxy = pd.Series(dtype=float)
    if eia_power_proxy is not None and not eia_power_proxy.empty:
        power_proxy = _monthly_last(eia_power_proxy, "value")
        monthly["power_proxy"] = power_proxy.reindex(monthly.index)

    inventory_parts: list[pd.Series] = []
    shfe_monthly = pd.Series(dtype=float)
    lme_stock_monthly = pd.Series(dtype=float)
    if shfe_inventory is not None and not shfe_inventory.empty:
        shfe_monthly = _monthly_sum_then_last(shfe_inventory, "inventory_tonnes")
        inventory_parts.append(shfe_monthly)
    if lme_stocks is not None and not lme_stocks.empty:
        lme_stock_monthly = _monthly_sum_then_last(lme_stocks, "stock_tonnes")
        inventory_parts.append(lme_stock_monthly)

    if inventory_parts:
        inventory = inventory_parts[0].copy()
        for part in inventory_parts[1:]:
            inventory = inventory.add(part, fill_value=0.0)
        monthly["total_inventory_tonnes"] = inventory.reindex(monthly.index)

    returns_1m_raw = monthly["aluminum_price_usd_tonne"].pct_change()
    raw_features = pd.DataFrame(index=monthly.index)
    raw_features["aluminum_return_1m"] = returns_1m_raw
    raw_features["aluminum_return_3m"] = monthly["aluminum_price_usd_tonne"].pct_change(3)
    raw_features["aluminum_return_6m"] = monthly["aluminum_price_usd_tonne"].pct_change(6)
    raw_features["aluminum_momentum_12m"] = monthly["aluminum_price_usd_tonne"].pct_change(12)
    raw_features["aluminum_volatility_6m"] = returns_1m_raw.rolling(6, min_periods=3).std() * np.sqrt(12.0)

    if "total_inventory_tonnes" in monthly.columns:
        raw_features["inventory_change_1m"] = monthly["total_inventory_tonnes"].pct_change()
        raw_features["inventory_change_3m"] = monthly["total_inventory_tonnes"].pct_change(3)
    else:
        raw_features["inventory_change_1m"] = np.nan
        raw_features["inventory_change_3m"] = np.nan

    if "power_proxy" in monthly.columns:
        raw_features["power_proxy_change_1m"] = monthly["power_proxy"].pct_change()
    else:
        raw_features["power_proxy_change_1m"] = np.nan

    lagged = raw_features.shift(feature_lag_months)
    lagged["month"] = lagged.index.month
    lagged["quarter"] = lagged.index.quarter

    out = monthly.join(lagged[FEATURE_COLUMNS])
    out[TARGET_COLUMN] = monthly["aluminum_price_usd_tonne"].pct_change().shift(-1)

    out["has_world_bank_price"] = monthly["aluminum_price_usd_tonne"].notna()
    out["has_eia_power_proxy"] = _source_flag(monthly.index, power_proxy)
    out["has_shfe_inventory"] = _source_flag(monthly.index, shfe_monthly)
    out["has_lme_price"] = (
        monthly.get("lme_aluminum_cash", pd.Series(index=monthly.index, dtype=float)).notna()
        | monthly.get("lme_aluminum_3m", pd.Series(index=monthly.index, dtype=float)).notna()
    )
    out["has_lme_stock"] = _source_flag(monthly.index, lme_stock_monthly)

    out = out.reset_index(names="date")
    if drop_missing_target:
        out = out.dropna(subset=[TARGET_COLUMN])
    return out.reset_index(drop=True)


def feature_metadata(features: pd.DataFrame) -> pd.DataFrame:
    """Return feature category/source metadata and observed availability."""
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        if feature in PRICE_FEATURES:
            category, source, lagged = "price_technical", "world_bank_pink_sheet", True
        elif feature in FUNDAMENTAL_FEATURES:
            category, source, lagged = "fundamental_inventory", "shfe_or_lme", True
        elif feature in COST_PROXY_FEATURES:
            category, source, lagged = "cost_proxy", "eia_api_v2", True
        else:
            category, source, lagged = "calendar", "calendar", False

        observed = feature in features.columns and features[feature].notna().any()
        rows.append(
            {
                "feature": feature,
                "category": category if observed else "unavailable",
                "source": source,
                "is_lagged": lagged,
                "available": bool(observed),
                "non_null_observations": int(features[feature].notna().sum()) if feature in features.columns else 0,
            }
        )
    return pd.DataFrame(rows)
