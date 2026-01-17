#!/usr/bin/env python3
"""
liquidity_model.py

Simple test

Builds a weekly "liquidity" dataset (no backtest/portfolio logic) using public FRED series:
- US balance sheet / plumbing: WALCL, WTREGEN, RRPONTSYD, WRESBAL
- Credit conditions: BAMLC0A0CM (IG OAS), BAMLH0A0HYM2 (HY OAS), NFCI
- Macro liquidity regime: M2SL / GDP
- Global CB balance sheets: ECBASSETSW (weekly), JPNASSETS (monthly)

Outputs:
- CSV with raw series (weekly-aligned), derived changes, rolling z-scores,
  and a composite LiquidityScore (higher = easier).

Install:
  pip install pandas numpy fredapi

Run:
  python3 liquidity_model.py --start 2003-01-01 --output liquidity_signals.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from fredapi import Fred
except ImportError as e:
    raise SystemExit(
        "Missing dependency fredapi. Install with:\n"
        "  pip install fredapi"
    ) from e

# Initialize FRED client with API key from environment
_api_key = os.environ.get("FRED_API_KEY")
if not _api_key:
    raise SystemExit(
        "FRED_API_KEY environment variable not set.\n"
        "Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
    )
fred = Fred(api_key=_api_key)


# ---------------------------
# Configuration
# ---------------------------

@dataclass(frozen=True)
class SeriesSpec:
    fred_id: str
    scale: float = 1.0  # multiply raw values by this to put into desired units
    label: Optional[str] = None


# FRED series (IDs documented in module docstring)
SERIES: Dict[str, SeriesSpec] = {
    # Credit / conditions
    "ig_oas_pct": SeriesSpec("BAMLC0A0CM", 1.0, "IG OAS (%)"),
    "hy_oas_pct": SeriesSpec("BAMLH0A0HYM2", 1.0, "HY OAS (%)"),
    "nfci": SeriesSpec("NFCI", 1.0, "NFCI (index; +tighter)"),

    # Fed balance sheet / plumbing (mostly in Millions of USD)
    "fed_assets_musd": SeriesSpec("WALCL", 1.0, "Fed Total Assets (Millions USD)"),
    "tga_musd": SeriesSpec("WTREGEN", 1.0, "Treasury General Account (Millions USD)"),
    # RRPONTSYD is in Billions USD (daily); convert to Millions USD
    "on_rrp_musd": SeriesSpec("RRPONTSYD", 1000.0, "ON RRP (Millions USD)"),
    "reserves_musd": SeriesSpec("WRESBAL", 1.0, "Reserve Balances (Millions USD)"),

    # Macro (M2 is monthly, billions USD; GDP is quarterly, billions USD SAAR)
    "m2_busd": SeriesSpec("M2SL", 1.0, "M2 (Billions USD)"),
    "gdp_busd_saar": SeriesSpec("GDP", 1.0, "GDP (Billions USD, SAAR)"),

    # Global CB balance sheets (local currency; standardized later)
    "ecb_assets_meur": SeriesSpec("ECBASSETSW", 1.0, "ECB Total Assets (Millions EUR)"),
    "boj_assets_100m_yen": SeriesSpec("JPNASSETS", 1.0, "BoJ Total Assets (100M JPY)"),
}


# Composite score weights:
# Convention: higher score = easier liquidity.
# Negative contributors: spreads and NFCI (higher = tighter).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "z_netliq_4w": 0.35,
    "z_reserves_4w": 0.10,
    "z_global_cb_4w": 0.15,
    "z_m2_gdp": 0.10,
    "z_hy_oas": 0.15,
    "z_ig_oas": 0.10,
    "z_nfci": 0.05,
}


# ---------------------------
# Helpers
# ---------------------------

def fetch_fred_series(fred_id: str, start: str, end: Optional[str]) -> pd.Series:
    """Fetch a single FRED series via fredapi."""
    s = fred.get_series(fred_id, observation_start=start, observation_end=end)
    s.name = fred_id
    return s


def to_weekly(s: pd.Series, week_ending: str = "W-WED", how: str = "last") -> pd.Series:
    """
    Align to a weekly index (default: week ending Wednesday, matching many Fed series).
    - Daily -> last observation within week
    - Weekly -> coerced to same weekly grid
    - Monthly/Quarterly -> forward-filled to weekly grid after resampling
    """
    s = s.sort_index()

    # If it's already weekly-ish, resample still works.
    if how == "last":
        w = s.resample(week_ending).last()
    elif how == "mean":
        w = s.resample(week_ending).mean()
    else:
        raise ValueError(f"Unsupported resample method: {how}")

    # Forward fill across gaps (needed for monthly/quarterly series after resample)
    return w.ffill()


def rolling_zscore(s: pd.Series, window: int = 104, min_periods: Optional[int] = None) -> pd.Series:
    """
    Rolling z-score: (x - rolling_mean) / rolling_std
    Default window ~ 2 years of weekly data.
    """
    if min_periods is None:
        min_periods = max(26, window // 4)
    m = s.rolling(window=window, min_periods=min_periods).mean()
    sd = s.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (s - m) / sd
    return z


def weighted_sum_with_renorm(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Weighted sum across columns in `weights`, renormalizing weights row-by-row
    when some components are NaN.
    """
    cols = [c for c in weights.keys() if c in df.columns]
    if not cols:
        raise ValueError("None of the weighted columns exist in the DataFrame.")

    w = pd.Series({c: weights[c] for c in cols}, dtype=float)
    x = df[cols].copy()

    # For each row, sum weights for available (non-NaN) components
    avail = ~x.isna()
    wsum = avail.mul(w, axis=1).sum(axis=1)

    # Weighted sum ignoring NaNs
    raw = x.mul(w, axis=1).sum(axis=1, min_count=1)

    # Renormalize where possible
    out = raw / wsum
    out.name = "LiquidityScore"
    return out


# ---------------------------
# Liquidity feature engineering
# ---------------------------

def build_liquidity_dataset(
    start: str,
    end: Optional[str],
    week_ending: str,
    z_window: int,
    output: str,
) -> pd.DataFrame:
    # 1) Fetch
    raw = {}
    for key, spec in SERIES.items():
        s = fetch_fred_series(spec.fred_id, start=start, end=end)
        s = s.astype(float) * float(spec.scale)
        s.name = key
        raw[key] = s

    # 2) Weekly align
    w = pd.DataFrame({k: to_weekly(v, week_ending=week_ending, how="last") for k, v in raw.items()})

    # 3) Core liquidity constructs
    # US "net liquidity" proxy (in Millions USD)
    # Note: common practitioner proxy = Fed assets - TGA - ON RRP
    w["net_liquidity_musd"] = w["fed_assets_musd"] - w["tga_musd"] - w["on_rrp_musd"]

    # M2/GDP ratio (dimensionless) using Billions USD series
    w["m2_gdp_ratio"] = w["m2_busd"] / w["gdp_busd_saar"]

    # 4) Changes (weekly cadence)
    def add_changes(prefix: str, col: str) -> None:
        w[f"{prefix}_1w"] = w[col].diff(1)
        w[f"{prefix}_4w"] = w[col].diff(4)
        w[f"{prefix}_13w"] = w[col].diff(13)

    add_changes("d_netliq", "net_liquidity_musd")
    add_changes("d_reserves", "reserves_musd")
    add_changes("d_fed_assets", "fed_assets_musd")
    add_changes("d_tga", "tga_musd")
    add_changes("d_rrp", "on_rrp_musd")
    add_changes("d_ecb_assets", "ecb_assets_meur")
    add_changes("d_boj_assets", "boj_assets_100m_yen")

    add_changes("d_ig_oas", "ig_oas_pct")
    add_changes("d_hy_oas", "hy_oas_pct")
    add_changes("d_nfci", "nfci")
    add_changes("d_m2gdp", "m2_gdp_ratio")

    # 5) Rolling z-scores (signals)
    # Sign conventions:
    # - spreads/NFCI: higher = tighter => invert in score
    # - net liquidity / CB assets: higher (or rising) = easier
    w["z_netliq_4w"] = rolling_zscore(w["d_netliq_4w"], window=z_window)
    w["z_reserves_4w"] = rolling_zscore(w["d_reserves_4w"], window=z_window)

    # Global CB balance sheet impulse (standardized within each currency series, then averaged)
    z_ecb = rolling_zscore(w["d_ecb_assets_4w"], window=z_window)
    z_boj = rolling_zscore(w["d_boj_assets_4w"], window=z_window)
    w["z_global_cb_4w"] = pd.concat([z_ecb, z_boj], axis=1).mean(axis=1)

    w["z_m2_gdp"] = rolling_zscore(w["m2_gdp_ratio"], window=z_window)

    # Invert "tightness" series so higher means easier
    w["z_hy_oas"] = -rolling_zscore(w["hy_oas_pct"], window=z_window)
    w["z_ig_oas"] = -rolling_zscore(w["ig_oas_pct"], window=z_window)
    w["z_nfci"] = -rolling_zscore(w["nfci"], window=z_window)

    # 6) Composite LiquidityScore + LiquidityImpulse
    w["LiquidityScore"] = weighted_sum_with_renorm(w, DEFAULT_WEIGHTS)
    w["LiquidityImpulse_4w"] = w["LiquidityScore"].diff(4)
    w["LiquidityImpulse_13w"] = w["LiquidityScore"].diff(13)

    # 7) Output
    w.to_csv(output, index=True)

    return w


# ---------------------------
# CLI
# ---------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build weekly liquidity dataset from FRED (no backtest).")
    p.add_argument("--start", default="2003-01-01", help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Default: today/latest available.")
    p.add_argument("--week-ending", default="W-WED",
                   help="Weekly anchor for resampling (pandas offset alias). Default W-WED.")
    p.add_argument("--z-window", type=int, default=104,
                   help="Rolling window (in weeks) for z-scores. Default 104 (~2 years).")
    p.add_argument("--output", default="liquidity_signals.csv", help="Output CSV file path.")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    df = build_liquidity_dataset(
        start=args.start,
        end=args.end,
        week_ending=args.week_ending,
        z_window=args.z_window,
        output=args.output,
    )

    # Minimal console output: latest observation
    latest = df.dropna(subset=["LiquidityScore"]).tail(1)
    if latest.empty:
        print("No LiquidityScore computed yet (insufficient history / missing data).")
        print(f"CSV still written to: {args.output}")
        return

    def last_valid(col: str) -> float:
        """Get most recent non-NaN value for a column."""
        return df[col].dropna().iloc[-1] if df[col].dropna().size > 0 else float("nan")

    dt = latest.index[0].date()
    row = latest.iloc[0]
    print(f"As of {dt}:")
    print(f"  LiquidityScore        : {row['LiquidityScore']:.3f}  (higher = easier)")
    print(f"  LiquidityImpulse_4w   : {row['LiquidityImpulse_4w']:.3f}")
    print(f"  US net liquidity (mn) : {last_valid('net_liquidity_musd'):.0f}")
    print(f"  HY OAS (%)            : {last_valid('hy_oas_pct'):.2f}")
    print(f"  IG OAS (%)            : {last_valid('ig_oas_pct'):.2f}")
    print(f"  NFCI                  : {last_valid('nfci'):.3f}")
    print(f"  M2/GDP                : {last_valid('m2_gdp_ratio'):.3f}")
    print(f"\nWrote full dataset to: {args.output}")


if __name__ == "__main__":
    main()
