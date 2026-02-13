#!/usr/bin/env python3
"""
Batch Signal Fetching Module

Provides batch data fetching functions for:
- Price momentum metrics (relative to benchmark)
- Quality metrics (profitability, growth, safety)
- EPS momentum metrics (YoY change, CAGR, acceleration)
- Revenue momentum metrics (YoY change, CAGR, acceleration)

All functions return DataFrames with tickers as index and metrics as columns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance")

# Import from existing single-ticker scripts
sys.path.insert(0, str(Path(__file__).parent.parent / "quality"))
sys.path.insert(0, str(Path(__file__).parent.parent / "momentum" / "fundamental_momentum"))

from quality_single import fetch_raw_metrics as fetch_quality_raw_metrics, RawMetrics
from eps_momentum_single import fetch_eps_metrics, EPSMetrics
from revenue_momentum_single import fetch_revenue_metrics, RevenueMetrics


# -------------------------
# Price Momentum Utilities
# -------------------------

MIN_DATA_POINTS = 83  # 63 + 20 days minimum for momentum calculation


def safe_div(a: float, b: float) -> float:
    """Safe division that returns NaN on invalid inputs."""
    if a is None or b is None or np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return float(a) / float(b)


def compute_momentum_metrics(
    ticker_prices: pd.Series,
    benchmark_prices: pd.Series,
) -> Optional[Dict[str, float]]:
    """
    Compute momentum metrics for a single ticker relative to benchmark.

    Returns dict with:
        - avg20_roc63: 20-day average of 63-day ROC (%)
        - rel_roc42: 42-day ROC of relative price (%)
        - avg10_rel_roc: 10-day average of relative ROC (%)

    Returns None if insufficient data.
    """
    # Align on common dates
    combined = pd.DataFrame({
        "ticker": ticker_prices,
        "benchmark": benchmark_prices
    }).dropna()

    if len(combined) < MIN_DATA_POINTS:
        return None

    prices = combined["ticker"]
    benchmark = combined["benchmark"]

    # 1. 20-day avg of 63-day ROC (%) - absolute price
    roc63 = (prices / prices.shift(63) - 1.0) * 100.0
    avg20_roc63 = roc63.rolling(window=20, min_periods=20).mean()

    # 2. Relative price calculations
    relative_price = prices / benchmark

    # 42-day ROC of relative price
    rel_roc42 = (relative_price / relative_price.shift(42) - 1.0) * 100.0

    # 3. 10-day avg of relative ROC
    avg10_rel_roc = rel_roc42.rolling(window=10, min_periods=10).mean()

    # Get latest values
    if pd.isna(avg20_roc63.iloc[-1]) or pd.isna(rel_roc42.iloc[-1]) or pd.isna(avg10_rel_roc.iloc[-1]):
        return None

    return {
        "avg20_roc63": float(avg20_roc63.iloc[-1]),
        "rel_roc42": float(rel_roc42.iloc[-1]),
        "avg10_rel_roc": float(avg10_rel_roc.iloc[-1]),
    }


# -------------------------
# Batch Fetching Functions
# -------------------------

def fetch_price_momentum_batch(
    tickers: List[str],
    benchmark_map: Dict[str, str],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute price momentum metrics for multiple tickers in batch.

    Args:
        tickers: List of ticker symbols
        benchmark_map: Dict mapping ticker -> benchmark ticker
        prices: DataFrame with tickers as columns, dates as index (already fetched)

    Returns:
        DataFrame with tickers as index and columns: avg20_roc63, rel_roc42, avg10_rel_roc
    """
    raw_metrics: Dict[str, Dict[str, float]] = {}
    failed_tickers: List[str] = []

    for ticker in tickers:
        if ticker not in prices.columns:
            failed_tickers.append(ticker)
            continue

        ticker_prices = prices[ticker].dropna()
        benchmark = benchmark_map[ticker]

        if benchmark not in prices.columns:
            failed_tickers.append(ticker)
            continue

        benchmark_prices = prices[benchmark].dropna()
        metrics = compute_momentum_metrics(ticker_prices, benchmark_prices)

        if metrics is None:
            failed_tickers.append(ticker)
            continue

        raw_metrics[ticker] = metrics

    if failed_tickers:
        print(f"[WARN] Price momentum failed for: {', '.join(failed_tickers)}", file=sys.stderr)

    return pd.DataFrame(raw_metrics).T if raw_metrics else pd.DataFrame()


def fetch_quality_batch(
    tickers: List[str],
    market: str = "SPY",
    growth_years: int = 5,
    beta_years: float = 3.0,
) -> pd.DataFrame:
    """
    Fetch quality metrics for multiple tickers in batch.

    Args:
        tickers: List of ticker symbols
        market: Market proxy ticker for beta calculation
        growth_years: Target growth window in years
        beta_years: Beta lookback window in years

    Returns:
        DataFrame with tickers as index and 15 columns for quality metrics
    """
    raws: Dict[str, RawMetrics] = {}

    for i, ticker in enumerate(tickers, 1):
        try:
            rm = fetch_quality_raw_metrics(ticker, market, growth_years, beta_years)
            raws[ticker] = rm
        except Exception as e:
            print(f"[WARN] {ticker}: Quality fetch failed ({e})", file=sys.stderr)

        if i % 5 == 0:
            print(f"  Quality: processed {i}/{len(tickers)}")

    if not raws:
        return pd.DataFrame()

    # Convert to DataFrame
    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T
    return raw_df


def fetch_eps_momentum_batch(
    tickers: List[str],
    growth_years: int = 5,
) -> pd.DataFrame:
    """
    Fetch EPS momentum metrics for multiple tickers in batch.

    Args:
        tickers: List of ticker symbols
        growth_years: Target EPS CAGR window in years

    Returns:
        DataFrame with tickers as index and columns:
            eps_yoy_change, eps_cagr, eps_growth_acceleration
    """
    raws: Dict[str, EPSMetrics] = {}

    for i, ticker in enumerate(tickers, 1):
        try:
            rm = fetch_eps_metrics(ticker, growth_years)
            raws[ticker] = rm
        except Exception as e:
            print(f"[WARN] {ticker}: EPS fetch failed ({e})", file=sys.stderr)

        if i % 5 == 0:
            print(f"  EPS: processed {i}/{len(tickers)}")

    if not raws:
        return pd.DataFrame()

    # Convert to DataFrame
    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T
    return raw_df


def fetch_revenue_momentum_batch(
    tickers: List[str],
    growth_years: int = 5,
) -> pd.DataFrame:
    """
    Fetch revenue momentum metrics for multiple tickers in batch.

    Args:
        tickers: List of ticker symbols
        growth_years: Target revenue CAGR window in years

    Returns:
        DataFrame with tickers as index and columns:
            revenue_yoy_change, revenue_cagr, revenue_growth_acceleration
    """
    raws: Dict[str, RevenueMetrics] = {}

    for i, ticker in enumerate(tickers, 1):
        try:
            rm = fetch_revenue_metrics(ticker, growth_years)
            raws[ticker] = rm
        except Exception as e:
            print(f"[WARN] {ticker}: Revenue fetch failed ({e})", file=sys.stderr)

        if i % 5 == 0:
            print(f"  Revenue: processed {i}/{len(tickers)}")

    if not raws:
        return pd.DataFrame()

    # Convert to DataFrame
    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T
    return raw_df
