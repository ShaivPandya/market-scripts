#!/usr/bin/env python3
"""
Single Ticker Composite Scorer

Scores a ticker (or multiple tickers) against the S&P 500 universe using
multi-factor composite signals:
- Quality (33%): Profitability, growth, safety metrics
- Price Momentum (33%): Relative price momentum vs benchmark
- Revenue Momentum (20%): Revenue growth and acceleration
- EPS Momentum (14%): Earnings growth and acceleration

Signals are z-scored across the universe and combined with weights.

Usage:
    python3 ticker_score.py AAPL
    python3 ticker_score.py AAPL,MSFT,GOOGL
    python3 ticker_score.py AAPL --universe nasdaq
    python3 ticker_score.py AAPL --benchmark QQQ
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_universe, list_universes, get_sp500_universe

from signal_fetchers import (
    fetch_price_momentum_batch,
    fetch_quality_batch,
    fetch_eps_momentum_batch,
    fetch_revenue_momentum_batch,
)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_BENCHMARK = "SPY"
DEFAULT_YEARS = 5
CLIP_BOUNDS = (-3.0, 3.0)
DEFAULT_WEIGHTS = {
    'quality': 0.33,
    'price_momentum': 0.33,
    'revenue_momentum': 0.20,
    'eps_momentum': 0.14,
}


# -----------------------------
# Z-Score Utilities
# -----------------------------
def zscore_of_ranks(values: pd.Series) -> pd.Series:
    """
    Convert a cross-sectional vector into z-scores of ranks.
    Missing values remain missing.
    """
    x = values.copy()
    mask = x.notna()
    if mask.sum() < 2:
        return pd.Series(index=x.index, dtype="float64")

    ranks = x[mask].rank(method="average", ascending=True)
    mu = ranks.mean()
    sigma = ranks.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        out = pd.Series(index=x.index, dtype="float64")
        out.loc[mask] = 0.0
        return out
    z = (ranks - mu) / sigma

    out = pd.Series(index=x.index, dtype="float64")
    out.loc[mask] = z
    return out


# -----------------------------
# Price Fetching
# -----------------------------
def fetch_prices(tickers: List[str], years: int = 5) -> pd.DataFrame:
    """Download adjusted close prices for multiple tickers from yfinance."""
    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    start = end - timedelta(days=365 * years)

    df = yf.download(
        tickers,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for tickers: {tickers}")

    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    else:
        prices = df[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.dropna(how="all")


# -----------------------------
# Benchmark Selection
# -----------------------------
def select_benchmark_ticker(ticker: str) -> str:
    """Auto-select benchmark based on ticker metadata."""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.get_info()
        if not info:
            info = yf_ticker.info

        market_cap = info.get("marketCap")
        sector = info.get("sector")
        quote_type = str(info.get("quoteType", "")).lower()
        is_etf = quote_type == "etf"

        if is_etf:
            return "SPY"
        if market_cap is not None and market_cap <= 20_000_000_000:
            return "IWM"
        if sector and "technology" in sector.lower():
            return "QQQ"
    except Exception:
        pass

    return "SPY"


# -----------------------------
# Signal Computation Functions
# -----------------------------
def compute_quality_scores(raw_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute quality z-scores from raw quality metrics.
    Returns (quality_signal, pillar_scores DataFrame).
    """
    if raw_df.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()

    df = raw_df.copy()

    # Orient each metric so that "higher is better"
    oriented = pd.DataFrame(index=df.index)

    for col in ["gpoa", "roe", "roa", "cfoa", "gmar"]:
        if col in df.columns:
            oriented[col] = df[col]
    if "acc_low_is_good" in df.columns:
        oriented["acc"] = df["acc_low_is_good"]

    for col in ["dgpoa", "droe", "droa", "dcfoa", "dgmar"]:
        if col in df.columns:
            oriented[col] = df[col]

    if "beta_low_is_good" in df.columns:
        oriented["bab"] = -df["beta_low_is_good"]
    if "leverage_low_is_good" in df.columns:
        oriented["lev"] = -df["leverage_low_is_good"]
    if "zscore_high_is_good" in df.columns:
        oriented["zscore"] = df["zscore_high_is_good"]
    if "roe_vol_low_is_good" in df.columns:
        oriented["evol"] = -df["roe_vol_low_is_good"]

    if oriented.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()

    z_metrics = oriented.apply(zscore_of_ranks, axis=0)

    def pillar(cols: List[str]) -> pd.Series:
        available = [c for c in cols if c in z_metrics.columns]
        if not available:
            return pd.Series(np.nan, index=z_metrics.index)
        tmp = z_metrics[available].mean(axis=1, skipna=True)
        return zscore_of_ranks(tmp)

    profitability = pillar(["gpoa", "roe", "roa", "cfoa", "gmar", "acc"])
    growth = pillar(["dgpoa", "droe", "droa", "dcfoa", "dgmar"])
    safety = pillar(["bab", "lev", "zscore", "evol"])

    pillars = pd.DataFrame({
        "profitability": profitability,
        "growth": growth,
        "safety": safety,
    }, index=df.index)

    combo = profitability + growth + safety
    quality = zscore_of_ranks(combo)

    return quality, pillars


def compute_price_momentum_scores(raw_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute price momentum z-scores from raw momentum metrics."""
    if raw_df.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()

    metrics = ["avg20_roc63", "rel_roc42", "avg10_rel_roc"]
    available = [m for m in metrics if m in raw_df.columns]

    if not available:
        return pd.Series(dtype="float64"), pd.DataFrame()

    z_metrics = raw_df[available].apply(zscore_of_ranks, axis=0)
    composite = z_metrics.mean(axis=1, skipna=True)

    return zscore_of_ranks(composite), z_metrics


def compute_eps_momentum_scores(raw_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute EPS momentum z-scores from raw EPS metrics."""
    if raw_df.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()

    metrics = ["eps_yoy_change", "eps_cagr", "eps_growth_acceleration"]
    available = [m for m in metrics if m in raw_df.columns]

    if not available:
        return pd.Series(dtype="float64"), pd.DataFrame()

    z_metrics = raw_df[available].apply(zscore_of_ranks, axis=0)
    composite = z_metrics.mean(axis=1, skipna=True)

    return zscore_of_ranks(composite), z_metrics


def compute_revenue_momentum_scores(raw_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute revenue momentum z-scores from raw revenue metrics."""
    if raw_df.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()

    metrics = ["revenue_yoy_change", "revenue_cagr", "revenue_growth_acceleration"]
    available = [m for m in metrics if m in raw_df.columns]

    if not available:
        return pd.Series(dtype="float64"), pd.DataFrame()

    z_metrics = raw_df[available].apply(zscore_of_ranks, axis=0)
    composite = z_metrics.mean(axis=1, skipna=True)

    return zscore_of_ranks(composite), z_metrics


# -----------------------------
# Signal Combination
# -----------------------------
def combine_signals(
    signal_dict: Dict[str, pd.Series],
    weights: Dict[str, float],
    tickers: List[str],
) -> pd.Series:
    """Weighted combination of signals with dynamic weight adjustment for missing data."""
    signals_df = pd.DataFrame(signal_dict, index=tickers)
    composite = pd.Series(index=tickers, dtype="float64")

    for ticker in tickers:
        ticker_signals = signals_df.loc[ticker]
        available = ticker_signals.dropna()

        if available.empty:
            composite[ticker] = 0.0
            continue

        available_weights = {k: weights[k] for k in available.index if k in weights}
        if not available_weights:
            composite[ticker] = 0.0
            continue

        weight_sum = sum(available_weights.values())
        normalized_weights = {k: v / weight_sum for k, v in available_weights.items()}

        weighted_sum = sum(
            normalized_weights[k] * available[k]
            for k in normalized_weights.keys()
        )
        composite[ticker] = weighted_sum

    return zscore_of_ranks(composite)


def clip_signal(signal: pd.Series, lower: float = -3.0, upper: float = 3.0) -> pd.Series:
    """Clip signal to specified bounds."""
    return signal.clip(lower=lower, upper=upper)


# -----------------------------
# Display Functions
# -----------------------------
def display_ticker_scores(
    ticker: str,
    composite: float,
    composite_pct: float,
    quality: float,
    quality_pct: float,
    quality_pillars: pd.Series,
    price_mom: float,
    price_mom_pct: float,
    price_z_metrics: pd.Series,
    eps_mom: float,
    eps_mom_pct: float,
    eps_z_metrics: pd.Series,
    rev_mom: float,
    rev_mom_pct: float,
    rev_z_metrics: pd.Series,
    weights: Dict[str, float],
) -> None:
    """Display detailed scores for a single ticker."""
    print(f"\n{'='*50}")
    print(f"  {ticker} Composite Score")
    print(f"{'='*50}")

    if pd.notna(composite):
        print(f"Composite:  {composite:+.2f} z-score | {composite_pct:.0f}th percentile")
    else:
        print("Composite:  N/A")

    # Quality
    print(f"\nQuality:        {quality:+.2f} (weight: {weights['quality']:.0%}) | {quality_pct:.0f}th pctl" if pd.notna(quality) else "\nQuality:        N/A")
    if pd.notna(quality) and not quality_pillars.empty:
        for pillar, val in quality_pillars.items():
            if pd.notna(val):
                print(f"  {pillar:14s}: {val:+.2f}")

    # Price Momentum
    print(f"\nPrice Momentum: {price_mom:+.2f} (weight: {weights['price_momentum']:.0%}) | {price_mom_pct:.0f}th pctl" if pd.notna(price_mom) else "\nPrice Momentum: N/A")
    if pd.notna(price_mom) and not price_z_metrics.empty:
        for metric, val in price_z_metrics.items():
            if pd.notna(val):
                print(f"  {metric:14s}: {val:+.2f}")

    # EPS Momentum
    print(f"\nEPS Momentum:   {eps_mom:+.2f} (weight: {weights['eps_momentum']:.0%}) | {eps_mom_pct:.0f}th pctl" if pd.notna(eps_mom) else "\nEPS Momentum:   N/A")
    if pd.notna(eps_mom) and not eps_z_metrics.empty:
        for metric, val in eps_z_metrics.items():
            if pd.notna(val):
                print(f"  {metric:14s}: {val:+.2f}")

    # Revenue Momentum
    print(f"\nRev Momentum:   {rev_mom:+.2f} (weight: {weights['revenue_momentum']:.0%}) | {rev_mom_pct:.0f}th pctl" if pd.notna(rev_mom) else "\nRev Momentum:   N/A")
    if pd.notna(rev_mom) and not rev_z_metrics.empty:
        for metric, val in rev_z_metrics.items():
            if pd.notna(val):
                print(f"  {metric:14s}: {val:+.2f}")


# -----------------------------
# Main Function
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score ticker(s) against S&P 500 universe using multi-factor composite signals."
    )
    ap.add_argument(
        "tickers",
        help="Ticker or comma-separated tickers to score (e.g., AAPL or AAPL,MSFT,GOOGL)",
    )
    ap.add_argument(
        "--universe",
        default="sp500",
        help="Universe for comparison: 'sp500', universe name, or path (default: sp500)",
    )
    ap.add_argument(
        "--list-universes",
        action="store_true",
        help="List available universe files and exit",
    )
    ap.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark ticker override for price momentum (auto-selects if not specified)",
    )
    ap.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help=f"Years of history (default: {DEFAULT_YEARS})",
    )
    ap.add_argument(
        "--max-universe",
        type=int,
        default=0,
        help="If >0, limit universe size (for faster testing)",
    )
    args = ap.parse_args()

    if args.list_universes:
        universes = list_universes()
        print("Available universes:", ", ".join(universes) if universes else "(none)")
        return 0

    # Parse target tickers
    target_tickers = [t.strip().upper() for t in args.tickers.split(',')]

    # Load universe
    print(f"Loading universe: {args.universe}...")
    if args.universe.lower() == "sp500":
        universe = get_sp500_universe()
    else:
        universe = load_universe(args.universe)

    # Ensure target tickers are in universe
    for ticker in target_tickers:
        if ticker not in universe:
            universe = [ticker] + universe

    # Optional trim for testing
    if args.max_universe and args.max_universe > 0:
        # Keep targets at front
        universe = target_tickers + [t for t in universe if t not in target_tickers]
        universe = universe[:args.max_universe]

    print(f"Universe size: {len(universe)} tickers")
    print(f"Target ticker(s): {', '.join(target_tickers)}")

    weights = DEFAULT_WEIGHTS.copy()
    print(f"Weights: Quality={weights['quality']:.0%}, Price={weights['price_momentum']:.0%}, "
          f"Revenue={weights['revenue_momentum']:.0%}, EPS={weights['eps_momentum']:.0%}")

    # Determine benchmarks
    ticker_benchmarks: Dict[str, str] = {}
    if args.benchmark:
        print(f"\nUsing benchmark override: {args.benchmark}")
        for ticker in universe:
            ticker_benchmarks[ticker] = args.benchmark
    else:
        print("\nAuto-selecting benchmarks for targets...")
        for ticker in target_tickers:
            benchmark = select_benchmark_ticker(ticker)
            ticker_benchmarks[ticker] = benchmark
            print(f"  {ticker} -> {benchmark}")
        # Use SPY for rest of universe
        for ticker in universe:
            if ticker not in ticker_benchmarks:
                ticker_benchmarks[ticker] = "SPY"

    # Fetch prices for all tickers + benchmarks
    unique_benchmarks = set(ticker_benchmarks.values())
    all_price_tickers = list(set(universe) | unique_benchmarks)
    print(f"\nFetching prices for {len(all_price_tickers)} tickers...")

    try:
        prices = fetch_prices(all_price_tickers, years=args.years)
    except Exception as e:
        print(f"[ERROR] Failed to fetch prices: {e}", file=sys.stderr)
        return 1

    # Fetch all metrics
    print("\nFetching quality metrics...")
    quality_raw = fetch_quality_batch(universe, market="SPY", growth_years=args.years)

    print("\nFetching EPS momentum metrics...")
    eps_raw = fetch_eps_momentum_batch(universe, growth_years=args.years)

    print("\nFetching revenue momentum metrics...")
    rev_raw = fetch_revenue_momentum_batch(universe, growth_years=args.years)

    print("\nComputing price momentum...")
    price_raw = fetch_price_momentum_batch(universe, ticker_benchmarks, prices)

    # Compute scores
    print("\nComputing z-scores...")
    quality_signal, quality_pillars_df = compute_quality_scores(quality_raw)
    price_signal, price_z_df = compute_price_momentum_scores(price_raw)
    eps_signal, eps_z_df = compute_eps_momentum_scores(eps_raw)
    rev_signal, rev_z_df = compute_revenue_momentum_scores(rev_raw)

    # Reindex to universe
    quality_signal = quality_signal.reindex(universe)
    price_signal = price_signal.reindex(universe, fill_value=0.0)
    eps_signal = eps_signal.reindex(universe)
    rev_signal = rev_signal.reindex(universe)

    # Combine signals
    signal_dict = {
        'quality': quality_signal,
        'eps_momentum': eps_signal,
        'revenue_momentum': rev_signal,
        'price_momentum': price_signal,
    }
    composite_signal = combine_signals(signal_dict, weights, universe)
    composite_signal = clip_signal(composite_signal, *CLIP_BOUNDS)

    # Compute percentiles
    def percentile(series: pd.Series, ticker: str) -> float:
        if pd.isna(series.get(ticker)):
            return np.nan
        return series.rank(pct=True).get(ticker, np.nan) * 100

    # Display results for each target ticker
    for ticker in target_tickers:
        display_ticker_scores(
            ticker=ticker,
            composite=composite_signal.get(ticker, np.nan),
            composite_pct=percentile(composite_signal, ticker),
            quality=quality_signal.get(ticker, np.nan),
            quality_pct=percentile(quality_signal, ticker),
            quality_pillars=quality_pillars_df.loc[ticker] if ticker in quality_pillars_df.index else pd.Series(),
            price_mom=price_signal.get(ticker, np.nan),
            price_mom_pct=percentile(price_signal, ticker),
            price_z_metrics=price_z_df.loc[ticker] if ticker in price_z_df.index else pd.Series(),
            eps_mom=eps_signal.get(ticker, np.nan),
            eps_mom_pct=percentile(eps_signal, ticker),
            eps_z_metrics=eps_z_df.loc[ticker] if ticker in eps_z_df.index else pd.Series(),
            rev_mom=rev_signal.get(ticker, np.nan),
            rev_mom_pct=percentile(rev_signal, ticker),
            rev_z_metrics=rev_z_df.loc[ticker] if ticker in rev_z_df.index else pd.Series(),
            weights=weights,
        )

    print(f"\n{'='*50}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
