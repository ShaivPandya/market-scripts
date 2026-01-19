#!/usr/bin/env python3
"""
Momentum Signal Generator for Portfolio Optimization

Generates standardized, clipped momentum signals for each ticker in the portfolio.
Signals are z-scored across the portfolio and can be used by portfolio_optimizer.py
to inform raw target weights.

Momentum metrics (equal-weighted):
1. avg20_roc63: 20-day average of 63-day ROC (absolute price momentum)
2. rel_roc42: 42-day ROC of relative price vs benchmark
3. avg10_rel_roc: 10-day average of relative ROC (smoothed)

Usage:
    python3 signal.py
    python3 signal.py --benchmark QQQ
    python3 signal.py --portfolio /path/to/portfolio.csv
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

# -----------------------------
# Configuration
# -----------------------------
PORTFOLIO_CSV = Path(__file__).parent.parent / "universes" / "portfolio.csv"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_YEARS = 5
CLIP_BOUNDS = (-3.0, 3.0)
MIN_DATA_POINTS = 83  # 63 + 20 days minimum for momentum calculation


# -----------------------------
# Z-Score Utilities
# -----------------------------
def zscore_of_ranks(values: pd.Series) -> pd.Series:
    """
    Convert a cross-sectional vector into z-scores of ranks.
    Missing values remain missing.

    Uses rank-based z-scores which are more robust to outliers than
    direct value z-scores.
    """
    x = values.copy()
    mask = x.notna()
    if mask.sum() < 2:
        return pd.Series(index=x.index, dtype="float64")

    # Rank in ascending order; highest value gets highest rank
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
    """
    Download adjusted close prices for multiple tickers from yfinance.
    Returns DataFrame with tickers as columns, dates as index.
    """
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

    # Handle multi-index columns when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    else:
        # Single ticker case
        prices = df[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.dropna(how="all")


# -----------------------------
# Momentum Metrics
# -----------------------------
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


# -----------------------------
# Signal Computation
# -----------------------------
def compute_momentum_signal(raw_df: pd.DataFrame) -> pd.Series:
    """
    Compute z-scored momentum signal from raw metrics DataFrame.

    Process:
    1. Z-score each metric across portfolio (rank-based)
    2. Average the 3 z-scores per ticker
    3. Final z-score of the composite
    """
    if raw_df.empty:
        return pd.Series(dtype="float64")

    # Z-score each metric across portfolio
    z_metrics = raw_df.apply(zscore_of_ranks, axis=0)

    # Equal-weighted average of z-scores
    composite = z_metrics.mean(axis=1, skipna=True)

    # Final z-score of composite
    return zscore_of_ranks(composite)


def clip_signal(signal: pd.Series, lower: float = -3.0, upper: float = 3.0) -> pd.Series:
    """Clip signal to specified bounds."""
    return signal.clip(lower=lower, upper=upper)


# -----------------------------
# Main Entry Point
# -----------------------------
def generate_portfolio_signals(
    tickers: List[str],
    benchmark: str = DEFAULT_BENCHMARK,
    years: int = DEFAULT_YEARS,
    clip_bounds: Tuple[float, float] = CLIP_BOUNDS,
) -> pd.Series:
    """
    Generate momentum signals for a list of tickers.

    Args:
        tickers: List of ticker symbols
        benchmark: Benchmark ticker for relative momentum (default: SPY)
        years: Years of price history to fetch (default: 5)
        clip_bounds: (lower, upper) bounds for signal clipping (default: (-3, 3))

    Returns:
        pd.Series of clipped z-scored signals indexed by ticker
    """
    if not tickers:
        return pd.Series(dtype="float64")

    # Fetch all prices at once (more efficient than per-ticker)
    all_tickers = list(set(tickers + [benchmark]))
    print(f"Fetching prices for {len(all_tickers)} tickers...")

    try:
        prices = fetch_prices(all_tickers, years=years)
    except Exception as e:
        print(f"[ERROR] Failed to fetch prices: {e}", file=sys.stderr)
        return pd.Series(0.0, index=tickers)

    # Check benchmark exists
    if benchmark not in prices.columns:
        print(f"[ERROR] Benchmark {benchmark} not found in downloaded data", file=sys.stderr)
        return pd.Series(0.0, index=tickers)

    benchmark_prices = prices[benchmark].dropna()

    # Compute momentum metrics for each ticker
    raw_metrics: Dict[str, Dict[str, float]] = {}
    failed_tickers: List[str] = []

    for ticker in tickers:
        if ticker not in prices.columns:
            print(f"[WARN] {ticker}: No price data available", file=sys.stderr)
            failed_tickers.append(ticker)
            continue

        ticker_prices = prices[ticker].dropna()
        metrics = compute_momentum_metrics(ticker_prices, benchmark_prices)

        if metrics is None:
            print(f"[WARN] {ticker}: Insufficient data for momentum calculation", file=sys.stderr)
            failed_tickers.append(ticker)
            continue

        raw_metrics[ticker] = metrics

    if not raw_metrics:
        print("[ERROR] No valid momentum data for any ticker", file=sys.stderr)
        return pd.Series(0.0, index=tickers)

    # Build DataFrame and compute signal
    raw_df = pd.DataFrame(raw_metrics).T
    signal = compute_momentum_signal(raw_df)

    # Clip signal
    signal = clip_signal(signal, *clip_bounds)

    # Reindex to include failed tickers with 0.0 signal
    signal = signal.reindex(tickers, fill_value=0.0)

    return signal


# -----------------------------
# CLI
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate momentum signals for portfolio tickers."
    )
    ap.add_argument(
        "--portfolio",
        default=str(PORTFOLIO_CSV),
        help=f"Path to portfolio CSV (default: {PORTFOLIO_CSV})",
    )
    ap.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help=f"Benchmark ticker for relative momentum (default: {DEFAULT_BENCHMARK})",
    )
    ap.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help=f"Years of price history (default: {DEFAULT_YEARS})",
    )
    ap.add_argument(
        "--clip-lower",
        type=float,
        default=CLIP_BOUNDS[0],
        help=f"Lower bound for signal clipping (default: {CLIP_BOUNDS[0]})",
    )
    ap.add_argument(
        "--clip-upper",
        type=float,
        default=CLIP_BOUNDS[1],
        help=f"Upper bound for signal clipping (default: {CLIP_BOUNDS[1]})",
    )
    ap.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path (default: portfolio_signals.csv in same directory)",
    )
    args = ap.parse_args()

    # Load portfolio metadata
    portfolio_path = Path(args.portfolio)
    if not portfolio_path.exists():
        print(f"[ERROR] Portfolio file not found: {portfolio_path}", file=sys.stderr)
        return 1

    meta = pd.read_csv(portfolio_path)
    meta["direction"] = meta["direction"].fillna("")

    # Filter to tickers with direction (exclude SPY which has no direction)
    active_mask = meta["direction"].str.strip().ne("")
    active_tickers = meta.loc[active_mask, "ticker"].tolist()

    if not active_tickers:
        print("[ERROR] No active tickers found in portfolio (need direction specified)", file=sys.stderr)
        return 1

    print(f"Portfolio: {len(active_tickers)} active tickers")
    print(f"Benchmark: {args.benchmark}")
    print()

    # Generate signals
    signals = generate_portfolio_signals(
        tickers=active_tickers,
        benchmark=args.benchmark,
        years=args.years,
        clip_bounds=(args.clip_lower, args.clip_upper),
    )

    # Build output DataFrame
    output = pd.DataFrame({
        "direction": meta.set_index("ticker").loc[active_tickers, "direction"],
        "signal": signals,
    })
    output.index.name = "ticker"

    # Print results
    print("\n=== Momentum Signals ===")
    print(output.to_string(float_format=lambda x: f"{x: .4f}"))

    # Summary stats
    print(f"\nSignal stats: mean={signals.mean():.4f}, std={signals.std():.4f}, "
          f"min={signals.min():.4f}, max={signals.max():.4f}")

    # Save to CSV
    output_path = Path(args.out_csv) if args.out_csv else (Path(__file__).parent / "portfolio_signals.csv")
    output.to_csv(output_path)
    print(f"\nWrote signals to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
