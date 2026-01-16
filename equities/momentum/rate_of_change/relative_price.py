#!/usr/bin/env python3
"""
Relative Price Rate of Change Analysis

Computes:
1) Relative price of a ticker vs a benchmark (ticker / benchmark)
2) 1-month (21 trading days) Rate of Change of the relative price
3) 5-day rolling average of that ROC
4) Zero crossings of the 5-day average (sign changes)

Usage:
    python3 relative_price.py AAPL SPY
    python3 relative_price.py MSFT QQQ --years 10
    python3 relative_price.py NVDA SPY --roc-period 21 --avg-period 5

Requirements:
    pip install yfinance pandas

Use --roc-period 42 --avg-period 10
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd


def fetch_prices_yfinance(ticker: str, years: int = 5) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Missing dependency: yfinance. Install with: pip install yfinance")

    end = datetime.now(UTC).date() + timedelta(days=1)
    start = end - timedelta(days=365 * years)

    df = yf.download(
        ticker,
        start=str(start),
        end=str(end),
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for ticker '{ticker}'.")

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].dropna().copy()
    # Squeeze in case of multi-level columns from single ticker download
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def main() -> int:
    p = argparse.ArgumentParser(
        description="Relative price ROC analysis with zero-crossing detection"
    )
    p.add_argument("ticker", help="Ticker symbol to analyze (e.g., AAPL)")
    p.add_argument("benchmark", help="Benchmark ticker (e.g., SPY, QQQ)")
    p.add_argument("--years", type=int, default=5, help="Years of history (default: 5)")
    p.add_argument("--roc-period", type=int, default=42, help="ROC lookback in trading days (default: 42 ~2 months)")
    p.add_argument("--avg-period", type=int, default=10, help="Rolling average period (default: 10 days)")
    args = p.parse_args()

    ticker = args.ticker.strip().upper()
    benchmark = args.benchmark.strip().upper()

    try:
        ticker_prices = fetch_prices_yfinance(ticker, years=args.years)
        benchmark_prices = fetch_prices_yfinance(benchmark, years=args.years)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Align the two series on common dates
    combined = pd.DataFrame({"ticker": ticker_prices, "benchmark": benchmark_prices}).dropna()

    if len(combined) < args.roc_period + args.avg_period:
        print(
            f"Not enough data: need at least {args.roc_period + args.avg_period} trading days, got {len(combined)}.",
            file=sys.stderr,
        )
        return 2

    # Calculate relative price
    combined["relative_price"] = combined["ticker"] / combined["benchmark"]

    # Calculate 1-month ROC of relative price
    combined["roc"] = (combined["relative_price"] / combined["relative_price"].shift(args.roc_period) - 1.0) * 100.0

    # Calculate 5-day rolling average of ROC
    combined["roc_avg"] = combined["roc"].rolling(window=args.avg_period, min_periods=args.avg_period).mean()

    # Get latest values
    latest = combined.dropna().iloc[-1]
    latest_date = combined.dropna().index[-1]

    print(f"Ticker: {ticker}")
    print(f"Benchmark: {benchmark}")
    print(f"As of: {latest_date.date().isoformat()}")
    print(f"Relative Price: {latest['relative_price']:.4f}")
    print(f"{args.roc_period}-day ROC (%): {latest['roc']:.4f}")
    print(f"{args.avg_period}-day avg of ROC (%): {latest['roc_avg']:.4f}")

    # Detect zero crossings in the rolling average
    roc_avg = combined["roc_avg"].dropna()
    if len(roc_avg) < 2:
        print("\nNot enough data to detect zero crossings.")
        return 0

    # Sign change detection
    sign_changes = (roc_avg.shift(1) * roc_avg < 0)
    cross_dates = roc_avg[sign_changes].index

    if len(cross_dates) == 0:
        print(f"\nNo zero crossings found in the {args.avg_period}-day avg ROC within the data range.")
        return 0

    # Get the last 10 crossings (most recent first)
    last_10 = cross_dates[-10:][::-1]

    print(f"\nLast {len(last_10)} Zero Crossings (most recent first):")
    print(f"{'Date':<12} {'Rel Price':>12} {f'{args.avg_period}d Avg ROC':>14} {'Direction':<12}")
    print("-" * 54)

    for date in last_10:
        rel_price = combined.loc[date, "relative_price"]
        avg_val = combined.loc[date, "roc_avg"]
        direction = "Cross Above" if avg_val > 0 else "Cross Below"
        print(f"{date.date().isoformat():<12} {rel_price:>12.4f} {avg_val:>14.4f} {direction:<12}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
