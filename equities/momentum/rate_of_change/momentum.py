#!/usr/bin/env python3
"""
Momentum ROC Analysis

Computes:
1) 20-day avg of 63-day ROC (%) - absolute price
2) 42-day ROC (%) of relative price to benchmark
3) 10-day avg of ROC (%) of relative price to benchmark

Usage:
    python3 momentum.py AAPL SPY
    python3 momentum.py MSFT QQQ --years 10

Requirements:
    pip install yfinance pandas
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[38;2;52;199;89m"
RESET = "\033[0m"


def colorize(value: float, threshold: float, below_is_red: bool = True) -> str:
    """Return colored string based on threshold comparison."""
    if below_is_red:
        color = RED if value < threshold else GREEN
    else:
        color = GREEN if value < threshold else RED
    return f"{color}{value:.6f}{RESET}"


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

    if isinstance(s, pd.DataFrame):
        s = s.squeeze()

    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="Momentum ROC analysis")
    p.add_argument("ticker", help="Ticker symbol (e.g., AAPL)")
    p.add_argument("benchmark", help="Benchmark ticker (e.g., SPY)")
    p.add_argument("--years", type=int, default=5, help="Years of history to download (default: 5)")
    args = p.parse_args()

    ticker = args.ticker.strip().upper()
    benchmark = args.benchmark.strip().upper()

    try:
        ticker_prices = fetch_prices_yfinance(ticker, years=args.years)
        benchmark_prices = fetch_prices_yfinance(benchmark, years=args.years)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Align on common dates
    combined = pd.DataFrame({"ticker": ticker_prices, "benchmark": benchmark_prices}).dropna()

    # Need enough data: 63 + 20 = 83 days minimum
    min_points = 63 + 20
    if len(combined) < min_points:
        print(
            f"Not enough data: need at least {min_points} trading days, got {len(combined)}.",
            file=sys.stderr,
        )
        return 2

    prices = combined["ticker"]

    # 1. 20-day avg of 63-day ROC (%) - absolute price
    roc63 = (prices / prices.shift(63) - 1.0) * 100.0
    avg20_roc63 = roc63.rolling(window=20, min_periods=20).mean()

    # Relative price calculations
    relative_price = combined["ticker"] / combined["benchmark"]

    # 2. 42-day ROC (%) of relative price
    rel_roc42 = (relative_price / relative_price.shift(42) - 1.0) * 100.0

    # 3. 10-day avg of ROC (%) of relative price
    avg10_rel_roc = rel_roc42.rolling(window=10, min_periods=10).mean()

    # Extract latest values
    latest_date = combined.index[-1]
    latest_close = float(prices.iloc[-1])
    latest_avg20_roc63 = float(avg20_roc63.iloc[-1])
    latest_rel_roc42 = float(rel_roc42.iloc[-1])
    latest_avg10_rel_roc = float(avg10_rel_roc.iloc[-1])

    print(f"Ticker: {ticker}")
    print(f"Benchmark: {benchmark}")
    print(f"As of:  {latest_date.date().isoformat()}")
    print(f"Close:  {latest_close:.4f}")
    print()
    print(f"20-day avg of 63-day ROC (%):        {colorize(latest_avg20_roc63, 1.5)}")
    print(f"42-day ROC of relative price (%):   {colorize(latest_rel_roc42, 0)}")
    print(f"10-day avg of relative ROC (%):     {colorize(latest_avg10_rel_roc, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
