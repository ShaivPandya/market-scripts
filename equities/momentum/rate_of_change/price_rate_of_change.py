#!/usr/bin/env python3
"""
Compute:
1) The 20-trading-day average of the 63-trading-day Rate of Change (ROC) (%), as of the most recent trading day.
2) The most recent date (and close price) when that 20-day average met specified conditions.

ROC_63(%) = (Price / Price.shift(63) - 1) * 100
Avg20_ROC63(%) = rolling mean of ROC_63 over 20 trading days

Usage:
  # Default: Find when avg20(ROC63) was below 1%
  python3 price_rate_of_change.py AAPL

  # Find when avg20(ROC63) was below 2%
  python3 price_rate_of_change.py AAPL --upper 2

  # Find when avg20(ROC63) was between -0.5% and 0.5%
  python3 price_rate_of_change.py AAPL --lower -0.5 --upper 0.5

  # Find when avg20(ROC63) was above 5% (no upper bound)
  python3 price_rate_of_change.py AAPL --lower 5 --upper inf

  # Use more historical data
  python3 price_rate_of_change.py AAPL --years 10

  # Show the last 5 times the condition was met
  python3 price_rate_of_change.py AAPL --last 5

Requirements:
  pip install yfinance pandas
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

    # Prefer Adj Close when present, else Close.
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].dropna().copy()

    # Normalize index to date (keeps it as Timestamp, no timezone issues).
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    p.add_argument("--lower", type=float, default=None, help="Lower bound for condition test (percent). If not specified, no lower bound.")
    p.add_argument("--upper", type=float, default=1.0, help="Upper bound for condition test (percent). Default is 1.0 (finds values <= 1%%).")
    p.add_argument("--years", type=int, default=5, help="How many years of history to download (default: 5)")
    p.add_argument("--last", type=int, default=1, help="Number of most recent times to show when condition was met (default: 1)")
    args = p.parse_args()

    ticker = args.ticker.strip().upper()

    try:
        prices = fetch_prices_yfinance(ticker, years=args.years)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Need at least 63 + 20 data points to get the latest 20-day average.
    min_points = 63 + 20
    if len(prices) < min_points:
        print(
            f"Not enough data for {ticker}: need at least {min_points} trading days, got {len(prices)}.",
            file=sys.stderr,
        )
        return 2

    roc63 = (prices / prices.shift(63) - 1.0) * 100.0
    avg20_roc63 = roc63.rolling(window=20, min_periods=20).mean()

    latest_date = prices.index[-1]
    latest_close = float(prices.iloc[-1].iloc[0]) if isinstance(prices.iloc[-1], pd.Series) else float(prices.iloc[-1])
    latest_avg = float(avg20_roc63.iloc[-1].iloc[0]) if isinstance(avg20_roc63.iloc[-1], pd.Series) else float(avg20_roc63.iloc[-1])

    print(f"Ticker: {ticker}")
    print(f"As of:  {latest_date.date().isoformat()}")
    print(f"Close:  {latest_close:.4f}")
    print(f"20-day avg of 63-day ROC (%): {latest_avg:.6f}")

    # Ensure Series
    avg20_roc63 = avg20_roc63.squeeze()

    # Build condition based on specified bounds
    condition_mask = avg20_roc63.notna()

    if args.lower is not None:
        condition_mask = condition_mask & (avg20_roc63 >= args.lower)

    if args.upper is not None:
        condition_mask = condition_mask & (avg20_roc63 <= args.upper)

    # Build description of condition
    if args.lower is not None and args.upper is not None:
        condition_desc = f"between {args.lower} and {args.upper}"
    elif args.lower is not None:
        condition_desc = f">= {args.lower}"
    elif args.upper is not None:
        condition_desc = f"<= {args.upper}"
    else:
        condition_desc = "any value"

    if not condition_mask.any():
        print(f"Last time avg20(ROC63) {condition_desc} (%): None found in downloaded history.")
        return 0

    # Get all matching indices and take the last N
    matching_indices = avg20_roc63[condition_mask].index
    num_to_show = min(args.last, len(matching_indices))
    last_n_indices = matching_indices[-num_to_show:][::-1]  # Most recent first

    print(f"\nLast {num_to_show} time(s) avg20(ROC63) {condition_desc} (%):")
    for idx in last_n_indices:
        avg_val = avg20_roc63.loc[idx]
        avg_float = float(avg_val.iloc[0]) if isinstance(avg_val, pd.Series) else float(avg_val)
        close_val = prices.loc[idx]
        close_float = float(close_val.iloc[0]) if isinstance(close_val, pd.Series) else float(close_val)
        print(f"  {idx.date().isoformat()} | Close {close_float:.4f} | Avg {avg_float:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
