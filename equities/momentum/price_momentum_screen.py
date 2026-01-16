#!/usr/bin/env python3
"""
Price Momentum Screen: Find stocks with highest and lowest price momentum in a given universe.

Uses Jegadeesh-Titman style momentum scoring from price_momentum_single.py to rank
a set of tickers and output the top 10 and bottom 10 momentum names.

Usage:
    python3 price_momentum_screen.py consumer_discretionary.csv
    python3 price_momentum_screen.py tickers.csv --out_csv results.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

import pandas as pd

from price_momentum_single import (
    compute_price_momentum,
    compute_momentum_scores,
    load_universe,
    MomentumMetrics,
)


def main():
    ap = argparse.ArgumentParser(
        description="Screen a universe of tickers for highest/lowest price momentum."
    )
    ap.add_argument("input", help="Path to CSV/txt file with tickers")
    ap.add_argument("--out_csv", default="", help="Optional path to save full results as CSV")
    args = ap.parse_args()

    # Load tickers
    universe = load_universe(args.input)
    print(f"Loaded {len(universe)} tickers from {args.input}")

    if len(universe) < 20:
        print(f"[WARN] Small universe ({len(universe)} tickers). Momentum scores are relative rankings,")
        print("       so results may be less meaningful with fewer stocks.")

    # Fetch momentum metrics for each ticker
    print("\nFetching price data for each ticker...")
    raws: Dict[str, MomentumMetrics] = {}
    failed = []

    for i, ticker in enumerate(universe, 1):
        try:
            mm = compute_price_momentum(ticker)
            raws[ticker] = mm
        except Exception as e:
            failed.append(ticker)
            print(f"[WARN] {ticker}: failed ({e})", file=sys.stderr)

        if i % 10 == 0 or i == len(universe):
            print(f"  Processed {i}/{len(universe)}")

    if len(raws) < 10:
        raise SystemExit(f"Only {len(raws)} tickers succeeded. Need at least 10 for meaningful ranking.")

    print(f"\nSuccessfully fetched data for {len(raws)}/{len(universe)} tickers")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")

    # Build DataFrame and compute scores
    raw_df = pd.DataFrame(
        {k: {"mom_3m": v.mom_3m, "mom_12_1": v.mom_12_1} for k, v in raws.items()}
    ).T
    z_metrics, scores = compute_momentum_scores(raw_df)

    # Sort by momentum score
    scores_sorted = scores.sort_values("momentum", ascending=False)

    # Display top 10
    print("\n" + "=" * 50)
    print("TOP 10 MOMENTUM")
    print("=" * 50)
    print(f"{'Rank':<6}{'Ticker':<10}{'Momentum':>10}{'3-Month':>10}{'12-1':>10}")
    print("-" * 50)

    top10 = scores_sorted.head(10)
    for rank, (ticker, row) in enumerate(top10.iterrows(), 1):
        print(f"{rank:<6}{ticker:<10}{row['momentum']:>10.3f}{row['mom_3m']:>10.3f}"
              f"{row['mom_12_1']:>10.3f}")

    # Display bottom 10
    print("\n" + "=" * 50)
    print("BOTTOM 10 MOMENTUM")
    print("=" * 50)
    print(f"{'Rank':<6}{'Ticker':<10}{'Momentum':>10}{'3-Month':>10}{'12-1':>10}")
    print("-" * 50)

    bottom10 = scores_sorted.tail(10).iloc[::-1]  # Reverse to show worst first
    for rank, (ticker, row) in enumerate(bottom10.iterrows(), len(raws) - 9):
        print(f"{rank:<6}{ticker:<10}{row['momentum']:>10.3f}{row['mom_3m']:>10.3f}"
              f"{row['mom_12_1']:>10.3f}")

    # Save full results if requested
    if args.out_csv:
        out = raw_df.join(z_metrics.add_prefix("z_")).join(scores)
        out = out.sort_values("momentum", ascending=False)
        out.to_csv(args.out_csv, index=True)
        print(f"\nWrote full results to: {args.out_csv}")


if __name__ == "__main__":
    main()
