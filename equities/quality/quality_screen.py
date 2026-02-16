#!/usr/bin/env python3
"""
Quality Screen: Find highest and lowest quality stocks in a given universe.

Uses QMJ-style quality scoring from quality.py to rank a set of tickers
and output the top 10 and bottom 10 quality names.

Usage:
    python3 quality_screen.py consumer_discretionary.csv
    python3 quality_screen.py tickers.csv --out_csv results.csv
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_universe, list_universes

from quality_single import fetch_raw_metrics, compute_scores, RawMetrics


def main():
    ap = argparse.ArgumentParser(
        description="Screen a universe of tickers for highest/lowest quality."
    )
    ap.add_argument("input", nargs="?", help="Universe name or path to CSV/txt file with tickers")
    ap.add_argument("--list-universes", action="store_true",
                    help="List available universe files and exit")
    ap.add_argument("--market", default="SPY", help="Market proxy for beta (default: SPY)")
    ap.add_argument("--growth_years", type=int, default=5, help="Growth window in years (default: 5)")
    ap.add_argument("--beta_years", type=float, default=3.0, help="Beta lookback in years (default: 3)")
    ap.add_argument("--out_csv", default="", help="Optional path to save full results as CSV")
    args = ap.parse_args()

    if args.list_universes:
        universes = list_universes()
        print("Available universes:", ", ".join(universes) if universes else "(none)")
        sys.exit(0)

    if not args.input:
        ap.error("input is required unless using --list-universes")

    # Load tickers
    universe = load_universe(args.input)
    print(f"Loaded {len(universe)} tickers from {args.input}")

    if len(universe) < 20:
        print(f"[WARN] Small universe ({len(universe)} tickers). Quality scores are relative rankings,")
        print("       so results may be less meaningful with fewer stocks.")

    # Fetch raw metrics for each ticker
    print("\nFetching data for each ticker...")
    raws: Dict[str, RawMetrics] = {}
    failed = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                fetch_raw_metrics,
                ticker,
                market=args.market,
                growth_years=args.growth_years,
                beta_years=args.beta_years,
            ): ticker
            for ticker in universe
        }
        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                raws[ticker] = future.result()
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
    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T
    z_metrics, scores = compute_scores(raw_df)

    # Sort by quality score
    scores_sorted = scores.sort_values("quality", ascending=False)

    # Display top 10
    print("\n" + "=" * 60)
    print("TOP 10 QUALITY")
    print("=" * 60)
    print(f"{'Rank':<6}{'Ticker':<10}{'Quality':>10}{'Profit':>10}{'Growth':>10}{'Safety':>10}")
    print("-" * 60)

    top10 = scores_sorted.head(10)
    for rank, (ticker, row) in enumerate(top10.iterrows(), 1):
        print(f"{rank:<6}{ticker:<10}{row['quality']:>10.3f}{row['profitability']:>10.3f}"
              f"{row['growth']:>10.3f}{row['safety']:>10.3f}")

    # Display bottom 10
    print("\n" + "=" * 60)
    print("BOTTOM 10 QUALITY")
    print("=" * 60)
    print(f"{'Rank':<6}{'Ticker':<10}{'Quality':>10}{'Profit':>10}{'Growth':>10}{'Safety':>10}")
    print("-" * 60)

    bottom10 = scores_sorted.tail(10).iloc[::-1]  # Reverse to show worst first
    for rank, (ticker, row) in enumerate(bottom10.iterrows(), len(raws) - 9):
        print(f"{rank:<6}{ticker:<10}{row['quality']:>10.3f}{row['profitability']:>10.3f}"
              f"{row['growth']:>10.3f}{row['safety']:>10.3f}")

    # Save full results if requested
    if args.out_csv:
        out = raw_df.join(z_metrics.add_prefix("z_")).join(scores)
        out = out.sort_values("quality", ascending=False)
        out.to_csv(args.out_csv, index=True)
        print(f"\nWrote full results to: {args.out_csv}")


if __name__ == "__main__":
    main()
