#!/usr/bin/env python3
"""
Combined Quality + Momentum Screen

Ranks stocks by combining QMJ-style quality scores with Jegadeesh-Titman style
momentum scores. Outputs top 10 and bottom 10 ranked stocks.

Methodology:
    1. Quality scoring (QMJ-style):
       - Profitability: ROE, ROA, gross profit margin, cash flow
       - Growth: 5-year growth in profitability metrics
       - Safety: Low volatility, low beta, low leverage
       - Converts raw metrics to z-scores, then ranks-to-z-scores for composite

    2. Price momentum scoring (Jegadeesh-Titman):
       - Short-term: 3-month price momentum
       - Medium-term: 12-month price momentum skipping most recent month (12-1)
       - Converts to z-scores, then ranks-to-z-scores for composite

    3. Revenue momentum scoring:
       - Revenue YoY change: most recent quarter vs same quarter 1 year ago
       - Revenue CAGR: compound annual growth rate over ~5 years
       - Revenue growth acceleration: slope of QoQ growth rates (second derivative)
       - Converts to z-scores, then ranks-to-z-scores for composite

    4. Combined scoring:
       - Weighted average of quality, price momentum, and revenue momentum scores
       - Default weights: Quality 50%, Price Momentum 25%, Revenue Momentum 25%
       - Final score converted to z-score of ranks for cross-sectional comparison
       - Higher scores indicate stocks with strong fundamentals, price trends, and revenue growth

The z-score of ranks approach ensures each factor contributes equally regardless of
the distribution of raw values, and makes scores comparable across different universes.

Usage:
    cd equities
    python3 combined/quality_momentum_screen.py momentum/consumer_discretionary.csv
    python3 combined/quality_momentum_screen.py momentum/consumer_discretionary.csv --quality-weight 0.6 --price-mom-weight 0.2 --rev-mom-weight 0.2
    python3 combined/quality_momentum_screen.py tickers.csv --out_csv results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from momentum.price_momentum_single import (
    compute_price_momentum,
    compute_momentum_scores as compute_price_momentum_scores,
    load_universe,
    MomentumMetrics as PriceMomentumMetrics,
)
from momentum.revenue_momentum_single import (
    fetch_revenue_metrics,
    compute_universe_scores as compute_revenue_momentum_scores,
    RevenueMetrics,
)
from quality.quality_single import (
    fetch_raw_metrics as fetch_quality_metrics,
    compute_scores as compute_quality_scores,
    RawMetrics as QualityMetrics,
)


def zscore_of_ranks(values: pd.Series) -> pd.Series:
    """Convert a cross-sectional vector into z-scores of ranks."""
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


def main():
    ap = argparse.ArgumentParser(
        description="Screen a universe for combined quality + momentum ranking."
    )
    ap.add_argument("input", help="Path to CSV/txt file with tickers")
    ap.add_argument(
        "--quality-weight",
        type=float,
        default=0.5,
        help="Weight for quality score (default: 0.5)",
    )
    ap.add_argument(
        "--price-mom-weight",
        type=float,
        default=0.25,
        help="Weight for price momentum score (default: 0.25)",
    )
    ap.add_argument(
        "--rev-mom-weight",
        type=float,
        default=0.25,
        help="Weight for revenue momentum score (default: 0.25)",
    )
    ap.add_argument(
        "--market", default="SPY", help="Market proxy for beta (default: SPY)"
    )
    ap.add_argument(
        "--growth_years", type=int, default=5, help="Growth window in years (default: 5)"
    )
    ap.add_argument(
        "--beta_years", type=float, default=3.0, help="Beta lookback in years (default: 3)"
    )
    ap.add_argument(
        "--out_csv", default="", help="Optional path to save full results as CSV"
    )
    args = ap.parse_args()

    # Normalize weights
    total_weight = args.quality_weight + args.price_mom_weight + args.rev_mom_weight
    if total_weight <= 0:
        raise SystemExit("Weights must sum to a positive number")
    q_weight = args.quality_weight / total_weight
    pm_weight = args.price_mom_weight / total_weight
    rm_weight = args.rev_mom_weight / total_weight

    print(f"Weights: Quality={q_weight:.1%}, Price Momentum={pm_weight:.1%}, Revenue Momentum={rm_weight:.1%}")

    # Load tickers
    universe = load_universe(args.input)
    print(f"Loaded {len(universe)} tickers from {args.input}")

    if len(universe) < 20:
        print(
            f"[WARN] Small universe ({len(universe)} tickers). Scores are relative rankings,"
        )
        print("       so results may be less meaningful with fewer stocks.")

    # Fetch metrics for each ticker
    print("\nFetching data for each ticker...")
    quality_raws: Dict[str, QualityMetrics] = {}
    price_mom_raws: Dict[str, PriceMomentumMetrics] = {}
    rev_mom_raws: Dict[str, RevenueMetrics] = {}
    failed = []

    for i, ticker in enumerate(universe, 1):
        try:
            # Fetch quality metrics
            qm = fetch_quality_metrics(
                ticker,
                market=args.market,
                growth_years=args.growth_years,
                beta_years=args.beta_years,
            )
            quality_raws[ticker] = qm

            # Fetch price momentum metrics
            pm = compute_price_momentum(ticker)
            price_mom_raws[ticker] = pm

            # Fetch revenue momentum metrics
            rm = fetch_revenue_metrics(ticker, growth_years=args.growth_years)
            rev_mom_raws[ticker] = rm

        except Exception as e:
            failed.append(ticker)
            print(f"[WARN] {ticker}: failed ({e})", file=sys.stderr)

        if i % 10 == 0 or i == len(universe):
            print(f"  Processed {i}/{len(universe)}")

    # Only keep tickers that succeeded in all three
    successful = set(quality_raws.keys()) & set(price_mom_raws.keys()) & set(rev_mom_raws.keys())
    if len(successful) < 10:
        raise SystemExit(
            f"Only {len(successful)} tickers succeeded. Need at least 10 for meaningful ranking."
        )

    print(f"\nSuccessfully fetched data for {len(successful)}/{len(universe)} tickers")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")

    # Build DataFrames for scoring
    quality_raw_df = pd.DataFrame(
        {k: vars(v) for k, v in quality_raws.items() if k in successful}
    ).T
    price_mom_raw_df = pd.DataFrame(
        {k: {"mom_3m": v.mom_3m, "mom_12_1": v.mom_12_1}
         for k, v in price_mom_raws.items() if k in successful}
    ).T
    rev_mom_raw_df = pd.DataFrame(
        {k: {
            "revenue_yoy_change": v.revenue_yoy_change,
            "revenue_cagr": v.revenue_cagr,
            "revenue_growth_acceleration": v.revenue_growth_acceleration,
        } for k, v in rev_mom_raws.items() if k in successful}
    ).T

    # Compute individual scores
    quality_z_metrics, quality_scores = compute_quality_scores(quality_raw_df)
    price_mom_z_metrics, price_mom_scores = compute_price_momentum_scores(price_mom_raw_df)
    rev_mom_z_metrics, rev_mom_score = compute_revenue_momentum_scores(rev_mom_raw_df)

    # Combine scores with weights
    combined_raw = (
        q_weight * quality_scores["quality"]
        + pm_weight * price_mom_scores["momentum"]
        + rm_weight * rev_mom_score
    )
    combined_score = zscore_of_ranks(combined_raw)

    # Build final scores DataFrame
    scores = pd.DataFrame(
        {
            "combined": combined_score,
            "quality": quality_scores["quality"],
            "price_mom": price_mom_scores["momentum"],
            "rev_mom": rev_mom_score,
            "profitability": quality_scores["profitability"],
            "growth": quality_scores["growth"],
            "safety": quality_scores["safety"],
            "mom_3m": price_mom_scores["mom_3m"],
            "mom_12_1": price_mom_scores["mom_12_1"],
        }
    )

    # Sort by combined score
    scores_sorted = scores.sort_values("combined", ascending=False)

    # Display top 10
    print("\n" + "=" * 90)
    print(f"TOP 10 COMBINED (Quality {q_weight:.0%} + Price Mom {pm_weight:.0%} + Rev Mom {rm_weight:.0%})")
    print("=" * 90)
    print(
        f"{'Rank':<6}{'Ticker':<10}{'Combined':>10}{'Quality':>10}{'PriceMom':>10}{'RevMom':>10}"
    )
    print("-" * 90)

    top10 = scores_sorted.head(10)
    for rank, (ticker, row) in enumerate(top10.iterrows(), 1):
        print(
            f"{rank:<6}{ticker:<10}{row['combined']:>10.3f}"
            f"{row['quality']:>10.3f}{row['price_mom']:>10.3f}{row['rev_mom']:>10.3f}"
        )

    # Display bottom 10
    print("\n" + "=" * 90)
    print(f"BOTTOM 10 COMBINED (Quality {q_weight:.0%} + Price Mom {pm_weight:.0%} + Rev Mom {rm_weight:.0%})")
    print("=" * 90)
    print(
        f"{'Rank':<6}{'Ticker':<10}{'Combined':>10}{'Quality':>10}{'PriceMom':>10}{'RevMom':>10}"
    )
    print("-" * 90)

    bottom10 = scores_sorted.tail(10).iloc[::-1]  # Reverse to show worst first
    for rank, (ticker, row) in enumerate(bottom10.iterrows(), len(successful) - 9):
        print(
            f"{rank:<6}{ticker:<10}{row['combined']:>10.3f}"
            f"{row['quality']:>10.3f}{row['price_mom']:>10.3f}{row['rev_mom']:>10.3f}"
        )

    # Save full results if requested
    if args.out_csv:
        out = (
            quality_raw_df.add_prefix("q_raw_")
            .join(price_mom_raw_df.add_prefix("pm_raw_"))
            .join(rev_mom_raw_df.add_prefix("rm_raw_"))
            .join(quality_z_metrics.add_prefix("q_z_"))
            .join(price_mom_z_metrics.add_prefix("pm_z_"))
            .join(rev_mom_z_metrics.add_prefix("rm_z_"))
            .join(scores)
        )
        out = out.sort_values("combined", ascending=False)
        out.to_csv(args.out_csv, index=True)
        print(f"\nWrote full results to: {args.out_csv}")


if __name__ == "__main__":
    main()
