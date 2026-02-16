#!/usr/bin/env python3
"""
Unified Quality Screen

Score a set of tickers (or an entire universe) using QMJ-style quality metrics,
ranked against a benchmark universe for cross-sectional z-scoring.

Input modes:
  - Specific tickers:  quality.py AAPL MSFT GOOG
  - A universe file:   quality.py --universe consumer_discretionary

Benchmark (the peer group used for z-scoring):
  - S&P 500 (default): --benchmark sp500
  - A universe file:   --benchmark consumer_discretionary
  - Self (input=bench): --benchmark self

Usage:
    python3 quality.py AAPL MSFT GOOG
    python3 quality.py --universe consumer_discretionary --benchmark self
    python3 quality.py --universe sp400 --benchmark sp500 --out_csv results.csv
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_universe, list_universes, get_sp500_universe, clean_ticker

from quality_single import fetch_raw_metrics, compute_scores, RawMetrics


def _build_universe(
    tickers: List[str],
    benchmark: Optional[str],
) -> tuple[List[str], List[str], str]:
    """
    Build the full scoring universe from input tickers + benchmark.

    Returns:
        (scoring_universe, input_tickers, benchmark_name)
    """
    input_tickers = [clean_ticker(t) for t in tickers]

    if benchmark is None or benchmark.lower() == "self":
        # Use input tickers as their own benchmark
        return list(dict.fromkeys(input_tickers)), input_tickers, "Self"

    if benchmark.lower() == "sp500":
        bench_tickers = get_sp500_universe()
        bench_name = "S&P 500"
    else:
        bench_tickers = load_universe(benchmark)
        bench_name = benchmark

    # Union: input tickers first, then benchmark (deduplicated, order-preserving)
    combined = list(dict.fromkeys(input_tickers + bench_tickers))
    return combined, input_tickers, bench_name


def get_data(
    tickers: List[str],
    benchmark: str = "sp500",
    market: str = "SPY",
    growth_years: int = 5,
    beta_years: float = 3.0,
    progress_callback=None,
) -> dict:
    """
    Score tickers against a benchmark universe.

    Args:
        tickers: list of ticker symbols to score
        benchmark: "sp500", a universe name, "self", or None
        market: market proxy for beta (default SPY)
        growth_years: growth window in years
        beta_years: beta lookback in years
        progress_callback: optional callable(current, total) for progress updates

    Returns:
        dict with keys: results_df, z_metrics_df, failed, benchmark_name,
                        input_count, universe_size, scored_count
    """
    if not tickers:
        return {"error": "No tickers provided"}

    try:
        scoring_universe, input_tickers, benchmark_name = _build_universe(
            tickers, benchmark
        )
    except Exception as e:
        return {"error": f"Failed to build universe: {e}"}

    # Fetch raw metrics for the full scoring universe (parallelized)
    raws: Dict[str, RawMetrics] = {}
    failed = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                fetch_raw_metrics,
                ticker,
                market=market,
                growth_years=growth_years,
                beta_years=beta_years,
            ): ticker
            for ticker in scoring_universe
        }
        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                raws[ticker] = future.result()
            except Exception:
                failed.append(ticker)

            if progress_callback and (i % 10 == 0 or i == len(scoring_universe)):
                progress_callback(i, len(scoring_universe))

    if len(raws) < 3:
        return {
            "error": f"Only {len(raws)} tickers succeeded. Need at least 3 for scoring.",
            "failed": failed,
        }

    # Compute scores across full universe
    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T
    z_metrics, scores = compute_scores(raw_df)

    # Filter to input tickers that succeeded
    scored_inputs = [t for t in input_tickers if t in scores.index]
    if not scored_inputs:
        return {
            "error": "None of the input tickers were successfully scored.",
            "failed": failed,
        }

    # Results for input tickers, sorted by quality
    results_df = scores.loc[scored_inputs].sort_values("quality", ascending=False)
    z_metrics_filtered = z_metrics.loc[scored_inputs]

    # Add percentile ranks (relative to full universe)
    pct = scores.rank(pct=True)
    pct_filtered = pct.loc[scored_inputs]
    results_df = results_df.join(pct_filtered.add_suffix("_pct"))

    return {
        "results_df": results_df,
        "z_metrics_df": z_metrics_filtered,
        "failed": [t for t in failed if t in input_tickers],
        "benchmark_name": benchmark_name,
        "input_count": len(input_tickers),
        "universe_size": len(scoring_universe),
        "scored_count": len(raws),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Score tickers for QMJ-style quality against a benchmark.",
        epilog="Examples:\n"
        "  python3 quality.py AAPL MSFT GOOG\n"
        "  python3 quality.py --universe consumer_discretionary --benchmark self\n"
        "  python3 quality.py --universe sp400 --benchmark sp500 --out_csv results.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("tickers", nargs="*", help="Tickers to score (e.g. AAPL MSFT)")
    ap.add_argument(
        "--universe",
        default="",
        help="Universe name or file to score (alternative to listing tickers)",
    )
    ap.add_argument(
        "--benchmark",
        default="sp500",
        help="Benchmark universe: 'sp500' (default), universe name, or 'self'",
    )
    ap.add_argument(
        "--list-universes",
        action="store_true",
        help="List available universe files and exit",
    )
    ap.add_argument(
        "--market", default="SPY", help="Market proxy for beta (default: SPY)"
    )
    ap.add_argument(
        "--growth_years",
        type=int,
        default=5,
        help="Growth window in years (default: 5)",
    )
    ap.add_argument(
        "--beta_years",
        type=float,
        default=3.0,
        help="Beta lookback in years (default: 3)",
    )
    ap.add_argument(
        "--out_csv", default="", help="Optional path to save full results as CSV"
    )
    args = ap.parse_args()

    if args.list_universes:
        universes = list_universes()
        print(
            "Available universes:",
            ", ".join(universes) if universes else "(none)",
        )
        sys.exit(0)

    # Determine input tickers
    if args.universe:
        if args.universe.lower() == "sp500":
            tickers = get_sp500_universe()
            print(f"Loaded S&P 500 universe ({len(tickers)} tickers)")
        else:
            tickers = load_universe(args.universe)
            print(f"Loaded {len(tickers)} tickers from {args.universe}")
    elif args.tickers:
        tickers = [t.upper().strip() for t in args.tickers]
        print(f"Scoring {len(tickers)} ticker(s): {', '.join(tickers)}")
    else:
        ap.error(
            "Provide tickers as arguments or use --universe. Use --list-universes to see options."
        )

    benchmark = args.benchmark
    print(f"Benchmark: {benchmark}")

    def progress(current, total):
        print(f"  Processed {current}/{total}")

    result = get_data(
        tickers=tickers,
        benchmark=benchmark,
        market=args.market,
        growth_years=args.growth_years,
        beta_years=args.beta_years,
        progress_callback=progress,
    )

    if "error" in result:
        print(f"\nError: {result['error']}", file=sys.stderr)
        if result.get("failed"):
            print(f"Failed tickers: {', '.join(result['failed'])}", file=sys.stderr)
        sys.exit(1)

    results_df = result["results_df"]
    failed = result["failed"]

    print(f"\nScored {len(results_df)} tickers against {result['benchmark_name']} "
          f"({result['scored_count']}/{result['universe_size']} universe tickers succeeded)")

    if failed:
        print(f"Failed input tickers: {', '.join(failed)}")

    # Display ranked results
    print("\n" + "=" * 80)
    print("QUALITY RANKING")
    print("=" * 80)
    print(
        f"{'Rank':<6}{'Ticker':<10}{'Quality':>10}{'Pctl':>8}"
        f"{'Profit':>10}{'Growth':>10}{'Safety':>10}"
    )
    print("-" * 80)

    for rank, (ticker, row) in enumerate(results_df.iterrows(), 1):
        pctl = row.get("quality_pct", 0)
        print(
            f"{rank:<6}{ticker:<10}{row['quality']:>10.3f}{pctl*100:>7.1f}%"
            f"{row['profitability']:>10.3f}{row['growth']:>10.3f}{row['safety']:>10.3f}"
        )

    # Save if requested
    if args.out_csv:
        results_df.to_csv(args.out_csv, index=True)
        print(f"\nWrote results to: {args.out_csv}")


if __name__ == "__main__":
    main()
