#!/usr/bin/env python3
"""
Revenue Momentum Screen

Score a set of tickers (or an entire universe) using revenue momentum metrics,
ranked against a benchmark universe for cross-sectional z-scoring.

Input modes:
  - Specific tickers:  revenue_screen.py AAPL MSFT GOOG
  - A universe file:   revenue_screen.py --universe consumer_discretionary

Benchmark (the peer group used for z-scoring):
  - S&P 500 (default): --benchmark sp500
  - A universe file:   --benchmark consumer_discretionary
  - Self (input=bench): --benchmark self

Usage:
    python3 revenue_screen.py AAPL MSFT GOOG
    python3 revenue_screen.py --universe consumer_discretionary --benchmark self
    python3 revenue_screen.py --universe sp400 --benchmark sp500 --out_csv results.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional  # noqa: UP035

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Ensure repo modules (equities/common) are importable when running as a script.
_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT / "equities", _ROOT):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from common import (
    clean_ticker,
    get_sp500_universe,
    get_universe_tickers,
    list_universes,
    load_universe,
)
from revenue_momentum_single import RevenueMetrics, compute_universe_scores, fetch_revenue_metrics


def _build_universe(
    tickers: list[str],
    benchmark: str | None,
) -> tuple[list[str], list[str], str]:
    """
    Build the full scoring universe from input tickers + benchmark.

    Returns:
        (scoring_universe, input_tickers, benchmark_name)
    """
    input_tickers = [clean_ticker(t) for t in tickers]

    if benchmark is None or benchmark.lower() == "self":
        return list(dict.fromkeys(input_tickers)), input_tickers, "Self"

    bench_tickers = get_universe_tickers(benchmark)
    if not bench_tickers:
        raise ValueError(f"Benchmark universe '{benchmark}' resolved to 0 tickers")

    bench_name = "S&P 500" if benchmark.lower() == "sp500" else benchmark

    combined = list(dict.fromkeys(input_tickers + bench_tickers))
    return combined, input_tickers, bench_name


def get_data(
    tickers: list[str],
    benchmark: str = "sp500",
    growth_years: int = 3,
    progress_callback=None,
) -> dict:
    """
    Score tickers for revenue momentum against a benchmark universe.

    Args:
        tickers: list of ticker symbols to score
        benchmark: "sp500", a universe name, "self", or None
        growth_years: revenue CAGR window in years
        progress_callback: optional callable(current, total) for progress updates

    Returns:
        dict with keys: results_df, z_metrics_df, raw_metrics_df, failed,
                        benchmark_name, input_count, universe_size, scored_count
    """
    if not tickers:
        return {"error": "No tickers provided"}

    try:
        scoring_universe, input_tickers, benchmark_name = _build_universe(tickers, benchmark)
    except Exception as e:
        return {"error": f"Failed to build universe: {e}"}

    # Fetch raw metrics for the full scoring universe (parallelized)
    raws: dict[str, RevenueMetrics] = {}
    failed = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                fetch_revenue_metrics,
                ticker,
                growth_years=growth_years,
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
    z_metrics, score = compute_universe_scores(raw_df)

    # Filter to input tickers that succeeded
    scored_inputs = [t for t in input_tickers if t in score.index]
    if not scored_inputs:
        return {
            "error": "None of the input tickers were successfully scored.",
            "failed": failed,
        }

    # Results for input tickers, sorted by revenue momentum
    score_df = score.rename("revenue_momentum").to_frame()
    results_df = score_df.loc[scored_inputs].sort_values("revenue_momentum", ascending=False)
    z_metrics_filtered = z_metrics.loc[scored_inputs]

    # Add percentile ranks (relative to full universe)
    pct = score.rank(pct=True).rename("revenue_momentum_pct").to_frame()
    pct_filtered = pct.loc[scored_inputs]
    results_df = results_df.join(pct_filtered)

    # Raw metrics for input tickers
    raw_filtered = raw_df.loc[[t for t in scored_inputs if t in raw_df.index]]

    return {
        "results_df": results_df,
        "z_metrics_df": z_metrics_filtered,
        "raw_metrics_df": raw_filtered,
        "failed": [t for t in failed if t in input_tickers],
        "benchmark_name": benchmark_name,
        "input_count": len(input_tickers),
        "universe_size": len(scoring_universe),
        "scored_count": len(raws),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Score tickers for revenue momentum against a benchmark.",
        epilog="Examples:\n"
        "  python3 revenue_screen.py AAPL MSFT GOOG\n"
        "  python3 revenue_screen.py --universe consumer_discretionary --benchmark self\n"
        "  python3 revenue_screen.py --universe sp400 --benchmark sp500 --out_csv results.csv",
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
        "--growth_years",
        type=int,
        default=3,
        help="Revenue CAGR window in years (default: 3)",
    )
    ap.add_argument("--out_csv", default="", help="Optional path to save full results as CSV")
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
        tickers = get_universe_tickers(args.universe)
        if not tickers:
            ap.error(f"Universe '{args.universe}' resolved to 0 tickers.")
        label = "S&P 500" if args.universe.lower() == "sp500" else args.universe
        print(f"Loaded {label} universe ({len(tickers)} tickers)")
    elif args.tickers:
        tickers = [t.upper().strip() for t in args.tickers]
        print(f"Scoring {len(tickers)} ticker(s): {', '.join(tickers)}")
    else:
        ap.error("Provide tickers as arguments or use --universe. Use --list-universes to see options.")

    benchmark = args.benchmark
    print(f"Benchmark: {benchmark}")

    def progress(current, total):
        print(f"  Processed {current}/{total}")

    result = get_data(
        tickers=tickers,
        benchmark=benchmark,
        growth_years=args.growth_years,
        progress_callback=progress,
    )

    if "error" in result:
        LOGGER.error("%s", result["error"])
        if result.get("failed"):
            LOGGER.error("Failed tickers: %s", ", ".join(result["failed"]))
        sys.exit(1)

    results_df = result["results_df"]
    failed = result["failed"]

    print(
        f"\nScored {len(results_df)} tickers against {result['benchmark_name']} "
        f"({result['scored_count']}/{result['universe_size']} universe tickers succeeded)"
    )

    if failed:
        print(f"Failed input tickers: {', '.join(failed)}")

    # Display ranked results
    print("\n" + "=" * 84)
    print("REVENUE MOMENTUM RANKING")
    print("=" * 84)
    print(f"{'Rank':<6}{'Ticker':<10}{'Rev Mom':>10}{'Pctl':>8}{'Rev YoY':>12}{'Rev CAGR':>12}{'Growth Accel':>14}")
    print("-" * 84)

    z_metrics_df = result["z_metrics_df"]
    raw_df = result["raw_metrics_df"]

    for rank, (ticker, row) in enumerate(results_df.iterrows(), 1):
        pctl = row.get("revenue_momentum_pct", float("nan"))
        rev_yoy = raw_df.loc[ticker, "revenue_yoy_change"] if ticker in raw_df.index else float("nan")
        rev_cagr = raw_df.loc[ticker, "revenue_cagr"] if ticker in raw_df.index else float("nan")
        growth_accel = raw_df.loc[ticker, "revenue_growth_acceleration"] if ticker in raw_df.index else float("nan")

        import math

        pctl_str = f"{pctl * 100:.1f}%" if not math.isnan(pctl) else "NA"
        yoy_str = f"{rev_yoy * 100:.1f}%" if not math.isnan(rev_yoy) else "NA"
        cagr_str = f"{rev_cagr * 100:.1f}%" if not math.isnan(rev_cagr) else "NA"
        accel_str = f"{growth_accel:.3f}" if not math.isnan(growth_accel) else "NA"

        print(
            f"{rank:<6}{ticker:<10}{row['revenue_momentum']:>10.3f}{pctl_str:>8}"
            f"{yoy_str:>12}{cagr_str:>12}{accel_str:>14}"
        )

    # Save if requested
    if args.out_csv:
        out = results_df.join(z_metrics_df.add_prefix("z_")).join(
            raw_df[["revenue_yoy_change", "revenue_cagr", "revenue_growth_acceleration"]]
        )
        out.to_csv(args.out_csv, index=True)
        print(f"\nWrote results to: {args.out_csv}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("Starting script execution: %s", __file__)
    main()
