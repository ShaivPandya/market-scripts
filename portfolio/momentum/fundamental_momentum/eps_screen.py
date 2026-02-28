#!/usr/bin/env python3
"""
EPS Momentum Screen

Score a set of tickers (or an entire universe) using EPS momentum metrics,
ranked against a benchmark universe for cross-sectional z-scoring.

Input modes:
  - Specific tickers:  eps_screen.py AAPL MSFT GOOG
  - A universe file:   eps_screen.py --universe consumer_discretionary

Benchmark (the peer group used for z-scoring):
  - S&P 500 (default): --benchmark sp500
  - A universe file:   --benchmark consumer_discretionary
  - Self (input=bench): --benchmark self

Usage:
    python3 eps_screen.py AAPL MSFT GOOG
    python3 eps_screen.py --universe consumer_discretionary --benchmark self
    python3 eps_screen.py --universe sp400 --benchmark sp500 --out_csv results.csv
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add portfolio/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import load_universe, list_universes, get_sp500_universe, clean_ticker

from eps_momentum_single import fetch_eps_metrics, compute_universe_scores, EPSMetrics


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
        return list(dict.fromkeys(input_tickers)), input_tickers, "Self"

    if benchmark.lower() == "sp500":
        bench_tickers = get_sp500_universe()
        bench_name = "S&P 500"
    else:
        bench_tickers = load_universe(benchmark)
        bench_name = benchmark

    combined = list(dict.fromkeys(input_tickers + bench_tickers))
    return combined, input_tickers, bench_name


def get_data(
    tickers: List[str],
    benchmark: str = "sp500",
    growth_years: int = 3,
    progress_callback=None,
) -> dict:
    """
    Score tickers against a benchmark universe using EPS momentum metrics.

    Args:
        tickers: list of ticker symbols to score
        benchmark: "sp500", a universe name, "self", or None
        growth_years: EPS CAGR window in years
        progress_callback: optional callable(current, total) for progress updates

    Returns:
        dict with keys: results_df, z_metrics_df, raw_metrics_df, failed,
                        benchmark_name, input_count, universe_size, scored_count
    """
    if not tickers:
        return {"error": "No tickers provided"}

    try:
        scoring_universe, input_tickers, benchmark_name = _build_universe(
            tickers, benchmark
        )
    except Exception as e:
        return {"error": f"Failed to build universe: {e}"}

    raws: Dict[str, EPSMetrics] = {}
    failed = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(fetch_eps_metrics, ticker, growth_years=growth_years): ticker
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

    # Build numeric-only DataFrame for scoring
    numeric_cols = ["eps_yoy_change", "eps_cagr", "eps_growth_acceleration"]
    raw_df = pd.DataFrame(
        {
            ticker: {col: getattr(m, col) for col in numeric_cols}
            for ticker, m in raws.items()
        }
    ).T

    z_metrics, score = compute_universe_scores(raw_df)

    # Filter to input tickers that succeeded
    scored_inputs = [t for t in input_tickers if t in score.index]
    if not scored_inputs:
        return {
            "error": "None of the input tickers were successfully scored.",
            "failed": failed,
        }

    # Results for input tickers, sorted by EPS momentum score
    results_df = score.loc[scored_inputs].rename("eps_momentum_z").to_frame()
    results_df = results_df.sort_values("eps_momentum_z", ascending=False)

    z_metrics_filtered = z_metrics.loc[scored_inputs]

    # Add percentile ranks (relative to full universe)
    pct = score.rank(pct=True)
    results_df["eps_momentum_pct"] = pct.loc[scored_inputs]

    # Raw metrics for input tickers
    raw_metrics_df = raw_df.loc[[t for t in scored_inputs if t in raw_df.index]]

    return {
        "results_df": results_df,
        "z_metrics_df": z_metrics_filtered,
        "raw_metrics_df": raw_metrics_df,
        "failed": [t for t in failed if t in input_tickers],
        "benchmark_name": benchmark_name,
        "input_count": len(input_tickers),
        "universe_size": len(scoring_universe),
        "scored_count": len(raws),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Score tickers for EPS momentum against a benchmark.",
        epilog="Examples:\n"
        "  python3 eps_screen.py AAPL MSFT GOOG\n"
        "  python3 eps_screen.py --universe consumer_discretionary --benchmark self\n"
        "  python3 eps_screen.py --universe sp400 --benchmark sp500 --out_csv results.csv",
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
        help="EPS CAGR window in years (default: 3)",
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
        growth_years=args.growth_years,
        progress_callback=progress,
    )

    if "error" in result:
        print(f"\nError: {result['error']}", file=sys.stderr)
        if result.get("failed"):
            print(f"Failed tickers: {', '.join(result['failed'])}", file=sys.stderr)
        sys.exit(1)

    results_df = result["results_df"]
    z_metrics_df = result["z_metrics_df"]
    raw_metrics_df = result["raw_metrics_df"]
    failed = result["failed"]

    print(
        f"\nScored {len(results_df)} tickers against {result['benchmark_name']} "
        f"({result['scored_count']}/{result['universe_size']} universe tickers succeeded)"
    )

    if failed:
        print(f"Failed input tickers: {', '.join(failed)}")

    print("\n" + "=" * 90)
    print("EPS MOMENTUM RANKING")
    print("=" * 90)
    print(
        f"{'Rank':<6}{'Ticker':<10}{'EPS Mom z':>10}{'Pctl':>8}"
        f"{'EPS YoY':>12}{'EPS CAGR':>12}{'Accel':>10}"
    )
    print("-" * 90)

    for rank, (ticker, row) in enumerate(results_df.iterrows(), 1):
        pctl = row.get("eps_momentum_pct", float("nan"))
        pctl_str = f"{pctl * 100:.1f}%" if not pd.isna(pctl) else "NA"

        yoy = raw_metrics_df.loc[ticker, "eps_yoy_change"] if ticker in raw_metrics_df.index else float("nan")
        cagr = raw_metrics_df.loc[ticker, "eps_cagr"] if ticker in raw_metrics_df.index else float("nan")
        accel = raw_metrics_df.loc[ticker, "eps_growth_acceleration"] if ticker in raw_metrics_df.index else float("nan")

        yoy_str = f"{yoy * 100:.2f}%" if not pd.isna(yoy) else "NA"
        cagr_str = f"{cagr * 100:.2f}%" if not pd.isna(cagr) else "NA"
        accel_str = f"{accel:.3f}" if not pd.isna(accel) else "NA"

        print(
            f"{rank:<6}{ticker:<10}{row['eps_momentum_z']:>10.3f}{pctl_str:>8}"
            f"{yoy_str:>12}{cagr_str:>12}{accel_str:>10}"
        )

    if args.out_csv:
        out = results_df.join(z_metrics_df.add_prefix("z_")).join(
            raw_metrics_df.add_prefix("raw_")
        )
        out.to_csv(args.out_csv, index=True)
        print(f"\nWrote results to: {args.out_csv}")


if __name__ == "__main__":
    main()
