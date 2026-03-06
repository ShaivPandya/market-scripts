#!/usr/bin/env python3
"""
Quality Screen: Find highest and lowest quality stocks in a given universe.

Supported universes:
  sp500       - S&P 500 (fetched from Wikipedia)
  russell2000 - Russell 2000 (from universes/russell2000.csv)
  sp400       - S&P 400 (from universes/sp400.csv)
  xlb         - Materials (SPDR XLB holdings)
  xlc         - Communication Services (SPDR XLC holdings)
  xle         - Energy (SPDR XLE holdings)
  xlf         - Financials (SPDR XLF holdings)
  xli         - Industrials (SPDR XLI holdings)
  xlk         - Technology (SPDR XLK holdings)
  xlp         - Consumer Staples (SPDR XLP holdings)
  xlre        - Real Estate (SPDR XLRE holdings)
  xlu         - Utilities (SPDR XLU holdings)
  xlv         - Health Care (SPDR XLV holdings)
  xly         - Consumer Discretionary (SPDR XLY holdings)
  <file>      - path to CSV/txt file, or any name in universes/

Usage:
    python3 quality_screen.py sp500
    python3 quality_screen.py russell2000
    python3 quality_screen.py xlk
    python3 quality_screen.py sp400 --out_csv results.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Download configuration
CHUNK_SIZE = 50  # Tickers per batch
BATCH_DELAY = 1.0  # Seconds between batches
MAX_WORKERS = 8  # Threads per batch

try:
    import yfinance as yf  # used indirectly by quality_single
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_sp500_universe, get_universe_tickers, list_universes, load_universe
from quality_single import RawMetrics, compute_scores, fetch_raw_metrics

# GICS sectors and their SPDR ETF tickers
SECTOR_ETFS: dict[str, tuple] = {
    "xlb": ("XLB", "Materials"),
    "xlc": ("XLC", "Communication Services"),
    "xle": ("XLE", "Energy"),
    "xlf": ("XLF", "Financials"),
    "xli": ("XLI", "Industrials"),
    "xlk": ("XLK", "Technology"),
    "xlp": ("XLP", "Consumer Staples"),
    "xlre": ("XLRE", "Real Estate"),
    "xlu": ("XLU", "Utilities"),
    "xlv": ("XLV", "Health Care"),
    "xly": ("XLY", "Consumer Discretionary"),
}

_INTL_SUFFIXES = (
    ".HE",
    ".L",
    ".TO",
    ".AX",
    ".PA",
    ".DE",
    ".MI",
    ".AS",
    ".SW",
    ".MC",
    ".SI",
    ".HK",
    ".T",
    ".NS",
    ".BO",
    ".KS",
    ".KQ",
    ".TW",
    ".TWO",
    ".SA",
)


def _clean_ticker(tk: str) -> str:
    tk = str(tk).strip().upper()
    if not tk or tk == "NAN":
        return ""
    if any(tk.endswith(s) for s in _INTL_SUFFIXES):
        return tk
    return tk.replace(".", "-")


def load_screen_universe(name: str) -> tuple[list[str], str]:
    """
    Resolve the user-supplied universe name to a ticker list and display label.

    Returns (tickers, label).
    """
    key = name.strip().lower()

    # Built-in index universes
    if key == "sp500":
        print("Fetching S&P 500 tickers from Wikipedia...")
        tickers = get_sp500_universe()
        return tickers, "S&P 500"

    if key == "russell2000":
        tickers = load_universe("russell2000")
        return tickers, "Russell 2000"

    if key == "sp400":
        tickers = load_universe("sp400")
        return tickers, "S&P 400"

    # GICS sector ETFs
    if key in SECTOR_ETFS:
        etf_ticker, sector_name = SECTOR_ETFS[key]
        print(f"Fetching all holdings for {etf_ticker} ({sector_name})...")
        tickers = get_universe_tickers(etf_ticker)
        if not tickers:
            raise SystemExit(f"Failed to fetch holdings for {etf_ticker}. Cannot proceed.")
        return tickers, f"{sector_name} ({etf_ticker})"

    # Fallback: file path or named universe in universes/
    tickers = load_universe(name)
    return tickers, name


def main():
    ap = argparse.ArgumentParser(
        description="Screen a universe of tickers for highest/lowest quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Universe options:
  sp500, russell2000, sp400
  xlb  xlc  xle  xlf  xli  xlk  xlp  xlre  xlu  xlv  xly  (GICS sector ETFs)
  <path or name>  (CSV/txt file, or named universe from universes/)
        """,
    )
    ap.add_argument(
        "input", nargs="?", help="Universe: sp500, russell2000, sp400, sector ETF (xlk, xle, ...), or file path"
    )
    ap.add_argument("--list-universes", action="store_true", help="List available universe files and exit")
    ap.add_argument("--market", default="SPY", help="Market proxy for beta (default: SPY)")
    ap.add_argument("--growth_years", type=int, default=5, help="Growth window in years (default: 5)")
    ap.add_argument("--beta_years", type=float, default=3.0, help="Beta lookback in years (default: 3)")
    ap.add_argument("--out_csv", default="", help="Optional path to save full results as CSV")
    args = ap.parse_args()

    if args.list_universes:
        universes = list_universes()
        built_in = ["sp500", "russell2000", "sp400"] + sorted(SECTOR_ETFS.keys())
        print("Built-in universes:", ", ".join(built_in))
        print("File-based universes:", ", ".join(universes) if universes else "(none)")
        sys.exit(0)

    if not args.input:
        ap.error("input is required unless using --list-universes")

    # Load universe
    universe, label = load_screen_universe(args.input)
    print(f"Universe: {label} | {len(universe)} tickers")

    if len(universe) < 10:
        print(f"[WARN] Very small universe ({len(universe)} tickers).")

    # Fetch raw metrics for each ticker in batches
    print("\nFetching data for each ticker...")
    raws: dict[str, RawMetrics] = {}
    failed = []

    chunks = [universe[i : i + CHUNK_SIZE] for i in range(0, len(universe), CHUNK_SIZE)]
    total_chunks = len(chunks)
    processed = 0

    for batch_idx, chunk in enumerate(chunks, 1):
        print(f"  Fetching batch {batch_idx}/{total_chunks} ({len(chunk)} tickers)...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    fetch_raw_metrics,
                    ticker,
                    market=args.market,
                    growth_years=args.growth_years,
                    beta_years=args.beta_years,
                ): ticker
                for ticker in chunk
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    raws[ticker] = future.result()
                except Exception as e:
                    failed.append(ticker)
                    print(f"[WARN] {ticker}: failed ({e})", file=sys.stderr)
                processed += 1

        print(f"  Processed {processed}/{len(universe)}")
        if batch_idx < total_chunks:
            time.sleep(BATCH_DELAY)

    if len(raws) < 10:
        raise SystemExit(f"Only {len(raws)} tickers succeeded. Need at least 10 for meaningful ranking.")

    print(f"\nSuccessfully fetched data for {len(raws)}/{len(universe)} tickers")
    if failed:
        print(f"Failed tickers ({len(failed)}): {', '.join(failed)}")

    # Build DataFrame and compute scores
    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T
    z_metrics, scores = compute_scores(raw_df)

    # Sort by quality score descending
    scores_sorted = scores.sort_values("quality", ascending=False)
    n = len(scores_sorted)

    # Display top 10
    print("\n" + "=" * 60)
    print(f"TOP 10 QUALITY  —  {label}")
    print("=" * 60)
    print(f"{'Rank':<6}{'Ticker':<10}{'Quality':>10}{'Profit':>10}{'Growth':>10}{'Safety':>10}")
    print("-" * 60)
    for rank, (ticker, row) in enumerate(scores_sorted.head(10).iterrows(), 1):
        print(
            f"{rank:<6}{ticker:<10}{row['quality']:>10.3f}{row['profitability']:>10.3f}"
            f"{row['growth']:>10.3f}{row['safety']:>10.3f}"
        )

    # Display bottom 10
    print("\n" + "=" * 60)
    print(f"BOTTOM 10 QUALITY  —  {label}")
    print("=" * 60)
    print(f"{'Rank':<6}{'Ticker':<10}{'Quality':>10}{'Profit':>10}{'Growth':>10}{'Safety':>10}")
    print("-" * 60)
    bottom10 = scores_sorted.tail(10).iloc[::-1]
    for rank, (ticker, row) in enumerate(bottom10.iterrows(), n - 9):
        print(
            f"{rank:<6}{ticker:<10}{row['quality']:>10.3f}{row['profitability']:>10.3f}"
            f"{row['growth']:>10.3f}{row['safety']:>10.3f}"
        )

    # Save full results if requested
    if args.out_csv:
        out = raw_df.join(z_metrics.add_prefix("z_")).join(scores)
        out = out.sort_values("quality", ascending=False)
        out.to_csv(args.out_csv, index=True)
        print(f"\nWrote full results to: {args.out_csv}")


if __name__ == "__main__":
    main()
