#!/usr/bin/env python3
"""
Price Momentum Score (Jegadeesh-Titman style)

Implements:
- 3-Month Momentum: Price return over past 3 months
- 12-1 Momentum: 12-month return excluding most recent month (classic momentum factor)

Key method:
- For each metric: cross-sectional rank -> z-score of ranks
- Momentum: z( 3-month + 12-1 ) across universe

Notes:
- The 12-1 formulation skips the most recent month to avoid short-term reversal effects
- Uses monthly prices for stable cross-sectional comparisons

pip install yfinance pandas numpy lxml html5lib
python3 price_momentum_single.py AAPL --universe sp500
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_universe, list_universes, get_sp500_universe


# ----------------------------
# Utilities
# ----------------------------

def zscore_of_ranks(values: pd.Series) -> pd.Series:
    """
    Convert a cross-sectional vector into z-scores of ranks.
    Missing values remain missing.
    """
    x = values.copy()
    mask = x.notna()
    if mask.sum() < 2:
        return pd.Series(index=x.index, dtype="float64")

    # Rank in ascending order; highest value gets highest rank.
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


@dataclass
class MomentumMetrics:
    mom_3m: float   # 3-month price return
    mom_12_1: float # 12-month return excluding most recent month


def compute_price_momentum(ticker: str) -> MomentumMetrics:
    """
    Fetch price history and compute momentum metrics.

    - mom_3m: Return from T-3m to T (most recent 3 months)
    - mom_12_1: Return from T-12m to T-1m (skip most recent month)
    """
    t = yf.Ticker(ticker)

    # Fetch ~15 months of daily data to ensure we have enough for 12-1 calculation
    hist = t.history(period="15mo", interval="1d", auto_adjust=True)

    if hist is None or hist.empty or len(hist) < 20:
        return MomentumMetrics(mom_3m=np.nan, mom_12_1=np.nan)

    # Resample to monthly (end of month prices)
    monthly = hist["Close"].resample("ME").last().dropna()

    if len(monthly) < 4:
        return MomentumMetrics(mom_3m=np.nan, mom_12_1=np.nan)

    # Current price (most recent month end)
    p_now = monthly.iloc[-1]

    # 3-month momentum: return from T-3 to T
    mom_3m = np.nan
    if len(monthly) >= 4:
        p_3m_ago = monthly.iloc[-4]  # 3 months before current
        if p_3m_ago > 0:
            mom_3m = (p_now - p_3m_ago) / p_3m_ago

    # 12-1 momentum: return from T-12 to T-1 (skip most recent month)
    mom_12_1 = np.nan
    if len(monthly) >= 13:
        p_12m_ago = monthly.iloc[-13]  # 12 months before current
        p_1m_ago = monthly.iloc[-2]     # 1 month before current (skip most recent)
        if p_12m_ago > 0:
            mom_12_1 = (p_1m_ago - p_12m_ago) / p_12m_ago

    return MomentumMetrics(mom_3m=mom_3m, mom_12_1=mom_12_1)


def compute_momentum_scores(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given raw momentum metrics across universe, compute z-scores of ranks.
    Returns:
      - z_metrics: per-metric z-scores
      - scores: component scores + composite momentum score
    """
    df = raw.copy()

    # Z-scores of ranks for each metric
    z_metrics = pd.DataFrame(index=df.index)
    z_metrics["mom_3m"] = zscore_of_ranks(df["mom_3m"])
    z_metrics["mom_12_1"] = zscore_of_ranks(df["mom_12_1"])

    # Composite: average of available z-scores, then z-score again
    combo = z_metrics.mean(axis=1, skipna=True)
    momentum = zscore_of_ranks(combo)

    scores = pd.DataFrame({
        "mom_3m": z_metrics["mom_3m"],
        "mom_12_1": z_metrics["mom_12_1"],
        "momentum": momentum
    }, index=df.index)

    return z_metrics, scores


def main():
    ap = argparse.ArgumentParser(
        description="Compute price momentum score for a ticker relative to a universe."
    )
    ap.add_argument("ticker", nargs="?", help="Ticker to score (e.g., AAPL)")
    ap.add_argument("--universe", default="sp500",
                    help="Universe: 'sp500', universe name, or path to file (txt/csv)")
    ap.add_argument("--list-universes", action="store_true",
                    help="List available universe files and exit")
    ap.add_argument("--max_universe", type=int, default=0,
                    help="If >0, limit universe size (debug/speed)")
    ap.add_argument("--out_csv", default="",
                    help="Optional path to save full universe scores as CSV")
    args = ap.parse_args()

    if args.list_universes:
        universes = list_universes()
        print("Available universes:", ", ".join(universes) if universes else "(none)")
        sys.exit(0)

    if not args.ticker:
        ap.error("ticker is required unless using --list-universes")

    ticker = args.ticker.upper().strip()

    # Load universe
    if args.universe.lower() == "sp500":
        universe = get_sp500_universe()
    else:
        universe = load_universe(args.universe)

    # Ensure target is included
    if ticker not in universe:
        universe = [ticker] + universe

    # Optional trim
    if args.max_universe and args.max_universe > 0:
        # Make sure ticker is still included
        if ticker in universe[:args.max_universe]:
            universe = universe[:args.max_universe]
        else:
            universe = [ticker] + universe[:args.max_universe - 1]

    print(f"Universe size: {len(universe)} | Target: {ticker}")

    # Collect momentum metrics
    raws: Dict[str, MomentumMetrics] = {}
    for i, tk in enumerate(universe, 1):
        try:
            mm = compute_price_momentum(tk)
            raws[tk] = mm
        except Exception as e:
            print(f"[WARN] {tk}: failed ({e})", file=sys.stderr)

        if i % 25 == 0:
            print(f"  processed {i}/{len(universe)}")

    if ticker not in raws:
        raise SystemExit(f"Failed to fetch data for target ticker: {ticker}")

    # Build DataFrame from raw metrics
    raw_df = pd.DataFrame({k: {"mom_3m": v.mom_3m, "mom_12_1": v.mom_12_1}
                           for k, v in raws.items()}).T

    # Compute scores
    z_metrics, scores = compute_momentum_scores(raw_df)

    # Percentiles
    pct = scores.rank(pct=True)

    # Output for ticker
    t_scores = scores.loc[ticker]
    t_pct = pct.loc[ticker]

    print("\n--- Price Momentum Score (relative to universe) ---")
    print(f"Ticker: {ticker}")
    print(f"Momentum z-score: {t_scores['momentum']:.3f} | Percentile: {100*t_pct['momentum']:.1f}%")
    print(f"  3-Month:  {t_scores['mom_3m']:.3f} | {100*t_pct['mom_3m']:.1f}%")
    print(f"  12-1:     {t_scores['mom_12_1']:.3f} | {100*t_pct['mom_12_1']:.1f}%")

    # Show raw returns for context
    raw_ticker = raw_df.loc[ticker]
    print("\n--- Raw Returns ---")
    if not np.isnan(raw_ticker["mom_3m"]):
        print(f"  3-Month return:  {100*raw_ticker['mom_3m']:.1f}%")
    else:
        print("  3-Month return:  N/A")
    if not np.isnan(raw_ticker["mom_12_1"]):
        print(f"  12-1 return:     {100*raw_ticker['mom_12_1']:.1f}%")
    else:
        print("  12-1 return:     N/A")

    # Save full results if requested
    if args.out_csv:
        out = raw_df.join(z_metrics.add_prefix("z_")).join(scores).join(pct.add_prefix("pct_"))
        out.to_csv(args.out_csv, index=True)
        print(f"\nWrote full universe output to: {args.out_csv}")


if __name__ == "__main__":
    main()
