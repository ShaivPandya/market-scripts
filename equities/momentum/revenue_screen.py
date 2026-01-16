#!/usr/bin/env python3
"""
Revenue Momentum Score + Percentile - Universe Screener

This script calculates revenue momentum scores for all tickers in a specified universe
and displays the top 10 and bottom 10 performers.

=== METRICS ===

1) Quarterly Revenue YoY Change (most recent quarter vs same quarter 1 year ago):
      revenue_yoy_change = (Revenue_q0 - Revenue_q4) / Revenue_q4

   where:
   - q0 = most recent reported quarter
   - q4 = 4 quarters earlier (same quarter last year)
   - Returns NaN if Revenue_q4 is 0 or missing

2) Annual Revenue CAGR (Compound Annual Growth Rate over ~5 fiscal years):
      revenue_cagr = (Revenue_t / Revenue_{t-n}) ** (1/n) - 1

   where:
   - n = min(requested_years, available_years)
   - Only computed when both Revenue_t and Revenue_{t-n} are positive

=== SCORING METHODOLOGY ===

1. For each metric (revenue_yoy_change, revenue_cagr):
   - Compute cross-sectional rank across all tickers
   - Transform ranks into z-scores

2. Revenue Momentum Score:
   - Average the two metric z-scores
   - Compute rank-based z-score of the average (final score)

3. Percentile:
   - rank(pct=True) on the final Revenue Momentum score
   - Higher percentile = stronger revenue momentum

=== OUTPUT ===

The script displays:
- Top 10 tickers by revenue momentum score (highest scores)
- Bottom 10 tickers by revenue momentum score (lowest scores)
- For each ticker: Rank, Ticker, Revenue Momentum Score, Percentile, Revenue YoY %, Revenue CAGR %
- Optional CSV export with full universe results

=== DATA SOURCE ===

yfinance (Yahoo Finance API)
- Statement availability varies by ticker
- Some tickers may fail due to missing or incomplete data

=== USAGE EXAMPLES ===

# Process S&P 500 universe (default)
python3 revenue_screen.py

# Use a custom universe file (txt or csv)
python3 revenue_screen.py --universe tickers.txt

# Save full results to CSV (sorted by score)
python3 revenue_screen.py --universe sp500 --out_csv results.csv

# Use 3-year CAGR instead of 5-year
python3 revenue_screen.py --growth_years 3

# Test with limited universe (first 50 tickers)
python3 revenue_screen.py --max_universe 50

# Combine options
python3 revenue_screen.py --universe sp500 --growth_years 3 --out_csv sp500_revenue_3y.csv

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
except ImportError as e:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance") from e

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_universe, list_universes, get_sp500_universe, clean_ticker


# -------------------------
# Utilities
# -------------------------

def zscore_of_ranks(values: pd.Series) -> pd.Series:
    """Convert a cross-sectional vector into z-scores of ranks. Missing values remain missing."""
    x = values.copy()
    mask = x.notna()
    if mask.sum() < 2:
        return pd.Series(index=x.index, dtype="float64")

    ranks = x[mask].rank(method="average", ascending=True)
    mu = ranks.mean()
    sigma = ranks.std(ddof=0)

    out = pd.Series(index=x.index, dtype="float64")
    if sigma == 0 or np.isnan(sigma):
        out.loc[mask] = 0.0
        return out

    out.loc[mask] = (ranks - mu) / sigma
    return out


def col_at(df: Optional[pd.DataFrame], idx: int) -> Optional[pd.Series]:
    """Return the column at position idx (0 is the most recent)."""
    if df is None or df.empty or df.shape[1] <= idx:
        return None
    return df.iloc[:, idx]


def get_item(s: Optional[pd.Series], keys: List[str]) -> float:
    """Return the first matching line item value from a statement column, or NaN if missing."""
    if s is None or s.empty:
        return np.nan
    for k in keys:
        if k in s.index:
            v = s.loc[k]
            try:
                return float(v)
            except Exception:
                try:
                    return float(pd.to_numeric(v, errors="coerce"))
                except Exception:
                    return np.nan
    return np.nan


def safe_div(a: float, b: float) -> float:
    """Safe division that returns NaN on invalid inputs or division by zero."""
    if a is None or b is None:
        return np.nan
    if b == 0 or np.isnan(b) or np.isinf(b):
        return np.nan
    if np.isnan(a) or np.isinf(a):
        return np.nan
    return float(a) / float(b)


def try_get_income_stmt(t: yf.Ticker, freq: str) -> Optional[pd.DataFrame]:
    """Retrieve an income statement for the requested frequency if available."""
    freq = freq.lower()

    if freq == "annual":
        for attr in ("financials", "income_stmt", "get_income_stmt"):
            if hasattr(t, attr):
                obj = getattr(t, attr)
                try:
                    if callable(obj):
                        return obj(freq="annual")
                    return obj
                except TypeError:
                    try:
                        return obj()
                    except Exception:
                        pass
                except Exception:
                    pass

    if freq == "quarterly":
        for attr in ("quarterly_financials", "quarterly_income_stmt", "get_income_stmt"):
            if hasattr(t, attr):
                obj = getattr(t, attr)
                try:
                    if callable(obj):
                        return obj(freq="quarterly")
                    return obj
                except TypeError:
                    try:
                        return obj()
                    except Exception:
                        pass
                except Exception:
                    pass

    return None


# -------------------------
# Revenue metrics
# -------------------------

REVENUE_KEYS = [
    "Total Revenue", "TotalRevenue", "Revenue",
    "Net Sales", "NetSales", "Sales",
]


@dataclass
class RevenueMetrics:
    revenue_q0: float = np.nan
    revenue_q4: float = np.nan
    revenue_yoy_change: float = np.nan
    revenue_yoy_diff: float = np.nan

    revenue_a0: float = np.nan
    revenue_aN: float = np.nan
    revenue_cagr: float = np.nan
    years_used: int = 0

    q0_end: Optional[pd.Timestamp] = None
    q4_end: Optional[pd.Timestamp] = None
    a0_end: Optional[pd.Timestamp] = None
    aN_end: Optional[pd.Timestamp] = None


def get_revenue_from_stmt(stmt_col: pd.Series) -> float:
    """Extract revenue from an income statement column."""
    return get_item(stmt_col, REVENUE_KEYS)


def fetch_revenue_metrics(ticker: str, growth_years: int = 5) -> RevenueMetrics:
    """Fetch quarterly YoY revenue change and annual revenue CAGR for a ticker."""
    t = yf.Ticker(ticker)

    out = RevenueMetrics()

    q_inc = try_get_income_stmt(t, "quarterly")
    if q_inc is not None and not q_inc.empty and q_inc.shape[1] >= 5:
        out.q0_end = pd.to_datetime(q_inc.columns[0])
        out.q4_end = pd.to_datetime(q_inc.columns[4])

        q0 = col_at(q_inc, 0)
        q4 = col_at(q_inc, 4)

        if q0 is not None and q4 is not None:
            out.revenue_q0 = get_revenue_from_stmt(q0)
            out.revenue_q4 = get_revenue_from_stmt(q4)
            out.revenue_yoy_diff = out.revenue_q0 - out.revenue_q4
            out.revenue_yoy_change = safe_div(out.revenue_yoy_diff, out.revenue_q4)

    a_inc = try_get_income_stmt(t, "annual")
    if a_inc is not None and not a_inc.empty and a_inc.shape[1] >= 2:
        max_years_available = a_inc.shape[1] - 1
        years_used = min(growth_years, max_years_available)

        if years_used >= 1:
            out.years_used = int(years_used)
            out.a0_end = pd.to_datetime(a_inc.columns[0])
            out.aN_end = pd.to_datetime(a_inc.columns[years_used])

            a0 = col_at(a_inc, 0)
            aN = col_at(a_inc, years_used)

            if a0 is not None and aN is not None:
                out.revenue_a0 = get_revenue_from_stmt(a0)
                out.revenue_aN = get_revenue_from_stmt(aN)

                if out.revenue_a0 > 0 and out.revenue_aN > 0:
                    out.revenue_cagr = (out.revenue_a0 / out.revenue_aN) ** (1.0 / years_used) - 1.0
                else:
                    out.revenue_cagr = np.nan

    return out


# -------------------------
# Scoring + reporting
# -------------------------

def compute_universe_scores(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute rank-based z-scores for metrics and a combined Revenue Momentum score.

    Expects raw_df to contain:
      - revenue_yoy_change
      - revenue_cagr
    """
    metrics = ["revenue_yoy_change", "revenue_cagr"]

    z_metrics = pd.DataFrame(index=raw_df.index)
    for m in metrics:
        z_metrics[m] = zscore_of_ranks(raw_df[m])

    combo = z_metrics[metrics].mean(axis=1, skipna=True)
    revenue_momentum_score = zscore_of_ranks(combo)

    return z_metrics, revenue_momentum_score


def fmt_pct(x: float) -> str:
    if x is None or np.isnan(x):
        return "NA"
    return f"{100.0 * x:,.2f}%"


def fmt_num(x: float) -> str:
    if x is None or np.isnan(x):
        return "NA"
    return f"{x:,.3f}"


def main():
    ap = argparse.ArgumentParser(
        description="Compute revenue momentum (quarterly YoY revenue change + ~5y revenue CAGR) and percentile for a universe, showing top 10 and bottom 10."
    )
    ap.add_argument("--universe", default="sp500", help="Universe: 'sp500', universe name, or path to file (txt/csv)")
    ap.add_argument("--list-universes", action="store_true",
                    help="List available universe files and exit")
    ap.add_argument("--growth_years", type=int, default=5, help="Target revenue CAGR window in years")
    ap.add_argument("--max_universe", type=int, default=0, help="If >0, limit universe size")
    ap.add_argument("--out_csv", default="", help="Optional path to save full universe output as CSV")
    args = ap.parse_args()

    if args.list_universes:
        universes = list_universes()
        print("Available universes:", ", ".join(universes) if universes else "(none)")
        sys.exit(0)

    if args.universe.lower() == "sp500":
        universe = get_sp500_universe()
    else:
        universe = load_universe(args.universe)

    if args.max_universe and args.max_universe > 0:
        universe = universe[: args.max_universe]

    print(f"Universe size: {len(universe)}")
    print(f"Fetching revenue data for all tickers...\n")

    raws: Dict[str, RevenueMetrics] = {}
    for i, tk in enumerate(universe, 1):
        try:
            raws[tk] = fetch_revenue_metrics(tk, growth_years=args.growth_years)
        except Exception as e:
            print(f"[WARN] {tk}: failed ({e})", file=sys.stderr)

        if i % 25 == 0:
            print(f"  processed {i}/{len(universe)}")

    if not raws:
        raise SystemExit("Failed to fetch data for any tickers in the universe")

    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T

    z_metrics, score = compute_universe_scores(raw_df)
    pct_score = score.rank(pct=True)

    # Create combined output dataframe
    out = raw_df.copy()
    out["z_revenue_yoy_change"] = z_metrics["revenue_yoy_change"]
    out["z_revenue_cagr"] = z_metrics["revenue_cagr"]
    out["revenue_momentum_z"] = score
    out["revenue_momentum_pct"] = pct_score

    # Sort by revenue momentum score (descending)
    out_sorted = out.sort_values("revenue_momentum_z", ascending=False, na_position="last")

    # Get top 10 and bottom 10
    valid_scores = out_sorted[out_sorted["revenue_momentum_z"].notna()]
    n_valid = len(valid_scores)

    if n_valid == 0:
        raise SystemExit("No valid revenue momentum scores computed")

    top_10 = valid_scores.head(10)
    bottom_10 = valid_scores.tail(10)

    # Display top 10
    print("\n" + "=" * 100)
    print("TOP 10 TICKERS BY REVENUE MOMENTUM SCORE")
    print("=" * 100)
    print(f"{'Rank':<6} {'Ticker':<8} {'Revenue Momentum':<18} {'Percentile':<12} {'Revenue YoY':<14} {'Revenue CAGR':<14}")
    print("-" * 100)

    for i, (ticker, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:<6} {ticker:<8} {row['revenue_momentum_z']:>14.3f}   {row['revenue_momentum_pct']:>10.1%}  {fmt_pct(row['revenue_yoy_change']):>13}  {fmt_pct(row['revenue_cagr']):>13}")

    # Display bottom 10
    print("\n" + "=" * 100)
    print("BOTTOM 10 TICKERS BY REVENUE MOMENTUM SCORE")
    print("=" * 100)
    print(f"{'Rank':<6} {'Ticker':<8} {'Revenue Momentum':<18} {'Percentile':<12} {'Revenue YoY':<14} {'Revenue CAGR':<14}")
    print("-" * 100)

    bottom_rank_start = n_valid - 9
    for i, (ticker, row) in enumerate(bottom_10.iterrows(), bottom_rank_start):
        print(f"{i:<6} {ticker:<8} {row['revenue_momentum_z']:>14.3f}   {row['revenue_momentum_pct']:>10.1%}  {fmt_pct(row['revenue_yoy_change']):>13}  {fmt_pct(row['revenue_cagr']):>13}")

    print("\n" + "=" * 100)
    print(f"Total tickers with valid scores: {n_valid}/{len(universe)}")
    print("=" * 100)

    if args.out_csv:
        out_sorted.to_csv(args.out_csv, index=True)
        print(f"\nWrote full universe output to: {args.out_csv}")


if __name__ == "__main__":
    main()
