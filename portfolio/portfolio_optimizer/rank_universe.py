#!/usr/bin/env python3
"""
Universe Ranker

Ranks all constituents in a universe by composite score and displays top/bottom performers.

Usage:
    python3 rank_universe.py --universe sp400
    python3 rank_universe.py --universe sp400 --top 20
    python3 rank_universe.py --universe sp500 --output rankings.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_universe, list_universes, get_sp500_universe

from signal_fetchers import (
    fetch_price_momentum_batch,
    fetch_quality_batch,
    fetch_eps_momentum_batch,
    fetch_revenue_momentum_batch,
)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_BENCHMARK = "SPY"
DEFAULT_YEARS = 5
CLIP_BOUNDS = (-3.0, 3.0)
DEFAULT_WEIGHTS = {
    'quality': 0.33,
    'price_momentum': 0.33,
    'revenue_momentum': 0.20,
    'eps_momentum': 0.14,
}


# -----------------------------
# Z-Score Utilities
# -----------------------------
def zscore_of_ranks(values: pd.Series) -> pd.Series:
    """
    Convert a cross-sectional vector into z-scores of ranks.
    Missing values remain missing.
    """
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


# -----------------------------
# Price Fetching
# -----------------------------
def fetch_prices_batch(tickers: List[str], years: int = 5, batch_size: int = 100, delay: float = 1.0) -> pd.DataFrame:
    """
    Download adjusted close prices for multiple tickers from yfinance in batches.

    Args:
        tickers: List of ticker symbols
        years: Number of years of history to fetch
        batch_size: Number of tickers to download per batch
        delay: Seconds to wait between batches

    Returns:
        DataFrame with dates as index and tickers as columns
    """
    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    start = end - timedelta(days=365 * years)

    all_prices = []
    failed_tickers = []

    # Split tickers into batches
    num_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        print(f"  Batch {batch_num}/{num_batches}: Downloading {len(batch)} tickers...")

        try:
            df = yf.download(
                batch,
                start=str(start),
                end=str(end),
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    prices = df["Close"].copy()
                else:
                    # Single ticker case
                    prices = df[["Close"]].copy()
                    prices.columns = [batch[0]]

                prices.index = pd.to_datetime(prices.index).tz_localize(None)
                all_prices.append(prices)
            else:
                failed_tickers.extend(batch)

        except Exception as e:
            print(f"  Warning: Batch {batch_num} failed with error: {e}")
            failed_tickers.extend(batch)

        # Delay between batches to avoid rate limiting (except for last batch)
        if i + batch_size < len(tickers):
            time.sleep(delay)

    if not all_prices:
        raise RuntimeError(f"No price data downloaded for any tickers")

    # Combine all batches
    combined = pd.concat(all_prices, axis=1)

    # Remove duplicate columns if any
    combined = combined.loc[:, ~combined.columns.duplicated()]

    if failed_tickers:
        print(f"  Warning: Failed to download data for {len(failed_tickers)} tickers")

    return combined.dropna(how="all")


def fetch_prices(tickers: List[str], years: int = 5) -> pd.DataFrame:
    """Download adjusted close prices for multiple tickers from yfinance."""
    # For small batches (<=100), use direct download
    if len(tickers) <= 100:
        end = datetime.now(timezone.utc).date() + timedelta(days=1)
        start = end - timedelta(days=365 * years)

        df = yf.download(
            tickers,
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if df is None or df.empty:
            raise RuntimeError(f"No data returned for tickers: {tickers}")

        if isinstance(df.columns, pd.MultiIndex):
            prices = df["Close"].copy()
        else:
            prices = df[["Close"]].copy()
            prices.columns = [tickers[0]]

        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        return prices.dropna(how="all")
    else:
        # For large universes, use batched download
        return fetch_prices_batch(tickers, years=years)


# -----------------------------
# Signal Computation Functions
# -----------------------------
def compute_quality_scores(raw_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute quality z-scores from raw quality metrics.
    Returns (quality_signal, pillar_scores DataFrame).
    """
    if raw_df.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()

    df = raw_df.copy()

    # Orient each metric so that "higher is better"
    oriented = pd.DataFrame(index=df.index)

    for col in ["gpoa", "roe", "roa", "cfoa", "gmar"]:
        if col in df.columns:
            oriented[col] = df[col]
    if "acc_low_is_good" in df.columns:
        oriented["acc"] = df["acc_low_is_good"]

    for col in ["dgpoa", "droe", "droa", "dcfoa", "dgmar"]:
        if col in df.columns:
            oriented[col] = df[col]

    if "beta_low_is_good" in df.columns:
        oriented["bab"] = -df["beta_low_is_good"]
    if "leverage_low_is_good" in df.columns:
        oriented["lev"] = -df["leverage_low_is_good"]
    if "zscore_high_is_good" in df.columns:
        oriented["zscore"] = df["zscore_high_is_good"]
    if "roe_vol_low_is_good" in df.columns:
        oriented["evol"] = -df["roe_vol_low_is_good"]

    if oriented.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()

    z_metrics = oriented.apply(zscore_of_ranks, axis=0)

    def pillar(cols: List[str]) -> pd.Series:
        available = [c for c in cols if c in z_metrics.columns]
        if not available:
            return pd.Series(np.nan, index=z_metrics.index)
        tmp = z_metrics[available].mean(axis=1, skipna=True)
        return zscore_of_ranks(tmp)

    profitability = pillar(["gpoa", "roe", "roa", "cfoa", "gmar", "acc"])
    growth = pillar(["dgpoa", "droe", "droa", "dcfoa", "dgmar"])
    safety = pillar(["bab", "lev", "zscore", "evol"])

    pillars = pd.DataFrame({
        "profitability": profitability,
        "growth": growth,
        "safety": safety,
    }, index=df.index)

    combo = profitability + growth + safety
    quality = zscore_of_ranks(combo)

    return quality, pillars


def compute_price_momentum_scores(raw_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute price momentum z-scores from raw momentum metrics."""
    if raw_df.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()

    metrics = ["avg20_roc63", "rel_roc42", "avg10_rel_roc"]
    available = [m for m in metrics if m in raw_df.columns]

    if not available:
        return pd.Series(dtype="float64"), pd.DataFrame()

    z_metrics = raw_df[available].apply(zscore_of_ranks, axis=0)
    composite = z_metrics.mean(axis=1, skipna=True)

    return zscore_of_ranks(composite), z_metrics


def compute_eps_momentum_scores(raw_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute EPS momentum z-scores from raw EPS metrics."""
    if raw_df.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()

    metrics = ["eps_yoy_change", "eps_cagr", "eps_growth_acceleration"]
    available = [m for m in metrics if m in raw_df.columns]

    if not available:
        return pd.Series(dtype="float64"), pd.DataFrame()

    z_metrics = raw_df[available].apply(zscore_of_ranks, axis=0)
    composite = z_metrics.mean(axis=1, skipna=True)

    return zscore_of_ranks(composite), z_metrics


def compute_revenue_momentum_scores(raw_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute revenue momentum z-scores from raw revenue metrics."""
    if raw_df.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()

    metrics = ["revenue_yoy_change", "revenue_cagr", "revenue_growth_acceleration"]
    available = [m for m in metrics if m in raw_df.columns]

    if not available:
        return pd.Series(dtype="float64"), pd.DataFrame()

    z_metrics = raw_df[available].apply(zscore_of_ranks, axis=0)
    composite = z_metrics.mean(axis=1, skipna=True)

    return zscore_of_ranks(composite), z_metrics


# -----------------------------
# Signal Combination
# -----------------------------
def combine_signals(
    signal_dict: Dict[str, pd.Series],
    weights: Dict[str, float],
    tickers: List[str],
) -> pd.Series:
    """Weighted combination of signals with dynamic weight adjustment for missing data."""
    signals_df = pd.DataFrame(signal_dict, index=tickers)
    composite = pd.Series(index=tickers, dtype="float64")

    for ticker in tickers:
        ticker_signals = signals_df.loc[ticker]
        available = ticker_signals.dropna()

        if available.empty:
            composite[ticker] = 0.0
            continue

        available_weights = {k: weights[k] for k in available.index if k in weights}
        if not available_weights:
            composite[ticker] = 0.0
            continue

        weight_sum = sum(available_weights.values())
        normalized_weights = {k: v / weight_sum for k, v in available_weights.items()}

        weighted_sum = sum(
            normalized_weights[k] * available[k]
            for k in normalized_weights.keys()
        )
        composite[ticker] = weighted_sum

    return zscore_of_ranks(composite)


def clip_signal(signal: pd.Series, lower: float = -3.0, upper: float = 3.0) -> pd.Series:
    """Clip signal to specified bounds."""
    return signal.clip(lower=lower, upper=upper)


# -----------------------------
# Display Functions
# -----------------------------
def display_rankings(
    composite: pd.Series,
    quality: pd.Series,
    price_mom: pd.Series,
    eps_mom: pd.Series,
    rev_mom: pd.Series,
    top_n: int = 10,
) -> None:
    """Display top and bottom ranked tickers."""
    # Create rankings dataframe
    rankings_df = pd.DataFrame({
        'Ticker': composite.index,
        'Composite': composite.values,
        'Quality': quality.reindex(composite.index).values,
        'Price Mom': price_mom.reindex(composite.index).values,
        'EPS Mom': eps_mom.reindex(composite.index).values,
        'Rev Mom': rev_mom.reindex(composite.index).values,
    })

    # Sort by composite score
    rankings_df = rankings_df.sort_values('Composite', ascending=False)
    rankings_df['Rank'] = range(1, len(rankings_df) + 1)

    # Get top and bottom
    top = rankings_df.head(top_n)
    bottom = rankings_df.tail(top_n).iloc[::-1]  # Reverse to show worst first

    # Display top performers
    print(f"\n{'='*80}")
    print(f"TOP {top_n} PERFORMERS (by Composite Score)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Ticker':<8} {'Composite':>10} {'Quality':>10} {'Price Mom':>10} {'EPS Mom':>10} {'Rev Mom':>10}")
    print(f"{'-'*80}")

    for _, row in top.iterrows():
        print(f"{int(row['Rank']):<6} {row['Ticker']:<8} "
              f"{row['Composite']:>10.2f} "
              f"{row['Quality']:>10.2f} "
              f"{row['Price Mom']:>10.2f} "
              f"{row['EPS Mom']:>10.2f} "
              f"{row['Rev Mom']:>10.2f}")

    # Display bottom performers
    print(f"\n{'='*80}")
    print(f"BOTTOM {top_n} PERFORMERS (by Composite Score)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Ticker':<8} {'Composite':>10} {'Quality':>10} {'Price Mom':>10} {'EPS Mom':>10} {'Rev Mom':>10}")
    print(f"{'-'*80}")

    for _, row in bottom.iterrows():
        print(f"{int(row['Rank']):<6} {row['Ticker']:<8} "
              f"{row['Composite']:>10.2f} "
              f"{row['Quality']:>10.2f} "
              f"{row['Price Mom']:>10.2f} "
              f"{row['EPS Mom']:>10.2f} "
              f"{row['Rev Mom']:>10.2f}")

    print(f"{'='*80}\n")

    return rankings_df


# -----------------------------
# Main Function
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rank all constituents in a universe by composite score."
    )
    ap.add_argument(
        "--universe",
        default="sp500",
        help="Universe to rank: 'sp500', universe name, or path (default: sp500)",
    )
    ap.add_argument(
        "--list-universes",
        action="store_true",
        help="List available universe files and exit",
    )
    ap.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark ticker for price momentum (default: SPY)",
    )
    ap.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help=f"Years of history (default: {DEFAULT_YEARS})",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top/bottom performers to display (default: 20)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Optional: save full rankings to CSV file",
    )
    ap.add_argument(
        "--max-universe",
        type=int,
        default=0,
        help="If >0, limit universe size (for faster testing)",
    )
    args = ap.parse_args()

    if args.list_universes:
        universes = list_universes()
        print("Available universes:", ", ".join(universes) if universes else "(none)")
        return 0

    # Load universe
    print(f"Loading universe: {args.universe}...")
    if args.universe.lower() == "sp500":
        universe = get_sp500_universe()
    else:
        universe = load_universe(args.universe)

    # Optional trim for testing
    if args.max_universe and args.max_universe > 0:
        universe = universe[:args.max_universe]

    print(f"Universe size: {len(universe)} tickers")

    weights = DEFAULT_WEIGHTS.copy()
    print(f"Weights: Quality={weights['quality']:.0%}, Price={weights['price_momentum']:.0%}, "
          f"Revenue={weights['revenue_momentum']:.0%}, EPS={weights['eps_momentum']:.0%}")

    # Use single benchmark for all
    print(f"Benchmark: {args.benchmark}")
    ticker_benchmarks = {ticker: args.benchmark for ticker in universe}

    # Fetch prices for all tickers + benchmark
    all_price_tickers = list(set(universe) | {args.benchmark})
    print(f"\nFetching prices for {len(all_price_tickers)} tickers...")

    try:
        prices = fetch_prices(all_price_tickers, years=args.years)
    except Exception as e:
        print(f"[ERROR] Failed to fetch prices: {e}", file=sys.stderr)
        return 1

    # Fetch all metrics
    print("\nFetching quality metrics...")
    quality_raw = fetch_quality_batch(universe, market="SPY", growth_years=args.years)

    print("\nFetching EPS momentum metrics...")
    eps_raw = fetch_eps_momentum_batch(universe, growth_years=args.years)

    print("\nFetching revenue momentum metrics...")
    rev_raw = fetch_revenue_momentum_batch(universe, growth_years=args.years)

    print("\nComputing price momentum...")
    price_raw = fetch_price_momentum_batch(universe, ticker_benchmarks, prices)

    # Compute scores
    print("\nComputing z-scores...")
    quality_signal, quality_pillars_df = compute_quality_scores(quality_raw)
    price_signal, price_z_df = compute_price_momentum_scores(price_raw)
    eps_signal, eps_z_df = compute_eps_momentum_scores(eps_raw)
    rev_signal, rev_z_df = compute_revenue_momentum_scores(rev_raw)

    # Reindex to universe
    quality_signal = quality_signal.reindex(universe)
    price_signal = price_signal.reindex(universe, fill_value=0.0)
    eps_signal = eps_signal.reindex(universe)
    rev_signal = rev_signal.reindex(universe)

    # Combine signals
    signal_dict = {
        'quality': quality_signal,
        'eps_momentum': eps_signal,
        'revenue_momentum': rev_signal,
        'price_momentum': price_signal,
    }
    composite_signal = combine_signals(signal_dict, weights, universe)
    composite_signal = clip_signal(composite_signal, *CLIP_BOUNDS)

    # Display rankings
    rankings_df = display_rankings(
        composite=composite_signal,
        quality=quality_signal,
        price_mom=price_signal,
        eps_mom=eps_signal,
        rev_mom=rev_signal,
        top_n=args.top,
    )

    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        rankings_df.to_csv(output_path, index=False)
        print(f"Full rankings saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
