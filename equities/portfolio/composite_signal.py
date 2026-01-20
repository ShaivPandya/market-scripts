#!/usr/bin/env python3
"""
Multi-Factor Composite Signal Generator for Portfolio Optimization

Generates standardized, clipped composite signals combining:
- Quality (33%): Profitability, growth, safety metrics
- Price Momentum (33%): Relative price momentum vs benchmark
- Revenue Momentum (20%): Revenue growth and acceleration
- EPS Momentum (14%): Earnings growth and acceleration

Signals are z-scored across the portfolio and can be used by portfolio_optimizer.py
to inform raw target weights.

Usage:
    python3 composite_signal.py
    python3 composite_signal.py --benchmark QQQ
    python3 composite_signal.py --quality-weight 0.5 --price-weight 0.3 --revenue-weight 0.1 --eps-weight 0.1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance")

from signal_fetchers import (
    fetch_price_momentum_batch,
    fetch_quality_batch,
    fetch_eps_momentum_batch,
    fetch_revenue_momentum_batch,
)

# -----------------------------
# Configuration
# -----------------------------
PORTFOLIO_CSV = Path(__file__).parent.parent / "universes" / "portfolio.csv"
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

    Uses rank-based z-scores which are more robust to outliers than
    direct value z-scores.
    """
    x = values.copy()
    mask = x.notna()
    if mask.sum() < 2:
        return pd.Series(index=x.index, dtype="float64")

    # Rank in ascending order; highest value gets highest rank
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
def fetch_prices(tickers: List[str], years: int = 5) -> pd.DataFrame:
    """
    Download adjusted close prices for multiple tickers from yfinance.
    Returns DataFrame with tickers as columns, dates as index.
    """
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

    # Handle multi-index columns when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    else:
        # Single ticker case
        prices = df[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.dropna(how="all")


# -----------------------------
# Benchmark Selection
# -----------------------------
def fetch_ticker_metadata(ticker: str) -> Tuple[Optional[float], Optional[str], bool]:
    """
    Fetch market cap, sector, and ETF status from yfinance.
    """
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.get_info()
    if not info:
        info = yf_ticker.info

    market_cap = info.get("marketCap")
    sector = info.get("sector")
    quote_type = str(info.get("quoteType", "")).lower()
    is_etf = quote_type == "etf"
    return market_cap, sector, is_etf


def select_benchmark_ticker(ticker: str, asset_type: Optional[str] = None) -> str:
    """
    Auto-select benchmark based on ticker metadata and asset type.

    Selection logic:
        - Commodities (including commodity ETFs) → ^BCOM (Bloomberg Commodity Index)
        - ETFs → SPY
        - Market cap <= $20B → IWM (small-cap)
        - Technology sector → QQQ
        - Default → SPY
    """
    # Check if this is a commodity first
    if asset_type == "commodity":
        return "^BCOM"

    try:
        market_cap, sector, is_etf = fetch_ticker_metadata(ticker)
    except Exception as e:
        print(f"Warning: failed to fetch metadata for {ticker}: {e}. Defaulting to SPY.", file=sys.stderr)
        return "SPY"

    if is_etf:
        return "SPY"

    if market_cap is not None and market_cap <= 20_000_000_000:
        return "IWM"

    if sector and "technology" in sector.lower():
        return "QQQ"

    return "SPY"


# -----------------------------
# Signal Computation Functions
# -----------------------------
def compute_price_momentum_signal(raw_df: pd.DataFrame) -> pd.Series:
    """
    Compute price momentum z-score from raw momentum metrics.

    Expects columns: avg20_roc63, rel_roc42, avg10_rel_roc
    """
    if raw_df.empty:
        return pd.Series(dtype="float64")

    metrics = ["avg20_roc63", "rel_roc42", "avg10_rel_roc"]
    available = [m for m in metrics if m in raw_df.columns]

    if not available:
        return pd.Series(dtype="float64")

    # Z-score each metric across portfolio
    z_metrics = raw_df[available].apply(zscore_of_ranks, axis=0)

    # Equal-weighted average of z-scores
    composite = z_metrics.mean(axis=1, skipna=True)

    # Final z-score of composite
    return zscore_of_ranks(composite)


def compute_quality_signal(raw_df: pd.DataFrame) -> pd.Series:
    """
    Compute quality z-score from raw quality metrics.

    Implements QMJ-style scoring:
    1. Orient metrics (higher = better)
    2. Z-score each metric
    3. Average by pillar: profitability, growth, safety
    4. Z-score each pillar
    5. Sum pillars and final z-score
    """
    if raw_df.empty:
        return pd.Series(dtype="float64")

    df = raw_df.copy()

    # Orient each metric so that "higher is better"
    oriented = pd.DataFrame(index=df.index)

    # Profitability metrics (already higher = better)
    for col in ["gpoa", "roe", "roa", "cfoa", "gmar"]:
        if col in df.columns:
            oriented[col] = df[col]
    if "acc_low_is_good" in df.columns:
        oriented["acc"] = df["acc_low_is_good"]  # Already inverted in fetch

    # Growth metrics (already higher = better)
    for col in ["dgpoa", "droe", "droa", "dcfoa", "dgmar"]:
        if col in df.columns:
            oriented[col] = df[col]

    # Safety metrics (need to invert low-is-good)
    if "beta_low_is_good" in df.columns:
        oriented["bab"] = -df["beta_low_is_good"]
    if "leverage_low_is_good" in df.columns:
        oriented["lev"] = -df["leverage_low_is_good"]
    if "zscore_high_is_good" in df.columns:
        oriented["zscore"] = df["zscore_high_is_good"]
    if "roe_vol_low_is_good" in df.columns:
        oriented["evol"] = -df["roe_vol_low_is_good"]

    if oriented.empty:
        return pd.Series(dtype="float64")

    # Per-metric z-scores of ranks
    z_metrics = oriented.apply(zscore_of_ranks, axis=0)

    # Pillars: average available z's, then z-score across universe
    def pillar(cols: List[str]) -> pd.Series:
        available = [c for c in cols if c in z_metrics.columns]
        if not available:
            return pd.Series(np.nan, index=z_metrics.index)
        tmp = z_metrics[available].mean(axis=1, skipna=True)
        return zscore_of_ranks(tmp)

    profitability = pillar(["gpoa", "roe", "roa", "cfoa", "gmar", "acc"])
    growth = pillar(["dgpoa", "droe", "droa", "dcfoa", "dgmar"])
    safety = pillar(["bab", "lev", "zscore", "evol"])

    combo = profitability + growth + safety
    return zscore_of_ranks(combo)


def compute_eps_momentum_signal(raw_df: pd.DataFrame) -> pd.Series:
    """
    Compute EPS momentum z-score from raw EPS metrics.

    Expects columns: eps_yoy_change, eps_cagr, eps_growth_acceleration
    """
    if raw_df.empty:
        return pd.Series(dtype="float64")

    metrics = ["eps_yoy_change", "eps_cagr", "eps_growth_acceleration"]
    available = [m for m in metrics if m in raw_df.columns]

    if not available:
        return pd.Series(dtype="float64")

    # Z-score each metric
    z_metrics = raw_df[available].apply(zscore_of_ranks, axis=0)

    # Average z-scores
    composite = z_metrics.mean(axis=1, skipna=True)

    # Final z-score of composite
    return zscore_of_ranks(composite)


def compute_revenue_momentum_signal(raw_df: pd.DataFrame) -> pd.Series:
    """
    Compute revenue momentum z-score from raw revenue metrics.

    Expects columns: revenue_yoy_change, revenue_cagr, revenue_growth_acceleration
    """
    if raw_df.empty:
        return pd.Series(dtype="float64")

    metrics = ["revenue_yoy_change", "revenue_cagr", "revenue_growth_acceleration"]
    available = [m for m in metrics if m in raw_df.columns]

    if not available:
        return pd.Series(dtype="float64")

    # Z-score each metric
    z_metrics = raw_df[available].apply(zscore_of_ranks, axis=0)

    # Average z-scores
    composite = z_metrics.mean(axis=1, skipna=True)

    # Final z-score of composite
    return zscore_of_ranks(composite)


# -----------------------------
# Signal Combination
# -----------------------------
def combine_signals(
    signal_dict: Dict[str, pd.Series],
    weights: Dict[str, float],
    tickers: List[str],
) -> pd.Series:
    """
    Weighted combination of signals with dynamic weight adjustment for missing data.

    Args:
        signal_dict: Dict of {signal_name: pd.Series of z-scores}
        weights: Dict of {signal_name: weight} (should sum to 1.0)
        tickers: List of all tickers

    Returns:
        pd.Series of composite signals indexed by ticker
    """
    # Build DataFrame of all signals
    signals_df = pd.DataFrame(signal_dict, index=tickers)

    # For each ticker, compute weighted average using only available signals
    composite = pd.Series(index=tickers, dtype="float64")

    for ticker in tickers:
        ticker_signals = signals_df.loc[ticker]
        available = ticker_signals.dropna()

        if available.empty:
            composite[ticker] = 0.0
            continue

        # Get weights for available signals
        available_weights = {k: weights[k] for k in available.index if k in weights}

        if not available_weights:
            composite[ticker] = 0.0
            continue

        # Normalize weights to sum to 1.0
        weight_sum = sum(available_weights.values())
        normalized_weights = {k: v / weight_sum for k, v in available_weights.items()}

        # Weighted average
        weighted_sum = sum(
            normalized_weights[k] * available[k]
            for k in normalized_weights.keys()
        )
        composite[ticker] = weighted_sum

    # Final z-score for cross-sectional ranking
    return zscore_of_ranks(composite)


def clip_signal(signal: pd.Series, lower: float = -3.0, upper: float = 3.0) -> pd.Series:
    """Clip signal to specified bounds."""
    return signal.clip(lower=lower, upper=upper)


# -----------------------------
# Main Generation Function
# -----------------------------
def generate_composite_signals(
    tickers: List[str],
    asset_map: Dict[str, str],
    benchmark_override: Optional[str] = None,
    weights: Dict[str, float] = None,
    years: int = DEFAULT_YEARS,
    clip_bounds: Tuple[float, float] = CLIP_BOUNDS,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Generate multi-factor composite signals for portfolio.

    Args:
        tickers: List of ticker symbols
        asset_map: Dict mapping ticker -> asset type (equity, commodity)
        benchmark_override: If specified, use this benchmark for all tickers
        weights: Dict of signal weights (default: quality=0.33, price=0.33, revenue=0.20, eps=0.14)
        years: Years of price history to fetch
        clip_bounds: (lower, upper) bounds for signal clipping

    Returns:
        Tuple of:
        - DataFrame with columns: quality_signal, eps_mom_signal,
          rev_mom_signal, price_mom_signal, composite_signal
        - Dict mapping ticker -> benchmark used
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    if not tickers:
        return pd.DataFrame(), {}

    # Determine benchmark for each ticker
    ticker_benchmarks: Dict[str, str] = {}
    if benchmark_override:
        print(f"Using benchmark override: {benchmark_override}")
        for ticker in tickers:
            ticker_benchmarks[ticker] = benchmark_override
    else:
        print("Auto-selecting benchmarks per ticker...")
        for ticker in tickers:
            asset_type = asset_map.get(ticker, "equity")
            benchmark = select_benchmark_ticker(ticker, asset_type)
            ticker_benchmarks[ticker] = benchmark
            print(f"  {ticker} -> {benchmark}")

    # Separate equities from commodities
    equities = [t for t in tickers if asset_map.get(t, "equity") == "equity"]
    commodities = [t for t in tickers if asset_map.get(t) == "commodity"]

    if commodities:
        print(f"\nCommodities detected ({len(commodities)}): {', '.join(commodities)}")
        print("  -> Will use price momentum only for commodities")

    # Fetch all unique tickers + benchmarks for price data
    unique_benchmarks = set(ticker_benchmarks.values())
    all_tickers = list(set(tickers) | unique_benchmarks)
    print(f"\nFetching prices for {len(all_tickers)} tickers...")

    try:
        prices = fetch_prices(all_tickers, years=years)
    except Exception as e:
        print(f"[ERROR] Failed to fetch prices: {e}", file=sys.stderr)
        empty_df = pd.DataFrame({
            'quality_signal': pd.Series(np.nan, index=tickers),
            'eps_mom_signal': pd.Series(np.nan, index=tickers),
            'rev_mom_signal': pd.Series(np.nan, index=tickers),
            'price_mom_signal': pd.Series(0.0, index=tickers),
            'composite_signal': pd.Series(0.0, index=tickers),
        })
        return empty_df, ticker_benchmarks

    # Verify all benchmarks exist
    missing_benchmarks = [b for b in unique_benchmarks if b not in prices.columns]
    if missing_benchmarks:
        print(f"[ERROR] Missing benchmark(s): {missing_benchmarks}", file=sys.stderr)

    # 1. Compute price momentum for ALL tickers
    print("\nComputing price momentum...")
    price_raw = fetch_price_momentum_batch(tickers, ticker_benchmarks, prices)
    price_signal = compute_price_momentum_signal(price_raw)
    price_signal = price_signal.reindex(tickers, fill_value=0.0)

    # 2. Compute fundamental signals for EQUITIES only
    quality_signal = pd.Series(np.nan, index=tickers)
    eps_mom_signal = pd.Series(np.nan, index=tickers)
    rev_mom_signal = pd.Series(np.nan, index=tickers)

    if equities:
        print(f"\nComputing fundamental signals for {len(equities)} equities...")

        # Quality
        print("  Fetching quality metrics...")
        quality_raw = fetch_quality_batch(equities, market="SPY", growth_years=years)
        if not quality_raw.empty:
            quality_scores = compute_quality_signal(quality_raw)
            for ticker in quality_scores.index:
                if ticker in quality_signal.index:
                    quality_signal[ticker] = quality_scores[ticker]

        # EPS Momentum
        print("  Fetching EPS momentum metrics...")
        eps_raw = fetch_eps_momentum_batch(equities, growth_years=years)
        if not eps_raw.empty:
            eps_scores = compute_eps_momentum_signal(eps_raw)
            for ticker in eps_scores.index:
                if ticker in eps_mom_signal.index:
                    eps_mom_signal[ticker] = eps_scores[ticker]

        # Revenue Momentum
        print("  Fetching revenue momentum metrics...")
        rev_raw = fetch_revenue_momentum_batch(equities, growth_years=years)
        if not rev_raw.empty:
            rev_scores = compute_revenue_momentum_signal(rev_raw)
            for ticker in rev_scores.index:
                if ticker in rev_mom_signal.index:
                    rev_mom_signal[ticker] = rev_scores[ticker]

    if commodities:
        print(f"\nSkipping fundamental signals for {len(commodities)} commodities")

    # 3. Combine signals with dynamic weight adjustment
    signal_dict = {
        'quality': quality_signal,
        'eps_momentum': eps_mom_signal,
        'revenue_momentum': rev_mom_signal,
        'price_momentum': price_signal,
    }
    composite_signal = combine_signals(signal_dict, weights, tickers)

    # 4. Clip composite signal
    composite_signal = clip_signal(composite_signal, *clip_bounds)

    # 5. Build output DataFrame
    output = pd.DataFrame({
        'quality_signal': quality_signal,
        'eps_mom_signal': eps_mom_signal,
        'rev_mom_signal': rev_mom_signal,
        'price_mom_signal': price_signal,
        'composite_signal': composite_signal,
    }, index=tickers)

    return output, ticker_benchmarks


# -----------------------------
# CLI
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate multi-factor composite signals for portfolio tickers."
    )
    ap.add_argument(
        "--portfolio",
        default=str(PORTFOLIO_CSV),
        help=f"Path to portfolio CSV (default: {PORTFOLIO_CSV})",
    )
    ap.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark ticker override (auto-selects per ticker if not specified)",
    )
    ap.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help=f"Years of history (default: {DEFAULT_YEARS})",
    )
    # Weight configuration arguments
    ap.add_argument(
        "--quality-weight",
        type=float,
        default=DEFAULT_WEIGHTS['quality'],
        help=f"Quality weight (default: {DEFAULT_WEIGHTS['quality']})",
    )
    ap.add_argument(
        "--price-weight",
        type=float,
        default=DEFAULT_WEIGHTS['price_momentum'],
        help=f"Price momentum weight (default: {DEFAULT_WEIGHTS['price_momentum']})",
    )
    ap.add_argument(
        "--revenue-weight",
        type=float,
        default=DEFAULT_WEIGHTS['revenue_momentum'],
        help=f"Revenue momentum weight (default: {DEFAULT_WEIGHTS['revenue_momentum']})",
    )
    ap.add_argument(
        "--eps-weight",
        type=float,
        default=DEFAULT_WEIGHTS['eps_momentum'],
        help=f"EPS momentum weight (default: {DEFAULT_WEIGHTS['eps_momentum']})",
    )
    ap.add_argument(
        "--clip-lower",
        type=float,
        default=CLIP_BOUNDS[0],
        help=f"Lower bound for signal clipping (default: {CLIP_BOUNDS[0]})",
    )
    ap.add_argument(
        "--clip-upper",
        type=float,
        default=CLIP_BOUNDS[1],
        help=f"Upper bound for signal clipping (default: {CLIP_BOUNDS[1]})",
    )
    ap.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path (default: composite_signals.csv)",
    )
    args = ap.parse_args()

    # Build weights dict from CLI args
    weights = {
        'quality': args.quality_weight,
        'price_momentum': args.price_weight,
        'revenue_momentum': args.revenue_weight,
        'eps_momentum': args.eps_weight,
    }

    # Validate weights sum to 1.0 (warn if not)
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        print(f"[WARN] Weights sum to {weight_sum:.3f}, normalizing to 1.0")
        weights = {k: v / weight_sum for k, v in weights.items()}

    # Load portfolio
    portfolio_path = Path(args.portfolio)
    if not portfolio_path.exists():
        print(f"[ERROR] Portfolio file not found: {portfolio_path}", file=sys.stderr)
        return 1

    meta = pd.read_csv(portfolio_path)
    meta["direction"] = meta["direction"].fillna("")

    # Build asset map
    asset_map = dict(zip(meta["ticker"], meta["asset"]))

    # Filter to active tickers (has direction)
    active_mask = meta["direction"].str.strip().ne("")
    active_tickers = meta.loc[active_mask, "ticker"].tolist()

    if not active_tickers:
        print("[ERROR] No active tickers in portfolio", file=sys.stderr)
        return 1

    print(f"Portfolio: {len(active_tickers)} active tickers")
    print(f"Weights: Quality={weights['quality']:.1%}, Price={weights['price_momentum']:.1%}, "
          f"Revenue={weights['revenue_momentum']:.1%}, EPS={weights['eps_momentum']:.1%}")

    # Generate signals
    signals_df, ticker_benchmarks = generate_composite_signals(
        tickers=active_tickers,
        asset_map=asset_map,
        benchmark_override=args.benchmark,
        weights=weights,
        years=args.years,
        clip_bounds=(args.clip_lower, args.clip_upper),
    )

    # Add metadata columns
    output = pd.DataFrame({
        "direction": meta.set_index("ticker").loc[active_tickers, "direction"],
        "benchmark": pd.Series(ticker_benchmarks),
    })
    output = output.join(signals_df)
    output.index.name = "ticker"

    # Print results
    print("\n=== Multi-Factor Composite Signals ===")
    print(output.to_string(float_format=lambda x: f"{x: .4f}" if pd.notna(x) else "NaN"))

    # Summary stats per signal
    print("\n=== Signal Statistics ===")
    for col in ['quality_signal', 'eps_mom_signal', 'rev_mom_signal', 'price_mom_signal', 'composite_signal']:
        valid = signals_df[col].dropna()
        if len(valid) > 0:
            print(f"{col:20s}: n={len(valid):2d}, mean={valid.mean(): .4f}, std={valid.std(): .4f}, "
                  f"min={valid.min(): .4f}, max={valid.max(): .4f}")
        else:
            print(f"{col:20s}: no valid data")

    # Save to CSV
    # output_path = Path(args.out_csv) if args.out_csv else (Path(__file__).parent / "composite_signals.csv")
    # output.to_csv(output_path)
    # print(f"\nWrote signals to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
