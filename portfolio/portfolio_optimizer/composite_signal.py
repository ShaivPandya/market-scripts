#!/usr/bin/env python3
"""
Multi-Factor Composite Signal Generator for Portfolio Optimization

Generates standardized, clipped composite signals combining:
- Price Momentum (40%): Relative price momentum vs benchmark
- Quality (30%): Profitability, growth, safety metrics
- Revenue Momentum (20%): Revenue growth and acceleration
- EPS Momentum (10%): Earnings growth and acceleration

Signals are z-scored across the portfolio and can be used by portfolio_analyzer.py
to inform raw target weights.

Usage:
    python3 composite_signal.py
    python3 composite_signal.py --ticker AAPL
    python3 composite_signal.py --ticker AAPL,MSFT,GOOGL
    python3 composite_signal.py --ticker GLD --asset commodity
    python3 composite_signal.py --benchmark QQQ
    python3 composite_signal.py --quality-weight 0.5 --price-weight 0.3 --revenue-weight 0.1 --eps-weight 0.1
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance")  # noqa: B904

try:
    from .signal_fetchers import (
        fetch_eps_momentum_batch,
        fetch_etf_lookthrough_fundamentals_batch,
        fetch_price_momentum_batch,
        fetch_quality_batch,
        fetch_revenue_momentum_batch,
        fetch_spdr_sector_anchor_universe,
    )
except ImportError:
    from signal_fetchers import (
        fetch_eps_momentum_batch,
        fetch_etf_lookthrough_fundamentals_batch,
        fetch_price_momentum_batch,
        fetch_quality_batch,
        fetch_revenue_momentum_batch,
        fetch_spdr_sector_anchor_universe,
    )

# -----------------------------
# Configuration
# -----------------------------
PORTFOLIO_CSV = Path(__file__).parent.parent / "portfolio.csv"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_YEARS = 5
CLIP_BOUNDS = (-3.0, 3.0)
DEFAULT_WEIGHTS = {
    "quality": 0.30,
    "price_momentum": 0.40,
    "revenue_momentum": 0.20,
    "eps_momentum": 0.10,
}
DEFAULT_WEIGHTS_SHORT = {
    "quality": 0.30,
    "price_momentum": 0.40,
    "revenue_momentum": 0.20,
    "eps_momentum": 0.10,
}
DEFAULT_ANCHOR_TOP_N = 10
DEFAULT_ANCHOR_MIN_UNIQUE = 60


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
def fetch_prices(tickers: list[str], years: int = 5) -> pd.DataFrame:
    """
    Download adjusted close prices for multiple tickers from yfinance.
    Returns DataFrame with tickers as columns, dates as index.
    """
    end = datetime.now(UTC).date() + timedelta(days=1)
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
def fetch_ticker_metadata(ticker: str) -> tuple[float | None, str | None, bool]:
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


def select_benchmark_ticker(ticker: str, asset_type: str | None = None) -> str:
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
        LOGGER.warning("failed to fetch metadata for %s: %s. Defaulting to SPY.", ticker, e)
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
    def pillar(cols: list[str]) -> pd.Series:
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
    signal_dict: dict[str, pd.Series],
    weights: dict[str, float],
    tickers: list[str],
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
        weighted_sum = sum(normalized_weights[k] * available[k] for k in normalized_weights.keys())
        composite[ticker] = weighted_sum

    # Final z-score for cross-sectional ranking
    return zscore_of_ranks(composite)


def clip_signal(signal: pd.Series, lower: float = -3.0, upper: float = 3.0) -> pd.Series:
    """Clip signal to specified bounds."""
    return signal.clip(lower=lower, upper=upper)


def generate_anchor_normalized_long_equity_signals(
    long_equity_tickers: list[str],
    years: int = DEFAULT_YEARS,
    use_edgar: bool = True,
    benchmark: str = DEFAULT_BENCHMARK,
    weights: dict[str, float] | None = None,
    clip_bounds: tuple[float, float] = CLIP_BOUNDS,
    anchor_top_n: int = DEFAULT_ANCHOR_TOP_N,
    anchor_min_unique: int = DEFAULT_ANCHOR_MIN_UNIQUE,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Compute long-equity signals against a broad anchor universe.

    The scoring universe is:
      deduped top-N holdings from 11 SPDR sector ETFs union long_equity_tickers.
    Signal normalization (rank-zscore) is performed across that full scoring universe.

    Returns:
      - DataFrame indexed by `long_equity_tickers` with factor/composite signal columns
      - Metadata describing anchor/fallback state
    """
    tickers = list(dict.fromkeys([str(t).strip().upper() for t in long_equity_tickers if str(t).strip()]))
    mode = "spdr_sector_top10_anchor"
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    if not tickers:
        return pd.DataFrame(), {
            "signal_anchor_mode": mode,
            "signal_anchor_universe_size": 0,
            "signal_anchor_fallback_used": True,
            "reason": "no_long_equities",
        }

    anchor_universe, anchor_meta = fetch_spdr_sector_anchor_universe(
        top_n=anchor_top_n,
        min_unique=anchor_min_unique,
    )
    anchor_universe_size = int(anchor_meta.get("anchor_universe_size", 0))
    if not anchor_universe:
        return pd.DataFrame(), {
            "signal_anchor_mode": mode,
            "signal_anchor_universe_size": anchor_universe_size,
            "signal_anchor_fallback_used": True,
            "reason": "anchor_universe_unavailable",
            "anchor_metadata": anchor_meta,
        }

    scoring_universe = list(dict.fromkeys(anchor_universe + tickers))
    scoring_size = len(scoring_universe)
    ticker_benchmarks = {t: benchmark for t in scoring_universe}
    all_tickers = list(set(scoring_universe + [benchmark]))

    try:
        prices = fetch_prices(all_tickers, years=years)
    except Exception as e:
        LOGGER.warning("Anchor signal pricing failed: %s", e)
        return pd.DataFrame(), {
            "signal_anchor_mode": mode,
            "signal_anchor_universe_size": anchor_universe_size,
            "signal_anchor_fallback_used": True,
            "reason": "price_fetch_failed",
            "anchor_metadata": anchor_meta,
        }

    price_raw = fetch_price_momentum_batch(scoring_universe, ticker_benchmarks, prices)
    price_signal = compute_price_momentum_signal(price_raw).reindex(scoring_universe, fill_value=0.0)

    quality_raw = fetch_quality_batch(scoring_universe, market=benchmark, growth_years=years)
    quality_signal = (
        compute_quality_signal(quality_raw).reindex(scoring_universe)
        if quality_raw is not None and not quality_raw.empty
        else pd.Series(np.nan, index=scoring_universe)
    )

    eps_raw = fetch_eps_momentum_batch(scoring_universe, growth_years=3, use_edgar=use_edgar)
    eps_signal = (
        compute_eps_momentum_signal(eps_raw).reindex(scoring_universe)
        if eps_raw is not None and not eps_raw.empty
        else pd.Series(np.nan, index=scoring_universe)
    )

    rev_raw = fetch_revenue_momentum_batch(scoring_universe, growth_years=3, use_edgar=use_edgar)
    rev_signal = (
        compute_revenue_momentum_signal(rev_raw).reindex(scoring_universe)
        if rev_raw is not None and not rev_raw.empty
        else pd.Series(np.nan, index=scoring_universe)
    )

    signal_dict = {
        "quality": quality_signal,
        "eps_momentum": eps_signal,
        "revenue_momentum": rev_signal,
        "price_momentum": price_signal,
    }
    composite_signal = combine_signals(signal_dict, weights, scoring_universe)
    composite_signal = clip_signal(composite_signal, *clip_bounds)

    full_output = pd.DataFrame(
        {
            "quality_signal": quality_signal,
            "eps_mom_signal": eps_signal,
            "rev_mom_signal": rev_signal,
            "price_mom_signal": price_signal,
            "composite_signal": composite_signal,
        },
        index=scoring_universe,
    )
    target_output = full_output.reindex(tickers)
    valid_count = (
        int(target_output["composite_signal"].notna().sum()) if "composite_signal" in target_output.columns else 0
    )

    metadata: dict[str, object] = {
        "signal_anchor_mode": mode,
        "signal_anchor_universe_size": anchor_universe_size,
        "signal_anchor_scoring_universe_size": scoring_size,
        "signal_anchor_fallback_used": valid_count == 0,
        "anchor_metadata": anchor_meta,
    }
    if valid_count == 0:
        metadata["reason"] = "no_target_signals"
    return target_output, metadata


# -----------------------------
# Main Generation Function
# -----------------------------
def generate_composite_signals(
    tickers: list[str],
    asset_map: dict[str, str],
    benchmark_override: str | None = None,
    weights: dict[str, float] = None,
    weights_short: dict[str, float] | None = None,
    direction_map: dict[str, str] | None = None,
    years: int = DEFAULT_YEARS,
    etf_lookthrough_top_n: int = 10,
    clip_bounds: tuple[float, float] = CLIP_BOUNDS,
    use_edgar: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Generate multi-factor composite signals for portfolio.

    Args:
        tickers: List of ticker symbols
        asset_map: Dict mapping ticker -> asset type (equity, commodity)
        benchmark_override: If specified, use this benchmark for all tickers
        weights: Dict of signal weights for longs (default: quality=0.30, price=0.40, revenue=0.20, eps=0.10)
        weights_short: Dict of signal weights for shorts (default: None, uses same as longs)
        direction_map: Dict mapping ticker -> direction ("long" or "short")
        years: Years of price history to fetch
        etf_lookthrough_top_n: If >0, compute ETF fundamentals by looking through to top N holdings
        clip_bounds: (lower, upper) bounds for signal clipping
        use_edgar: If True, try SEC EDGAR first for EPS/revenue data. If False, use yfinance only.

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
    ticker_benchmarks: dict[str, str] = {}
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
        LOGGER.error("Failed to fetch prices: %s", e)
        empty_df = pd.DataFrame(
            {
                "quality_signal": pd.Series(np.nan, index=tickers),
                "eps_mom_signal": pd.Series(np.nan, index=tickers),
                "rev_mom_signal": pd.Series(np.nan, index=tickers),
                "price_mom_signal": pd.Series(0.0, index=tickers),
                "composite_signal": pd.Series(0.0, index=tickers),
            }
        )
        return empty_df, ticker_benchmarks

    # Verify all benchmarks exist
    missing_benchmarks = [b for b in unique_benchmarks if b not in prices.columns]
    if missing_benchmarks:
        LOGGER.error("Missing benchmark(s): %s", missing_benchmarks)

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

        # ETF look-through fundamentals (top holdings)
        etf_quality_raw = pd.DataFrame()
        etf_eps_raw = pd.DataFrame()
        etf_rev_raw = pd.DataFrame()
        etf_holdings_map: dict[str, pd.Series] = {}

        if etf_lookthrough_top_n and etf_lookthrough_top_n > 0:
            print(f"  ETF look-through: top {etf_lookthrough_top_n} holdings...")
            etf_quality_raw, etf_eps_raw, etf_rev_raw, etf_holdings_map = fetch_etf_lookthrough_fundamentals_batch(
                equities,
                top_n=etf_lookthrough_top_n,
                market="SPY",
                use_edgar=use_edgar,
                growth_years=years,
            )

            if etf_holdings_map:
                print(f"  ETF look-through enabled for: {', '.join(sorted(etf_holdings_map.keys()))}")

        etf_tickers = set(etf_holdings_map.keys())
        stock_equities = [t for t in equities if t not in etf_tickers]

        # Quality
        print("  Fetching quality metrics...")
        quality_raw_stock = (
            fetch_quality_batch(stock_equities, market="SPY", growth_years=years) if stock_equities else pd.DataFrame()
        )
        quality_raw = (
            pd.concat([quality_raw_stock, etf_quality_raw], axis=0) if not etf_quality_raw.empty else quality_raw_stock
        )
        if quality_raw is not None and not quality_raw.empty:
            quality_scores = compute_quality_signal(quality_raw)
            for ticker in quality_scores.index:
                if ticker in quality_signal.index:
                    quality_signal[ticker] = quality_scores[ticker]

        # EPS Momentum
        print("  Fetching EPS momentum metrics...")
        eps_raw_stock = (
            fetch_eps_momentum_batch(stock_equities, growth_years=3, use_edgar=use_edgar)
            if stock_equities
            else pd.DataFrame()
        )
        eps_raw = pd.concat([eps_raw_stock, etf_eps_raw], axis=0) if not etf_eps_raw.empty else eps_raw_stock
        if eps_raw is not None and not eps_raw.empty:
            eps_scores = compute_eps_momentum_signal(eps_raw)
            for ticker in eps_scores.index:
                if ticker in eps_mom_signal.index:
                    eps_mom_signal[ticker] = eps_scores[ticker]

        # Revenue Momentum
        print("  Fetching revenue momentum metrics...")
        rev_raw_stock = (
            fetch_revenue_momentum_batch(stock_equities, growth_years=3, use_edgar=use_edgar)
            if stock_equities
            else pd.DataFrame()
        )
        rev_raw = pd.concat([rev_raw_stock, etf_rev_raw], axis=0) if not etf_rev_raw.empty else rev_raw_stock
        if rev_raw is not None and not rev_raw.empty:
            rev_scores = compute_revenue_momentum_signal(rev_raw)
            for ticker in rev_scores.index:
                if ticker in rev_mom_signal.index:
                    rev_mom_signal[ticker] = rev_scores[ticker]

    if commodities:
        print(f"\nSkipping fundamental signals for {len(commodities)} commodities")

    # 3. Combine signals with dynamic weight adjustment
    signal_dict = {
        "quality": quality_signal,
        "eps_momentum": eps_mom_signal,
        "revenue_momentum": rev_mom_signal,
        "price_momentum": price_signal,
    }

    # Direction-specific composite signals
    if direction_map is not None and weights_short is not None:
        longs = [t for t in tickers if direction_map.get(t, "").lower() == "long"]
        shorts = [t for t in tickers if direction_map.get(t, "").lower() == "short"]
        others = [t for t in tickers if t not in longs and t not in shorts]

        composite_signal = pd.Series(index=tickers, dtype="float64")

        if longs:
            composite_long = combine_signals(signal_dict, weights, longs)
            composite_signal.loc[longs] = composite_long
        if shorts:
            composite_short = combine_signals(signal_dict, weights_short, shorts)
            composite_signal.loc[shorts] = composite_short
        if others:
            composite_other = combine_signals(signal_dict, weights, others)
            composite_signal.loc[others] = composite_other
    else:
        composite_signal = combine_signals(signal_dict, weights, tickers)

    # 4. Clip composite signal
    composite_signal = clip_signal(composite_signal, *clip_bounds)

    # 5. Build output DataFrame
    output = pd.DataFrame(
        {
            "quality_signal": quality_signal,
            "eps_mom_signal": eps_mom_signal,
            "rev_mom_signal": rev_mom_signal,
            "price_mom_signal": price_signal,
            "composite_signal": composite_signal,
        },
        index=tickers,
    )

    return output, ticker_benchmarks


# -----------------------------
# CLI
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Generate multi-factor composite signals for portfolio tickers.")
    ap.add_argument(
        "--portfolio",
        default=str(PORTFOLIO_CSV),
        help=f"Path to portfolio CSV (default: {PORTFOLIO_CSV})",
    )
    ap.add_argument(
        "--ticker",
        default=None,
        help="Single ticker or comma-separated tickers (overrides --portfolio)",
    )
    ap.add_argument(
        "--asset",
        default="equity",
        help="Asset type when using --ticker: 'equity' or 'commodity' (default: equity)",
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
        default=DEFAULT_WEIGHTS["quality"],
        help=f"Quality weight (default: {DEFAULT_WEIGHTS['quality']})",
    )
    ap.add_argument(
        "--price-weight",
        type=float,
        default=DEFAULT_WEIGHTS["price_momentum"],
        help=f"Price momentum weight (default: {DEFAULT_WEIGHTS['price_momentum']})",
    )
    ap.add_argument(
        "--revenue-weight",
        type=float,
        default=DEFAULT_WEIGHTS["revenue_momentum"],
        help=f"Revenue momentum weight (default: {DEFAULT_WEIGHTS['revenue_momentum']})",
    )
    ap.add_argument(
        "--eps-weight",
        type=float,
        default=DEFAULT_WEIGHTS["eps_momentum"],
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
        "quality": args.quality_weight,
        "price_momentum": args.price_weight,
        "revenue_momentum": args.revenue_weight,
        "eps_momentum": args.eps_weight,
    }

    # Validate weights sum to 1.0 (warn if not)
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        LOGGER.warning("Weights sum to %.3f, normalizing to 1.0", weight_sum)
        weights = {k: v / weight_sum for k, v in weights.items()}

    # Handle --ticker argument (overrides portfolio CSV)
    if args.ticker:
        active_tickers = [t.strip() for t in args.ticker.split(",")]
        asset_map = {ticker: args.asset for ticker in active_tickers}
        # Create dummy direction for output
        direction_map = {ticker: "long" for ticker in active_tickers}
    else:
        # Load portfolio
        portfolio_path = Path(args.portfolio)
        if not portfolio_path.exists():
            LOGGER.error("Portfolio file not found: %s", portfolio_path)
            return 1

        meta = pd.read_csv(portfolio_path)
        meta["direction"] = meta["direction"].fillna("")

        # Build asset map
        asset_map = dict(zip(meta["ticker"], meta["asset"]))  # noqa: B905

        # Filter to active tickers (has direction)
        active_mask = meta["direction"].str.strip().ne("")
        active_tickers = meta.loc[active_mask, "ticker"].tolist()
        direction_map = dict(zip(meta["ticker"], meta["direction"]))  # noqa: B905

        if not active_tickers:
            LOGGER.error("No active tickers in portfolio")
            return 1

    print(f"Portfolio: {len(active_tickers)} active tickers")
    print(
        f"Weights: Quality={weights['quality']:.1%}, Price={weights['price_momentum']:.1%}, "
        f"Revenue={weights['revenue_momentum']:.1%}, EPS={weights['eps_momentum']:.1%}"
    )

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
    output = pd.DataFrame(
        {
            "direction": pd.Series(direction_map),
            "benchmark": pd.Series(ticker_benchmarks),
        }
    )
    output = output.join(signals_df)
    output.index.name = "ticker"

    # Print results
    print("\n=== Multi-Factor Composite Signals ===")
    print(output.to_string(float_format=lambda x: f"{x: .4f}" if pd.notna(x) else "NaN"))

    # Summary stats per signal
    print("\n=== Signal Statistics ===")
    for col in ["quality_signal", "eps_mom_signal", "rev_mom_signal", "price_mom_signal", "composite_signal"]:
        valid = signals_df[col].dropna()
        if len(valid) > 0:
            print(
                f"{col:20s}: n={len(valid):2d}, mean={valid.mean(): .4f}, std={valid.std(): .4f}, "
                f"min={valid.min(): .4f}, max={valid.max(): .4f}"
            )
        else:
            print(f"{col:20s}: no valid data")

    # Save to CSV
    # output_path = Path(args.out_csv) if args.out_csv else (Path(__file__).parent / "composite_signals.csv")
    # output.to_csv(output_path)
    # print(f"\nWrote signals to: {output_path}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("Starting script execution: %s", __file__)
    raise SystemExit(main())
