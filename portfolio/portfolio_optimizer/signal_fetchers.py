#!/usr/bin/env python3
"""
Batch Signal Fetching Module

Provides batch data fetching functions for:
- Price momentum metrics (relative to benchmark)
- Quality metrics (profitability, growth, safety)
- EPS momentum metrics (YoY change, CAGR, acceleration)
- Revenue momentum metrics (YoY change, CAGR, acceleration)

All functions return DataFrames with tickers as index and metrics as columns.
"""

from __future__ import annotations
import logging

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance")

# Import from existing single-ticker scripts
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "equities" / "quality"))
sys.path.insert(0, str(Path(__file__).parent.parent / "momentum" / "fundamental_momentum"))

from quality_single import fetch_raw_metrics as fetch_quality_raw_metrics, RawMetrics
from eps_momentum_single import fetch_eps_metrics, EPSMetrics
from revenue_momentum_single import fetch_revenue_metrics, RevenueMetrics


# -------------------------
# Price Momentum Utilities
# -------------------------

MIN_DATA_POINTS = 83  # 63 + 20 days minimum for momentum calculation
MAX_BATCH_WORKERS = 6


def safe_div(a: float, b: float) -> float:
    """Safe division that returns NaN on invalid inputs."""
    if a is None or b is None or np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return float(a) / float(b)


def compute_momentum_metrics(
    ticker_prices: pd.Series,
    benchmark_prices: pd.Series,
    ticker_volume: Optional[pd.Series] = None,
) -> Optional[Dict[str, float]]:
    """
    Compute momentum metrics for a single ticker relative to benchmark.

    Returns dict with:
        - avg20_roc63: 20-day average of 63-day ROC (%)
        - avg20_vol_roc63: 20-day average of 63-day volume ROC (%)
        - rel_roc42: 42-day ROC of relative price (%)
        - avg10_rel_roc: 10-day average of relative ROC (%)

    Returns None if insufficient data.
    """
    # Align on common dates
    combined = pd.DataFrame({
        "ticker": ticker_prices,
        "benchmark": benchmark_prices
    }).dropna()

    if len(combined) < MIN_DATA_POINTS:
        return None

    prices = combined["ticker"]
    benchmark = combined["benchmark"]

    # 1. 20-day avg of 63-day ROC (%) - absolute price
    roc63 = (prices / prices.shift(63) - 1.0) * 100.0
    avg20_roc63 = roc63.rolling(window=20, min_periods=20).mean()

    # 2. 20-day avg of 63-day ROC (%) - volume
    avg20_vol_roc63 = np.nan
    if ticker_volume is not None:
        vol = ticker_volume.reindex(combined.index)
        vol = vol[vol > 0].reindex(combined.index)
        if vol.notna().sum() >= MIN_DATA_POINTS:
            vol_roc63 = (vol / vol.shift(63) - 1.0) * 100.0
            avg20_vol_roc63_series = vol_roc63.rolling(window=20, min_periods=20).mean()
            if not pd.isna(avg20_vol_roc63_series.iloc[-1]):
                avg20_vol_roc63 = float(avg20_vol_roc63_series.iloc[-1])

    # 3. Relative price calculations
    relative_price = prices / benchmark

    # 42-day ROC of relative price
    rel_roc42 = (relative_price / relative_price.shift(42) - 1.0) * 100.0

    # 4. 10-day avg of relative ROC
    avg10_rel_roc = rel_roc42.rolling(window=10, min_periods=10).mean()

    # Get latest values
    if pd.isna(avg20_roc63.iloc[-1]) or pd.isna(rel_roc42.iloc[-1]) or pd.isna(avg10_rel_roc.iloc[-1]):
        return None

    return {
        "avg20_roc63": float(avg20_roc63.iloc[-1]),
        "avg20_vol_roc63": float(avg20_vol_roc63),
        "rel_roc42": float(rel_roc42.iloc[-1]),
        "avg10_rel_roc": float(avg10_rel_roc.iloc[-1]),
    }


# -------------------------
# Batch Fetching Functions
# -------------------------

def fetch_price_momentum_batch(
    tickers: List[str],
    benchmark_map: Dict[str, str],
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute price momentum metrics for multiple tickers in batch.

    Args:
        tickers: List of ticker symbols
        benchmark_map: Dict mapping ticker -> benchmark ticker
        prices: DataFrame with tickers as columns, dates as index (already fetched)
        volumes: DataFrame with tickers as columns, dates as index (already fetched)

    Returns:
        DataFrame with tickers as index and columns: avg20_roc63, avg20_vol_roc63, rel_roc42, avg10_rel_roc
    """
    raw_metrics: Dict[str, Dict[str, float]] = {}
    failed_tickers: List[str] = []

    for ticker in tickers:
        if ticker not in prices.columns:
            failed_tickers.append(ticker)
            continue

        ticker_prices = prices[ticker].dropna()
        benchmark = benchmark_map[ticker]

        if benchmark not in prices.columns:
            failed_tickers.append(ticker)
            continue

        benchmark_prices = prices[benchmark].dropna()
        ticker_volume = None
        if volumes is not None and ticker in volumes.columns:
            ticker_volume = volumes[ticker].dropna()
        metrics = compute_momentum_metrics(ticker_prices, benchmark_prices, ticker_volume)

        if metrics is None:
            failed_tickers.append(ticker)
            continue

        raw_metrics[ticker] = metrics

    if failed_tickers:
        LOGGER.warning(f"[WARN] Price momentum failed for: {', '.join(failed_tickers)}")

    return pd.DataFrame(raw_metrics).T if raw_metrics else pd.DataFrame()


def fetch_quality_batch(
    tickers: List[str],
    market: str = "SPY",
    growth_years: int = 5,
    beta_years: float = 3.0,
) -> pd.DataFrame:
    """
    Fetch quality metrics for multiple tickers in batch.

    Args:
        tickers: List of ticker symbols
        market: Market proxy ticker for beta calculation
        growth_years: Target growth window in years
        beta_years: Beta lookback window in years

    Returns:
        DataFrame with tickers as index and 15 columns for quality metrics
    """
    raws: Dict[str, RawMetrics] = {}
    failed_tickers: List[str] = []

    if tickers:
        max_workers = min(MAX_BATCH_WORKERS, len(tickers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(fetch_quality_raw_metrics, ticker, market, growth_years, beta_years): ticker
                for ticker in tickers
            }
            for i, future in enumerate(as_completed(futures), 1):
                ticker = futures[future]
                try:
                    raws[ticker] = future.result()
                except Exception as e:
                    failed_tickers.append(ticker)
                    LOGGER.warning(f"[WARN] {ticker}: Quality fetch failed ({e})")
                if i % 5 == 0 or i == len(tickers):
                    print(f"  Quality: processed {i}/{len(tickers)}")

    if not raws:
        return pd.DataFrame()

    if failed_tickers:
        LOGGER.warning(f"[WARN] Quality failed for: {', '.join(sorted(failed_tickers))}")

    # Convert to DataFrame
    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T
    return raw_df


def fetch_eps_momentum_batch(
    tickers: List[str],
    growth_years: int = 3,
    use_edgar: bool = True,
) -> pd.DataFrame:
    """
    Fetch EPS momentum metrics for multiple tickers in batch.

    Args:
        tickers: List of ticker symbols
        growth_years: Target EPS CAGR window in years
        use_edgar: If True, try SEC EDGAR first then fall back to yfinance. If False, use yfinance only.

    Returns:
        DataFrame with tickers as index and columns:
            eps_yoy_change, eps_cagr, eps_growth_acceleration
    """
    raws: Dict[str, EPSMetrics] = {}
    failed_tickers: List[str] = []

    if tickers:
        max_workers = min(MAX_BATCH_WORKERS, len(tickers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_eps_metrics, ticker, growth_years, use_edgar=use_edgar): ticker for ticker in tickers}
            for i, future in enumerate(as_completed(futures), 1):
                ticker = futures[future]
                try:
                    raws[ticker] = future.result()
                except Exception as e:
                    failed_tickers.append(ticker)
                    LOGGER.warning(f"[WARN] {ticker}: EPS fetch failed ({e})")
                if i % 5 == 0 or i == len(tickers):
                    print(f"  EPS: processed {i}/{len(tickers)}")

    if not raws:
        return pd.DataFrame()

    if failed_tickers:
        LOGGER.warning(f"[WARN] EPS failed for: {', '.join(sorted(failed_tickers))}")

    # Convert to DataFrame
    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T
    return raw_df


def fetch_revenue_momentum_batch(
    tickers: List[str],
    growth_years: int = 3,
    use_edgar: bool = True,
) -> pd.DataFrame:
    """
    Fetch revenue momentum metrics for multiple tickers in batch.

    Args:
        tickers: List of ticker symbols
        growth_years: Target revenue CAGR window in years
        use_edgar: If True, try SEC EDGAR first then fall back to yfinance. If False, use yfinance only.

    Returns:
        DataFrame with tickers as index and columns:
            revenue_yoy_change, revenue_cagr, revenue_growth_acceleration
    """
    raws: Dict[str, RevenueMetrics] = {}
    failed_tickers: List[str] = []

    if tickers:
        max_workers = min(MAX_BATCH_WORKERS, len(tickers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_revenue_metrics, ticker, growth_years, use_edgar=use_edgar): ticker for ticker in tickers}
            for i, future in enumerate(as_completed(futures), 1):
                ticker = futures[future]
                try:
                    raws[ticker] = future.result()
                except Exception as e:
                    failed_tickers.append(ticker)
                    LOGGER.warning(f"[WARN] {ticker}: Revenue fetch failed ({e})")
                if i % 5 == 0 or i == len(tickers):
                    print(f"  Revenue: processed {i}/{len(tickers)}")

    if not raws:
        return pd.DataFrame()

    if failed_tickers:
        LOGGER.warning(f"[WARN] Revenue failed for: {', '.join(sorted(failed_tickers))}")

    # Convert to DataFrame
    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T
    return raw_df


# -------------------------
# ETF Look-through Utilities
# -------------------------

_INTL_SUFFIXES = (
    ".HE", ".L", ".TO", ".AX", ".PA", ".DE", ".MI", ".AS", ".SW", ".MC",
    ".SI", ".HK", ".T", ".NS", ".BO",
    ".KS", ".KQ", ".TW", ".TWO", ".SA",
)


def clean_ticker(tk: str) -> str:
    """
    Normalize ticker to Yahoo Finance format.

    Preserves dots for international exchange suffixes (e.g., METSO.HE).
    Converts dots to dashes for US share classes (e.g., BRK.B -> BRK-B).
    """
    tk = str(tk).strip().upper()
    if not tk or tk == "NAN":
        return ""
    if any(tk.endswith(suffix) for suffix in _INTL_SUFFIXES):
        return tk
    return tk.replace(".", "-")


def _normalize_holding_weights(raw: pd.Series) -> pd.Series:
    w = pd.to_numeric(raw, errors="coerce").astype("float64")
    w = w.replace([np.inf, -np.inf], np.nan).dropna()
    w = w[w > 0]
    if w.empty:
        return w
    if float(w.max()) > 1.0:
        w = w / 100.0
    s = float(w.sum())
    if s <= 0:
        return pd.Series(dtype="float64")
    return w / s


def fetch_etf_top_holdings(etf_ticker: str, top_n: int = 10) -> pd.Series:
    """
    Fetch top holdings weights for an ETF/mutual fund using yfinance.

    Returns:
        pd.Series indexed by holding ticker, values are normalized weights that sum to 1.0.
        Empty series if unavailable or not a fund.
    """
    try:
        t = yf.Ticker(etf_ticker)
        df = t.funds_data.top_holdings
    except Exception:
        return pd.Series(dtype="float64")

    if df is None or df.empty:
        return pd.Series(dtype="float64")

    weight_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("holding percent", "holding_percent", "holdingpercent"):
            weight_col = c
            break
    if weight_col is None:
        return pd.Series(dtype="float64")

    raw_w = df[weight_col].copy()
    raw_w.index = [clean_ticker(x) for x in df.index]
    raw_w = raw_w[raw_w.index != ""]
    raw_w = raw_w.groupby(level=0).sum()
    raw_w = raw_w.sort_values(ascending=False)
    raw_w = raw_w.head(int(top_n)) if top_n and top_n > 0 else raw_w
    return _normalize_holding_weights(raw_w)


def fetch_etf_top_holdings_batch(etf_tickers: List[str], top_n: int = 10) -> Dict[str, pd.Series]:
    """
    Fetch top holdings for multiple ETFs.

    Only ETFs with non-empty holdings are returned in the dict.
    """
    out: Dict[str, pd.Series] = {}
    for etf in etf_tickers:
        w = fetch_etf_top_holdings(etf, top_n=top_n)
        if not w.empty:
            out[etf] = w
    return out


def _weighted_average_row(metrics: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Weighted average of each metrics column, skipping NaNs and re-normalizing weights per column.
    """
    if metrics is None or metrics.empty or weights is None or weights.empty:
        return pd.Series(dtype="float64")

    weights = weights.copy()
    weights.index = [clean_ticker(x) for x in weights.index]
    weights = weights[weights.index != ""]
    weights = weights.groupby(level=0).sum()
    weights = _normalize_holding_weights(weights)
    if weights.empty:
        return pd.Series(dtype="float64")

    available = [t for t in weights.index if t in metrics.index]
    if not available:
        return pd.Series({c: np.nan for c in metrics.columns}, dtype="float64")

    m = metrics.reindex(available)
    w = weights.reindex(available)

    out: Dict[str, float] = {}
    for col in m.columns:
        v = pd.to_numeric(m[col], errors="coerce").astype("float64")
        mask = v.notna() & w.notna()
        if not mask.any():
            out[col] = np.nan
            continue
        w_col = w[mask]
        s = float(w_col.sum())
        if s <= 0:
            out[col] = np.nan
            continue
        w_col = w_col / s
        out[col] = float((v[mask] * w_col).sum())
    return pd.Series(out, index=list(m.columns), dtype="float64")


def compute_lookthrough_raw_metrics(
    etf_to_holdings: Dict[str, pd.Series],
    holdings_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-holding raw metrics into ETF-level raw metrics via holding weights.

    Args:
        etf_to_holdings: Dict of ETF ticker -> weights series (index holding ticker, values sum to 1)
        holdings_raw: DataFrame of raw metrics indexed by holding ticker

    Returns:
        DataFrame indexed by ETF ticker with same columns as holdings_raw.
    """
    if not etf_to_holdings or holdings_raw is None or holdings_raw.empty:
        return pd.DataFrame()

    rows: Dict[str, pd.Series] = {}
    for etf, w in etf_to_holdings.items():
        rows[etf] = _weighted_average_row(holdings_raw, w)

    out = pd.DataFrame(rows).T
    # Preserve column order when possible
    out = out.reindex(columns=list(holdings_raw.columns))
    return out


def fetch_etf_lookthrough_fundamentals_batch(
    etf_tickers: List[str],
    top_n: int = 10,
    market: str = "SPY",
    growth_years: int = 5,
    beta_years: float = 3.0,
    use_edgar: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
    """
    Compute ETF look-through fundamentals using the top N holdings.

    Returns:
        (quality_raw, eps_raw, revenue_raw, etf_to_holdings)
        - Each *_raw is indexed by ETF ticker.
        - etf_to_holdings maps ETF ticker -> normalized holding weights.
    """
    etf_to_holdings = fetch_etf_top_holdings_batch(etf_tickers, top_n=top_n)
    if not etf_to_holdings:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    holding_universe = sorted({t for w in etf_to_holdings.values() for t in w.index})
    if not holding_universe:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), etf_to_holdings

    quality_holdings = fetch_quality_batch(holding_universe, market=market, growth_years=growth_years, beta_years=beta_years)
    eps_holdings = fetch_eps_momentum_batch(holding_universe, growth_years=3, use_edgar=use_edgar)
    rev_holdings = fetch_revenue_momentum_batch(holding_universe, growth_years=3, use_edgar=use_edgar)

    quality_etf = compute_lookthrough_raw_metrics(etf_to_holdings, quality_holdings) if not quality_holdings.empty else pd.DataFrame()
    eps_etf = compute_lookthrough_raw_metrics(etf_to_holdings, eps_holdings) if not eps_holdings.empty else pd.DataFrame()
    rev_etf = compute_lookthrough_raw_metrics(etf_to_holdings, rev_holdings) if not rev_holdings.empty else pd.DataFrame()

    return quality_etf, eps_etf, rev_etf, etf_to_holdings
