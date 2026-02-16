#!/usr/bin/env python3
"""
QMJ-style Quality Score

Implements:
- Profitability: GPOA, ROE, ROA, CFOA, GMAR, ACC (low accruals)
- Growth: ~5-year growth for GPOA/ROE/ROA/CFOA/GMAR (falls back to shorter if data limited)
- Safety: low beta, low leverage, low bankruptcy risk (Altman Z), low ROE volatility

Key method:
- For each metric: cross-sectional rank -> z-score of ranks
- For each pillar: average of component z-scores, then z-score across universe (to match z(·) style)
- Quality: z( Profitability + Growth + Safety ) across universe

Notes:
- yfinance often provides only ~4 annual periods; growth window will shrink automatically.
- Per-share adjustments and exact Compustat definitions are approximated due to data limitations.
- Your results will depend heavily on the chosen universe and data availability.

pip install yfinance pandas numpy lxml html5lib
python3 quality.py AAPL --universe sp500
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def safe_div(a: float, b: float) -> float:
    if a is None or b is None:
        return np.nan
    if b == 0 or np.isnan(b):
        return np.nan
    return a / b


def last_col(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """
    yfinance financial statement tables are usually in columns of dates.
    Returns most recent column as a Series.
    """
    if df is None or df.empty:
        return None
    return df.iloc[:, 0]


def get_item(s: Optional[pd.Series], keys: List[str]) -> float:
    """
    Try multiple label keys in a yfinance statement series.
    Returns float or NaN.
    """
    if s is None:
        return np.nan
    for k in keys:
        if k in s.index:
            v = s.get(k)
            try:
                return float(v)
            except Exception:
                return np.nan
    return np.nan


_market_cache: Dict[str, pd.Series] = {}
_market_cache_lock = threading.Lock()


def _get_market_prices(market: str, window_years: float) -> pd.Series:
    """Download and cache market prices for beta computation."""
    cache_key = f"{market}_{window_years}"
    with _market_cache_lock:
        if cache_key in _market_cache:
            return _market_cache[cache_key]

    hist = yf.download(market, period=f"{int(window_years * 365)}d", interval="1d", auto_adjust=True, progress=False)
    if hist is None or hist.empty:
        return pd.Series(dtype="float64")

    px = hist["Close"].squeeze().dropna()

    with _market_cache_lock:
        _market_cache[cache_key] = px
    return px


def compute_beta(stock: str, market: str = "SPY", window_years: float = 3.0) -> float:
    """
    Estimate beta by regressing daily returns of stock on market over a lookback window.
    """
    # Get cached market prices
    px_m = _get_market_prices(market, window_years)
    if px_m.empty:
        return np.nan

    # Download only the stock
    hist = yf.download(stock, period=f"{int(window_years * 365)}d", interval="1d", auto_adjust=True, progress=False)
    if hist is None or hist.empty:
        return np.nan

    px_s = hist["Close"].squeeze().dropna()

    df = pd.concat([px_s, px_m], axis=1).dropna()
    if df.shape[0] < 60:
        return np.nan

    rets = df.pct_change().dropna()
    rs = rets.iloc[:, 0].values
    rm = rets.iloc[:, 1].values
    if np.std(rm) == 0:
        return np.nan

    # OLS beta = cov(rs, rm)/var(rm)
    beta = float(np.cov(rs, rm, ddof=0)[0, 1] / np.var(rm, ddof=0))
    return beta


def approx_altman_z(
    ca: float, cl: float, ta: float, re: float, ebit: float,
    mve: float, tl: float, sales: float
) -> float:
    """
    Classic Altman Z-score (public manufacturing form):
      Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MVE/TL) + 1.0*(Sales/TA)

    If inputs missing -> NaN.
    """
    vals = [ca, cl, ta, re, ebit, mve, tl, sales]
    if any(v is None or np.isnan(v) for v in vals) or ta == 0 or tl == 0:
        return np.nan
    wc = ca - cl
    return float(
        1.2 * (wc / ta) +
        1.4 * (re / ta) +
        3.3 * (ebit / ta) +
        0.6 * (mve / tl) +
        1.0 * (sales / ta)
    )


@dataclass
class RawMetrics:
    # Profitability
    gpoa: float
    roe: float
    roa: float
    cfoa: float
    gmar: float
    acc_low_is_good: float  # define as negative accruals, so higher is better

    # Growth (computed later)
    dgpoa: float
    droe: float
    droa: float
    dcfoa: float
    dgmar: float

    # Safety
    beta_low_is_good: float  # store beta; will invert later
    leverage_low_is_good: float  # debt/assets
    zscore_high_is_good: float
    roe_vol_low_is_good: float


def compute_growth_change_over_lag(
    num_series: pd.Series,
    den_series: pd.Series,
    lag_years: int
) -> float:
    """
    Approximate "five-year change in numerator divided by lagged denominator".
    Uses the earliest available point that is >= lag_years back; otherwise uses max available lag.
    """
    if num_series is None or den_series is None:
        return np.nan
    s = pd.concat([num_series, den_series], axis=1).dropna()
    if s.shape[0] < 2:
        return np.nan

    # Sort by date descending (yfinance often gives most recent first)
    s = s.sort_index(ascending=False)

    # Determine usable lag
    max_lag = s.shape[0] - 1
    use_lag = min(lag_years, max_lag)
    if use_lag < 1:
        return np.nan

    num_t = float(s.iloc[0, 0])
    num_l = float(s.iloc[use_lag, 0])
    den_l = float(s.iloc[use_lag, 1])
    if den_l == 0 or np.isnan(den_l):
        return np.nan
    return float((num_t - num_l) / den_l)


def compute_roe_vol(net_income_series: pd.Series, equity_series: pd.Series) -> float:
    """
    Rolling volatility of ROE using available annual data (std dev).
    """
    if net_income_series is None or equity_series is None:
        return np.nan
    df = pd.concat([net_income_series, equity_series], axis=1).dropna()
    if df.shape[0] < 3:
        return np.nan
    roe = df.iloc[:, 0] / df.iloc[:, 1].replace(0, np.nan)
    roe = roe.replace([np.inf, -np.inf], np.nan).dropna()
    if roe.shape[0] < 3:
        return np.nan
    return float(roe.std(ddof=0))


def fetch_raw_metrics(ticker: str, market: str, growth_years: int, beta_years: float) -> RawMetrics:
    t = yf.Ticker(ticker)

    # Annual statements
    inc = t.financials  # income statement (annual)
    bal = t.balance_sheet  # balance sheet (annual)
    cfs = t.cashflow  # cash flow (annual)

    inc_last = last_col(inc)
    bal_last = last_col(bal)
    cfs_last = last_col(cfs)

    # Items (try multiple keys because labels vary)
    revenue = get_item(inc_last, ["Total Revenue", "TotalRevenue", "Revenue"])
    cogs = get_item(inc_last, ["Cost Of Revenue", "CostOfRevenue"])
    gross_profit = get_item(inc_last, ["Gross Profit", "GrossProfit"])
    if np.isnan(gross_profit) and (not np.isnan(revenue)) and (not np.isnan(cogs)):
        gross_profit = revenue - cogs

    net_income = get_item(inc_last, ["Net Income", "NetIncome"])
    ebit = get_item(inc_last, ["EBIT", "Ebit", "Operating Income", "OperatingIncome"])

    total_assets = get_item(bal_last, ["Total Assets", "TotalAssets"])
    equity = get_item(bal_last, ["Total Stockholder Equity", "TotalStockholderEquity", "Stockholders Equity", "StockholdersEquity"])
    total_liab = get_item(bal_last, ["Total Liab", "TotalLiab", "Total Liabilities Net Minority Interest", "TotalLiabilitiesNetMinorityInterest"])
    current_assets = get_item(bal_last, ["Total Current Assets", "TotalCurrentAssets"])
    current_liab = get_item(bal_last, ["Total Current Liabilities", "TotalCurrentLiabilities"])
    retained_earnings = get_item(bal_last, ["Retained Earnings", "RetainedEarnings"])

    st_debt = get_item(bal_last, ["Short Long Term Debt", "ShortLongTermDebt", "Short Term Debt", "ShortTermDebt", "Current Debt", "CurrentDebt"])
    lt_debt = get_item(bal_last, ["Long Term Debt", "LongTermDebt"])
    total_debt = np.nan
    if not np.isnan(st_debt) or not np.isnan(lt_debt):
        total_debt = (0.0 if np.isnan(st_debt) else st_debt) + (0.0 if np.isnan(lt_debt) else lt_debt)

    op_cf = get_item(cfs_last, ["Total Cash From Operating Activities", "TotalCashFromOperatingActivities", "Operating Cash Flow", "OperatingCashFlow"])

    # Profitability metrics
    gpoa = safe_div(gross_profit, total_assets)
    roe = safe_div(net_income, equity)
    roa = safe_div(net_income, total_assets)
    cfoa = safe_div(op_cf, total_assets)
    gmar = safe_div(gross_profit, revenue)

    # Accruals: (NI - CFO)/Assets; "low accruals" is good -> multiply by -1
    accruals = safe_div((net_income - op_cf), total_assets) if (not np.isnan(net_income) and not np.isnan(op_cf)) else np.nan
    acc_low_is_good = -accruals if not np.isnan(accruals) else np.nan

    # Growth inputs: need time series. yfinance columns are dates; we want aligned series.
    # Build series for numerator/denominator pairs
    def series_from_table(df: pd.DataFrame, keys: List[str]) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        for k in keys:
            if k in df.index:
                s = df.loc[k].copy()
                try:
                    s = s.astype(float)
                    s.index = pd.to_datetime(s.index)
                    return s.sort_index()
                except Exception:
                    return None
        return None

    gp_series = series_from_table(inc, ["Gross Profit", "GrossProfit"])
    if gp_series is None and (inc is not None and not inc.empty):
        # Try reconstruct from revenue - cogs if both exist
        rev_s = series_from_table(inc, ["Total Revenue", "TotalRevenue", "Revenue"])
        cogs_s = series_from_table(inc, ["Cost Of Revenue", "CostOfRevenue"])
        if rev_s is not None and cogs_s is not None:
            gp_series = (rev_s - cogs_s).dropna()

    ni_series = series_from_table(inc, ["Net Income", "NetIncome"])
    rev_series = series_from_table(inc, ["Total Revenue", "TotalRevenue", "Revenue"])

    assets_series = series_from_table(bal, ["Total Assets", "TotalAssets"])
    equity_series = series_from_table(bal, ["Total Stockholder Equity", "TotalStockholderEquity", "Stockholders Equity", "StockholdersEquity"])

    opcf_series = series_from_table(cfs, ["Total Cash From Operating Activities", "TotalCashFromOperatingActivities", "Operating Cash Flow", "OperatingCashFlow"])

    # Approx growth per paper idea: change in numerator over lagged denominator
    dgpoa = compute_growth_change_over_lag(gp_series, assets_series, growth_years)
    droe = compute_growth_change_over_lag(ni_series, equity_series, growth_years)
    droa = compute_growth_change_over_lag(ni_series, assets_series, growth_years)
    dcfoa = compute_growth_change_over_lag(opcf_series, assets_series, growth_years)
    dgmar = compute_growth_change_over_lag(gp_series, rev_series, growth_years)

    # Safety metrics
    beta = compute_beta(ticker, market=market, window_years=beta_years)

    leverage = safe_div(total_debt, total_assets)  # debt/assets (lower is better)

    # Altman Z-score needs market value of equity; approximate from current price and shares outstanding
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    shares_out = info.get("sharesOutstanding", np.nan)
    last_px = np.nan
    try:
        last_px = float(t.history(period="5d", interval="1d", auto_adjust=True)["Close"].dropna().iloc[-1])
    except Exception:
        last_px = np.nan

    mve = np.nan
    if not np.isnan(shares_out) and not np.isnan(last_px):
        mve = float(shares_out) * float(last_px)

    z = approx_altman_z(
        ca=current_assets,
        cl=current_liab,
        ta=total_assets,
        re=retained_earnings,
        ebit=ebit,
        mve=mve,
        tl=total_liab,
        sales=revenue,
    )

    roe_vol = compute_roe_vol(ni_series, equity_series)

    return RawMetrics(
        gpoa=gpoa, roe=roe, roa=roa, cfoa=cfoa, gmar=gmar, acc_low_is_good=acc_low_is_good,
        dgpoa=dgpoa, droe=droe, droa=droa, dcfoa=dcfoa, dgmar=dgmar,
        beta_low_is_good=beta, leverage_low_is_good=leverage, zscore_high_is_good=z,
        roe_vol_low_is_good=roe_vol
    )


def compute_scores(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given raw metrics across universe, compute z-scores of ranks for each metric,
    then pillars, then Quality.
    Returns:
      - z_metrics: per-metric z-scores
      - scores: pillars + quality (z-scored composites)
    """
    df = raw.copy()

    # Orient each metric so that "higher is better" before ranking/z-scoring.
    oriented = pd.DataFrame(index=df.index)
    oriented["gpoa"] = df["gpoa"]
    oriented["roe"] = df["roe"]
    oriented["roa"] = df["roa"]
    oriented["cfoa"] = df["cfoa"]
    oriented["gmar"] = df["gmar"]
    oriented["acc"] = df["acc_low_is_good"]

    oriented["dgpoa"] = df["dgpoa"]
    oriented["droe"] = df["droe"]
    oriented["droa"] = df["droa"]
    oriented["dcfoa"] = df["dcfoa"]
    oriented["dgmar"] = df["dgmar"]

    oriented["bab"] = -df["beta_low_is_good"]          # low beta => higher is better
    oriented["lev"] = -df["leverage_low_is_good"]      # low leverage => higher is better
    oriented["zscore"] = df["zscore_high_is_good"]     # high Z => higher is better
    oriented["evol"] = -df["roe_vol_low_is_good"]      # low ROE vol => higher is better

    # Per-metric z-scores of ranks
    z_metrics = oriented.apply(zscore_of_ranks, axis=0)

    # Pillars: average available z's, then z-score across universe (as z(sum) style)
    def pillar(cols: List[str]) -> pd.Series:
        tmp = z_metrics[cols].mean(axis=1, skipna=True)
        return zscore_of_ranks(tmp)

    profitability = pillar(["gpoa", "roe", "roa", "cfoa", "gmar", "acc"])
    growth = pillar(["dgpoa", "droe", "droa", "dcfoa", "dgmar"])
    safety = pillar(["bab", "lev", "zscore", "evol"])  # omits O-score due to data limits

    combo = profitability + growth + safety
    quality = zscore_of_ranks(combo)

    scores = pd.DataFrame({
        "profitability": profitability,
        "growth": growth,
        "safety": safety,
        "quality": quality
    }, index=df.index)

    return z_metrics, scores


def main():
    ap = argparse.ArgumentParser(description="Compute QMJ-style Quality score for a ticker relative to a universe.")
    ap.add_argument("ticker", nargs="?", help="Ticker to score (e.g., AAPL)")
    ap.add_argument("--universe", default="sp500", help="Universe: 'sp500', universe name, or path to file (txt/csv)")
    ap.add_argument("--list-universes", action="store_true",
                    help="List available universe files and exit")
    ap.add_argument("--market", default="SPY", help="Market proxy ticker for beta (default: SPY)")
    ap.add_argument("--growth_years", type=int, default=5, help="Target growth window in years (default: 5)")
    ap.add_argument("--beta_years", type=float, default=3.0, help="Beta lookback window in years (default: 3)")
    ap.add_argument("--max_universe", type=int, default=0, help="If >0, limit universe size (debug/speed)")
    ap.add_argument("--out_csv", default="", help="Optional path to save full universe scores as CSV")
    args = ap.parse_args()

    if args.list_universes:
        universes = list_universes()
        print("Available universes:", ", ".join(universes) if universes else "(none)")
        sys.exit(0)

    if not args.ticker:
        ap.error("ticker is required unless using --list-universes")

    ticker = args.ticker.upper().strip()

    # Universe
    if args.universe.lower() == "sp500":
        universe = get_sp500_universe()
    else:
        universe = load_universe(args.universe)

    # Ensure target is included
    if ticker not in universe:
        universe = [ticker] + universe

    # Optional trim
    if args.max_universe and args.max_universe > 0:
        universe = universe[: args.max_universe]

    print(f"Universe size: {len(universe)} | Target: {ticker}")

    # Collect raw metrics (parallelized)
    raws: Dict[str, RawMetrics] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(fetch_raw_metrics, tk, market=args.market, growth_years=args.growth_years, beta_years=args.beta_years): tk
            for tk in universe
        }
        for i, future in enumerate(as_completed(futures), 1):
            tk = futures[future]
            try:
                raws[tk] = future.result()
            except Exception as e:
                print(f"[WARN] {tk}: failed ({e})", file=sys.stderr)

            if i % 25 == 0:
                print(f"  processed {i}/{len(universe)}")

    if ticker not in raws:
        raise SystemExit(f"Failed to fetch required data for target ticker: {ticker}")

    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T

    # Compute scores
    z_metrics, scores = compute_scores(raw_df)

    # Percentiles
    pct = scores.rank(pct=True)

    # Output for ticker
    t_scores = scores.loc[ticker]
    t_pct = pct.loc[ticker]
    print("\n--- QMJ-style Quality Score (relative to universe) ---")
    print(f"Ticker: {ticker}")
    print(f"Quality z-score: {t_scores['quality']:.3f} | Percentile: {100*t_pct['quality']:.1f}%")
    print(f"  Profitability: {t_scores['profitability']:.3f} | {100*t_pct['profitability']:.1f}%")
    print(f"  Growth:        {t_scores['growth']:.3f} | {100*t_pct['growth']:.1f}%")
    print(f"  Safety:        {t_scores['safety']:.3f} | {100*t_pct['safety']:.1f}%")

    # Show underlying metric z-scores for the ticker
    print("\n--- Underlying metric z-scores (rank->z) ---")
    t_z = z_metrics.loc[ticker].sort_values(ascending=False)
    for k, v in t_z.items():
        if np.isnan(v):
            continue
        print(f"{k:>8s}: {v: .3f}")

    # Save full results if requested
    if args.out_csv:
        out = raw_df.join(z_metrics.add_prefix("z_")).join(scores).join(pct.add_prefix("pct_"))
        out.to_csv(args.out_csv, index=True)
        print(f"\nWrote full universe output to: {args.out_csv}")


if __name__ == "__main__":
    main()
