#!/usr/bin/env python3
"""
EPS Momentum Score + Percentile
Analyzes a specific ticker's EPS momentum relative to a universe

Metrics:
1) EPS YoY (most recent quarter vs same quarter 1 year ago):
      eps_yoy_change = (EPS_q0 - EPS_q4) / abs(EPS_q4)
   where q0 is the most recent reported quarter and q4 is 4 quarters earlier.
   If EPS_q4 is 0 or missing, the result is NaN.

2) Earnings growth (CAGR) over the prior ~5 fiscal years:
      eps_cagr = (EPS_t / EPS_{t-n}) ** (1/n) - 1
   where n is min(requested_years, available_years).

3) EPS growth acceleration (second derivative) over the last 5 quarters:
      - Compute QoQ growth rates from 5 quarters of EPS data:
        growth_i = (EPS_Qi - EPS_Q{i+1}) / |EPS_Q{i+1}| for i in [0,1,2,3]
      - Fit linear regression to growth rates vs time
      - eps_growth_acceleration = slope of regression (negated so positive = improving)
      - Positive value means EPS growth is accelerating (improving trend)
      - Negative value means EPS growth is decelerating (worsening trend)

Scoring:
- For each metric: cross-sectional rank transformed into a z-score
- EPS Momentum score: rank-based z-score of the average of the three metric z-scores
- Percentile: rank(pct=True) on the final score

Data source: yfinance (Yahoo Finance). Statement availability varies by ticker.

python3 eps_momentum.py AAPL --universe sp500
python3 eps_momentum.py AAPL --universe sp500 --growth_years 3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance") from e


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


def clean_ticker(tk: str) -> str:
    """Normalize tickers to the Yahoo Finance format."""
    tk = tk.strip().upper()
    return tk.replace(".", "-")


def read_universe_file(path: str) -> List[str]:
    """Read tickers from a txt (one per line) or csv (any column) file."""
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        tickers: List[str] = []
        for c in df.columns:
            tickers += [str(x) for x in df[c].dropna().tolist()]
        return sorted({clean_ticker(t) for t in tickers if str(t).strip()})

    with open(path, "r", encoding="utf-8") as f:
        tickers = [clean_ticker(line) for line in f if line.strip()]
    return sorted(set(tickers))


def get_sp500_universe() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    import urllib.request

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        html = resp.read()

    tables = pd.read_html(html)
    df = tables[0]
    syms = [clean_ticker(x) for x in df["Symbol"].astype(str).tolist()]
    return sorted(set(syms))


# -------------------------
# EPS metrics
# -------------------------

EPS_KEYS = [
    "Diluted EPS", "DilutedEPS", "EPS Diluted", "Earnings Per Share Diluted",
    "Basic EPS", "BasicEPS", "EPS Basic", "Earnings Per Share Basic",
]

DILUTED_SHARES_KEYS = [
    "Diluted Average Shares", "DilutedAverageShares",
    "Weighted Average Shares Diluted", "WeightedAverageSharesDiluted",
    "Diluted Shares", "DilutedShares",
]

BASIC_SHARES_KEYS = [
    "Basic Average Shares", "BasicAverageShares",
    "Weighted Average Shares Basic", "WeightedAverageSharesBasic",
    "Basic Shares", "BasicShares",
]

NET_INCOME_KEYS = [
    "Net Income", "NetIncome",
    "Net Income Common Stockholders", "NetIncomeCommonStockholders",
    "Net Income Applicable To Common Shares", "NetIncomeApplicableToCommonShares",
    "Net Income Continuous Operations", "NetIncomeContinuousOperations",
]


@dataclass
class EPSMetrics:
    eps_q0: float = np.nan
    eps_q4: float = np.nan
    eps_yoy_change: float = np.nan
    eps_yoy_diff: float = np.nan

    eps_a0: float = np.nan
    eps_aN: float = np.nan
    eps_cagr: float = np.nan
    years_used: int = 0

    # Second derivative of EPS growth (acceleration)
    eps_growth_rates: Optional[List[float]] = None  # 4 QoQ growth rates (Q0-Q1, Q1-Q2, Q2-Q3, Q3-Q4)
    eps_growth_acceleration: float = np.nan  # Slope of growth rates over time

    q0_end: Optional[pd.Timestamp] = None
    q4_end: Optional[pd.Timestamp] = None
    a0_end: Optional[pd.Timestamp] = None
    aN_end: Optional[pd.Timestamp] = None


def compute_eps_from_stmt(stmt_col: pd.Series, fallback_shares: Optional[float] = None) -> float:
    """
    Compute EPS for one statement column.

    Priority:
    - Direct EPS line item (diluted first, then basic)
    - Net income divided by diluted shares (or basic shares)
    - Net income divided by shares outstanding from ticker info
    """
    eps = get_item(stmt_col, EPS_KEYS)
    if not np.isnan(eps):
        return eps

    net_income = get_item(stmt_col, NET_INCOME_KEYS)
    if np.isnan(net_income):
        return np.nan

    shares = get_item(stmt_col, DILUTED_SHARES_KEYS)
    if np.isnan(shares):
        shares = get_item(stmt_col, BASIC_SHARES_KEYS)
    if np.isnan(shares) and fallback_shares is not None and not np.isnan(fallback_shares):
        shares = float(fallback_shares)

    return safe_div(net_income, shares)


def fetch_eps_metrics(ticker: str, growth_years: int = 5) -> EPSMetrics:
    """Fetch quarterly YoY EPS change, annual EPS CAGR, and EPS growth acceleration for a ticker."""
    t = yf.Ticker(ticker)

    shares_out = np.nan
    try:
        info = t.info or {}
        shares_out = float(info.get("sharesOutstanding", np.nan))
    except Exception:
        shares_out = np.nan

    out = EPSMetrics()

    q_inc = try_get_income_stmt(t, "quarterly")
    if q_inc is not None and not q_inc.empty and q_inc.shape[1] >= 5:
        out.q0_end = pd.to_datetime(q_inc.columns[0])
        out.q4_end = pd.to_datetime(q_inc.columns[4])

        q0 = col_at(q_inc, 0)
        q4 = col_at(q_inc, 4)

        if q0 is not None and q4 is not None:
            out.eps_q0 = compute_eps_from_stmt(q0, fallback_shares=shares_out)
            out.eps_q4 = compute_eps_from_stmt(q4, fallback_shares=shares_out)
            out.eps_yoy_diff = out.eps_q0 - out.eps_q4
            out.eps_yoy_change = safe_div(out.eps_yoy_diff, abs(out.eps_q4))

        # Compute EPS growth acceleration (second derivative) using 5 quarters
        if q_inc.shape[1] >= 5:
            eps_values = []
            for i in range(5):
                q_col = col_at(q_inc, i)
                if q_col is not None:
                    eps_values.append(compute_eps_from_stmt(q_col, fallback_shares=shares_out))
                else:
                    eps_values.append(np.nan)

            # Compute 4 QoQ growth rates: (Q0-Q1), (Q1-Q2), (Q2-Q3), (Q3-Q4)
            growth_rates = []
            for i in range(4):
                eps_recent = eps_values[i]
                eps_prior = eps_values[i + 1]
                growth = safe_div(eps_recent - eps_prior, abs(eps_prior))
                growth_rates.append(growth)

            out.eps_growth_rates = growth_rates

            # Fit linear regression to growth rates over time to get acceleration
            # x = [0, 1, 2, 3] where 0 is most recent, 3 is oldest
            valid_growth = [(i, g) for i, g in enumerate(growth_rates) if not np.isnan(g)]
            if len(valid_growth) >= 2:
                x = np.array([v[0] for v in valid_growth])
                y = np.array([v[1] for v in valid_growth])
                # Slope is negative when growth is accelerating (improving over time)
                # since x=0 is most recent. Negate to make positive = accelerating.
                slope = np.polyfit(x, y, 1)[0]
                out.eps_growth_acceleration = -slope

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
                out.eps_a0 = compute_eps_from_stmt(a0, fallback_shares=shares_out)
                out.eps_aN = compute_eps_from_stmt(aN, fallback_shares=shares_out)

                if out.eps_a0 > 0 and out.eps_aN > 0:
                    out.eps_cagr = (out.eps_a0 / out.eps_aN) ** (1.0 / years_used) - 1.0
                else:
                    out.eps_cagr = np.nan

    return out


# -------------------------
# Scoring + reporting
# -------------------------

def compute_universe_scores(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute rank-based z-scores for metrics and a combined EPS Momentum score.

    Expects raw_df to contain:
      - eps_yoy_change
      - eps_cagr
      - eps_growth_acceleration
    """
    metrics = ["eps_yoy_change", "eps_cagr", "eps_growth_acceleration"]

    z_metrics = pd.DataFrame(index=raw_df.index)
    for m in metrics:
        z_metrics[m] = zscore_of_ranks(raw_df[m])

    combo = z_metrics[metrics].mean(axis=1, skipna=True)
    eps_momentum_score = zscore_of_ranks(combo)

    return z_metrics, eps_momentum_score


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
        description="Compute EPS momentum (quarterly YoY EPS change + ~5y EPS CAGR) and percentile in a universe."
    )
    ap.add_argument("ticker", help="Ticker to score (e.g., AAPL)")
    ap.add_argument("--universe", default="sp500", help="Universe: 'sp500' or path to file (txt/csv) with tickers")
    ap.add_argument("--growth_years", type=int, default=5, help="Target EPS CAGR window in years")
    ap.add_argument("--max_universe", type=int, default=0, help="If >0, limit universe size")
    ap.add_argument("--out_csv", default="", help="Optional path to save full universe output as CSV")
    args = ap.parse_args()

    ticker = clean_ticker(args.ticker)

    if args.universe.lower() == "sp500":
        universe = get_sp500_universe()
    else:
        universe = read_universe_file(args.universe)

    if ticker not in universe:
        universe = [ticker] + universe

    if args.max_universe and args.max_universe > 0:
        universe = universe[: args.max_universe]

    print(f"Universe size: {len(universe)} | Target: {ticker}")

    raws: Dict[str, EPSMetrics] = {}
    for i, tk in enumerate(universe, 1):
        try:
            raws[tk] = fetch_eps_metrics(tk, growth_years=args.growth_years)
        except Exception as e:
            print(f"[WARN] {tk}: failed ({e})", file=sys.stderr)

        if i % 25 == 0:
            print(f"  processed {i}/{len(universe)}")

    if ticker not in raws:
        raise SystemExit(f"Failed to fetch required data for target ticker: {ticker}")

    raw_df = pd.DataFrame({k: vars(v) for k, v in raws.items()}).T

    z_metrics, score = compute_universe_scores(raw_df)
    pct_score = score.rank(pct=True)

    tm = raws[ticker]
    print("\nTarget ticker raw metrics:")
    if tm.q0_end is not None:
        print(f"  Latest quarter end: {tm.q0_end.date()} | Year-ago quarter end: {tm.q4_end.date() if tm.q4_end is not None else 'NA'}")
    if tm.a0_end is not None:
        print(f"  Latest fiscal year end: {tm.a0_end.date()} | {tm.years_used}y-ago fiscal year end: {tm.aN_end.date() if tm.aN_end is not None else 'NA'}")

    print(f"  EPS (latest quarter):          {fmt_num(tm.eps_q0)}")
    print(f"  EPS (same qtr 1y ago):         {fmt_num(tm.eps_q4)}")
    print(f"  EPS YoY change (q/q-4):        {fmt_pct(tm.eps_yoy_change)}")
    print(f"  EPS YoY difference (q-q-4):    {fmt_num(tm.eps_yoy_diff)}")
    print(f"  EPS CAGR ({tm.years_used}y):               {fmt_pct(tm.eps_cagr)}")
    print(f"  EPS growth acceleration:       {fmt_num(tm.eps_growth_acceleration)}")
    if tm.eps_growth_rates is not None:
        rates_str = ", ".join([fmt_pct(r) for r in tm.eps_growth_rates])
        print(f"  EPS QoQ growth rates (5Q):     [{rates_str}]")

    print("\nUniverse-relative scores:")
    print(f"  z_eps_yoy_change:        {z_metrics.loc[ticker, 'eps_yoy_change'] if ticker in z_metrics.index else np.nan: .3f}")
    print(f"  z_eps_cagr:              {z_metrics.loc[ticker, 'eps_cagr'] if ticker in z_metrics.index else np.nan: .3f}")
    print(f"  z_eps_growth_accel:      {z_metrics.loc[ticker, 'eps_growth_acceleration'] if ticker in z_metrics.index else np.nan: .3f}")
    print(f"  EPS Momentum z:          {score.loc[ticker] if ticker in score.index else np.nan: .3f}")
    print(f"  EPS Momentum pct:        {pct_score.loc[ticker] if ticker in pct_score.index else np.nan: .3%}")

    if args.out_csv:
        out = raw_df.join(z_metrics.add_prefix("z_"))
        out["eps_momentum_z"] = score
        out["eps_momentum_pct"] = pct_score
        out.to_csv(args.out_csv, index=True)
        print(f"\nWrote full universe output to: {args.out_csv}")


if __name__ == "__main__":
    main()
