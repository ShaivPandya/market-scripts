"""
Signal Aggregator Backtest
==========================
Validates regime thresholds and factor weights against 10 years of historical data.

Replicates the exact scoring formulas from api/signal_aggregator.py using raw
data from FRED and yfinance. Five factors: VIX, Breadth, Liquidity, Sector, Momentum.

Usage:
    python backtest/signal_backtest.py
    python backtest/signal_backtest.py --start 2016-01-01 --end 2026-01-01
    python backtest/signal_backtest.py --force   # re-download all data

Limitations:
    - Uses current S&P 500 constituents (survivorship bias, acceptable for
      threshold calibration since we measure aggregate regime, not stock picks)
    - Liquidity uses US-only FRED data (ECB/BoJ data unavailable for 10yr)
    - Top-50 breadth metrics are excluded (30% of breadth weight; remaining 70%
      is reweighted to 100%)
    - Sector weights use current approximations as proxy for historical
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from load_env import load_env

load_env()

log = logging.getLogger(__name__)

# ── Constants (from api/signal_aggregator.py) ──────────────────────────
CONFIGURED_WEIGHTS: dict[str, float] = {
    "vix": 0.20,
    "breadth": 0.20,
    "liquidity": 0.35,
    "sector": 0.15,
    "momentum": 0.10,
}

DEFAULT_LO, DEFAULT_HI = 40.0, 65.0

FRED_LIQUIDITY = {
    "fed_assets": "WALCL",
    "reserves": "WRESBAL",
    "tga": "WTREGEN",
    "rrp": "RRPONTSYD",
    "ig_oas": "BAMLC0A0CM",
    "hy_oas": "BAMLH0A0HYM2",
    "nfci": "NFCI",
    "m2": "M2SL",
    "gdp": "GDP",
}

# liquidity.py US_COMPONENTS: (key, polarity, weight)
US_LIQ_COMPONENTS = [
    ("net_liquidity_change_4w", 1, 0.25),
    ("net_liquidity", 1, 0.20),
    ("reserves_change_4w", 1, 0.20),
    ("ig_oas", -1, 0.15),
    ("hy_oas", -1, 0.10),
    ("nfci", -1, 0.05),
    ("m2_gdp", 1, 0.05),
]

SECTOR_ETFS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
SECTOR_WEIGHTS = {
    "XLK": 30.0,
    "XLF": 14.0,
    "XLV": 12.0,
    "XLY": 10.0,
    "XLC": 9.0,
    "XLI": 8.0,
    "XLP": 6.0,
    "XLE": 4.0,
    "XLB": 2.5,
    "XLRE": 2.5,
    "XLU": 2.0,
}

CACHE_DIR = Path(__file__).parent / "cache"

# ── Helpers ────────────────────────────────────────────────────────────


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def regime_label(score: float, lo: float = DEFAULT_LO, hi: float = DEFAULT_HI) -> str:
    if score < lo:
        return "risk-on"
    if score < hi:
        return "transitional"
    return "risk-off"


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _load_or_download(name: str, download_fn, force: bool = False) -> pd.DataFrame:
    path = _cache_path(name)
    if path.exists() and not force:
        log.info("Loading cached %s", name)
        return pd.read_parquet(path)
    log.info("Downloading %s …", name)
    df = download_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    log.info("Cached %s → %s (%d rows)", name, path.name, len(df))
    return df


# ── Data Download ──────────────────────────────────────────────────────


def _download_vix(start: str) -> pd.DataFrame:
    import yfinance as yf

    vix = yf.download("^VIX", start=start, auto_adjust=True, progress=False)["Close"]
    vix3m = yf.download("^VIX3M", start=start, auto_adjust=True, progress=False)["Close"]
    if vix3m.empty:
        vix3m = yf.download("^VXV", start=start, auto_adjust=True, progress=False)["Close"]
    # Flatten MultiIndex if yfinance returns one
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    if isinstance(vix3m, pd.DataFrame):
        vix3m = vix3m.iloc[:, 0]
    df = pd.DataFrame({"VIX": vix, "VIX3M": vix3m}).dropna()
    df["Ratio"] = df["VIX3M"] / df["VIX"]
    return df


def _download_spy(start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.DataFrame({"Close": close})


def _get_sp500_tickers() -> list[str]:
    """Get current S&P 500 tickers from Wikipedia."""
    from io import StringIO

    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        tickers = tables[0]["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
        return sorted(pd.unique(tickers).tolist())
    except Exception as e:
        log.warning("Wikipedia SP500 fetch failed: %s", e)
        raise RuntimeError("Cannot get S&P 500 ticker list") from e


def _download_spx_constituents(start: str) -> pd.DataFrame:
    import yfinance as yf

    tickers = _get_sp500_tickers()
    log.info("Downloading %d S&P 500 constituents (this takes a few minutes)…", len(tickers))
    all_closes: dict[str, pd.Series] = {}
    chunk_size = 50

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        batch_num = i // chunk_size + 1
        total_batches = (len(tickers) + chunk_size - 1) // chunk_size
        log.info("  Batch %d/%d (%d tickers)", batch_num, total_batches, len(chunk))
        try:
            data = yf.download(chunk, start=start, auto_adjust=True, progress=False)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data["Close"]
                else:
                    closes = data[["Close"]]
                    closes.columns = chunk[: len(closes.columns)]
                if isinstance(closes, pd.Series):
                    closes = closes.to_frame(chunk[0])
                for col in closes.columns:
                    if closes[col].notna().sum() > 200:
                        all_closes[str(col)] = closes[col]
            if i + chunk_size < len(tickers):
                time.sleep(1.5)
        except Exception as e:
            log.warning("  Batch %d failed: %s", batch_num, e)

    if not all_closes:
        raise RuntimeError("Failed to download any S&P 500 constituent data")
    return pd.DataFrame(all_closes)


def _download_sector_etfs(start: str) -> pd.DataFrame:
    import yfinance as yf

    tickers = SECTOR_ETFS + ["SPY"]
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data
    # Ensure column names are strings
    closes.columns = [str(c) for c in closes.columns]
    return closes


def _download_fred(start: str) -> pd.DataFrame:
    from fredapi import Fred

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set")
    fred = Fred(api_key=api_key)

    series_data: dict[str, pd.Series] = {}
    for name, sid in FRED_LIQUIDITY.items():
        try:
            s = fred.get_series(sid, observation_start=start)
            series_data[name] = s
            log.info("  FRED %s (%s): %d obs", name, sid, len(s))
        except Exception as e:
            log.warning("  FRED %s (%s) failed: %s", name, sid, e)

    df = pd.DataFrame(series_data)
    df.index = pd.to_datetime(df.index)
    return df


# ── Factor Scoring (vectorised where possible) ────────────────────────


def score_vix_series(vix_df: pd.DataFrame) -> pd.Series:
    """VIX risk score: higher = more risk-off (fear)."""
    ratio_comp = ((1.0 - vix_df["Ratio"]) / 0.2).clip(0.0, 1.0)
    vix_comp = ((vix_df["VIX"] - 18.0) / 12.0).clip(0.0, 1.0)
    return 70.0 * ratio_comp + 30.0 * vix_comp


def score_breadth_series(prices: pd.DataFrame) -> pd.Series:
    """Breadth risk score: higher = more risk-off (weak breadth).

    Uses SPX-wide metrics only (top-50 excluded for historical simplicity).
    Three inputs reweighted from 70/100 to 100%.
    """
    close = prices.copy()
    ma_200 = close.rolling(200, min_periods=200).mean()
    ma_20 = close.rolling(20, min_periods=20).mean()
    low_20 = close.rolling(20, min_periods=20).min()

    n_valid = close.notna().sum(axis=1)
    n_valid = n_valid.replace(0, np.nan)

    pct_above_200 = (close > ma_200).sum(axis=1) / n_valid * 100
    pct_above_20 = (close > ma_20).sum(axis=1) / n_valid * 100
    pct_at_low = (close <= low_20).sum(axis=1) / n_valid * 100

    # _score_breadth formula (first 3 inputs, weights 30+20+20=70, reweighted)
    c1 = ((55.0 - pct_above_200) / 35.0).clip(0.0, 1.0) * 30.0
    c2 = ((55.0 - pct_above_20) / 35.0).clip(0.0, 1.0) * 20.0
    c3 = ((pct_at_low - 20.0) / 40.0).clip(0.0, 1.0) * 20.0

    total_w = 70.0  # 30 + 20 + 20
    score = (c1 + c2 + c3) * (100.0 / total_w)
    return score


def score_liquidity_series(fred_df: pd.DataFrame) -> pd.Series:
    """US-only liquidity risk score: higher = more risk-off (tight liquidity)."""
    df = fred_df.copy().sort_index()
    df = df.resample("W-WED").last().ffill()

    # Scale RRP from billions → millions to match Fed assets
    if "rrp" in df.columns:
        df["rrp"] = df["rrp"] * 1000

    # Derived series. FRED can occasionally return partial data; keep the
    # backtest running with the components that are available.
    if {"fed_assets", "tga", "rrp"}.issubset(df.columns):
        df["net_liquidity"] = df["fed_assets"] - df["tga"] - df["rrp"]
        df["net_liquidity_change_4w"] = df["net_liquidity"].diff(4)
    if "reserves" in df.columns:
        df["reserves_change_4w"] = df["reserves"].diff(4)
    if {"m2", "gdp"}.issubset(df.columns):
        df["m2_gdp"] = df["m2"] / df["gdp"]

    # Rolling z-scores → weighted composite (replicates liquidity.py)
    Z_WINDOW = 104
    MIN_PERIODS = 13

    contributions: list[pd.Series] = []
    for key, polarity, weight in US_LIQ_COMPONENTS:
        if key not in df.columns:
            continue
        s = df[key].astype(float)
        mu = s.rolling(Z_WINDOW, min_periods=MIN_PERIODS).mean()
        std = s.rolling(Z_WINDOW, min_periods=MIN_PERIODS).std()
        std = std.replace(0, np.nan)
        z = (s - mu) / std
        contributions.append(z * polarity * weight)

    if not contributions:
        return pd.Series(dtype=float)

    # US composite (this IS the composite since we skip ECB/BoJ)
    us_composite = pd.concat(contributions, axis=1).sum(axis=1)

    # Convert composite z-score → signal_aggregator score
    def _to_score(c):
        if pd.isna(c):
            return np.nan
        if c > 1.0:
            base = 20.0  # ample
        elif c >= -0.5:
            base = 45.0  # normal
        elif c >= -1.5:
            base = 75.0  # tight
        else:
            base = 90.0  # stress
        return max(0.0, min(100.0, base + (-c * 10.0)))

    return us_composite.apply(_to_score)


def score_sector_series(etf_prices: pd.DataFrame) -> pd.Series:
    """Sector risk score: higher = sector weakness (risk-off)."""
    if etf_prices.empty or "SPY" not in etf_prices.columns:
        return pd.Series(dtype=float)

    df = etf_prices.copy().sort_index()
    fridays = df.resample("W-FRI").last().dropna(how="all")

    scores: dict[pd.Timestamp, float] = {}
    for fri_idx in range(len(fridays)):
        fri = fridays.index[fri_idx]
        window = df.loc[:fri]
        if len(window) < 200:
            continue

        spy = window["SPY"]
        weighted_sum = 0.0
        total_weight = 0.0

        for etf in SECTOR_ETFS:
            if etf not in window.columns:
                continue
            prices = window[etf].dropna()
            if len(prices) < 200:
                continue

            # 3-month relative performance (63 trading days)
            n_3m = min(63, len(prices) - 1)
            if n_3m < 20:
                continue
            etf_ret = (float(prices.iloc[-1]) / float(prices.iloc[-n_3m]) - 1) * 100
            spy_ret = (float(spy.iloc[-1]) / float(spy.iloc[-n_3m]) - 1) * 100
            rel_perf = etf_ret - spy_ret

            # 3-month absolute change
            chg = etf_ret

            # % distance from 200 DMA
            sma200 = float(prices.rolling(200).mean().iloc[-1])
            pct_200 = (float(prices.iloc[-1]) - sma200) / sma200 * 100 if sma200 > 0 else 0

            # _score_sector formula
            comp_vals = [
                clamp01((-rel_perf) / 8.0),
                clamp01((-chg) / 1.5),
                clamp01((-pct_200) / 12.0),
            ]
            local = sum(comp_vals) / len(comp_vals)
            weight = SECTOR_WEIGHTS.get(etf, 5.0)
            weighted_sum += local * weight
            total_weight += weight

        if total_weight > 0:
            scores[fri] = (weighted_sum / total_weight) * 100.0

    return pd.Series(scores)


def score_momentum_series(prices: pd.DataFrame, spy: pd.Series) -> pd.Series:
    """Momentum risk score: higher = weak momentum (risk-off).

    Counts bullish tickers (avg10_rel_roc > 0 AND rel_roc42 > 0) and inverts.
    """
    close = prices.copy()
    spy_aligned = spy.reindex(close.index)

    # Relative price = ticker / SPY
    relative = close.div(spy_aligned, axis=0)

    # 42-day ROC of relative price
    rel_roc42 = (relative / relative.shift(42) - 1) * 100

    # 10-day rolling mean of rel_roc42
    avg10 = rel_roc42.rolling(10, min_periods=10).mean()

    # Bullish = both avg10 > 0 AND rel_roc42 > 0
    bullish = (avg10 > 0) & (rel_roc42 > 0)
    n_valid = avg10.notna().sum(axis=1)
    n_valid = n_valid.replace(0, np.nan)
    bullish_ratio = bullish.sum(axis=1) / n_valid

    # Invert: low bullish ratio = high risk score
    score = ((0.55 - bullish_ratio) / 0.55).clip(0.0, 1.0) * 100.0
    return score


# ── Main Backtest ──────────────────────────────────────────────────────


def run_backtest(
    start: str = "2016-03-01",
    end: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    end = end or date.today().isoformat()

    # Need extra history for MAs and z-scores
    data_start = (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

    print("=" * 70)
    print("SIGNAL AGGREGATOR BACKTEST")
    print("=" * 70)
    print(f"  Period:  {start} → {end}")
    print(f"  Data from: {data_start} (extra lookback for MAs/z-scores)")

    # ── Download ───────────────────────────────────────────────────────
    print("\n── Downloading Data ──")
    vix_df = _load_or_download("vix", lambda: _download_vix(data_start), force_download)
    spy_df = _load_or_download("spy", lambda: _download_spy(data_start), force_download)
    spx_df = _load_or_download("spx_constituents", lambda: _download_spx_constituents(data_start), force_download)
    etfs_df = _load_or_download("sector_etfs", lambda: _download_sector_etfs(data_start), force_download)
    fred_df = _load_or_download("fred_liquidity", lambda: _download_fred(data_start), force_download)

    print(f"\n  SPX constituents: {spx_df.shape[1]} tickers, {len(spx_df)} days")
    print(f"  VIX:  {len(vix_df)} days")
    print(f"  FRED: {len(fred_df)} rows, {list(fred_df.columns)}")

    # ── Compute factor score series ────────────────────────────────────
    print("\n── Computing Factor Scores ──")

    spy_close = spy_df["Close"].sort_index()
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]

    print("  VIX …")
    vix_scores = score_vix_series(vix_df)

    print("  Breadth (vectorised over 500 stocks) …")
    breadth_scores = score_breadth_series(spx_df)

    print("  Liquidity (US-only FRED) …")
    liq_scores = score_liquidity_series(fred_df)

    print("  Sector (ETFs vs SPY) …")
    sect_scores = score_sector_series(etfs_df)

    print("  Momentum (relative ROC) …")
    mom_scores = score_momentum_series(spx_df, spy_close)

    # ── Sample on Fridays and build composite ──────────────────────────
    print("\n── Assembling Weekly Composite ──")
    spy_weekly = spy_close.resample("W-FRI").last().dropna()
    fridays = spy_weekly.loc[start:end].index
    print(f"  {len(fridays)} weekly observations")

    rows: list[dict] = []
    for fri in fridays:
        row: dict = {"date": fri.date(), "spy_close": float(spy_weekly.loc[fri])}

        # Sample each factor (use .asof for non-Friday-aligned series)
        row["vix"] = _safe_asof(vix_scores, fri)
        row["breadth"] = _safe_asof(breadth_scores, fri)
        row["liquidity"] = _safe_asof(liq_scores, fri)
        row["sector"] = _safe_asof(sect_scores, fri)
        row["momentum"] = _safe_asof(mom_scores, fri)

        # Weighted composite (reweight available factors)
        weighted = 0.0
        total_w = 0.0
        for key, w in CONFIGURED_WEIGHTS.items():
            v = row.get(key)
            if v is not None and not math.isnan(v):
                weighted += w * v
                total_w += w

        if total_w > 0:
            row["composite"] = weighted / total_w
            row["regime"] = regime_label(row["composite"])
            row["confidence"] = round(total_w, 4)
        else:
            row["composite"] = None
            row["regime"] = None
            row["confidence"] = 0.0

        # Forward SPX returns
        for weeks, col in [(1, "fwd_1w"), (4, "fwd_4w"), (12, "fwd_12w")]:
            target = fri + pd.DateOffset(weeks=weeks)
            if target <= spy_weekly.index[-1]:
                fwd_price = spy_weekly.asof(target)
                row[col] = (fwd_price / row["spy_close"] - 1) * 100
            else:
                row[col] = None

        rows.append(row)

    return pd.DataFrame(rows)


def _safe_asof(series: pd.Series, dt: pd.Timestamp) -> float | None:
    if series.empty:
        return None
    try:
        v = series.asof(dt)
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


# ── Analysis ───────────────────────────────────────────────────────────


def print_coverage(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("FACTOR DATA COVERAGE")
    print("=" * 70)
    for key in CONFIGURED_WEIGHTS:
        avail = df[key].notna().sum()
        total = len(df)
        pct = avail / total * 100
        first = df[df[key].notna()]["date"].iloc[0] if avail > 0 else "N/A"
        print(f"  {key:<14} {avail:>4}/{total} ({pct:>5.1f}%)  first valid: {first}")


def print_regime_returns(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["composite"])

    print("\n" + "=" * 70)
    print(f"REGIME FORWARD RETURNS  (thresholds: risk-on < {DEFAULT_LO}, risk-off ≥ {DEFAULT_HI})")
    print("=" * 70)

    for horizon, label in [("fwd_1w", "1-Week"), ("fwd_4w", "4-Week"), ("fwd_12w", "12-Week")]:
        subset = valid.dropna(subset=[horizon])
        if subset.empty:
            continue
        stats = subset.groupby("regime")[horizon].agg(["mean", "median", "std", "count"])
        stats = stats.reindex(["risk-on", "transitional", "risk-off"])

        print(f"\n  {label} Forward SPX Return (%):")
        print(f"  {'Regime':<16} {'Mean':>8} {'Median':>8} {'StdDev':>8} {'Count':>6}")
        print(f"  {'-' * 16} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")
        for r in ["risk-on", "transitional", "risk-off"]:
            if r in stats.index and not pd.isna(stats.loc[r, "mean"]):
                s = stats.loc[r]
                print(f"  {r:<16} {s['mean']:>8.3f} {s['median']:>8.3f} {s['std']:>8.3f} {int(s['count']):>6}")
        if "risk-on" in stats.index and "risk-off" in stats.index:
            spread = stats.loc["risk-on", "mean"] - stats.loc["risk-off", "mean"]
            print(f"  {'SPREAD (on-off)':<16} {spread:>8.3f}")


def print_threshold_sweep(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["composite", "fwd_4w"])

    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP  (4-Week Forward Return Spread: risk-on − risk-off)")
    print("=" * 70)
    print(
        f"\n  {'Lo':>4} {'Hi':>4} │ {'On Mean':>8} {'Trans':>8} {'Off Mean':>8} │ {'Spread':>8} {'On#':>5} {'Off#':>5}"
    )
    print(f"  {'─' * 4} {'─' * 4} │ {'─' * 8} {'─' * 8} {'─' * 8} │ {'─' * 8} {'─' * 5} {'─' * 5}")

    best_spread = -999.0
    best_lo, best_hi = DEFAULT_LO, DEFAULT_HI

    for lo in range(25, 55, 5):
        for hi in range(lo + 15, 85, 5):
            labels = valid["composite"].apply(lambda s, _lo=lo, _hi=hi: regime_label(s, _lo, _hi))
            groups = valid.groupby(labels)["fwd_4w"]
            means = groups.mean()
            counts = groups.count()

            if "risk-on" not in means or "risk-off" not in means:
                continue
            if counts.get("risk-on", 0) < 10 or counts.get("risk-off", 0) < 10:
                continue

            spread = means["risk-on"] - means["risk-off"]
            trans = means.get("transitional", float("nan"))

            marker = ""
            if lo == int(DEFAULT_LO) and hi == int(DEFAULT_HI):
                marker += " ◄current"
            if spread > best_spread:
                best_spread = spread
                best_lo, best_hi = lo, hi
                marker += " ★best"

            print(
                f"  {lo:>4} {hi:>4} │ {means['risk-on']:>8.3f} {trans:>8.3f} {means['risk-off']:>8.3f}"
                f" │ {spread:>8.3f} {int(counts['risk-on']):>5} {int(counts.get('risk-off', 0)):>5}{marker}"
            )

    print(f"\n  Best:    lo={best_lo}, hi={best_hi}  (spread={best_spread:+.3f})")
    print(f"  Current: lo={int(DEFAULT_LO)}, hi={int(DEFAULT_HI)}")


def print_factor_analysis(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["fwd_4w"])

    print("\n" + "=" * 70)
    print("PER-FACTOR PREDICTIVE POWER  (4-Week Forward Returns)")
    print("=" * 70)
    print(f"\n  {'Factor':<14} │ {'Corr':>7} {'IC':>7} │ {'Lo20%':>8} {'Hi80%':>8} {'Spread':>8} │ {'Weight':>6}")
    print(f"  {'─' * 14} │ {'─' * 7} {'─' * 7} │ {'─' * 8} {'─' * 8} {'─' * 8} │ {'─' * 6}")

    for key in CONFIGURED_WEIGHTS:
        subset = valid.dropna(subset=[key])
        if len(subset) < 30:
            print(f"  {key:<14} │ insufficient data ({len(subset)} obs)")
            continue

        factor = subset[key]
        fwd = subset["fwd_4w"]

        # Spearman rank correlation
        corr = factor.corr(fwd, method="spearman")
        ic = factor.rank().corr(fwd.rank())

        # Quintile spread (low factor = risk-on = should have higher returns)
        lo_thresh = factor.quantile(0.20)
        hi_thresh = factor.quantile(0.80)
        lo_mean = fwd[factor <= lo_thresh].mean()
        hi_mean = fwd[factor >= hi_thresh].mean()
        spread = lo_mean - hi_mean  # positive = signal works

        weight = CONFIGURED_WEIGHTS[key]
        print(
            f"  {key:<14} │ {corr:>7.3f} {ic:>7.3f} │ {lo_mean:>8.3f} {hi_mean:>8.3f} {spread:>8.3f} │ {weight:>6.2f}"
        )

    print("\n  Interpretation:")
    print("    Corr/IC: Negative = factor correctly predicts lower returns when elevated")
    print("    Spread:  Positive = low-risk-score periods outperform high-risk-score periods")
    print("    Both negative corr and positive spread = factor has real predictive signal")


def print_summary_stats(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["composite"])

    print("\n" + "=" * 70)
    print("COMPOSITE SCORE DISTRIBUTION")
    print("=" * 70)

    print(f"\n  {'Stat':<16} {'Value':>8}")
    print(f"  {'─' * 16} {'─' * 8}")
    for stat, val in [
        ("Mean", valid["composite"].mean()),
        ("Median", valid["composite"].median()),
        ("Std Dev", valid["composite"].std()),
        ("Min", valid["composite"].min()),
        ("Max", valid["composite"].max()),
        ("25th pct", valid["composite"].quantile(0.25)),
        ("75th pct", valid["composite"].quantile(0.75)),
    ]:
        print(f"  {stat:<16} {val:>8.2f}")

    regime_counts = valid["regime"].value_counts()
    total = len(valid)
    print(f"\n  {'Regime':<16} {'Count':>6} {'Pct':>6}")
    print(f"  {'─' * 16} {'─' * 6} {'─' * 6}")
    for r in ["risk-on", "transitional", "risk-off"]:
        n = regime_counts.get(r, 0)
        print(f"  {r:<16} {n:>6} {n / total * 100:>5.1f}%")


# ── Entrypoint ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Signal Aggregator Backtest")
    parser.add_argument("--start", default="2016-03-01", help="Backtest start (default: 2016-03-01)")
    parser.add_argument("--end", default=None, help="Backtest end (default: today)")
    parser.add_argument("--force", action="store_true", help="Force re-download all data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")

    df = run_backtest(start=args.start, end=args.end, force_download=args.force)

    # Save raw results
    output_path = CACHE_DIR / "backtest_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  Raw results → {output_path}")

    # Full analysis
    print_coverage(df)
    print_summary_stats(df)
    print_regime_returns(df)
    print_threshold_sweep(df)
    print_factor_analysis(df)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
