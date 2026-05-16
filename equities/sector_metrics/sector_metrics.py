#!/usr/bin/env python3
"""
Compute:
1) Relative weight of each of the 11 S&P 500 sectors (sector market cap / total market cap)
2) Change in relative weight vs ~1, ~3, ~6 months ago
3) Percent above the 200-day moving average for each sector (using SPDR sector ETFs as proxies)

Data sources:
- S&P 500 constituents + GICS sector: Wikipedia
- Prices + current market cap/shares: Yahoo Finance via yfinance

Notes / limitations:
- Yahoo Finance does NOT provide reliable historical market cap series for free.
  For past “market cap”, this script approximates:
      market_cap(t) ≈ shares_now * price(t)
  where shares_now is sharesOutstanding (if available) else inferred from marketCap/current_price.
  This ignores share count changes (buybacks, issuance, splits already handled in Adj Close).

python3 sector_metrics.py --outdir results
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import time
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance") from e

from utils.retry import yf_download

# Set User-Agent to avoid 403 errors from Yahoo Finance / Wikipedia
opener = urllib.request.build_opener()
opener.addheaders = [
    (
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    )
]
urllib.request.install_opener(opener)


SECTOR_ETFS: dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


BENCHMARK_ETF = "SPY"
DEFAULT_CACHE_TTL_HOURS = 24.0
DEFAULT_METADATA_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".sector_metrics_marketdata_cache.csv")


@dataclass(frozen=True)
class Lookbacks:
    one_month: int = 1
    three_month: int = 3
    six_month: int = 6
    twelve_month: int = 12


def _fix_yahoo_ticker(sym: str) -> str:
    # Wikipedia uses dots for share classes; Yahoo uses dashes.
    # Example: BRK.B -> BRK-B, BF.B -> BF-B
    return sym.replace(".", "-")


def get_sp500_constituents() -> pd.DataFrame:
    # Wikipedia page often changes layout; we select the first matching table with "Symbol".
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    table = None
    for t in tables:
        if "Symbol" in t.columns and ("GICS Sector" in t.columns or "Sector" in t.columns):
            table = t.copy()
            break
    if table is None:
        raise RuntimeError("Could not find S&P 500 constituents table on Wikipedia.")

    sector_col = "GICS Sector" if "GICS Sector" in table.columns else "Sector"
    out = table[["Symbol", sector_col]].rename(columns={sector_col: "Sector"})
    out["Ticker"] = out["Symbol"].astype(str).map(_fix_yahoo_ticker)
    out = out[["Ticker", "Sector"]].dropna()
    return out


def download_prices(
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
    batch_size: int = 100,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Returns dataframe indexed by date with columns=tickers (Adj Close if auto_adjust=True).
    Uses batching to reduce Yahoo throttling failures.
    """
    all_closes: list[pd.DataFrame] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        data = yf_download(
            tickers=" ".join(batch),
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
            threads=True,
        )
        if data.empty:
            continue

        # With multiple tickers, yfinance returns a multiindex columns df: (field, ticker)
        if isinstance(data.columns, pd.MultiIndex):
            # Prefer "Close" when auto_adjust=True. If not, use "Adj Close".
            field = "Close" if ("Close" in data.columns.get_level_values(0)) else "Adj Close"
            closes = data[field].copy()
        else:
            # Single ticker: columns like ["Open","High","Low","Close",...]
            field = "Close" if "Close" in data.columns else "Adj Close"
            closes = data[[field]].rename(columns={field: batch[0]})

        all_closes.append(closes)

    if not all_closes:
        raise RuntimeError("Price download failed for all batches.")

    closes_all = pd.concat(all_closes, axis=1)
    closes_all = closes_all.sort_index()
    # Drop duplicate columns if any batch overlaps
    closes_all = closes_all.loc[:, ~closes_all.columns.duplicated()]
    return closes_all


SECTOR_SERIES_TIMEFRAMES = {
    "This Week": {"period": "5d", "interval": "15m"},
    "Daily": {"period": "90d", "interval": "1d"},
    "Weekly": {"period": "2y", "interval": "1wk"},
    "Monthly": {"period": "5y", "interval": "1mo"},
}


def get_sector_etf_series(timeframe: str = "Daily") -> dict:
    """
    Return sector ETF price series and SPY-relative ratio series.

    The raw series are adjusted sector SPDR ETF prices. The relative series are
    ETF price divided by SPY price, so frontend chart normalization shows each
    sector's performance versus the benchmark over the selected window.
    """
    tf = SECTOR_SERIES_TIMEFRAMES.get(timeframe)
    if tf is None:
        return {"error": f"Invalid timeframe: {timeframe}"}

    etf_prices = fetch_etf_prices(
        SECTOR_ETFS,
        benchmark=BENCHMARK_ETF,
        period=tf["period"],
        interval=tf["interval"],
    ).ffill()

    if etf_prices.empty:
        return {"error": "No data returned from yfinance"}
    if hasattr(etf_prices.index, "tz") and etf_prices.index.tz is not None:
        etf_prices.index = etf_prices.index.tz_localize(None)

    sector_prices: dict[str, pd.Series] = {}
    sector_relative_prices: dict[str, pd.Series] = {}
    benchmark_raw = etf_prices[BENCHMARK_ETF] if BENCHMARK_ETF in etf_prices.columns else pd.Series(dtype=float)
    benchmark = pd.to_numeric(benchmark_raw, errors="coerce")

    for sector, ticker in SECTOR_ETFS.items():
        if ticker not in etf_prices.columns:
            continue

        prices = pd.to_numeric(etf_prices[ticker], errors="coerce").dropna()
        if prices.empty:
            continue

        sector_prices[sector] = prices

        aligned = pd.concat([prices.rename("sector"), benchmark.rename("benchmark")], axis=1).ffill().dropna()
        aligned = aligned[aligned["benchmark"] > 0]
        if not aligned.empty:
            sector_relative_prices[sector] = aligned["sector"] / aligned["benchmark"]

    return {
        "sector_prices": sector_prices,
        "sector_relative_prices": sector_relative_prices,
        "sector_order": list(SECTOR_ETFS.keys()),
        "benchmark": BENCHMARK_ETF,
        "timeframe": timeframe,
        "timestamp": dt.datetime.now(),
    }


def nearest_on_or_before(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    target = pd.Timestamp(target).tz_localize(None)
    idx = index[index <= target]
    if len(idx) == 0:
        return index.min()
    return idx.max()


def month_ago_dates(prices_index: pd.DatetimeIndex, months: int) -> pd.Timestamp:
    # Use calendar months, then snap to trading day <= target
    today = prices_index.max()
    target = today - pd.DateOffset(months=months)
    return nearest_on_or_before(prices_index, target)


def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def _extract_numeric_field(obj, keys: Iterable[str]) -> float | None:
    for key in keys:
        value = None
        try:
            if hasattr(obj, "get"):
                value = obj.get(key)
            else:
                value = getattr(obj, key, None)
        except Exception:
            value = getattr(obj, key, None)
        num = _safe_float(value)
        if num is not None and not math.isnan(num):
            return num
    return None


def _load_marketdata_cache(cache_path: str | None) -> pd.DataFrame:
    cols = ["MarketCap", "SharesOutstanding", "CacheTs"]
    if not cache_path or not os.path.exists(cache_path):
        return pd.DataFrame(columns=cols, index=pd.Index([], name="Ticker"))
    try:
        cache = pd.read_csv(cache_path)
    except Exception:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="Ticker"))

    required = {"Ticker", "MarketCap", "SharesOutstanding", "CacheTs"}
    if not required.issubset(cache.columns):
        return pd.DataFrame(columns=cols, index=pd.Index([], name="Ticker"))

    cache = cache[["Ticker", "MarketCap", "SharesOutstanding", "CacheTs"]].dropna(subset=["Ticker"])
    cache = cache.drop_duplicates(subset=["Ticker"], keep="last").set_index("Ticker")
    cache["MarketCap"] = pd.to_numeric(cache["MarketCap"], errors="coerce")
    cache["SharesOutstanding"] = pd.to_numeric(cache["SharesOutstanding"], errors="coerce")
    cache["CacheTs"] = pd.to_numeric(cache["CacheTs"], errors="coerce")
    return cache


def _write_marketdata_cache(cache: pd.DataFrame, cache_path: str | None) -> None:
    if not cache_path:
        return
    try:
        folder = os.path.dirname(cache_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        out = cache[["MarketCap", "SharesOutstanding", "CacheTs"]].copy()
        out = out[~out.index.duplicated(keep="last")]
        out.reset_index().to_csv(cache_path, index=False, float_format="%.10g")
    except Exception:
        # Cache write failures should never break the main data pipeline.
        return


def fetch_marketcap_and_shares(
    tickers: list[str],
    last_prices: pd.Series,
    max_workers: int = 12,
    cache_path: str | None = DEFAULT_METADATA_CACHE_PATH,
    cache_ttl_hours: float = DEFAULT_CACHE_TTL_HOURS,
) -> pd.DataFrame:
    """
    Fetch current marketCap + sharesOutstanding per ticker via yfinance.
    Uses a local cache to avoid refreshing unchanged metadata on every run.
    If sharesOutstanding missing, infer shares from marketCap / last_price if possible.
    """
    ordered_tickers = list(dict.fromkeys(tickers))
    cache = _load_marketdata_cache(cache_path)
    now_ts = time.time()
    stale_cutoff = now_ts - (cache_ttl_hours * 3600.0)

    cached_subset = cache.reindex(ordered_tickers)
    if cached_subset.empty:
        fresh_mask = pd.Series(False, index=pd.Index(ordered_tickers, name="Ticker"))
    else:
        fresh_mask = cached_subset["CacheTs"].ge(stale_cutoff).fillna(False)

    tickers_to_fetch = [t for t in ordered_tickers if not bool(fresh_mask.get(t, False))]

    def worker(t: str) -> tuple[str, float | None, float | None]:
        ticker_obj = yf.Ticker(t)
        mcap = None
        shares = None
        try:
            fast_info = ticker_obj.fast_info
            mcap = _extract_numeric_field(fast_info, ("market_cap", "marketCap"))
            shares = _extract_numeric_field(fast_info, ("shares", "shares_outstanding", "sharesOutstanding"))
        except Exception:
            pass

        # Fall back to get_info for fields unavailable in fast_info.
        if mcap is None or shares is None:
            try:
                info = ticker_obj.get_info()
                if mcap is None:
                    mcap = _safe_float(info.get("marketCap"))
                if shares is None:
                    shares = _safe_float(info.get("sharesOutstanding"))
            except Exception:
                pass

        return t, mcap, shares

    results: list[tuple[str, float | None, float | None]] = []
    if tickers_to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(worker, t): t for t in tickers_to_fetch}
            for fut in as_completed(futs):
                results.append(fut.result())

    fetched = pd.DataFrame(results, columns=["Ticker", "MarketCap", "SharesOutstanding"])
    if fetched.empty:
        fetched = pd.DataFrame(columns=["MarketCap", "SharesOutstanding"], index=pd.Index([], name="Ticker"))
    else:
        fetched = fetched.set_index("Ticker")
        fetched["MarketCap"] = pd.to_numeric(fetched["MarketCap"], errors="coerce")
        fetched["SharesOutstanding"] = pd.to_numeric(fetched["SharesOutstanding"], errors="coerce")

    # If a refresh fails, keep stale cached data as fallback.
    if not fetched.empty:
        stale_fallback = cached_subset.reindex(fetched.index)[["MarketCap", "SharesOutstanding"]]
        fetched[["MarketCap", "SharesOutstanding"]] = fetched[["MarketCap", "SharesOutstanding"]].where(
            fetched[["MarketCap", "SharesOutstanding"]].notna(),
            stale_fallback,
        )

    fresh_cached = cached_subset[fresh_mask][["MarketCap", "SharesOutstanding"]]
    raw = pd.concat([fresh_cached, fetched], axis=0)
    raw = raw[~raw.index.duplicated(keep="last")].reindex(ordered_tickers)

    # Update cache only with rows that actually returned at least one value.
    if not fetched.empty and cache_path:
        valid = fetched[fetched[["MarketCap", "SharesOutstanding"]].notna().any(axis=1)].copy()
        if not valid.empty:
            valid["CacheTs"] = now_ts
            cache_next = pd.concat([cache, valid], axis=0)
            cache_next = cache_next[~cache_next.index.duplicated(keep="last")]
            _write_marketdata_cache(cache_next, cache_path)

    df = raw.copy()
    df["MarketCap"] = pd.to_numeric(df["MarketCap"], errors="coerce")
    df["SharesOutstanding"] = pd.to_numeric(df["SharesOutstanding"], errors="coerce")

    px = pd.to_numeric(last_prices.reindex(df.index), errors="coerce")
    infer_mask = df["SharesOutstanding"].isna() & df["MarketCap"].notna() & px.notna() & (px > 0)
    df.loc[infer_mask, "SharesOutstanding"] = df.loc[infer_mask, "MarketCap"] / px[infer_mask]
    df["SharesInferred"] = infer_mask.fillna(False)
    return df


def compute_sector_weights_for_dates(
    constituents: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.Series,
    asof_dates: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """
    Compute sector weights for multiple as-of dates in one vectorized pass.
    Returns a DataFrame indexed by sector with columns from `asof_dates`.
    """
    if not asof_dates:
        return pd.DataFrame()

    snapped_dates = {
        label: nearest_on_or_before(prices.index, pd.Timestamp(asof).tz_localize(None))
        for label, asof in asof_dates.items()
    }
    labels = list(asof_dates.keys())

    base = constituents[["Ticker", "Sector"]].dropna().drop_duplicates(subset=["Ticker"], keep="last").copy()
    base = base.merge(shares.rename("Shares"), left_on="Ticker", right_index=True, how="left")
    base = base.dropna(subset=["Shares"])
    base = base[base["Ticker"].isin(prices.columns)].copy()
    if base.empty:
        return pd.DataFrame(index=pd.Index([], name="Sector"), columns=labels)

    tickers = base["Ticker"].tolist()
    px = prices.loc[[snapped_dates[label] for label in labels], tickers].copy()
    px.index = labels

    mapper = base.set_index("Ticker")
    caps = px.T.mul(mapper["Shares"], axis=0)
    caps["Sector"] = mapper["Sector"]

    sector_caps = caps.groupby("Sector").sum(min_count=1)
    totals = sector_caps.sum(axis=0)
    weights = sector_caps.div(totals, axis=1)
    return weights.sort_index()


def compute_sector_weights(
    constituents: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.Series,
    asof: pd.Timestamp,
) -> pd.Series:
    """
    Sector weight at date `asof`:
    sector_cap(asof) / total_cap(asof), where cap(asof) ≈ shares_now * price(asof)
    """
    col = "__asof__"
    out = compute_sector_weights_for_dates(
        constituents=constituents,
        prices=prices,
        shares=shares,
        asof_dates={col: asof},
    )
    if col not in out.columns:
        return pd.Series(dtype=float)
    return out[col].dropna()


def fetch_etf_prices(
    sector_etfs: dict[str, str],
    benchmark: str = BENCHMARK_ETF,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download all sector ETF prices plus benchmark in one pass."""
    tickers = list(sector_etfs.values()) + [benchmark]
    return download_prices(tickers, period=period, interval=interval, batch_size=50, auto_adjust=True)


def compute_pct_above_200dma(sector_etfs: dict[str, str], etf_prices: pd.DataFrame) -> pd.Series:
    """
    For each sector ETF: (last_close - SMA200) / SMA200 * 100
    """
    out = {}
    for sector, etf in sector_etfs.items():
        if etf not in etf_prices.columns:
            out[sector] = np.nan
            continue
        s = etf_prices[etf].dropna()
        if len(s) < 220:
            out[sector] = np.nan
            continue
        sma200 = s.rolling(200).mean()
        last = float(s.iloc[-1])
        ma = float(sma200.iloc[-1])
        out[sector] = (last - ma) / ma * 100.0 if ma and not math.isnan(ma) else np.nan

    return pd.Series(out).sort_index()


def compute_relative_performance(
    sector_etfs: dict[str, str],
    etf_prices: pd.DataFrame,
    benchmark: str = BENCHMARK_ETF,
    lookback_months: list[int] = [1, 3, 6, 12],  # noqa: B006
) -> pd.DataFrame:
    """
    Relative performance of each sector ETF vs the benchmark over each lookback period.

    rel_perf(sector, N months) = sector_return(N months) - benchmark_return(N months)

    Returns a DataFrame indexed by sector with columns RelPerf_1M_pp, RelPerf_3M_pp, etc.
    All values are in percentage points.
    """
    idx = etf_prices.index
    today = idx.max()
    prices_ffill = etf_prices.ffill()

    rows: dict[str, dict[str, float]] = {}
    for sector, etf in sector_etfs.items():
        row: dict[str, float] = {}
        for m in lookback_months:
            col = f"RelPerf_{m}M_pp"
            d_past = nearest_on_or_before(idx, today - pd.DateOffset(months=m))
            try:
                etf_now = float(prices_ffill.loc[today, etf])
                etf_then = float(prices_ffill.loc[d_past, etf])
                bench_now = float(prices_ffill.loc[today, benchmark])
                bench_then = float(prices_ffill.loc[d_past, benchmark])
                if any(math.isnan(v) for v in [etf_now, etf_then, bench_now, bench_then]):
                    row[col] = np.nan
                elif etf_then == 0 or bench_then == 0:
                    row[col] = np.nan
                else:
                    etf_ret = (etf_now - etf_then) / etf_then * 100.0
                    bench_ret = (bench_now - bench_then) / bench_then * 100.0
                    row[col] = etf_ret - bench_ret
            except (KeyError, ValueError):
                row[col] = np.nan
        rows[sector] = row

    return pd.DataFrame(rows).T.sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="2y", help="Price history window (default: 2y).")
    ap.add_argument("--batch-size", type=int, default=100, help="Ticker batch size for price downloads.")
    ap.add_argument("--max-workers", type=int, default=12, help="Threads for yfinance info calls.")
    ap.add_argument("--outdir", default=".", help="Directory to write CSV outputs.")
    ap.add_argument(
        "--cache-path",
        default=None,
        help="Path to market data cache CSV (default: <outdir>/sp500_ticker_marketdata_cache.csv).",
    )
    ap.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=DEFAULT_CACHE_TTL_HOURS,
        help=f"Refresh cached market data older than this many hours (default: {DEFAULT_CACHE_TTL_HOURS}).",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cache_path = args.cache_path or os.path.join(args.outdir, "sp500_ticker_marketdata_cache.csv")

    constituents = get_sp500_constituents()
    tickers = sorted(constituents["Ticker"].unique().tolist())

    prices = download_prices(tickers, period=args.period, batch_size=args.batch_size, auto_adjust=True)
    # Last available price per ticker
    last_prices = prices.ffill().iloc[-1].dropna()

    md = fetch_marketcap_and_shares(
        tickers=list(last_prices.index),
        last_prices=last_prices,
        max_workers=args.max_workers,
        cache_path=cache_path,
        cache_ttl_hours=args.cache_ttl_hours,
    )
    shares = md["SharesOutstanding"].dropna()

    # Compute weights now and at lookbacks
    idx = prices.index
    asof_now = idx.max()
    d_1m = month_ago_dates(idx, 1)
    d_3m = month_ago_dates(idx, 3)
    d_6m = month_ago_dates(idx, 6)

    weight_now_col = "Weight_Now"
    weight_1m_col = f"Weight_{d_1m.date()}"
    weight_3m_col = f"Weight_{d_3m.date()}"
    weight_6m_col = f"Weight_{d_6m.date()}"

    weights_by_date = compute_sector_weights_for_dates(
        constituents=constituents,
        prices=prices,
        shares=shares,
        asof_dates={
            weight_now_col: asof_now,
            weight_1m_col: d_1m,
            weight_3m_col: d_3m,
            weight_6m_col: d_6m,
        },
    )

    # Align on the 11 standard sectors (some may be absent if data missing)
    all_sectors = sorted(set(SECTOR_ETFS.keys()) | set(weights_by_date.index))
    weights = weights_by_date.reindex(all_sectors).reindex(
        columns=[weight_now_col, weight_1m_col, weight_3m_col, weight_6m_col]
    )

    # Changes in percentage points
    weights["Chg_1M_pp"] = (weights[weight_now_col] - weights[weight_1m_col]) * 100.0
    weights["Chg_3M_pp"] = (weights[weight_now_col] - weights[weight_3m_col]) * 100.0
    weights["Chg_6M_pp"] = (weights[weight_now_col] - weights[weight_6m_col]) * 100.0

    # Download sector ETF prices (+ SPY benchmark) once, share across computations
    etf_prices = fetch_etf_prices(SECTOR_ETFS, benchmark=BENCHMARK_ETF, period=args.period)

    # Percent above 200DMA (sector ETFs)
    pct_above_200 = compute_pct_above_200dma(SECTOR_ETFS, etf_prices)
    weights["Pct_Above_200DMA"] = pct_above_200.reindex(all_sectors)

    # Relative performance vs S&P 500 (SPY) over 1, 3, 6, 12 months
    rel_perf = compute_relative_performance(
        SECTOR_ETFS, etf_prices, benchmark=BENCHMARK_ETF, lookback_months=[1, 3, 6, 12]
    )
    for col in rel_perf.columns:
        weights[col] = rel_perf[col].reindex(all_sectors)

    # Pretty formatting columns (keep numeric raw too)
    weights_sorted = weights.loc[list(SECTOR_ETFS.keys())].copy()

    # Output
    path_weights = os.path.join(args.outdir, "sp500_sector_weights_and_changes.csv")
    path_marketdata = os.path.join(args.outdir, "sp500_ticker_marketdata_snapshot.csv")

    weights_sorted.to_csv(path_weights, float_format="%.6f")
    md.to_csv(path_marketdata, float_format="%.6f")

    # Print summary
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    display = weights_sorted.copy()
    # Display weights as % and changes as pp
    for c in display.columns:
        if c.startswith("Weight_"):
            display[c] = display[c] * 100.0
    print("\nSector weights (% of total S&P 500 market cap, approximated) and changes (pp):")
    print(
        display[
            [
                weight_now_col,
                weight_1m_col,
                weight_3m_col,
                weight_6m_col,
                "Chg_1M_pp",
                "Chg_3M_pp",
                "Chg_6M_pp",
                "Pct_Above_200DMA",
            ]
        ].round(3)
    )

    print("\nRelative performance vs S&P 500 / SPY (percentage points; positive = outperformed):")
    print(display[["RelPerf_1M_pp", "RelPerf_3M_pp", "RelPerf_6M_pp", "RelPerf_12M_pp"]].round(2))

    print(f"\nWrote:\n- {path_weights}\n- {path_marketdata}")
    print("\nInterpretation notes:")
    print(
        "- Sector weights are based on constituents’ current sharesOutstanding (or inferred) times historical prices."
    )
    print("- Pct_Above_200DMA is computed from sector ETF prices (SPDR Select Sector ETFs).")
    print(f"- Relative performance uses SPDR sector ETFs vs {BENCHMARK_ETF} as the S&P 500 benchmark.")


def get_data(
    period: str = "2y",
    batch_size: int = 100,
    max_workers: int = 12,
    cache_path: str | None = DEFAULT_METADATA_CACHE_PATH,
    cache_ttl_hours: float = DEFAULT_CACHE_TTL_HOURS,
    prices_df: pd.DataFrame | None = None,
) -> dict:
    """
    Return sector metrics as a dict for the frontend.

    Keys:
      weights_df  — DataFrame indexed by sector; weight columns already in %,
                    change columns in pp, relperf columns in pp, 200DMA column in %
      d_1m / d_3m / d_6m — date strings for the lookback snapshots
      timestamp   — datetime when data was fetched

    If *prices_df* is supplied (a MultiIndex DataFrame from yfinance), the
    constituent price download is skipped and the pre-fetched data is used.
    Metadata and ETF prices are still fetched independently.
    """
    constituents = get_sp500_constituents()
    tickers = sorted(constituents["Ticker"].unique().tolist())

    if prices_df is not None:
        # Extract Close prices from pre-fetched MultiIndex DataFrame
        if isinstance(prices_df.columns, pd.MultiIndex):
            field = "Close" if "Close" in prices_df.columns.get_level_values(0) else "Adj Close"
            prices = prices_df[field].copy()
        else:
            prices = prices_df.copy()
        # Keep only tickers in our constituents list
        available = [t for t in tickers if t in prices.columns]
        prices = prices[available].sort_index()
    else:
        prices = download_prices(tickers, period=period, batch_size=batch_size, auto_adjust=True)
    last_prices = prices.ffill().iloc[-1].dropna()

    md = fetch_marketcap_and_shares(
        tickers=list(last_prices.index),
        last_prices=last_prices,
        max_workers=max_workers,
        cache_path=cache_path,
        cache_ttl_hours=cache_ttl_hours,
    )
    shares = md["SharesOutstanding"].dropna()

    idx = prices.index
    asof_now = idx.max()
    d_1m = month_ago_dates(idx, 1)
    d_3m = month_ago_dates(idx, 3)
    d_6m = month_ago_dates(idx, 6)

    weights_by_date = compute_sector_weights_for_dates(
        constituents=constituents,
        prices=prices,
        shares=shares,
        asof_dates={
            "Weight_Now": asof_now,
            "Weight_1M": d_1m,
            "Weight_3M": d_3m,
            "Weight_6M": d_6m,
        },
    )

    all_sectors = sorted(set(SECTOR_ETFS.keys()) | set(weights_by_date.index))

    weights = weights_by_date.reindex(all_sectors).reindex(
        columns=["Weight_Now", "Weight_1M", "Weight_3M", "Weight_6M"]
    )

    weights["Chg_1M_pp"] = (weights["Weight_Now"] - weights["Weight_1M"]) * 100.0
    weights["Chg_3M_pp"] = (weights["Weight_Now"] - weights["Weight_3M"]) * 100.0
    weights["Chg_6M_pp"] = (weights["Weight_Now"] - weights["Weight_6M"]) * 100.0

    etf_prices = fetch_etf_prices(SECTOR_ETFS, benchmark=BENCHMARK_ETF, period=period)
    pct_above_200 = compute_pct_above_200dma(SECTOR_ETFS, etf_prices)
    weights["Pct_Above_200DMA"] = pct_above_200.reindex(all_sectors)

    rel_perf = compute_relative_performance(
        SECTOR_ETFS, etf_prices, benchmark=BENCHMARK_ETF, lookback_months=[1, 3, 6, 12]
    )
    for col in rel_perf.columns:
        weights[col] = rel_perf[col].reindex(all_sectors)

    weights_sorted = weights.loc[list(SECTOR_ETFS.keys())].copy()
    # Convert raw weight fractions → percentages for display
    for c in ["Weight_Now", "Weight_1M", "Weight_3M", "Weight_6M"]:
        weights_sorted[c] = weights_sorted[c] * 100.0

    return {
        "weights_df": weights_sorted,
        "d_1m": str(d_1m.date()),
        "d_3m": str(d_3m.date()),
        "d_6m": str(d_6m.date()),
        "timestamp": dt.datetime.now(),
    }


if __name__ == "__main__":
    main()
