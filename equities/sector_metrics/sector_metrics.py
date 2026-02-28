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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import urllib.request

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance") from e

# Set User-Agent to avoid 403 errors from Yahoo Finance / Wikipedia
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')]
urllib.request.install_opener(opener)


SECTOR_ETFS: Dict[str, str] = {
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
    tickers: List[str],
    period: str = "2y",
    batch_size: int = 100,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Returns dataframe indexed by date with columns=tickers (Adj Close if auto_adjust=True).
    Uses batching to reduce Yahoo throttling failures.
    """
    all_closes: List[pd.DataFrame] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        data = yf.download(
            tickers=" ".join(batch),
            period=period,
            interval="1d",
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


def _safe_float(x) -> Optional[float]:
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


def fetch_marketcap_and_shares(
    tickers: List[str],
    last_prices: pd.Series,
    max_workers: int = 12,
) -> pd.DataFrame:
    """
    Fetch current marketCap + sharesOutstanding per ticker via yfinance info.
    If sharesOutstanding missing, infer shares from marketCap / last_price if possible.
    """
    def worker(t: str) -> Tuple[str, Optional[float], Optional[float]]:
        try:
            info = yf.Ticker(t).get_info()
            mcap = _safe_float(info.get("marketCap"))
            shares = _safe_float(info.get("sharesOutstanding"))
            return t, mcap, shares
        except Exception:
            return t, None, None

    results: List[Tuple[str, Optional[float], Optional[float]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(worker, t): t for t in tickers}
        for fut in as_completed(futs):
            results.append(fut.result())

    df = pd.DataFrame(results, columns=["Ticker", "MarketCap", "SharesOutstanding"]).set_index("Ticker")

    # Infer shares if missing
    inferred = []
    for t in df.index:
        shares = df.at[t, "SharesOutstanding"]
        if shares is not None and not (isinstance(shares, float) and math.isnan(shares)):
            inferred.append(False)
            continue
        mcap = df.at[t, "MarketCap"]
        px = _safe_float(last_prices.get(t))
        if mcap is not None and px is not None and px > 0:
            df.at[t, "SharesOutstanding"] = mcap / px
            inferred.append(True)
        else:
            inferred.append(False)

    df["SharesInferred"] = inferred
    return df


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
    asof = pd.Timestamp(asof).tz_localize(None)
    asof = nearest_on_or_before(prices.index, asof)

    px = prices.loc[asof].copy()
    # Keep only tickers in constituents and available in price data
    df = constituents.copy()
    df = df[df["Ticker"].isin(px.index)].copy()

    df["Price"] = df["Ticker"].map(px.to_dict())
    df["Shares"] = df["Ticker"].map(shares.to_dict())
    df = df.dropna(subset=["Price", "Shares"])
    df["Cap"] = df["Price"].astype(float) * df["Shares"].astype(float)

    sector_caps = df.groupby("Sector")["Cap"].sum().sort_index()
    total = sector_caps.sum()
    weights = sector_caps / total
    return weights


def fetch_etf_prices(sector_etfs: Dict[str, str], benchmark: str = BENCHMARK_ETF, period: str = "2y") -> pd.DataFrame:
    """Download all sector ETF prices plus benchmark in one pass."""
    tickers = list(sector_etfs.values()) + [benchmark]
    return download_prices(tickers, period=period, batch_size=50, auto_adjust=True)


def compute_pct_above_200dma(sector_etfs: Dict[str, str], etf_prices: pd.DataFrame) -> pd.Series:
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
    sector_etfs: Dict[str, str],
    etf_prices: pd.DataFrame,
    benchmark: str = BENCHMARK_ETF,
    lookback_months: List[int] = [1, 3, 6, 12],
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

    rows: Dict[str, Dict[str, float]] = {}
    for sector, etf in sector_etfs.items():
        row: Dict[str, float] = {}
        for m in lookback_months:
            col = f"RelPerf_{m}M_pp"
            d_past = nearest_on_or_before(idx, today - pd.DateOffset(months=m))
            try:
                etf_now   = float(prices_ffill.loc[today,  etf])
                etf_then  = float(prices_ffill.loc[d_past, etf])
                bench_now  = float(prices_ffill.loc[today,  benchmark])
                bench_then = float(prices_ffill.loc[d_past, benchmark])
                if any(math.isnan(v) for v in [etf_now, etf_then, bench_now, bench_then]):
                    row[col] = np.nan
                elif etf_then == 0 or bench_then == 0:
                    row[col] = np.nan
                else:
                    etf_ret   = (etf_now  - etf_then)  / etf_then  * 100.0
                    bench_ret = (bench_now - bench_then) / bench_then * 100.0
                    row[col]  = etf_ret - bench_ret
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
    args = ap.parse_args()

    constituents = get_sp500_constituents()
    tickers = sorted(constituents["Ticker"].unique().tolist())

    prices = download_prices(tickers, period=args.period, batch_size=args.batch_size, auto_adjust=True)
    # Last available price per ticker
    last_prices = prices.ffill().iloc[-1].dropna()

    md = fetch_marketcap_and_shares(
        tickers=list(last_prices.index),
        last_prices=last_prices,
        max_workers=args.max_workers,
    )
    shares = md["SharesOutstanding"].dropna()

    # Compute weights now and at lookbacks
    idx = prices.index
    asof_now = idx.max()
    d_1m = month_ago_dates(idx, 1)
    d_3m = month_ago_dates(idx, 3)
    d_6m = month_ago_dates(idx, 6)

    w_now = compute_sector_weights(constituents, prices, shares, asof_now)
    w_1m = compute_sector_weights(constituents, prices, shares, d_1m)
    w_3m = compute_sector_weights(constituents, prices, shares, d_3m)
    w_6m = compute_sector_weights(constituents, prices, shares, d_6m)

    # Align on the 11 standard sectors (some may be absent if data missing)
    all_sectors = sorted(set(SECTOR_ETFS.keys()) | set(w_now.index) | set(w_1m.index) | set(w_3m.index) | set(w_6m.index))

    weights = pd.DataFrame(
        {
            "Weight_Now": w_now.reindex(all_sectors),
            f"Weight_{d_1m.date()}": w_1m.reindex(all_sectors),
            f"Weight_{d_3m.date()}": w_3m.reindex(all_sectors),
            f"Weight_{d_6m.date()}": w_6m.reindex(all_sectors),
        }
    )

    # Changes in percentage points
    weights["Chg_1M_pp"] = (weights["Weight_Now"] - weights[f"Weight_{d_1m.date()}"]) * 100.0
    weights["Chg_3M_pp"] = (weights["Weight_Now"] - weights[f"Weight_{d_3m.date()}"]) * 100.0
    weights["Chg_6M_pp"] = (weights["Weight_Now"] - weights[f"Weight_{d_6m.date()}"]) * 100.0

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
    os.makedirs(args.outdir, exist_ok=True)
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
    print(display[[
        "Weight_Now",
        f"Weight_{d_1m.date()}",
        f"Weight_{d_3m.date()}",
        f"Weight_{d_6m.date()}",
        "Chg_1M_pp",
        "Chg_3M_pp",
        "Chg_6M_pp",
        "Pct_Above_200DMA",
    ]].round(3))

    print("\nRelative performance vs S&P 500 / SPY (percentage points; positive = outperformed):")
    print(display[["RelPerf_1M_pp", "RelPerf_3M_pp", "RelPerf_6M_pp", "RelPerf_12M_pp"]].round(2))

    print(f"\nWrote:\n- {path_weights}\n- {path_marketdata}")
    print("\nInterpretation notes:")
    print("- Sector weights are based on constituents’ current sharesOutstanding (or inferred) times historical prices.")
    print("- Pct_Above_200DMA is computed from sector ETF prices (SPDR Select Sector ETFs).")
    print(f"- Relative performance uses SPDR sector ETFs vs {BENCHMARK_ETF} as the S&P 500 benchmark.")


def get_data(period: str = "2y", batch_size: int = 100, max_workers: int = 12) -> dict:
    """
    Return sector metrics as a dict for the Streamlit GUI.

    Keys:
      weights_df  — DataFrame indexed by sector; weight columns already in %,
                    change columns in pp, relperf columns in pp, 200DMA column in %
      d_1m / d_3m / d_6m — date strings for the lookback snapshots
      timestamp   — datetime when data was fetched
    """
    constituents = get_sp500_constituents()
    tickers = sorted(constituents["Ticker"].unique().tolist())

    prices = download_prices(tickers, period=period, batch_size=batch_size, auto_adjust=True)
    last_prices = prices.ffill().iloc[-1].dropna()

    md = fetch_marketcap_and_shares(
        tickers=list(last_prices.index),
        last_prices=last_prices,
        max_workers=max_workers,
    )
    shares = md["SharesOutstanding"].dropna()

    idx = prices.index
    asof_now = idx.max()
    d_1m = month_ago_dates(idx, 1)
    d_3m = month_ago_dates(idx, 3)
    d_6m = month_ago_dates(idx, 6)

    w_now = compute_sector_weights(constituents, prices, shares, asof_now)
    w_1m  = compute_sector_weights(constituents, prices, shares, d_1m)
    w_3m  = compute_sector_weights(constituents, prices, shares, d_3m)
    w_6m  = compute_sector_weights(constituents, prices, shares, d_6m)

    all_sectors = sorted(
        set(SECTOR_ETFS.keys()) | set(w_now.index) | set(w_1m.index) | set(w_3m.index) | set(w_6m.index)
    )

    weights = pd.DataFrame({
        "Weight_Now": w_now.reindex(all_sectors),
        "Weight_1M":  w_1m.reindex(all_sectors),
        "Weight_3M":  w_3m.reindex(all_sectors),
        "Weight_6M":  w_6m.reindex(all_sectors),
    })

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