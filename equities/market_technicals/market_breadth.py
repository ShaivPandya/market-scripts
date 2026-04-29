#!/usr/bin/env python3
"""
S&P 500 Market Breadth Analysis

Calculates:
1. % of stocks trading above their 200-day moving average
2. % of stocks trading above their 20-day moving average
3. % of stocks making 20-day highs
4. % of stocks making 20-day lows
5. % of stocks making 52-week (252-day) highs
6. % of stocks making 52-week (252-day) lows
7. % of stocks making 24-week (120-day) highs
8. % of stocks making 24-week (120-day) lows

Dependencies:
  pip install pandas yfinance requests lxml

Usage:
  python3 market_breadth.py
  python3 market_breadth.py --universe sp500
  python3 market_breadth.py --universe /path/to/tickers.txt
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, List, cast  # noqa: UP035

import pandas as pd

from utils.retry import requests_get, yf_download

logger = logging.getLogger(__name__)

# Download configuration
CHUNK_SIZE = 50  # Tickers per batch
BATCH_DELAY = 1.0  # Seconds between successful batches

# IBKR settings (via env). Distinct clientId from portfolio_news.py (10) so the
# two paths can coexist on a single TWS/Gateway session.
IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORT = int(os.environ.get("IB_PORT", "4001"))
IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID_BREADTH", "11"))
IB_FETCH_TIMEOUT_SECONDS = 600

CONSOLE: Any | None = None

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    CONSOLE = Console()
except ImportError:
    CONSOLE = None


WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / "data_cache" / "market_technicals"
_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_VERSION = 1
_CLOSE_PROBE_TICKER = "SPY"


def print_header() -> None:
    if CONSOLE:
        title = Text("Market Breadth", style="bold cyan")
        subtitle = Text("S&P 500 participation", style="dim")
        body = Text.assemble(title, "\n", subtitle)
        CONSOLE.print(Panel.fit(body, box=box.ASCII, padding=(1, 4), style="cyan"))
        return
    print("=" * 60)
    print("MARKET BREADTH")
    print("=" * 60)


def format_pct(value: float, highlight: bool):
    if value is None or pd.isna(value):
        return Text("N/A", style="dim")
    if highlight:
        return Text(f"{value:.1f}%", style="green")
    return Text(f"{value:.1f}%")


def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests_get(WIKI_SP500_URL, headers=headers, timeout=30)
    r.raise_for_status()

    df = pd.read_html(StringIO(r.text))[0]
    tickers = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    return cast(list[str], pd.unique(tickers).tolist())


def load_tickers_from_file(filepath: str) -> list[str]:
    """Load tickers from a text file (one per line) or CSV."""
    p = Path(filepath)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = df.columns[0]
        return cast(list[str], df[col].astype(str).str.strip().str.replace(".", "-", regex=False).tolist())
    else:
        with open(p) as f:
            return [line.strip().upper().replace(".", "-") for line in f if line.strip()]


def get_tickers(universe: str) -> list[str]:
    """Get tickers based on universe argument."""
    if universe.lower() == "sp500":
        print("Fetching S&P 500 tickers from Wikipedia...")
        return get_sp500_tickers()
    else:
        print(f"Loading tickers from {universe}...")
        return load_tickers_from_file(universe)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    return token[:80] or "default"


def _breadth_cache_path(universe: str, period: str) -> Path:
    filename = f"market_breadth_{_safe_token(universe)}_{_safe_token(period)}.json"
    return _CACHE_DIR / filename


def _load_breadth_cache(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        if raw.get("version") != _CACHE_VERSION:
            return None
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            return None
        fetched_at = raw.get("fetched_at")
        if not isinstance(fetched_at, str):
            return None
        datetime.fromisoformat(fetched_at)
        return raw
    except Exception:
        return None


def _write_breadth_cache(
    path: Path,
    payload: dict,
    universe: str,
    period: str,
    as_of_date: str | None,
    fetched_at: str | None = None,
) -> None:
    record = {
        "version": _CACHE_VERSION,
        "fetched_at": fetched_at or datetime.now().isoformat(),
        "as_of_date": as_of_date,
        "universe": universe,
        "period": period,
        "payload": payload,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Cache writes are best-effort and should not break data fetch.
        return


def _latest_market_close_date() -> str | None:
    try:
        probe = yf_download(
            tickers=_CLOSE_PROBE_TICKER,
            period="10d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if probe is None or probe.empty:
            return None
        idx = pd.to_datetime(probe.index, errors="coerce")
        idx = idx.dropna()
        if idx.empty:
            return None
        return cast(str, idx[-1].date().isoformat())
    except Exception:
        return None


def _period_to_ibkr_duration(period: str) -> str:
    """
    Map yfinance-style period strings to IBKR durationStr format.

    Floors at "1 Y" because the breadth metric needs >=252 bars for 52w highs/lows.
    """
    if not period:
        return "1 Y"

    s = period.strip().lower()
    m = re.fullmatch(r"(\d+)\s*([dwmoy]+)", s)
    if not m:
        if s == "max":
            return "30 Y"
        if s == "ytd":
            return "1 Y"
        logger.warning("Unrecognised period '%s' for IBKR; defaulting to 1 Y", period)
        return "1 Y"

    n = int(m.group(1))
    unit = m.group(2)

    if unit == "d":
        return f"{max(n, 365)} D"
    if unit == "w":
        weeks = max(n, 52)
        return f"{weeks} W"
    if unit in {"m", "mo"}:
        months = max(n, 12)
        return f"{months} M"
    if unit == "y":
        years = max(n, 1)
        return f"{years} Y"

    return "1 Y"


def _ticker_to_ibkr_symbol(ticker: str) -> str:
    """Convert yfinance-style class shares (BRK-B) to IBKR-style (BRK B)."""
    return ticker.replace("-", " ")


def _bars_to_dataframe(bars) -> pd.DataFrame | None:
    """Convert ib_insync BarDataList to a DataFrame indexed by date with Close/High/Low."""
    if not bars:
        return None
    rows = []
    for b in bars:
        date = getattr(b, "date", None)
        close = getattr(b, "close", None)
        high = getattr(b, "high", None)
        low = getattr(b, "low", None)
        if date is None or close is None or high is None or low is None:
            continue
        rows.append((pd.Timestamp(date), float(close), float(high), float(low)))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "Close", "High", "Low"]).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def _fetch_ibkr_prices(
    tickers: list[str],
    period: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Fetch daily ADJUSTED_LAST bars from IBKR for the given tickers.

    Runs ib_insync inside a dedicated worker thread with a fresh asyncio loop so
    it does not collide with uvloop under FastAPI/uvicorn (same idiom as
    portfolio/portfolio_news.py::_fetch_all_ibkr_news).

    Returns (per_ticker_frames, failed_tickers). On connection failure, all
    tickers are returned as failed so the caller falls back to yfinance.
    """

    duration = _period_to_ibkr_duration(period)

    def _run() -> tuple[dict[str, pd.DataFrame], list[str]]:
        import asyncio

        asyncio.set_event_loop(asyncio.new_event_loop())

        try:
            from ib_insync import IB, Stock
        except Exception as e:
            logger.warning("ib_insync unavailable (%s: %s); skipping IBKR fetch", type(e).__name__, e)
            return {}, list(tickers)

        ib = IB()
        try:
            ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=5, readonly=True)
        except Exception as e:
            logger.warning("IBKR connection unavailable (%s: %s); skipping IBKR fetch", type(e).__name__, e)
            return {}, list(tickers)

        try:
            contracts = [Stock(_ticker_to_ibkr_symbol(t), "SMART", "USD") for t in tickers]
            try:
                ib.qualifyContracts(*contracts)
            except Exception as e:
                logger.warning("IBKR qualifyContracts failed (%s: %s); skipping IBKR fetch", type(e).__name__, e)
                return {}, list(tickers)

            frames: dict[str, pd.DataFrame] = {}
            failed: list[str] = []
            request_count = 0

            for ticker, contract in zip(tickers, contracts, strict=True):
                if not getattr(contract, "conId", 0):
                    failed.append(ticker)
                    continue

                try:
                    bars = ib.reqHistoricalData(
                        contract,
                        endDateTime="",
                        durationStr=duration,
                        barSizeSetting="1 day",
                        whatToShow="ADJUSTED_LAST",
                        useRTH=True,
                        formatDate=1,
                        timeout=15,
                    )
                except Exception as e:
                    logger.debug("IBKR reqHistoricalData failed for %s: %s", ticker, e)
                    failed.append(ticker)
                else:
                    df = _bars_to_dataframe(bars)
                    if df is None:
                        failed.append(ticker)
                    else:
                        frames[ticker] = df

                request_count += 1
                if request_count % CHUNK_SIZE == 0 and request_count < len(tickers):
                    time.sleep(BATCH_DELAY)

            return frames, failed
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            return future.result(timeout=IB_FETCH_TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning("IBKR fetch thread failed (%s: %s); skipping IBKR fetch", type(e).__name__, e)
        return {}, list(tickers)


def _ibkr_frames_to_multiindex(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-ticker IBKR DataFrames into a (field, ticker) MultiIndex DataFrame."""
    if not frames:
        return pd.DataFrame()
    fields = ["Close", "High", "Low"]
    parts: dict[str, pd.DataFrame] = {}
    for field in fields:
        cols = {ticker: df[field] for ticker, df in frames.items() if field in df.columns}
        if cols:
            parts[field] = pd.DataFrame(cols)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, axis=1)
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)
    return combined.sort_index().sort_index(axis=1)


def _yfinance_download_chunked(
    tickers: list[str],
    period: str,
    chunk_size: int,
    batch_delay: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Original chunked yfinance fetch loop. Used as the fallback path."""
    all_data: list[pd.DataFrame] = []
    failed_tickers: list[str] = []

    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks, 1):
        print(f"  Downloading batch {idx}/{total_chunks} ({len(chunk)} tickers)...")

        try:
            df = yf_download(
                tickers=chunk,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )

            if df is not None and not df.empty:
                all_data.append(df)

                if isinstance(df.columns, pd.MultiIndex):
                    returned = set(df["Close"].columns.tolist())
                else:
                    returned = set(chunk[:1])

                missing = set(chunk) - returned
                if missing:
                    failed_tickers.extend(missing)

                if idx < total_chunks:
                    time.sleep(batch_delay)
            else:
                failed_tickers.extend(chunk)
        except Exception as e:
            print(f"    Batch {idx} failed: {e}")
            failed_tickers.extend(chunk)

    if not all_data:
        return pd.DataFrame(), failed_tickers

    if len(all_data) == 1:
        combined = all_data[0]
    else:
        combined_parts: dict[str, list[pd.DataFrame]] = {"Close": [], "High": [], "Low": []}
        for df in all_data:
            if isinstance(df.columns, pd.MultiIndex):
                for col in combined_parts:
                    if col in df.columns.get_level_values(0):
                        combined_parts[col].append(df[col])
            else:
                for col in combined_parts:
                    if col in df.columns:
                        combined_parts[col].append(df[[col]])

        merged: dict[str, pd.DataFrame] = {}
        for col, dfs in combined_parts.items():
            if dfs:
                merged[col] = pd.concat(dfs, axis=1)

        combined = pd.concat(merged, axis=1)

    return combined, failed_tickers


def download_with_retry(
    tickers: list[str],
    period: str = "1y",
    chunk_size: int = CHUNK_SIZE,
    batch_delay: float = BATCH_DELAY,
    **kwargs,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Download price data, preferring IBKR IB Gateway and falling back to yfinance.

    Returns:
        tuple of (combined MultiIndex DataFrame, list of failed tickers)
    """
    ibkr_frames, ibkr_failed = _fetch_ibkr_prices(tickers, period)

    yf_combined = pd.DataFrame()
    yf_failed: list[str] = []
    if ibkr_failed:
        yf_combined, yf_failed = _yfinance_download_chunked(ibkr_failed, period, chunk_size, batch_delay)

    ibkr_combined = _ibkr_frames_to_multiindex(ibkr_frames)

    if ibkr_combined.empty and yf_combined.empty:
        combined = pd.DataFrame()
    elif ibkr_combined.empty:
        combined = yf_combined
    elif yf_combined.empty:
        combined = ibkr_combined
    else:
        combined = pd.concat([ibkr_combined, yf_combined], axis=1).sort_index().sort_index(axis=1)

    logger.info(
        "market_breadth: fetched %d tickers from IBKR, %d from yfinance, %d failed",
        len(ibkr_frames),
        max(len(ibkr_failed) - len(yf_failed), 0),
        len(yf_failed),
    )

    return combined, yf_failed


def calculate_breadth_metrics(
    tickers: list[str],
    period: str = "1y",
    prices_df: pd.DataFrame | None = None,
) -> dict:
    """
    Calculate market breadth metrics for a list of tickers.

    Returns dict with:
      - above_200dma: count and percentage above 200-day MA
      - above_20dma: count and percentage above 20-day MA
      - at_20day_high: count and percentage at 20-day high
      - at_20day_low: count and percentage at 20-day low
      - at_52wk_high: count and percentage at 52-week high
      - at_52wk_low: count and percentage at 52-week low
      - at_24wk_high: count and percentage at 24-week high
      - at_24wk_low: count and percentage at 24-week low
      - total_analyzed: number of stocks with valid data
      - failed_tickers: list of tickers that failed to download

    If *prices_df* is supplied (a MultiIndex DataFrame from yfinance), the
    download step is skipped and the pre-fetched data is used directly.
    """
    if prices_df is not None:
        df = prices_df
        if isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.get_level_values(0):
            available = set(df["Close"].columns.tolist())
        else:
            available = set(df.columns.tolist()) if not df.empty else set()
        failed_tickers = [t for t in tickers if t not in available]
    else:
        print(f"Downloading price data for {len(tickers)} tickers...")
        df, failed_tickers = download_with_retry(tickers, period)

    if df.empty:
        raise RuntimeError("No data downloaded")

    if failed_tickers:
        print(f"  Warning: {len(failed_tickers)} tickers failed to download")

    # Extract Close, High, and Low prices
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
    else:
        # Single ticker case
        close = df[["Close"]]
        close.columns = tickers[:1]
        high = df[["High"]]
        high.columns = tickers[:1]
        low = df[["Low"]]
        low.columns = tickers[:1]

    idx = pd.to_datetime(close.index, errors="coerce").dropna()
    as_of_date = idx[-1].date().isoformat() if not idx.empty else None

    # Vectorized calculations for performance
    # Get the latest values
    current_close = close.iloc[-1]
    current_high = high.iloc[-1]
    current_low = low.iloc[-1]

    # Calculate moving averages (vectorized across all tickers)
    ma_200 = close.rolling(200).mean().iloc[-1]
    ma_20 = close.rolling(20).mean().iloc[-1]

    # Calculate 20-day highs and lows (vectorized)
    high_20 = high.tail(20).max()
    low_20 = low.tail(20).min()

    # Calculate 52-week (252-day) highs and lows (vectorized)
    high_252 = high.tail(252).max()
    low_252 = low.tail(252).min()

    # Calculate 24-week (120-day) highs and lows (vectorized)
    high_120 = high.tail(120).max()
    low_120 = low.tail(120).min()

    # Count valid tickers (at least 20 days of data)
    valid_counts = close.notna().sum()
    valid_tickers = valid_counts[valid_counts >= 20].index

    # Filter to only valid tickers
    current_close = current_close[valid_tickers]
    current_high = current_high[valid_tickers]
    current_low = current_low[valid_tickers]
    ma_200 = ma_200[valid_tickers]
    ma_20 = ma_20[valid_tickers]
    high_20 = high_20[valid_tickers]
    low_20 = low_20[valid_tickers]
    high_252 = high_252[valid_tickers]
    low_252 = low_252[valid_tickers]
    high_120 = high_120[valid_tickers]
    low_120 = low_120[valid_tickers]

    total_analyzed = len(valid_tickers)

    # Vectorized comparisons
    above_200dma = int((current_close > ma_200).sum())
    above_20dma = int((current_close > ma_20).sum())
    at_20day_high = int((current_high >= high_20).sum())
    at_20day_low = int((current_low <= low_20).sum())
    at_52wk_high = int((current_high >= high_252).sum())
    at_52wk_low = int((current_low <= low_252).sum())
    at_24wk_high = int((current_high >= high_120).sum())
    at_24wk_low = int((current_low <= low_120).sum())

    return {
        "above_200dma": above_200dma,
        "above_20dma": above_20dma,
        "at_20day_high": at_20day_high,
        "at_20day_low": at_20day_low,
        "at_52wk_high": at_52wk_high,
        "at_52wk_low": at_52wk_low,
        "at_24wk_high": at_24wk_high,
        "at_24wk_low": at_24wk_low,
        "total_analyzed": total_analyzed,
        "pct_above_200dma": (above_200dma / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_above_20dma": (above_20dma / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_at_20day_high": (at_20day_high / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_at_20day_low": (at_20day_low / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_at_52wk_high": (at_52wk_high / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_at_52wk_low": (at_52wk_low / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_at_24wk_high": (at_24wk_high / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_at_24wk_low": (at_24wk_low / total_analyzed * 100) if total_analyzed > 0 else 0,
        "as_of_date": as_of_date,
        "failed_tickers": failed_tickers,
    }


def colorize(text: str, color: str) -> str:
    """Wrap text with ANSI color codes."""
    colors = {
        "green": "\033[92m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def main():
    parser = argparse.ArgumentParser(description="Calculate S&P 500 market breadth metrics")
    parser.add_argument("--universe", default="sp500", help="Universe: 'sp500' or path to ticker file (default: sp500)")
    parser.add_argument("--period", default="1y", help="Data period for yfinance (default: 1y)")
    args = parser.parse_args()

    print_header()
    tickers = get_tickers(args.universe)
    print(f"Found {len(tickers)} tickers\n")

    metrics = calculate_breadth_metrics(tickers, args.period)

    # Determine color coding based on thresholds
    pct_200 = metrics["pct_above_200dma"]
    pct_20 = metrics["pct_above_20dma"]
    pct_highs = metrics["pct_at_20day_high"]
    pct_lows = metrics["pct_at_20day_low"]
    pct_52wk_highs = metrics["pct_at_52wk_high"]
    pct_52wk_lows = metrics["pct_at_52wk_low"]
    pct_24wk_highs = metrics["pct_at_24wk_high"]
    pct_24wk_lows = metrics["pct_at_24wk_low"]

    # Green if > 80% or < 15% (strong momentum or too much fear)
    line_200 = f"Above 200-day MA:  {metrics['above_200dma']:>4} / {metrics['total_analyzed']}  ({pct_200:.1f}%)"
    if pct_200 > 80 or pct_200 < 15:
        line_200 = colorize(line_200, "green")

    # Green if > 80% or < 20% (strong momentum or too much fear)
    line_20 = f"Above 20-day MA:   {metrics['above_20dma']:>4} / {metrics['total_analyzed']}  ({pct_20:.1f}%)"
    if pct_20 > 80 or pct_20 < 20:
        line_20 = colorize(line_20, "green")

    # Green if > 50% (strong momentum)
    line_highs = f"At 20-day highs:   {metrics['at_20day_high']:>4} / {metrics['total_analyzed']}  ({pct_highs:.1f}%)"
    if pct_highs > 50:
        line_highs = colorize(line_highs, "green")

    # Green if > 50% (capitulation signal)
    line_lows = f"At 20-day lows:    {metrics['at_20day_low']:>4} / {metrics['total_analyzed']}  ({pct_lows:.1f}%)"
    if pct_lows > 50:
        line_lows = colorize(line_lows, "green")

    # Green if > 15% (strong momentum)
    line_52wk_highs = (
        f"At 52-week highs:  {metrics['at_52wk_high']:>4} / {metrics['total_analyzed']}  ({pct_52wk_highs:.1f}%)"
    )
    if pct_52wk_highs > 15:
        line_52wk_highs = colorize(line_52wk_highs, "green")

    # Green if > 15%
    line_52wk_lows = (
        f"At 52-week lows:   {metrics['at_52wk_low']:>4} / {metrics['total_analyzed']}  ({pct_52wk_lows:.1f}%)"
    )
    if pct_52wk_lows > 15:
        line_52wk_lows = colorize(line_52wk_lows, "green")

    # Green if > 20% (strong momentum)
    line_24wk_highs = (
        f"At 24-week highs:  {metrics['at_24wk_high']:>4} / {metrics['total_analyzed']}  ({pct_24wk_highs:.1f}%)"
    )
    if pct_24wk_highs > 20:
        line_24wk_highs = colorize(line_24wk_highs, "green")

    # Green if > 20%
    line_24wk_lows = (
        f"At 24-week lows:   {metrics['at_24wk_low']:>4} / {metrics['total_analyzed']}  ({pct_24wk_lows:.1f}%)"
    )
    if pct_24wk_lows > 20:
        line_24wk_lows = colorize(line_24wk_lows, "green")

    failed = metrics.get("failed_tickers", [])

    if CONSOLE:
        summary = Table(title="Market Breadth Summary", box=box.ASCII)
        summary.add_column("Metric")
        summary.add_column("Count", justify="right")
        summary.add_column("Percent", justify="right")
        summary.add_row(
            "Above 200-day MA",
            f"{metrics['above_200dma']} / {metrics['total_analyzed']}",
            format_pct(pct_200, pct_200 > 80 or pct_200 < 15),
        )
        summary.add_row(
            "Above 20-day MA",
            f"{metrics['above_20dma']} / {metrics['total_analyzed']}",
            format_pct(pct_20, pct_20 > 80 or pct_20 < 20),
        )
        summary.add_row(
            "At 20-day highs",
            f"{metrics['at_20day_high']} / {metrics['total_analyzed']}",
            format_pct(pct_highs, pct_highs > 50),
        )
        summary.add_row(
            "At 20-day lows",
            f"{metrics['at_20day_low']} / {metrics['total_analyzed']}",
            format_pct(pct_lows, pct_lows > 50),
        )
        summary.add_row(
            "At 52-week highs",
            f"{metrics['at_52wk_high']} / {metrics['total_analyzed']}",
            format_pct(pct_52wk_highs, pct_52wk_highs > 15),
        )
        summary.add_row(
            "At 52-week lows",
            f"{metrics['at_52wk_low']} / {metrics['total_analyzed']}",
            format_pct(pct_52wk_lows, pct_52wk_lows > 15),
        )
        summary.add_row(
            "At 24-week highs",
            f"{metrics['at_24wk_high']} / {metrics['total_analyzed']}",
            format_pct(pct_24wk_highs, pct_24wk_highs > 20),
        )
        summary.add_row(
            "At 24-week lows",
            f"{metrics['at_24wk_low']} / {metrics['total_analyzed']}",
            format_pct(pct_24wk_lows, pct_24wk_lows > 20),
        )
        caption = f"Stocks analyzed: {metrics['total_analyzed']}"
        if failed:
            caption += f" | Failed: {len(failed)}"
        summary.caption = caption
        summary.caption_style = "dim"
        CONSOLE.print(summary)
        if failed:
            CONSOLE.print(
                f"[dim]Failed tickers: {', '.join(sorted(failed)[:20])}{'...' if len(failed) > 20 else ''}[/dim]"
            )
    else:
        print("\n" + "=" * 50)
        print("MARKET BREADTH SUMMARY")
        print("=" * 50)
        print(f"Stocks analyzed: {metrics['total_analyzed']}")
        if failed:
            print(f"Failed to download: {len(failed)}")
        print("-" * 50)
        print(line_200)
        print(line_20)
        print(line_highs)
        print(line_lows)
        print(line_52wk_highs)
        print(line_52wk_lows)
        print(line_24wk_highs)
        print(line_24wk_lows)
        print("=" * 50)
        if failed:
            print(f"\nFailed tickers: {', '.join(sorted(failed)[:20])}{'...' if len(failed) > 20 else ''}")


def get_data(
    universe: str = "sp500",
    period: str = "1y",
    prices_df: pd.DataFrame | None = None,
) -> dict:
    """
    Fetch market breadth data for GUI consumption.

    Returns dict with:
      - above_200dma, above_20dma, at_20day_high, at_20day_low: counts
      - at_52wk_high, at_52wk_low, at_24wk_high, at_24wk_low: counts
      - pct_above_200dma, pct_above_20dma, pct_at_20day_high, pct_at_20day_low: percentages
      - pct_at_52wk_high, pct_at_52wk_low, pct_at_24wk_high, pct_at_24wk_low: percentages
      - total_analyzed: number of stocks analyzed
      - tickers: list of tickers analyzed

    If *prices_df* is supplied, the yfinance download is skipped.
    """
    cache_enabled = prices_df is None and universe.lower() == "sp500"
    cache_path = _breadth_cache_path(universe, period) if cache_enabled else None
    cached_record = _load_breadth_cache(cache_path) if cache_path else None
    cached_payload = cached_record.get("payload") if cached_record else None

    if cached_record and isinstance(cached_payload, dict):
        try:
            fetched_at = datetime.fromisoformat(str(cached_record["fetched_at"]))
            age_seconds = (datetime.now() - fetched_at).total_seconds()
        except Exception:
            age_seconds = _CACHE_TTL_SECONDS + 1

        if age_seconds < _CACHE_TTL_SECONDS:
            return cached_payload

        cached_as_of = cached_record.get("as_of_date")
        latest_close = _latest_market_close_date()
        if isinstance(cached_as_of, str) and latest_close is not None and latest_close <= cached_as_of:
            assert cache_path is not None
            _write_breadth_cache(
                path=cache_path,
                payload=cached_payload,
                universe=universe,
                period=period,
                as_of_date=cached_as_of,
                fetched_at=datetime.now().isoformat(),
            )
            return cached_payload

    try:
        tickers = get_tickers(universe)
        metrics = calculate_breadth_metrics(tickers, period, prices_df=prices_df)
        metrics["tickers"] = tickers
    except Exception:
        if isinstance(cached_payload, dict):
            return cached_payload
        raise

    if cache_path:
        _write_breadth_cache(
            path=cache_path,
            payload=metrics,
            universe=universe,
            period=period,
            as_of_date=metrics.get("as_of_date") if isinstance(metrics.get("as_of_date"), str) else None,
        )

    return metrics


if __name__ == "__main__":
    main()
