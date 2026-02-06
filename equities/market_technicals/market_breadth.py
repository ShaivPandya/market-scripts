#!/usr/bin/env python3
"""
S&P 500 Market Breadth Analysis

Calculates:
1. % of stocks trading above their 200-day moving average
2. % of stocks trading above their 20-day moving average
3. % of stocks making 20-day highs
4. % of stocks making 20-day lows

Dependencies:
  pip install pandas yfinance requests lxml

Usage:
  python3 market_breadth.py
  python3 market_breadth.py --universe sp500
  python3 market_breadth.py --universe /path/to/tickers.txt
"""

from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd
import requests
import yfinance as yf

# Download configuration
CHUNK_SIZE = 50  # Tickers per batch
MAX_RETRIES = 2  # Retry attempts for failed tickers
RETRY_DELAY = 1.0  # Seconds between retries

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
except ImportError:
    Console = None

CONSOLE = Console() if Console else None


WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


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
    style = "green" if highlight else None
    return Text(f"{value:.1f}%", style=style)


def get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(WIKI_SP500_URL, headers=headers, timeout=30)
    r.raise_for_status()

    df = pd.read_html(StringIO(r.text))[0]
    tickers = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    return pd.unique(tickers).tolist()


def load_tickers_from_file(filepath: str) -> List[str]:
    """Load tickers from a text file (one per line) or CSV."""
    p = Path(filepath)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = df.columns[0]
        return df[col].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
    else:
        with open(p) as f:
            return [line.strip().upper().replace(".", "-") for line in f if line.strip()]


def get_tickers(universe: str) -> List[str]:
    """Get tickers based on universe argument."""
    if universe.lower() == "sp500":
        print("Fetching S&P 500 tickers from Wikipedia...")
        return get_sp500_tickers()
    else:
        print(f"Loading tickers from {universe}...")
        return load_tickers_from_file(universe)


def download_with_retry(
    tickers: List[str],
    period: str = "1y",
    chunk_size: int = CHUNK_SIZE,
    max_retries: int = MAX_RETRIES,
) -> tuple[pd.DataFrame, List[str]]:
    """
    Download price data in chunks with retry logic for reliability.

    Returns:
        tuple of (combined DataFrame, list of failed tickers)
    """
    all_data = []
    failed_tickers = []

    # Split into chunks
    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks, 1):
        print(f"  Downloading batch {idx}/{total_chunks} ({len(chunk)} tickers)...")

        for attempt in range(max_retries + 1):
            try:
                df = yf.download(
                    tickers=chunk,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    group_by="column",
                    threads=True,
                    progress=False,
                )

                if df is not None and not df.empty:
                    all_data.append(df)

                    # Check which tickers actually returned data
                    if isinstance(df.columns, pd.MultiIndex):
                        returned = set(df["Close"].columns.tolist())
                    else:
                        returned = set(chunk[:1])

                    missing = set(chunk) - returned
                    if missing:
                        failed_tickers.extend(missing)

                    break
                elif attempt < max_retries:
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"    Batch {idx} failed after {max_retries + 1} attempts: {e}")
                    failed_tickers.extend(chunk)

    if not all_data:
        return pd.DataFrame(), failed_tickers

    # Combine all chunks
    if len(all_data) == 1:
        combined = all_data[0]
    else:
        # Merge DataFrames along columns
        combined_parts = {"Close": [], "High": [], "Low": []}
        for df in all_data:
            if isinstance(df.columns, pd.MultiIndex):
                for col in combined_parts:
                    if col in df.columns.get_level_values(0):
                        combined_parts[col].append(df[col])
            else:
                # Single ticker in this chunk
                for col in combined_parts:
                    if col in df.columns:
                        combined_parts[col].append(df[[col]])

        merged = {}
        for col, dfs in combined_parts.items():
            if dfs:
                merged[col] = pd.concat(dfs, axis=1)

        combined = pd.concat(merged, axis=1)

    return combined, failed_tickers


def calculate_breadth_metrics(tickers: List[str], period: str = "1y") -> dict:
    """
    Calculate market breadth metrics for a list of tickers.

    Returns dict with:
      - above_200dma: count and percentage above 200-day MA
      - above_20dma: count and percentage above 20-day MA
      - at_20day_high: count and percentage at 20-day high
      - at_20day_low: count and percentage at 20-day low
      - total_analyzed: number of stocks with valid data
      - failed_tickers: list of tickers that failed to download
    """
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

    total_analyzed = len(valid_tickers)

    # Vectorized comparisons
    above_200dma = int((current_close > ma_200).sum())
    above_20dma = int((current_close > ma_20).sum())
    at_20day_high = int((current_high >= high_20).sum())
    at_20day_low = int((current_low <= low_20).sum())

    return {
        "above_200dma": above_200dma,
        "above_20dma": above_20dma,
        "at_20day_high": at_20day_high,
        "at_20day_low": at_20day_low,
        "total_analyzed": total_analyzed,
        "pct_above_200dma": (above_200dma / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_above_20dma": (above_20dma / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_at_20day_high": (at_20day_high / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_at_20day_low": (at_20day_low / total_analyzed * 100) if total_analyzed > 0 else 0,
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
    parser = argparse.ArgumentParser(
        description="Calculate S&P 500 market breadth metrics"
    )
    parser.add_argument(
        "--universe",
        default="sp500",
        help="Universe: 'sp500' or path to ticker file (default: sp500)"
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Data period for yfinance (default: 1y)"
    )
    args = parser.parse_args()

    print_header()
    tickers = get_tickers(args.universe)
    print(f"Found {len(tickers)} tickers\n")

    metrics = calculate_breadth_metrics(tickers, args.period)

    # Determine color coding based on thresholds
    pct_200 = metrics['pct_above_200dma']
    pct_20 = metrics['pct_above_20dma']
    pct_highs = metrics['pct_at_20day_high']
    pct_lows = metrics['pct_at_20day_low']

    # Green if > 80% or < 15%
    line_200 = f"Above 200-day MA:  {metrics['above_200dma']:>4} / {metrics['total_analyzed']}  ({pct_200:.1f}%)"
    if pct_200 > 80 or pct_200 < 15:
        line_200 = colorize(line_200, "green")

    # Green if > 80% or < 20%
    line_20 = f"Above 20-day MA:   {metrics['above_20dma']:>4} / {metrics['total_analyzed']}  ({pct_20:.1f}%)"
    if pct_20 > 80 or pct_20 < 20:
        line_20 = colorize(line_20, "green")

    # Green if > 50%
    line_highs = f"At 20-day highs:   {metrics['at_20day_high']:>4} / {metrics['total_analyzed']}  ({pct_highs:.1f}%)"
    if pct_highs > 50:
        line_highs = colorize(line_highs, "green")

    # Green if > 50% (capitulation signal)
    line_lows = f"At 20-day lows:    {metrics['at_20day_low']:>4} / {metrics['total_analyzed']}  ({pct_lows:.1f}%)"
    if pct_lows > 50:
        line_lows = colorize(line_lows, "green")

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
        caption = f"Stocks analyzed: {metrics['total_analyzed']}"
        if failed:
            caption += f" | Failed: {len(failed)}"
        summary.caption = caption
        summary.caption_style = "dim"
        CONSOLE.print(summary)
        if failed:
            CONSOLE.print(f"[dim]Failed tickers: {', '.join(sorted(failed)[:20])}{'...' if len(failed) > 20 else ''}[/dim]")
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
        print("=" * 50)
        if failed:
            print(f"\nFailed tickers: {', '.join(sorted(failed)[:20])}{'...' if len(failed) > 20 else ''}")


def get_data(universe: str = "sp500", period: str = "1y") -> dict:
    """
    Fetch market breadth data for GUI consumption.

    Returns dict with:
      - above_200dma, above_20dma, at_20day_high, at_20day_low: counts
      - pct_above_200dma, pct_above_20dma, pct_at_20day_high, pct_at_20day_low: percentages
      - total_analyzed: number of stocks analyzed
      - tickers: list of tickers analyzed
    """
    tickers = get_tickers(universe)
    metrics = calculate_breadth_metrics(tickers, period)
    metrics["tickers"] = tickers
    return metrics


if __name__ == "__main__":
    main()
