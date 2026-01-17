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
from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd
import requests
import yfinance as yf

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


def calculate_breadth_metrics(tickers: List[str], period: str = "1y") -> dict:
    """
    Calculate market breadth metrics for a list of tickers.

    Returns dict with:
      - above_200dma: count and percentage above 200-day MA
      - above_20dma: count and percentage above 20-day MA
      - at_20day_high: count and percentage at 20-day high
      - at_20day_low: count and percentage at 20-day low
      - total_analyzed: number of stocks with valid data
    """
    print(f"Downloading price data for {len(tickers)} tickers...")

    df = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=True,
    )

    if df.empty:
        raise RuntimeError("No data downloaded")

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

    above_200dma = 0
    above_20dma = 0
    at_20day_high = 0
    at_20day_low = 0
    total_analyzed = 0

    for ticker in close.columns:
        prices = close[ticker].dropna()
        highs = high[ticker].dropna()
        lows = low[ticker].dropna()

        if len(prices) < 20:
            continue

        current_price = prices.iloc[-1]

        # 200-day MA (need at least 200 days of data)
        ma_200 = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else None

        # 20-day MA
        ma_20 = prices.rolling(20).mean().iloc[-1]

        # 20-day high (using high prices)
        high_20 = highs.tail(20).max() if len(highs) >= 20 else highs.max()

        # 20-day low (using low prices)
        low_20 = lows.tail(20).min() if len(lows) >= 20 else lows.min()

        total_analyzed += 1

        if ma_200 is not None and current_price > ma_200:
            above_200dma += 1

        if current_price > ma_20:
            above_20dma += 1

        # Check if current high equals 20-day high (making new high)
        current_high = highs.iloc[-1]
        if current_high >= high_20:
            at_20day_high += 1

        # Check if current low equals 20-day low (making new low)
        current_low = lows.iloc[-1]
        if current_low <= low_20:
            at_20day_low += 1

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
        summary.caption = f"Stocks analyzed: {metrics['total_analyzed']}"
        summary.caption_style = "dim"
        CONSOLE.print(summary)
    else:
        print("\n" + "=" * 50)
        print("MARKET BREADTH SUMMARY")
        print("=" * 50)
        print(f"Stocks analyzed: {metrics['total_analyzed']}")
        print("-" * 50)
        print(line_200)
        print(line_20)
        print(line_highs)
        print(line_lows)
        print("=" * 50)


if __name__ == "__main__":
    main()
