#!/usr/bin/env python3
"""
S&P 500 Market Breadth Analysis

Calculates:
1. % of stocks trading above their 200-day moving average
2. % of stocks trading above their 20-day moving average
3. % of stocks making 20-day highs

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


WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


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

    # Extract Close and High prices
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        high = df["High"]
    else:
        # Single ticker case
        close = df[["Close"]]
        close.columns = tickers[:1]
        high = df[["High"]]
        high.columns = tickers[:1]

    above_200dma = 0
    above_20dma = 0
    at_20day_high = 0
    total_analyzed = 0

    for ticker in close.columns:
        prices = close[ticker].dropna()
        highs = high[ticker].dropna()

        if len(prices) < 20:
            continue

        current_price = prices.iloc[-1]

        # 200-day MA (need at least 200 days of data)
        ma_200 = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else None

        # 20-day MA
        ma_20 = prices.rolling(20).mean().iloc[-1]

        # 20-day high (using high prices)
        high_20 = highs.tail(20).max() if len(highs) >= 20 else highs.max()

        total_analyzed += 1

        if ma_200 is not None and current_price > ma_200:
            above_200dma += 1

        if current_price > ma_20:
            above_20dma += 1

        # Check if current high equals 20-day high (making new high)
        current_high = highs.iloc[-1]
        if current_high >= high_20:
            at_20day_high += 1

    return {
        "above_200dma": above_200dma,
        "above_20dma": above_20dma,
        "at_20day_high": at_20day_high,
        "total_analyzed": total_analyzed,
        "pct_above_200dma": (above_200dma / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_above_20dma": (above_20dma / total_analyzed * 100) if total_analyzed > 0 else 0,
        "pct_at_20day_high": (at_20day_high / total_analyzed * 100) if total_analyzed > 0 else 0,
    }


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

    tickers = get_tickers(args.universe)
    print(f"Found {len(tickers)} tickers\n")

    metrics = calculate_breadth_metrics(tickers, args.period)

    print("\n" + "=" * 50)
    print("MARKET BREADTH SUMMARY")
    print("=" * 50)
    print(f"Stocks analyzed: {metrics['total_analyzed']}")
    print("-" * 50)
    print(f"Above 200-day MA:  {metrics['above_200dma']:>4} / {metrics['total_analyzed']}  ({metrics['pct_above_200dma']:.1f}%)")
    print(f"Above 20-day MA:   {metrics['above_20dma']:>4} / {metrics['total_analyzed']}  ({metrics['pct_above_20dma']:.1f}%)")
    print(f"At 20-day highs:   {metrics['at_20day_high']:>4} / {metrics['total_analyzed']}  ({metrics['pct_at_20day_high']:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
