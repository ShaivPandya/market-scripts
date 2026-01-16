#!/usr/bin/env python3
"""
Top 50 S&P 500 performers over the past 6 months (total return proxy via adjusted prices).

Dependencies:
  pip install pandas yfinance lxml

Notes:
- Constituents come from Wikipedia (unofficial but commonly used). :contentReference[oaicite:2]{index=2}
- Prices come from Yahoo Finance via yfinance; may be rate-limited, so we download in chunks.

python3 get_top50.py
"""

from __future__ import annotations

import math
from io import StringIO
from typing import Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf
import requests


WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers():
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


def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def download_close_prices(
    tickers: List[str],
    period: str = "6mo",
    interval: str = "1d",
    chunk_size: int = 100,
) -> pd.DataFrame:
    closes = []

    for chunk in chunked(tickers, chunk_size):
        df = yf.download(
            tickers=chunk,
            period=period,
            interval=interval,
            auto_adjust=True,   # adjusted prices (splits/dividends) :contentReference[oaicite:4]{index=4}
            group_by="column",
            threads=True,
            progress=False,
        )

        # For multiple tickers, yfinance typically returns a MultiIndex column:
        # first level: ["Open","High","Low","Close","Volume"], second: ticker
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" not in df.columns.get_level_values(0):
                raise RuntimeError("Expected 'Close' in downloaded data.")
            close = df["Close"].copy()
        else:
            # Single ticker case
            if "Close" not in df.columns:
                raise RuntimeError("Expected 'Close' in downloaded data.")
            close = df[["Close"]].copy()
            close.columns = chunk

        closes.append(close)

    # Combine all chunks on the date index
    close_all = pd.concat(closes, axis=1)
    # Remove any duplicate columns if a ticker appears twice for any reason
    close_all = close_all.loc[:, ~close_all.columns.duplicated()]
    return close_all


def total_return_from_prices(close: pd.DataFrame) -> pd.Series:
    """
    Computes total return proxy per ticker:
      (last_valid_price / first_valid_price) - 1
    """
    def one_ticker_return(s: pd.Series) -> float:
        s2 = s.dropna()
        if len(s2) < 2:
            return np.nan
        return (s2.iloc[-1] / s2.iloc[0]) - 1.0

    return close.apply(one_ticker_return, axis=0)


def main():
    tickers = get_sp500_tickers()
    close = download_close_prices(tickers, period="6mo", interval="1d", chunk_size=100)

    rets = total_return_from_prices(close).dropna()
    top50 = rets.sort_values(ascending=False).head(50)

    out = (
        top50.rename("six_month_return")
        .to_frame()
        .assign(six_month_return_pct=lambda d: 100 * d["six_month_return"])
        .drop(columns=["six_month_return"])
    )

    # Save to CSV
    out.to_csv("sp500_top50_6mo.csv", index_label="ticker")


if __name__ == "__main__":
    main()
