#!/usr/bin/env python3
"""
Top 50 S&P 500 performers over the past year (total return proxy via adjusted prices).

Dependencies:
  pip install pandas yfinance lxml

Notes:
- Constituents come from Wikipedia (unofficial but commonly used).
- Prices come from Yahoo Finance via yfinance; may be rate-limited, so we download in chunks.
- The top-50 list is persisted to a Postgres-or-SQLite table (`sp500_top50_tickers`)
  on each refresh run. The list barely moves day-to-day, so a daily Cloud Run Job
  refreshes it and the API reads from the table at request time.

Run as a daily refresh:
  python -m equities.market_technicals.get_top50
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, List  # noqa: UP035

import numpy as np
import pandas as pd

from api.postgres import use_postgres_state
from api.postgres_compat import PostgresCompatConnection
from utils.retry import requests_get, yf_download

LOGGER = logging.getLogger(__name__)

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DB_FILENAME = "sp500_top50.sqlite3"
TABLE_NAME = "sp500_top50_tickers"


def get_sp500_tickers():
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
    return pd.unique(tickers).tolist()


def chunked(xs: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def download_close_prices(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
    chunk_size: int = 100,
) -> pd.DataFrame:
    closes = []

    for chunk in chunked(tickers, chunk_size):
        df = yf_download(
            tickers=chunk,
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="column",
            threads=True,
            progress=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            if "Close" not in df.columns.get_level_values(0):
                raise RuntimeError("Expected 'Close' in downloaded data.")
            close = df["Close"].copy()
        else:
            if "Close" not in df.columns:
                raise RuntimeError("Expected 'Close' in downloaded data.")
            close = df[["Close"]].copy()
            close.columns = chunk

        closes.append(close)

    close_all = pd.concat(closes, axis=1)
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
        return float((s2.iloc[-1] / s2.iloc[0]) - 1.0)

    return close.apply(one_ticker_return, axis=0)


def compute_top50(period: str = "1y") -> pd.DataFrame:
    """Fetch S&P 500 prices and rank the top-50 performers in memory.

    Returns a DataFrame with columns ``[ticker, rank, one_year_return_pct]``,
    sorted by rank ascending (rank 1 = best performer). Pure compute — no I/O
    other than the upstream HTTP/yfinance fetches.
    """
    tickers = get_sp500_tickers()
    close = download_close_prices(tickers, period=period, interval="1d", chunk_size=100)
    rets = total_return_from_prices(close).dropna()
    top50 = rets.sort_values(ascending=False).head(50)

    out = (
        top50.rename_axis("ticker")
        .rename("one_year_return")
        .to_frame()
        .assign(one_year_return_pct=lambda d: 100 * d["one_year_return"])
        .drop(columns=["one_year_return"])
        .reset_index()
    )
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out.insert(1, "rank", range(1, len(out) + 1))
    return out


def _resolve_db_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILENAME)


def _init_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            ticker TEXT PRIMARY KEY,
            rank INTEGER NOT NULL,
            one_year_return_pct REAL NOT NULL,
            refreshed_at TEXT NOT NULL
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_sp500_top50_tickers_rank ON {TABLE_NAME}(rank)")
    conn.commit()


def _connect_db():
    """Open a connection routed by ``use_postgres_state()``.

    Production (`ENVIRONMENT=production` or `STATE_DB_BACKEND=postgres`) → Postgres
    via :class:`PostgresCompatConnection`. Dev/test → local SQLite at
    ``equities/market_technicals/sp500_top50.sqlite3``. The SQLite path is created
    only outside production, so the production write guard is never tripped.
    """
    if use_postgres_state():
        return PostgresCompatConnection()
    conn = sqlite3.connect(_resolve_db_path())
    conn.row_factory = sqlite3.Row
    _init_sqlite(conn)
    return conn


def refresh_top50_in_db(conn) -> dict[str, Any]:
    """Recompute the top-50 list and replace the table contents in one transaction.

    Full replace (not upsert) keeps ranks contiguous and prunes tickers that
    fell out of the top 50 since the last run.
    """
    df = compute_top50()
    refreshed_at = datetime.now(UTC).isoformat()

    conn.execute(f"DELETE FROM {TABLE_NAME}")
    rows = [
        (
            str(row["ticker"]).upper(),
            int(row["rank"]),
            float(row["one_year_return_pct"]),
            refreshed_at,
        )
        for _, row in df.iterrows()
    ]
    conn.executemany(
        f"INSERT INTO {TABLE_NAME} (ticker, rank, one_year_return_pct, refreshed_at) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return {"count": len(rows), "refreshed_at": refreshed_at}


def read_top50_from_db(conn) -> list[str]:
    """Return the cached top-50 tickers ordered by rank, or ``[]`` if empty."""
    cur = conn.execute(f"SELECT ticker FROM {TABLE_NAME} ORDER BY rank ASC")
    rows = cur.fetchall()
    out: list[str] = []
    for row in rows:
        # PostgresCompatConnection / sqlite3.Row both support index access.
        ticker = row[0] if not isinstance(row, dict) else row["ticker"]
        if ticker is None:
            continue
        out.append(str(ticker).upper())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Refreshing top-50 S&P 500 leadership list")
    conn = _connect_db()
    try:
        result = refresh_top50_in_db(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    LOGGER.info("Refreshed sp500_top50_tickers: %s", result)


if __name__ == "__main__":
    main()
