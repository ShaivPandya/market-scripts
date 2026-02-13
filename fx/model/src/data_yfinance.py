"""Fetch market data from Yahoo Finance with CSV caching."""
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_yfinance_series(
    ticker: str,
    start: str,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.Series:
    """Fetch a daily price series from Yahoo Finance.

    Returns a pd.Series with DatetimeIndex (same contract as fetch_fred_series).
    """
    safe_name = ticker.replace("^", "").replace("=", "")
    cache_path = cache_dir / f"yf_{safe_name}.csv"

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        s = pd.to_numeric(df["value"], errors="coerce")
        return pd.Series(s.values, index=df["date"], name=ticker).sort_index()

    data = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"No data returned from Yahoo Finance for {ticker}")

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    s = data["Close"].dropna()
    s.index = pd.to_datetime(s.index)

    # Cache to CSV
    out = pd.DataFrame({"date": s.index, "value": s.values})
    out.to_csv(cache_path, index=False)

    return pd.Series(s.values, index=s.index, name=ticker).sort_index()
