"""
Market Sentiment Data Module

Fetches:
  - CBOE Put/Call Ratio (equity, index, total) via CBOE public CSV
  - AAII Investor Sentiment Survey (bull/bear/neutral %) via stooq.com
  - NAAIM Exposure Index via NAAIM website (HTML scrape → Excel download)
  - Volatility indices (VIX, ^VXN, ^VVIX) via yfinance
"""

from __future__ import annotations

import io
from datetime import timedelta

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency: yfinance") from e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CBOE_PC_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/PC_ratio.csv"

STOOQ_AAII_BULL = "https://stooq.com/q/d/l/?s=aaiibull.us&i=w"
STOOQ_AAII_BEAR = "https://stooq.com/q/d/l/?s=aaiibear.us&i=w"
STOOQ_AAII_NEUT = "https://stooq.com/q/d/l/?s=aaiineur.us&i=w"

NAAIM_PAGE_URL = "https://www.naaim.org/programs/naaim-exposure-index/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Put/Call Ratio
# ---------------------------------------------------------------------------


def get_put_call(lookback_days: int = 180) -> list[dict]:
    """
    Fetch CBOE daily Put/Call ratios.

    Returns a list of dicts:
      {date, equity_pc, index_pc, total_pc,
       equity_pc_5d, index_pc_5d, total_pc_5d}
    where *_5d are 5-day rolling averages.
    """
    resp = requests.get(CBOE_PC_URL, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names for reliable lookup
    col_upper = {c.upper().replace(" ", "_"): c for c in df.columns}

    date_col = col_upper.get("DATE") or df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    cutoff = pd.Timestamp.today() - timedelta(days=lookback_days)
    df = df[df[date_col] >= cutoff].copy()

    def _ratio(put_key: str, call_key: str) -> pd.Series:
        pc = col_upper.get(put_key)
        cc = col_upper.get(call_key)
        if pc and cc:
            puts = pd.to_numeric(df[pc], errors="coerce")
            calls = pd.to_numeric(df[cc], errors="coerce")
            return (puts / calls.replace(0, float("nan"))).round(3)
        return pd.Series([None] * len(df), index=df.index)

    def _direct(key: str) -> pd.Series:
        c = col_upper.get(key)
        if c:
            return pd.to_numeric(df[c], errors="coerce").round(3)
        return pd.Series([None] * len(df), index=df.index)

    # CBOE CSV columns (as of 2024):
    #   DATE, CALL, PUT, TOTAL, INDEX_CALL, INDEX_PUT, INDEX_TOTAL,
    #   EQUITY_CALL, EQUITY_PUT, EQUITY_TOTAL
    # TOTAL / INDEX_TOTAL / EQUITY_TOTAL are pre-computed ratios (put/call).
    total_s = _direct("TOTAL") if col_upper.get("TOTAL") else _ratio("PUT", "CALL")
    index_s = _direct("INDEX_TOTAL") if col_upper.get("INDEX_TOTAL") else _ratio("INDEX_PUT", "INDEX_CALL")
    equity_s = _direct("EQUITY_TOTAL") if col_upper.get("EQUITY_TOTAL") else _ratio("EQUITY_PUT", "EQUITY_CALL")

    df["total_pc"] = total_s.values
    df["index_pc"] = index_s.values
    df["equity_pc"] = equity_s.values

    df["total_pc_5d"] = df["total_pc"].rolling(5, min_periods=1).mean().round(3)
    df["index_pc_5d"] = df["index_pc"].rolling(5, min_periods=1).mean().round(3)
    df["equity_pc_5d"] = df["equity_pc"].rolling(5, min_periods=1).mean().round(3)

    records = []
    for _, row in df.iterrows():
        def _v(x):
            return None if (x is None or (isinstance(x, float) and (x != x))) else x

        records.append({
            "date": row[date_col].date().isoformat(),
            "equity_pc": _v(row["equity_pc"]),
            "index_pc": _v(row["index_pc"]),
            "total_pc": _v(row["total_pc"]),
            "equity_pc_5d": _v(row["equity_pc_5d"]),
            "index_pc_5d": _v(row["index_pc_5d"]),
            "total_pc_5d": _v(row["total_pc_5d"]),
        })
    return records


# ---------------------------------------------------------------------------
# Surveys: AAII + NAAIM
# ---------------------------------------------------------------------------


def _fetch_stooq_series(url: str, value_col: str) -> pd.DataFrame:
    """Fetch a single stooq weekly CSV and return (date, value_col) DataFrame."""
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    # stooq returns: Date, Open, High, Low, Close, Volume
    date_col = df.columns[0]
    close_col = [c for c in df.columns if c.lower() == "close"]
    if not close_col:
        raise ValueError(f"No Close column in stooq response for {url}")
    df = df[[date_col, close_col[0]]].copy()
    df.columns = ["date", value_col]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df


def get_aaii() -> list[dict]:
    """
    Fetch AAII weekly investor sentiment from stooq.com.
    Returns list of dicts: {date, bull, bear, neutral, spread}
    """
    bull_df = _fetch_stooq_series(STOOQ_AAII_BULL, "bull")
    bear_df = _fetch_stooq_series(STOOQ_AAII_BEAR, "bear")
    neut_df = _fetch_stooq_series(STOOQ_AAII_NEUT, "neutral")

    df = bull_df.merge(bear_df, on="date", how="outer").merge(neut_df, on="date", how="outer")
    df = df.sort_values("date").dropna(subset=["bull", "bear"])
    df["spread"] = (df["bull"] - df["bear"]).round(2)

    records = []
    for _, row in df.iterrows():
        records.append({
            "date": row["date"].date().isoformat(),
            "bull": round(float(row["bull"]), 2) if pd.notna(row["bull"]) else None,
            "bear": round(float(row["bear"]), 2) if pd.notna(row["bear"]) else None,
            "neutral": round(float(row["neutral"]), 2) if pd.notna(row.get("neutral")) else None,
            "spread": round(float(row["spread"]), 2) if pd.notna(row["spread"]) else None,
        })
    return records


def _fetch_naaim_excel(page_url: str) -> pd.DataFrame:
    """
    Scrape the NAAIM Exposure Index page for an Excel download link,
    then fetch and parse the Excel.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError("beautifulsoup4 is required for NAAIM scraping") from e

    page = requests.get(page_url, headers=_HEADERS, timeout=_TIMEOUT)
    page.raise_for_status()
    soup = BeautifulSoup(page.content, "lxml")

    # Find the Excel download link
    xlsx_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(ext in href.lower() for ext in (".xlsx", ".xls")):
            xlsx_url = href if href.startswith("http") else "https://www.naaim.org" + href
            break

    if not xlsx_url:
        raise ValueError("Could not find NAAIM Excel download link on page")

    resp = requests.get(xlsx_url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return pd.read_excel(io.BytesIO(resp.content))


def get_naaim() -> list[dict]:
    """
    Fetch NAAIM Exposure Index weekly data.
    Returns list of dicts: {date, exposure}
    """
    df = _fetch_naaim_excel(NAAIM_PAGE_URL)

    # Normalise column names: NAAIM Excel typically has Date + Exposure columns
    df.columns = [str(c).strip() for c in df.columns]
    col_upper = {c.upper(): c for c in df.columns}

    date_col = (
        col_upper.get("DATE")
        or col_upper.get("WEEK")
        or col_upper.get("SURVEY DATE")
        or df.columns[0]
    )
    exposure_col = next(
        (col_upper[k] for k in col_upper if "EXPOSURE" in k or "NAAIM" in k),
        df.columns[1],
    )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df[exposure_col] = pd.to_numeric(df[exposure_col], errors="coerce")

    records = []
    for _, row in df.iterrows():
        exp = row[exposure_col]
        records.append({
            "date": row[date_col].date().isoformat(),
            "exposure": round(float(exp), 2) if pd.notna(exp) else None,
        })
    return records


def get_surveys() -> dict:
    """
    Fetch AAII sentiment + NAAIM exposure.
    Returns {"aaii": [...], "naaim": [...]}
    """
    aaii = get_aaii()
    naaim = get_naaim()
    return {"aaii": aaii, "naaim": naaim}


# ---------------------------------------------------------------------------
# Volatility Indices
# ---------------------------------------------------------------------------


def _download_close(ticker: str, start: str) -> pd.Series:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Close" not in df.columns:
        return pd.Series(dtype=float, name=ticker)
    s = df["Close"].copy()
    s.name = ticker
    return s


def get_volatility(lookback_days: int = 365) -> list[dict]:
    """
    Fetch VIX, VXN, and VVIX historical close prices via yfinance.
    Returns list of dicts: {date, vix, vxn, vvix}
    """
    start = (pd.Timestamp.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    vix = _download_close("^VIX", start)
    vxn = _download_close("^VXN", start)
    vvix = _download_close("^VVIX", start)

    df = pd.concat([vix, vxn, vvix], axis=1, join="outer")
    df.columns = ["vix", "vxn", "vvix"]
    df = df.sort_index()

    records = []
    for dt, row in df.iterrows():
        def _v(x):
            try:
                f = float(x)
                return round(f, 2) if f == f else None  # NaN check
            except (TypeError, ValueError):
                return None

        records.append({
            "date": dt.date().isoformat() if hasattr(dt, "date") else str(dt)[:10],
            "vix": _v(row["vix"]),
            "vxn": _v(row["vxn"]),
            "vvix": _v(row["vvix"]),
        })
    return records
