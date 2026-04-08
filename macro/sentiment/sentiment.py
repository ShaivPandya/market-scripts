"""
Market Sentiment Data Module

Fetches:
  - Put/Call Ratio (current snapshot) computed from SPY + QQQ options chains via yfinance
  - AAII Investor Sentiment Survey (bull/bear/neutral %) via aaii.com XLS download
  - NAAIM Exposure Index via NAAIM website (HTML scrape → Excel download)
  - Volatility indices (VIX, ^VXN, ^VVIX) via yfinance
"""

from __future__ import annotations

import io
from datetime import date as date_type
from datetime import timedelta

import pandas as pd

from utils.retry import requests_get

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency: yfinance") from e

from utils.retry import yf_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Equity P/C proxies: SPY + QQQ options chains
_PC_EQUITY_TICKERS = ["SPY", "QQQ", "IWM"]
# Number of near-term expiries to sum over
_PC_EXPIRIES = 8

AAII_XLS_URL = "https://www.aaii.com/files/surveys/sentiment.xls"

NAAIM_PAGE_URL = "https://www.naaim.org/programs/naaim-exposure-index/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Put/Call Ratio  (computed from live options chains via yfinance)
# ---------------------------------------------------------------------------


def _pc_for_ticker(sym: str, max_expiries: int = _PC_EXPIRIES) -> dict | None:
    """Sum put/call volumes across the nearest N expiries for one ticker."""
    ticker = yf.Ticker(sym)
    exps = ticker.options
    if not exps:
        return None

    total_calls = total_puts = 0
    breakdown: list[dict] = []

    for exp in exps[:max_expiries]:
        try:
            chain = ticker.option_chain(exp)
            calls = int(chain.calls["volume"].fillna(0).sum())
            puts = int(chain.puts["volume"].fillna(0).sum())
            total_calls += calls
            total_puts += puts
            breakdown.append(
                {
                    "expiry": exp,
                    "calls": calls,
                    "puts": puts,
                    "ratio": round(puts / calls, 3) if calls > 0 else None,
                }
            )
        except Exception:
            continue

    if total_calls == 0:
        return None

    return {
        "ticker": sym,
        "calls": total_calls,
        "puts": total_puts,
        "ratio": round(total_puts / total_calls, 3),
        "breakdown": breakdown,
        "as_of": date_type.today().isoformat(),
    }


def get_put_call(lookback_days: int = 180) -> dict:
    """
    Compute current Put/Call ratios from live yfinance options chains.

    Note: CBOE's CDN (cdn.cboe.com) blocks all programmatic access with HTTP 403.
    This function computes equivalent ratios from Yahoo Finance options chains:
      - 'equity': SPY + QQQ + IWM combined (broad equity P/C proxy)
      - 'spy': SPY only
      - 'qqq': QQQ only
      - 'iwm': IWM only

    Returns a dict with keys:
      equity, spy, qqq, iwm  — each a {ticker, calls, puts, ratio, breakdown, as_of} dict
    """
    results: dict = {}

    for sym in _PC_EQUITY_TICKERS:
        pc = _pc_for_ticker(sym)
        if pc:
            results[sym.lower()] = pc

    # Aggregate equity P/C across SPY + QQQ + IWM
    agg_calls = sum(results[s.lower()]["calls"] for s in _PC_EQUITY_TICKERS if s.lower() in results)
    agg_puts = sum(results[s.lower()]["puts"] for s in _PC_EQUITY_TICKERS if s.lower() in results)
    if agg_calls > 0:
        results["equity"] = {
            "ticker": "+".join(_PC_EQUITY_TICKERS),
            "calls": agg_calls,
            "puts": agg_puts,
            "ratio": round(agg_puts / agg_calls, 3),
            "as_of": date_type.today().isoformat(),
        }

    return results


# ---------------------------------------------------------------------------
# Surveys: AAII + NAAIM
# ---------------------------------------------------------------------------


def get_aaii() -> list[dict]:
    """
    Fetch AAII weekly investor sentiment from aaii.com (XLS download).
    Returns list of dicts: {date, bull, bear, neutral, spread}

    Values in the XLS are stored as decimals (0.36 = 36%) — multiplied by 100
    to return percentage values consistent with the previous stooq feed.
    """
    try:
        import xlrd
    except ImportError as e:
        raise ImportError("xlrd is required for AAII data: pip install xlrd>=2.0.1") from e

    resp = requests_get(AAII_XLS_URL, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()

    wb = xlrd.open_workbook(file_contents=resp.content)
    sh = wb.sheet_by_index(0)

    # Row 3 (0-indexed) is the header row: Date, Bullish, Neutral, Bearish, ...
    # Data starts at row 5 (row 4 is blank)
    records = []
    for i in range(5, sh.nrows):
        row = sh.row_values(i)
        date_val = row[0]
        bull_val = row[1]
        neut_val = row[2]
        bear_val = row[3]

        # Date column is an Excel serial number for data rows; skip non-numeric rows
        if not isinstance(date_val, float) or date_val <= 0:
            continue
        # Skip rows where bull/bear are missing
        if not isinstance(bull_val, float) or not isinstance(bear_val, float):
            continue

        try:
            dt = xlrd.xldate_as_datetime(date_val, wb.datemode).date()
        except Exception:
            continue

        bull = round(bull_val * 100, 2)
        bear = round(bear_val * 100, 2)
        neut = round(neut_val * 100, 2) if isinstance(neut_val, float) else None

        records.append(
            {
                "date": dt.isoformat(),
                "bull": bull,
                "bear": bear,
                "neutral": neut,
                "spread": round(bull - bear, 2),
            }
        )

    return records


def _fetch_naaim_excel(page_url: str) -> pd.DataFrame:
    """
    Scrape the NAAIM Exposure Index page for an Excel download link,
    then fetch and parse the Excel.
    """
    try:
        from bs4 import BeautifulSoup, Tag
    except ImportError as e:
        raise ImportError("beautifulsoup4 is required for NAAIM scraping") from e

    page = requests_get(page_url, headers=_HEADERS, timeout=_TIMEOUT)
    page.raise_for_status()
    soup = BeautifulSoup(page.content, "lxml")

    # Find the Excel download link
    xlsx_url = None
    for a in soup.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        href_raw = a.get("href")
        if isinstance(href_raw, list):
            href = str(href_raw[0]) if href_raw else ""
        else:
            href = str(href_raw or "")
        if not href:
            continue
        if any(ext in href.lower() for ext in (".xlsx", ".xls")):
            xlsx_url = href if href.startswith("http") else "https://www.naaim.org" + href
            break

    if not xlsx_url:
        raise ValueError("Could not find NAAIM Excel download link on page")

    resp = requests_get(xlsx_url, headers=_HEADERS, timeout=_TIMEOUT)
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

    date_col = col_upper.get("DATE") or col_upper.get("WEEK") or col_upper.get("SURVEY DATE") or df.columns[0]
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
        records.append(
            {
                "date": row[date_col].date().isoformat(),
                "exposure": round(float(exp), 2) if pd.notna(exp) else None,
            }
        )
    return records


def get_surveys() -> dict:
    """
    Fetch AAII sentiment + NAAIM exposure.
    Returns {"aaii": [...], "naaim": [...], "errors": {...}}

    The two feeds are independent. If one source fails, return the other so the
    API can degrade gracefully instead of blanking the entire surveys view.
    """
    aaii: list[dict] = []
    naaim: list[dict] = []
    errors: dict[str, str] = {}

    try:
        aaii = get_aaii()
    except Exception as exc:
        errors["aaii"] = str(exc)

    try:
        naaim = get_naaim()
    except Exception as exc:
        errors["naaim"] = str(exc)

    return {"aaii": aaii, "naaim": naaim, "errors": errors}


# ---------------------------------------------------------------------------
# Volatility Indices
# ---------------------------------------------------------------------------


def _download_close(ticker: str, start: str) -> pd.Series:
    df = yf_download(ticker, start=start, auto_adjust=False, progress=False)
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

        records.append(
            {
                "date": dt.date().isoformat() if hasattr(dt, "date") else str(dt)[:10],
                "vix": _v(row["vix"]),
                "vxn": _v(row["vxn"]),
                "vvix": _v(row["vvix"]),
            }
        )
    return records
