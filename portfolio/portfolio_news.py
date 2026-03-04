"""
Portfolio News Feed via GDELT DOC 2.0 API + IBKR TWS API.

Queries GDELT and (optionally) IBKR for each ticker/company-name in
portfolio.csv and returns a unified feed with both grouped-by-ticker
and flat chronological views.
"""

import csv
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

PORTFOLIO_CSV = Path(__file__).parent / "portfolio.csv"

# ── GDELT settings ────────────────────────────────────────────────────────────
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TIMESPAN = "3d"
GDELT_MAX_RECORDS = 10
REQUEST_DELAY = 0.6  # seconds between GDELT calls to avoid rate-limiting

# ── IBKR settings (via env) ───────────────────────────────────────────────────
IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORT = int(os.environ.get("IB_PORT", "4001"))
IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "10"))
IB_NEWS_PROVIDERS = "BZ+FLY"   # Benzinga + Fly On The Wall
IB_MAX_HEADLINES = 10

# Cache ticker → company name so we don't re-fetch every call
_name_cache: dict[str, str] = {}


def _read_portfolio() -> list[dict[str, str]]:
    """Read portfolio.csv and return list of rows."""
    with open(PORTFOLIO_CSV, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("ticker")]


def _resolve_name(ticker: str, asset: str) -> str:
    """
    Resolve a human-readable company/fund name for a ticker.
    Uses yfinance for equities; returns ticker itself for commodities / ETFs.
    """
    if ticker in _name_cache:
        return _name_cache[ticker]

    name = ticker  # fallback
    if asset == "equity":
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info
            name = info.get("shortName") or info.get("longName") or ticker
        except Exception:
            logger.debug("yfinance lookup failed for %s, using ticker", ticker)

    _name_cache[ticker] = name
    return name


# ── GDELT ─────────────────────────────────────────────────────────────────────

def _query_gdelt(ticker: str, name: str) -> list[dict[str, Any]]:
    """
    Query GDELT DOC 2.0 API for articles matching a ticker or company name.
    Returns a list of article dicts.
    """
    # Build query: (TICKER OR "Company Name") if name differs from ticker
    if name and name.upper() != ticker.upper():
        query = f'({ticker} OR "{name}")'
    else:
        query = ticker

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "timespan": GDELT_TIMESPAN,
        "maxrecords": GDELT_MAX_RECORDS,
    }

    max_retries = 3
    data = {}
    for attempt in range(max_retries):
        try:
            resp = requests.get(GDELT_DOC_URL, params=params, timeout=25)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logger.debug("GDELT query for %s failed, retrying in %ds... (%s)", ticker, 2 ** attempt, e)
                time.sleep(2 ** attempt)
                continue
            logger.warning("GDELT query failed for %s after %d retries: %s", ticker, max_retries, e)
            return []
        except Exception as e:
            logger.warning("GDELT query failed for %s with unexpected error: %s", ticker, e)
            return []

    articles = data.get("articles") or []
    results = []
    for art in articles:
        results.append(
            {
                "ticker": ticker,
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": art.get("domain", ""),
                "seendate": _parse_gdelt_date(art.get("seendate", "")),
                "socialimage": art.get("socialimage", ""),
                "language": art.get("language", ""),
                "provider": "GDELT",
            }
        )
    return results


def _parse_gdelt_date(raw: str) -> str:
    """Parse GDELT seendate (YYYYMMDDTHHMMSSz) into ISO format."""
    if not raw:
        return ""
    try:
        dt = datetime.strptime(raw.rstrip("Z"), "%Y%m%dT%H%M%S")
        return dt.isoformat() + "Z"
    except ValueError:
        return raw


# ── IBKR ──────────────────────────────────────────────────────────────────────

def _connect_ib():
    """
    Try to connect to IB Gateway / TWS.  Returns an IB instance or None.
    """
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=5)
        return ib
    except Exception as e:
        logger.info("IBKR connection unavailable (%s), skipping IB news", e)
        return None


def _qualify_contract(ib, ticker: str, asset: str):
    """
    Build and qualify an IB contract for the given ticker.
    Returns the qualified contract or None.
    """
    from ib_insync import Stock, Contract

    # Skip non-equity for now (commodities / ETFs can be added later)
    # Determine exchange based on ticker suffix
    if "." in ticker:
        # International tickers like METSO.HE, LSEG.L
        local_symbol = ticker.split(".")[0]
        suffix = ticker.split(".")[-1]
        exchange_map = {
            "HE": "NASDAQ OMX",
            "L": "LSE",
        }
        exchange = exchange_map.get(suffix, "SMART")
        contract = Stock(local_symbol, exchange, "")
    else:
        contract = Stock(ticker, "SMART", "USD")

    try:
        qualified = ib.qualifyContracts(contract)
        if qualified:
            return qualified[0]
    except Exception as e:
        logger.debug("Failed to qualify contract for %s: %s", ticker, e)

    return None


def _query_ibkr(ib, ticker: str, asset: str) -> list[dict[str, Any]]:
    """
    Query IBKR TWS API for historical news headlines for a ticker.
    Returns a list of article dicts.
    """
    contract = _qualify_contract(ib, ticker, asset)
    if contract is None:
        return []

    start_dt = datetime.now() - timedelta(days=3)
    end_dt = ""  # current time

    try:
        headlines = ib.reqHistoricalNews(
            contract.conId,
            IB_NEWS_PROVIDERS,
            start_dt,
            end_dt,
            IB_MAX_HEADLINES,
        )
    except Exception as e:
        logger.warning("IBKR news query failed for %s: %s", ticker, e)
        return []

    results = []
    for h in headlines:
        # HistoricalNews has: time, providerCode, articleId, headline
        headline_text = getattr(h, "headline", "") or ""
        article_time = getattr(h, "time", None)
        provider_code = getattr(h, "providerCode", "") or ""

        seendate = ""
        if article_time:
            try:
                seendate = article_time.isoformat() + "Z"
            except Exception:
                seendate = str(article_time)

        results.append(
            {
                "ticker": ticker,
                "title": headline_text,
                "url": "",  # IBKR doesn't provide URLs for headlines
                "source": provider_code,
                "seendate": seendate,
                "socialimage": "",
                "language": "English",
                "provider": "IBKR",
            }
        )
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def get_data(refresh: bool = False) -> dict[str, Any]:
    """
    Main entry point. Returns structured news data:
      - items: flat list of all articles
      - by_ticker: dict mapping ticker -> list of articles
      - ticker_names: dict mapping ticker -> display name
      - counts: { total, tickers }
    """
    positions = _read_portfolio()

    all_items: list[dict[str, Any]] = []
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    ticker_names: dict[str, str] = {}

    # Try to establish IBKR connection (graceful failure)
    ib = _connect_ib()

    for i, pos in enumerate(positions):
        ticker = pos["ticker"]
        asset = pos.get("asset", "equity")
        name = _resolve_name(ticker, asset)
        ticker_names[ticker] = name

        # Fetch from GDELT
        gdelt_articles = _query_gdelt(ticker, name)

        # Fetch from IBKR (if connected)
        ibkr_articles = []
        if ib is not None:
            try:
                ibkr_articles = _query_ibkr(ib, ticker, asset)
            except Exception as e:
                logger.warning("IBKR query error for %s: %s", ticker, e)

        combined = gdelt_articles + ibkr_articles
        by_ticker[ticker] = combined
        all_items.extend(combined)

        # Rate-limit GDELT: sleep between requests (skip after last one)
        if i < len(positions) - 1:
            time.sleep(REQUEST_DELAY)

    # Disconnect IBKR if connected
    if ib is not None:
        try:
            ib.disconnect()
        except Exception:
            pass

    # Sort all items chronologically (newest first)
    all_items.sort(key=lambda x: x.get("seendate", ""), reverse=True)

    return {
        "items": all_items,
        "by_ticker": by_ticker,
        "ticker_names": ticker_names,
        "counts": {
            "total": len(all_items),
            "tickers": len(positions),
        },
    }


if __name__ == "__main__":
    import json

    result = get_data()
    print(json.dumps(result, indent=2, default=str))
