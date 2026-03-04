"""
Portfolio News Feed via GDELT DOC 2.0 API.

Queries GDELT for each ticker/company-name in portfolio.csv and returns
a unified feed with both grouped-by-ticker and flat chronological views.
"""

import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

PORTFOLIO_CSV = Path(__file__).parent / "portfolio.csv"

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TIMESPAN = "3d"
GDELT_MAX_RECORDS = 10
REQUEST_DELAY = 0.6  # seconds between GDELT calls to avoid rate-limiting

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

    try:
        resp = requests.get(GDELT_DOC_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("GDELT query failed for %s: %s", ticker, e)
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

    for i, pos in enumerate(positions):
        ticker = pos["ticker"]
        asset = pos.get("asset", "equity")
        name = _resolve_name(ticker, asset)
        ticker_names[ticker] = name

        articles = _query_gdelt(ticker, name)
        by_ticker[ticker] = articles
        all_items.extend(articles)

        # Rate-limit: sleep between requests (skip after last one)
        if i < len(positions) - 1:
            time.sleep(REQUEST_DELAY)

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
