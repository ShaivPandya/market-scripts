"""
Portfolio News Feed via IBKR TWS API.

Queries IBKR for each ticker/company-name in portfolio.csv and returns
a unified feed with both grouped-by-ticker and flat chronological views.
"""

import csv
import os
import logging
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import email.utils
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

PREMIUM_DOMAINS = [
    "bloomberg.com", "ft.com", "reuters.com", "wsj.com", 
    "nytimes.com", "marketwatch.com", "asia.nikkei.com", 
    "scmp.com", "caixinglobal.com", "axios.com", 
    "politico.com", "cnbc.com", "theglobeandmail.com"
]

PORTFOLIO_CSV = Path(__file__).parent / "portfolio.csv"

# ── IBKR settings (via env) ───────────────────────────────────────────────────
IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORT = int(os.environ.get("IB_PORT", "4001"))
IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "10"))
IB_MAX_HEADLINES = 10

IB_NEWS_PROVIDER_PREFERENCE = (
    "DJ-RTG",    # Dow Jones Top Stories Global
    "DJ-RT",     # Dow Jones Trader News
    "DJ-N",      # Dow Jones Global Equity Trader
    "DJNL",      # Dow Jones Newsletters
    "DJ-RTA",    # Dow Jones Top Stories Asia Pacific
    "DJ-RTE",    # Dow Jones Top Stories Europe
    "BRFG",      # Briefing.com General Market Columns
    "BRFUPDN",   # Briefing.com Analyst Actions
)

LEGAL_ENTITY_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "llc", "plc", "group", "sa", "ag", "nv", "lp", "holdings", "holding",
}


def _parse_strict_tickers(raw: str) -> set[str]:
    tickers = {part.strip().upper() for part in raw.split(",") if part.strip()}
    return tickers or {"FLY"}


NEWS_STRICT_TICKERS = _parse_strict_tickers(os.environ.get("NEWS_STRICT_TICKERS", "FLY"))
NEWS_LOOKBACK_DAYS = 30

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


# ── IBKR ──────────────────────────────────────────────────────────────────────

def _fetch_all_ibkr_news(positions: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    """
    Run all IBKR news fetching in a separate thread with its own event loop,
    avoiding conflicts with uvicorn's uvloop.
    Returns a dict mapping ticker -> list of articles.
    """
    def _run():
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            from ib_insync import IB
            ib = IB()
            ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=5)
        except Exception as e:
            logger.warning("IBKR connection unavailable (%s: %s), skipping IB news", type(e).__name__, e)
            return {}

        try:
            provider_codes = _select_ibkr_news_providers(ib)
            if not provider_codes:
                logger.warning("No IBKR news providers available")
                ib.disconnect()
                return {}

            results: dict[str, list[dict[str, Any]]] = {}
            for pos in positions:
                ticker = pos["ticker"]
                asset = pos.get("asset", "equity")
                try:
                    results[ticker] = _query_ibkr(ib, ticker, asset, provider_codes)
                except Exception as e:
                    logger.warning("IBKR query error for %s: %s", ticker, e)
                    results[ticker] = []
            return results
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            return future.result(timeout=30)
    except Exception as e:
        logger.warning("IBKR thread failed (%s: %s), skipping IB news", type(e).__name__, e)
        return {}


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
            "HE": "HEX",
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


def _select_ibkr_news_providers(ib) -> str:
    """
    Return a '+'-joined provider code string that this account is entitled to.
    Prefers a small curated set; falls back to all available providers.
    """
    try:
        providers = ib.reqNewsProviders() or []
    except Exception as e:
        logger.info("IBKR reqNewsProviders() failed (%s); skipping IB news", e)
        return ""

    available = [getattr(p, "code", "") for p in providers]
    available = [c for c in available if c]
    if not available:
        return ""

    available_set = set(available)
    chosen = [c for c in IB_NEWS_PROVIDER_PREFERENCE if c in available_set]
    if not chosen:
        chosen = available

    return "+".join(chosen)


def _query_ibkr(ib, ticker: str, asset: str, provider_codes: str) -> list[dict[str, Any]]:
    """
    Query IBKR TWS API for historical news headlines for a ticker.
    Returns a list of article dicts.
    """
    if not provider_codes:
        return []

    contract = _qualify_contract(ib, ticker, asset)
    if contract is None:
        return []

    start_dt = datetime.now(timezone.utc) - timedelta(days=3)
    end_dt = datetime.now(timezone.utc)

    try:
        headlines = ib.reqHistoricalNews(
            contract.conId,
            provider_codes,
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
        headline_text = re.sub(r"^\{[^}]*\}", "", getattr(h, "headline", "") or "").strip()
        article_time = getattr(h, "time", None)
        provider_code = getattr(h, "providerCode", "") or ""

        seendate = ""
        if article_time:
            try:
                if isinstance(article_time, datetime):
                    if article_time.tzinfo is None:
                        article_time = article_time.replace(tzinfo=timezone.utc)
                    else:
                        article_time = article_time.astimezone(timezone.utc)
                    seendate = article_time.isoformat().replace("+00:00", "Z")
                else:
                    seendate = str(article_time)
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


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"<[^>]+>", " ", text or "")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _build_company_aliases(name: str) -> list[str]:
    normalized_name = _normalize_text(name)
    if not normalized_name:
        return []

    aliases: list[str] = [normalized_name]
    tokens = normalized_name.split()
    while tokens and tokens[-1] in LEGAL_ENTITY_SUFFIXES:
        tokens = tokens[:-1]

    stripped_name = " ".join(tokens)
    if stripped_name and stripped_name not in aliases:
        aliases.append(stripped_name)

    # Keep only non-trivial aliases.
    return [alias for alias in aliases if len(alias) > 1]


def _is_reliable_alias(ticker: str, alias: str) -> bool:
    if not alias:
        return False

    ticker_norm = _normalize_text(ticker)
    if alias == ticker_norm:
        return False

    meaningful_tokens = [tok for tok in alias.split() if tok not in LEGAL_ENTITY_SUFFIXES]
    if not meaningful_tokens:
        return False

    return len("".join(meaningful_tokens)) >= 4


def _build_google_rss_query(ticker: str, name: str) -> tuple[str, bool, list[str]]:
    sites_query = " OR ".join(f"site:{domain}" for domain in PREMIUM_DOMAINS)
    strict_mode = ticker.upper() in NEWS_STRICT_TICKERS
    aliases = _build_company_aliases(name)

    if strict_mode:
        reliable_aliases = [alias for alias in aliases if _is_reliable_alias(ticker, alias)]
        if not reliable_aliases:
            return "", True, []

        alias_query = " OR ".join(f'"{alias}"' for alias in reliable_aliases)
        return f"({alias_query}) AND ({sites_query})", True, reliable_aliases

    return f'({ticker} OR "{name}") AND ({sites_query})', False, aliases


def _article_mentions_alias(title: str, description: str, aliases: list[str]) -> bool:
    article_text = _normalize_text(f"{title} {description}")
    article_text_padded = f" {article_text} "
    return any(f" {alias} " in article_text_padded for alias in aliases)


def _truncate_for_log(text: str, max_len: int = 100) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _is_within_lookback(seendate: str, now_utc: datetime) -> bool:
    if not seendate:
        return False
    parsed = None
    try:
        parsed = datetime.fromisoformat(seendate.replace("Z", "+00:00"))
    except Exception:
        # Some feeds use RFC 822 / RFC 1123 date strings.
        try:
            parsed = email.utils.parsedate_to_datetime(seendate)
        except Exception:
            return False
    if parsed is None:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    cutoff = now_utc - timedelta(days=NEWS_LOOKBACK_DAYS)
    return parsed >= cutoff


def _query_google_rss(ticker: str, name: str) -> list[dict[str, Any]]:
    """
    Query Google News RSS for the specified premium publisher domains.
    Returns a list of article dicts.
    """
    query, strict_mode, strict_aliases = _build_google_rss_query(ticker, name)
    if strict_mode and not strict_aliases:
        logger.info(
            "Skipping Google RSS for strict ticker %s: no reliable aliases derived from name '%s'",
            ticker,
            name,
        )
        return []

    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    results = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        for item in root.findall('./channel/item'):
            title = item.findtext('title') or ""
            description = item.findtext('description') or ""
            link = item.findtext('link') or ""
            source = item.findtext('source') or "Google News"
            pubDate = item.findtext('pubDate') or ""

            if strict_mode and not _article_mentions_alias(title, description, strict_aliases):
                logger.debug(
                    "Dropped Google RSS item for %s (reason=no_alias_match, title='%s')",
                    ticker,
                    _truncate_for_log(title),
                )
                continue
            
            try:
                parsed_date = email.utils.parsedate_to_datetime(pubDate)
                seendate = parsed_date.isoformat().replace("+00:00", "Z")
            except Exception:
                seendate = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            results.append({
                "ticker": ticker,
                "title": title,
                "url": link,
                "source": source,
                "seendate": seendate,
                "socialimage": "",
                "language": "English",
                "provider": "Google RSS",
            })
    except Exception as e:
        logger.warning("Google RSS query failed for %s: %s", ticker, e)
        
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

    now_utc = datetime.now(timezone.utc)

    # Fetch IBKR news in a separate thread (avoids uvloop conflicts)
    ibkr_by_ticker = _fetch_all_ibkr_news(positions)

    for pos in positions:
        ticker = pos["ticker"]
        asset = pos.get("asset", "equity")
        name = _resolve_name(ticker, asset)
        ticker_names[ticker] = name

        ibkr_articles = ibkr_by_ticker.get(ticker, [])
        rss_articles = _query_google_rss(ticker, name)

        combined_articles = [
            article
            for article in (ibkr_articles + rss_articles)
            if _is_within_lookback(str(article.get("seendate", "")), now_utc)
        ]
        by_ticker[ticker] = combined_articles
        all_items.extend(combined_articles)

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
