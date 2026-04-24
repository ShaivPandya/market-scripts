#!/usr/bin/env python3
"""
Short Screen: Identify potential short candidates from the Russell 2000.

Screening criteria (all must be met to pass):
  1. P/B ratio above threshold (priceToBook from yfinance; fallback: market_cap / book_equity)
  2. Gross loss OR operating loss (from most recent annual income statement)
  3. (Optional) Net equity issuance in the top quartile among Phase 1 passers (SEC EDGAR XBRL API)

Execution is phased:
  Phase 1 — parallel yfinance fetch for all ~1,948 Russell 2000 tickers, filter by P/B + loss
  Phase 2 — sequential SEC EDGAR calls ONLY for Phase 1 passers, reducing ~2,000 API calls to ~10-50
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
import pandas as pd
import yfinance as yf

from utils.retry import requests_get

LOGGER = logging.getLogger(__name__)

from equities.common import load_universe

# ---------------------------------------------------------------------------
# Module-level SEC caches
# ---------------------------------------------------------------------------
_cik_map: dict[str, str] = {}  # ticker.upper() -> zero-padded 10-digit CIK string
_cik_map_loaded: bool = False
_cik_map_lock = threading.Lock()

_edgar_facts_cache: dict[str, dict | None] = {}  # cik_str -> raw companyfacts JSON or None
_edgar_facts_lock = threading.Lock()

SEC_HEADERS = {"User-Agent": "market-scripts research@example.com"}
SEC_RATE_LIMIT_DELAY = 0.12  # comfortably under SEC's 10 requests/second limit


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


YF_CHUNK_SIZE = 100  # tickers per batch (Phase 1 and Phase 3)
YF_BATCH_DELAY = 0.5  # seconds between batches
PHASE1_WORKERS = max(1, _env_int("SCREEN_YF_WORKERS", 4))  # concurrent yfinance statement threads
YF_CALL_INTERVAL_SECONDS = max(0.0, _env_float("SCREEN_YF_CALL_INTERVAL_SECONDS", 0.08))
YF_RATE_LIMIT_COOLDOWN_SECONDS = max(0.0, _env_float("SCREEN_YF_RATE_LIMIT_COOLDOWN_SECONDS", 20.0))

_yf_call_lock = threading.Lock()
_yf_last_call_at = 0.0
_yf_cooldown_until = 0.0


class YahooRateLimitError(RuntimeError):
    """Raised when Yahoo explicitly rate limits a yfinance call."""


def _is_yahoo_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "too many requests" in msg or "rate limited" in msg or "rate limit" in msg


def _throttled_yf_call(label: str, func):
    """
    Rate-limit yfinance quote-summary/statement call starts across screen jobs.

    Yahoo's quote-summary endpoints rate-limit aggressively when the Russell
    2000 is fanned out across workers. This gate spaces request starts and
    honors cooldowns without serializing the entire network call.
    """
    global _yf_cooldown_until, _yf_last_call_at

    with _yf_call_lock:
        now = time.monotonic()
        wait_for_spacing = max(0.0, (_yf_last_call_at + YF_CALL_INTERVAL_SECONDS) - now)
        wait_for_cooldown = max(0.0, _yf_cooldown_until - now)
        delay = max(wait_for_spacing, wait_for_cooldown)
        if delay > 0:
            time.sleep(delay)
        _yf_last_call_at = time.monotonic()

    try:
        return func()
    except Exception as exc:
        if _is_yahoo_rate_limit_error(exc):
            with _yf_call_lock:
                _yf_cooldown_until = max(_yf_cooldown_until, time.monotonic() + YF_RATE_LIMIT_COOLDOWN_SECONDS)
            raise YahooRateLimitError(f"{label}: {exc}") from exc
        raise


# Label variants for quarterly revenue / EPS extraction
REVENUE_KEYS = ["Total Revenue", "TotalRevenue", "Revenue", "Operating Revenue", "Net Sales", "NetSales"]
EPS_KEYS = ["Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS"]
NET_INCOME_KEYS = ["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders"]
DILUTED_SHARES_KEYS = ["Diluted Average Shares", "DilutedAverageShares", "Weighted Average Shares Diluted"]


# ---------------------------------------------------------------------------
# yfinance helpers (pattern from equities/quality/quality_single.py)
# ---------------------------------------------------------------------------


def last_col(df: pd.DataFrame | None) -> pd.Series | None:
    """Return most recent column of a yfinance financial statement."""
    if df is None or df.empty:
        return None
    return df.iloc[:, 0]


def get_item(s: pd.Series | None, keys: list[str]) -> float:
    """Try multiple label variants in a yfinance statement series, return first match."""
    if s is None:
        return np.nan
    for k in keys:
        if k in s.index:
            v = s.get(k)
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan
    return np.nan


# ---------------------------------------------------------------------------
# Phase 1: yfinance data fetch
# ---------------------------------------------------------------------------


def fetch_yf_data(
    ticker: str,
    *,
    include_quarterly: bool = False,
    need_price_to_book: bool = True,
    need_profit: bool = True,
    need_market_cap: bool = True,
) -> dict:
    """
    Fetch P/B, gross profit, operating income, and market cap for one ticker.

    Returns a dict with keys:
        ticker, company_name, price_to_book, gross_profit, operating_income,
        market_cap, book_value

    Missing values are np.nan. Hard yfinance failures return {"ticker": ticker, "error": str}.
    """
    try:
        t = yf.Ticker(ticker)

        info: dict = {}
        if need_price_to_book or need_market_cap:
            try:
                info = _throttled_yf_call(f"{ticker}.info", lambda: dict(yf.Ticker(ticker).info or {}))
            except YahooRateLimitError as exc:
                return {"ticker": ticker, "error": str(exc)}
            except Exception:
                info = {}

        inc_last: pd.Series | None = None
        if need_profit:
            try:
                fin = _throttled_yf_call(f"{ticker}.financials", lambda: t.financials)
            except YahooRateLimitError as exc:
                return {"ticker": ticker, "error": str(exc)}
            except Exception:
                LOGGER.debug("Annual financials fetch failed for %s", ticker, exc_info=True)
                fin = pd.DataFrame()
            inc_last = last_col(fin)

        # P/B from info dict (primary source)
        price_to_book: float = np.nan
        if need_price_to_book:
            pb_raw = info.get("priceToBook")
            if pb_raw is not None:
                try:
                    price_to_book = float(pb_raw)
                except (TypeError, ValueError):
                    pass

        # Market cap — try fast_info first (uses chart API, immune to crumb invalidation)
        # then fall back to info dict. This matters because t.info returns {} when Yahoo
        # Finance rejects the request crumb, but fast_info uses a different endpoint.
        market_cap: float = np.nan
        mc_raw = info.get("marketCap")
        if mc_raw is not None:
            try:
                market_cap = float(mc_raw)
            except (TypeError, ValueError):
                pass
        if need_market_cap and np.isnan(market_cap):
            try:
                mc_fast = _throttled_yf_call(f"{ticker}.fast_info.market_cap", lambda: t.fast_info.market_cap)
                if mc_fast is not None:
                    market_cap = float(mc_fast)
            except YahooRateLimitError as exc:
                return {"ticker": ticker, "error": str(exc)}
            except Exception:
                LOGGER.debug("fast_info.market_cap failed for %s", ticker, exc_info=True)
        if np.isnan(market_cap):
            market_cap = np.nan

        book_value = np.nan
        if need_price_to_book and np.isnan(price_to_book):
            if np.isnan(market_cap):
                try:
                    mc_fast = _throttled_yf_call(f"{ticker}.fast_info.market_cap", lambda: t.fast_info.market_cap)
                    if mc_fast is not None:
                        market_cap = float(mc_fast)
                except YahooRateLimitError as exc:
                    return {"ticker": ticker, "error": str(exc)}
                except Exception:
                    LOGGER.debug("fast_info.market_cap failed for %s", ticker, exc_info=True)
            try:
                bal = _throttled_yf_call(f"{ticker}.balance_sheet", lambda: t.balance_sheet)
            except YahooRateLimitError as exc:
                return {"ticker": ticker, "error": str(exc)}
            except Exception:
                LOGGER.debug("Balance sheet fetch failed for %s", ticker, exc_info=True)
                bal = pd.DataFrame()
            bal_last = last_col(bal)
            book_value = get_item(
                bal_last,
                [
                    "Stockholders Equity",
                    "StockholdersEquity",
                    "Total Stockholder Equity",
                    "TotalStockholderEquity",
                    "Common Stock Equity",
                ],
            )

        # P/B fallback: market_cap / book_value when priceToBook is missing
        if np.isnan(price_to_book) and not np.isnan(market_cap) and not np.isnan(book_value) and book_value > 0:
            price_to_book = market_cap / book_value

        # Gross profit (most recent annual)
        gross_profit = get_item(inc_last, ["Gross Profit", "GrossProfit"]) if need_profit else np.nan

        # Operating income (most recent annual)
        operating_income = (
            get_item(
                inc_last,
                [
                    "Operating Income",
                    "OperatingIncome",
                    "EBIT",
                    "Ebit",
                ],
            )
            if need_profit
            else np.nan
        )

        # Quarterly revenue & EPS YoY growth (need 7 quarters: 3 recent + 4 year-ago)
        rev_yoy_q0 = rev_yoy_q1 = rev_yoy_q2 = rev_yoy_avg = np.nan
        eps_yoy_q0 = eps_yoy_q1 = eps_yoy_q2 = eps_yoy_avg = np.nan
        if include_quarterly:
            try:
                q_fin = _throttled_yf_call(f"{ticker}.quarterly_financials", lambda: t.quarterly_financials)
                if q_fin is not None and not q_fin.empty and q_fin.shape[1] >= 5:
                    # Extract per-quarter revenue
                    revs = [get_item(q_fin.iloc[:, i], REVENUE_KEYS) for i in range(min(q_fin.shape[1], 7))]
                    rev_yoys: list[float] = []
                    for idx, (recent_i, prior_i) in enumerate([(0, 4), (1, 5), (2, 6)]):
                        if prior_i < len(revs):
                            r, p = revs[recent_i], revs[prior_i]
                            if not np.isnan(r) and not np.isnan(p) and p != 0:
                                val = (r / p - 1) * 100
                                rev_yoys.append(val)
                                if idx == 0:
                                    rev_yoy_q0 = val
                                elif idx == 1:
                                    rev_yoy_q1 = val
                                else:
                                    rev_yoy_q2 = val
                    if rev_yoys:
                        rev_yoy_avg = float(np.mean(rev_yoys))

                    # Extract per-quarter EPS (direct label, then fallback to net income / shares)
                    def _get_eps(col_idx: int) -> float:
                        col = q_fin.iloc[:, col_idx]
                        eps_val = get_item(col, EPS_KEYS)
                        if not np.isnan(eps_val):
                            return eps_val
                        ni = get_item(col, NET_INCOME_KEYS)
                        sh = get_item(col, DILUTED_SHARES_KEYS)
                        if not np.isnan(ni) and not np.isnan(sh) and sh != 0:
                            return ni / sh
                        return np.nan

                    epss = [_get_eps(i) for i in range(min(q_fin.shape[1], 7))]
                    eps_yoys: list[float] = []
                    for idx, (recent_i, prior_i) in enumerate([(0, 4), (1, 5), (2, 6)]):
                        if prior_i < len(epss):
                            r, p = epss[recent_i], epss[prior_i]
                            if not np.isnan(r) and not np.isnan(p) and abs(p) > 0:
                                val = (r - p) / abs(p) * 100
                                eps_yoys.append(val)
                                if idx == 0:
                                    eps_yoy_q0 = val
                                elif idx == 1:
                                    eps_yoy_q1 = val
                                else:
                                    eps_yoy_q2 = val
                    if eps_yoys:
                        eps_yoy_avg = float(np.mean(eps_yoys))
            except YahooRateLimitError as exc:
                return {"ticker": ticker, "error": str(exc)}
            except Exception:
                LOGGER.debug("Quarterly financials fetch failed for %s", ticker, exc_info=True)

        company_name: str = info.get("longName") or info.get("shortName") or ""

        return {
            "ticker": ticker,
            "company_name": company_name,
            "price_to_book": price_to_book,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "market_cap": market_cap,
            "book_value": book_value,
            "rev_yoy_q0": rev_yoy_q0,
            "rev_yoy_q1": rev_yoy_q1,
            "rev_yoy_q2": rev_yoy_q2,
            "rev_yoy_avg": rev_yoy_avg,
            "eps_yoy_q0": eps_yoy_q0,
            "eps_yoy_q1": eps_yoy_q1,
            "eps_yoy_q2": eps_yoy_q2,
            "eps_yoy_avg": eps_yoy_avg,
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def screen_ticker(
    ticker: str,
    pb_threshold: float | None,
    loss_type: str | None,
    check_revenue: bool = False,
    max_revenue_growth: float = 0.0,
    check_eps: bool = False,
    max_eps_growth: float = 0.0,
    need_market_cap: bool = False,
) -> tuple[bool, dict]:
    """
    Apply Phase 1 criteria (P/B + loss type + optional revenue/EPS growth) using yfinance data.

    Returns:
        (passes: bool, data: dict)
        data["error"] is set if yfinance fetch failed hard.
    """
    data = fetch_yf_data(
        ticker,
        include_quarterly=check_revenue or check_eps,
        need_price_to_book=pb_threshold is not None,
        need_profit=loss_type is not None,
        need_market_cap=need_market_cap or pb_threshold is not None,
    )

    if "error" in data:
        return False, data

    pb = data.get("price_to_book", np.nan)
    gross = data.get("gross_profit", np.nan)
    operating = data.get("operating_income", np.nan)

    if pb_threshold is not None:
        pb_ok = (not (isinstance(pb, float) and np.isnan(pb))) and (pb > pb_threshold)
    else:
        pb_ok = True

    if loss_type is None:
        loss_ok = True
    elif loss_type == "Gross Loss":
        loss_ok = (not (isinstance(gross, float) and np.isnan(gross))) and (gross < 0)
    else:
        loss_ok = (not (isinstance(operating, float) and np.isnan(operating))) and (operating < 0)

    # Revenue growth filter: average YoY revenue growth across 3 quarters must be <= threshold
    if check_revenue:
        avg = data.get("rev_yoy_avg", np.nan)
        rev_ok = not (isinstance(avg, float) and np.isnan(avg)) and avg <= max_revenue_growth
    else:
        rev_ok = True

    # EPS growth filter: average YoY EPS growth across 3 quarters must be <= threshold
    if check_eps:
        avg = data.get("eps_yoy_avg", np.nan)
        eps_ok = not (isinstance(avg, float) and np.isnan(avg)) and avg <= max_eps_growth
    else:
        eps_ok = True

    return (pb_ok and loss_ok and rev_ok and eps_ok), data


# ---------------------------------------------------------------------------
# Phase 2: SEC EDGAR equity issuance
# ---------------------------------------------------------------------------


def _load_cik_map() -> None:
    """
    Download SEC's full company-ticker-CIK mapping once per process lifetime.
    Sets _cik_map_loaded=True even on failure to prevent retry loops.
    """
    global _cik_map_loaded
    with _cik_map_lock:
        if _cik_map_loaded:
            return
        try:
            resp = requests_get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=SEC_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            for entry in data.values():
                tk = str(entry.get("ticker", "")).upper()
                cik_int = entry.get("cik_str", 0)
                if tk and cik_int:
                    _cik_map[tk] = f"{int(cik_int):010d}"
        except Exception:
            LOGGER.debug("SEC CIK map load failed", exc_info=True)
        finally:
            _cik_map_loaded = True


def _fetch_edgar_facts(cik_str: str) -> dict | None:
    """
    Fetch XBRL company facts from SEC EDGAR for one CIK.
    Rate-limited via SEC_RATE_LIMIT_DELAY sleep before each network request.
    Results cached in _edgar_facts_cache for the process lifetime.
    """
    with _edgar_facts_lock:
        if cik_str in _edgar_facts_cache:
            return _edgar_facts_cache[cik_str]

    # Sleep outside the lock so other threads are not blocked during the wait
    time.sleep(SEC_RATE_LIMIT_DELAY)

    result: dict | None = None
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
        resp = requests_get(url, headers=SEC_HEADERS, timeout=20)
        if resp.status_code == 200:
            result = resp.json()
        # 404 → company has no XBRL facts; result stays None
    except Exception:
        LOGGER.debug("EDGAR facts fetch failed for CIK %s", cik_str, exc_info=True)

    with _edgar_facts_lock:
        _edgar_facts_cache[cik_str] = result

    return result


def _extract_annual_10k(entries: list) -> float | None:
    """
    From a list of XBRL fact entries, return the value from the most recent 10-K
    whose 'end' date falls within the past 18 months.

    18 months is used (rather than 12) because small-caps often have non-December
    fiscal year ends and can file their 10-Ks up to 9 months later.
    """
    cutoff = date.today() - timedelta(days=548)
    best_end: date | None = None
    best_val: float | None = None

    for entry in entries:
        if entry.get("form") != "10-K":
            continue
        try:
            end_date = date.fromisoformat(entry["end"])
        except (KeyError, ValueError):
            continue
        if end_date < cutoff:
            continue
        if best_end is None or end_date > best_end:
            best_end = end_date
            best_val = float(entry["val"])

    return best_val


def fetch_sec_issuance(ticker: str) -> dict:
    """
    Fetch net equity issuance from SEC EDGAR for one ticker.

    Uses:
        ProceedsFromIssuanceOfCommonStock (most recent 10-K)
        PaymentsForRepurchaseOfCommonStock (most recent 10-K)
        net_issuance = proceeds - repurchases

    Returns dict with {proceeds_issuance, repurchases, net_issuance}
    or {"error": str} on failure.
    """
    _load_cik_map()

    cik_str = _cik_map.get(ticker.upper())
    if not cik_str:
        return {"error": f"CIK not found for {ticker}"}

    facts = _fetch_edgar_facts(cik_str)
    if facts is None:
        return {"error": f"No EDGAR facts available for CIK {cik_str}"}

    try:
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        proceeds_entries = us_gaap.get("ProceedsFromIssuanceOfCommonStock", {}).get("units", {}).get("USD", [])
        repurchase_entries = us_gaap.get("PaymentsForRepurchaseOfCommonStock", {}).get("units", {}).get("USD", [])

        proceeds = _extract_annual_10k(proceeds_entries)
        repurchases = _extract_annual_10k(repurchase_entries)

        proceeds_f = float(proceeds) if proceeds is not None else np.nan
        repurchases_f = float(repurchases) if repurchases is not None else np.nan

        if not (np.isnan(proceeds_f) and np.isnan(repurchases_f)):
            net = (0.0 if np.isnan(proceeds_f) else proceeds_f) - (0.0 if np.isnan(repurchases_f) else repurchases_f)
        else:
            net = np.nan

        return {
            "proceeds_issuance": proceeds_f,
            "repurchases": repurchases_f,
            "net_issuance": net,
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Phase 3: Price-based filters (batch yf_download)
# ---------------------------------------------------------------------------


def _apply_price_filters(
    passers: list[dict],
    *,
    check_52w_positive: bool,
    check_min_drawdown: bool,
    min_drawdown_pct: float,
    check_max_drawdown: bool,
    max_drawdown_pct: float,
    check_3m_neg_momentum: bool,
    check_2m_neg_rel_momentum: bool,
    benchmark_ticker: str,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Apply optional price-based filters to Phase 1/2 passers.

    Uses a single batch yf_download for all passers + benchmark.

    Returns:
        (filtered_passers, price_metrics)
        price_metrics maps ticker -> {return_52w, drawdown_pct, return_3m, rel_return_2m}
    """
    from utils.retry import yf_download

    passer_tickers = [d["ticker"] for d in passers]

    # Build download list: passers + benchmark (if needed for relative momentum)
    download_tickers = list(passer_tickers)
    need_benchmark = check_2m_neg_rel_momentum and benchmark_ticker
    if need_benchmark and benchmark_ticker not in download_tickers:
        download_tickers.append(benchmark_ticker)

    # Chunked download with pauses to avoid rate limits
    chunks = [download_tickers[i : i + YF_CHUNK_SIZE] for i in range(0, len(download_tickers), YF_CHUNK_SIZE)]
    all_dfs: list[pd.DataFrame] = []
    for i, chunk in enumerate(chunks):
        try:
            df = yf_download(
                chunk,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if df is not None and not df.empty:
                all_dfs.append(df)
        except Exception:
            LOGGER.warning("Price filter batch %d/%d failed", i + 1, len(chunks), exc_info=True)
        if i < len(chunks) - 1:
            time.sleep(YF_BATCH_DELAY)

    if not all_dfs:
        LOGGER.warning("All price filter downloads failed; skipping price filters")
        return passers, {}

    if len(all_dfs) == 1:
        raw = all_dfs[0]
    else:
        parts: dict[str, list[pd.DataFrame]] = {}
        for df in all_dfs:
            if isinstance(df.columns, pd.MultiIndex):
                for level in df.columns.get_level_values(0).unique():
                    parts.setdefault(str(level), []).append(df[level])
            else:
                for col in df.columns:
                    parts.setdefault(str(col), []).append(df[[col]])
        raw = pd.concat({k: pd.concat(v, axis=1) for k, v in parts.items() if v}, axis=1)

    # Extract close prices per ticker
    def _get_close(df: pd.DataFrame, ticker: str) -> pd.Series | None:
        if df is None or df.empty:
            return None
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.get_level_values(1):
                    s = df[("Close", ticker)].dropna()
                else:
                    return None
            else:
                s = df["Close"].dropna()
            return s if len(s) > 0 else None
        except (KeyError, TypeError):
            return None

    # Pre-extract benchmark close if needed
    bench_close: pd.Series | None = None
    if need_benchmark:
        bench_close = _get_close(raw, benchmark_ticker)

    filtered: list[dict] = []
    metrics: dict[str, dict] = {}

    for data in passers:
        tk = data["ticker"]
        close = _get_close(raw, tk)
        if close is None or len(close) < 10:
            continue  # Insufficient data — exclude

        current = float(close.iloc[-1])
        m: dict[str, float | None] = {}

        # 52-week return
        ret_52w: float | None = None
        if len(close) >= 200:  # need ~1 year of data
            price_52w = float(close.iloc[0])
            ret_52w = (current / price_52w - 1) * 100
        m["return_52w"] = ret_52w

        # Drawdown from 52-week high
        peak = float(close.max())
        dd_pct = (current - peak) / peak * 100 if peak > 0 else None
        m["drawdown_pct"] = dd_pct

        # 3-month return (~63 trading days)
        ret_3m: float | None = None
        if len(close) >= 63:
            price_3m = float(close.iloc[-63])
            ret_3m = (current / price_3m - 1) * 100
        m["return_3m"] = ret_3m

        # 2-month relative return (~42 trading days)
        rel_ret_2m: float | None = None
        if len(close) >= 42 and bench_close is not None and len(bench_close) >= 42:
            price_2m = float(close.iloc[-42])
            stock_ret = (current / price_2m - 1) * 100
            bench_current = float(bench_close.iloc[-1])
            bench_2m = float(bench_close.iloc[-42])
            bench_ret = (bench_current / bench_2m - 1) * 100
            rel_ret_2m = stock_ret - bench_ret
        m["rel_return_2m"] = rel_ret_2m

        # Apply filters — stock must pass ALL enabled filters
        passes = True

        if check_52w_positive:
            if ret_52w is None or ret_52w <= 0:
                passes = False

        if check_min_drawdown and passes:
            if dd_pct is None or abs(dd_pct) < min_drawdown_pct:
                passes = False

        if check_max_drawdown and passes:
            if dd_pct is None or abs(dd_pct) > max_drawdown_pct:
                passes = False

        if check_3m_neg_momentum and passes:
            if ret_3m is None or ret_3m >= 0:
                passes = False

        if check_2m_neg_rel_momentum and passes:
            if rel_ret_2m is None or rel_ret_2m >= 0:
                passes = False

        if passes:
            filtered.append(data)
            metrics[tk] = m

    return filtered, metrics


# ---------------------------------------------------------------------------
# Result row builder
# ---------------------------------------------------------------------------


def _build_result_row(data: dict, price_metrics: dict | None = None) -> dict:
    """Convert raw yfinance data dict to a display-ready row (values in $M)."""

    def to_m(val) -> float | None:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val) / 1e6, 1)

    def fmt_pb(val) -> float | None:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 2)

    def fmt_pct(val) -> float | None:
        if val is None:
            return None
        return round(float(val), 1)

    row: dict = {
        "Ticker": data["ticker"],
        "Company": data.get("company_name") or "",
        "P/B Ratio": fmt_pb(data.get("price_to_book")),
        "Gross Profit ($M)": to_m(data.get("gross_profit")),
        "Operating Income ($M)": to_m(data.get("operating_income")),
        "Market Cap ($M)": to_m(data.get("market_cap")),
    }
    # Revenue / EPS YoY growth columns (shown when filters are active)
    for key, label in [
        ("rev_yoy_q0", "Rev YoY Q0 (%)"),
        ("rev_yoy_q1", "Rev YoY Q1 (%)"),
        ("rev_yoy_q2", "Rev YoY Q2 (%)"),
    ]:
        val = data.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            row[label] = fmt_pct(val)
    for key, label in [
        ("eps_yoy_q0", "EPS YoY Q0 (%)"),
        ("eps_yoy_q1", "EPS YoY Q1 (%)"),
        ("eps_yoy_q2", "EPS YoY Q2 (%)"),
    ]:
        val = data.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            row[label] = fmt_pct(val)

    if price_metrics:
        if price_metrics.get("return_52w") is not None:
            row["52w Return (%)"] = fmt_pct(price_metrics["return_52w"])
        if price_metrics.get("drawdown_pct") is not None:
            row["Drawdown (%)"] = fmt_pct(price_metrics["drawdown_pct"])
        if price_metrics.get("return_3m") is not None:
            row["3m Return (%)"] = fmt_pct(price_metrics["return_3m"])
        if price_metrics.get("rel_return_2m") is not None:
            row["2m Rel Return (%)"] = fmt_pct(price_metrics["rel_return_2m"])
    return row


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _report_progress(progress_callback, phase: str, done: int, total: int) -> None:
    if not progress_callback:
        return
    try:
        progress_callback(phase, done, total)
    except TypeError:
        progress_callback(done, total)


def _any_short_price_filter(
    *,
    check_52w_positive: bool,
    check_min_drawdown: bool,
    check_max_drawdown: bool,
    check_3m_neg_momentum: bool,
    check_2m_neg_rel_momentum: bool,
) -> bool:
    return (
        check_52w_positive
        or check_min_drawdown
        or check_max_drawdown
        or check_3m_neg_momentum
        or check_2m_neg_rel_momentum
    )


def _needs_short_fundamentals(
    *,
    pb_threshold: float | None,
    loss_type: str | None,
    check_revenue: bool,
    check_eps: bool,
) -> bool:
    return pb_threshold is not None or loss_type is not None or check_revenue or check_eps


def _screen_short_fundamentals(
    universe: list[str],
    *,
    pb_threshold: float | None,
    loss_type: str | None,
    check_revenue: bool,
    max_revenue_growth: float,
    check_eps: bool,
    max_eps_growth: float,
    need_market_cap: bool = False,
    progress_callback=None,
) -> tuple[list[dict], list[str]]:
    phase1_pass_data: list[dict] = []
    failed_tickers: list[str] = []
    done_count = 0
    total = len(universe)
    batches = [universe[i : i + YF_CHUNK_SIZE] for i in range(0, total, YF_CHUNK_SIZE)]

    with ThreadPoolExecutor(max_workers=PHASE1_WORKERS) as pool:
        for batch_idx, batch in enumerate(batches):
            futures = {
                pool.submit(
                    screen_ticker,
                    tk,
                    pb_threshold,
                    loss_type,
                    check_revenue=check_revenue,
                    max_revenue_growth=max_revenue_growth,
                    check_eps=check_eps,
                    max_eps_growth=max_eps_growth,
                    need_market_cap=need_market_cap,
                ): tk
                for tk in batch
            }
            for future in as_completed(futures):
                tk = futures[future]
                try:
                    passes, data = future.result()
                    if passes:
                        phase1_pass_data.append(data)
                    elif "error" in data:
                        failed_tickers.append(f"{tk}: {data.get('error') or 'data fetch failed'}")
                except Exception as exc:
                    failed_tickers.append(f"{tk}: {exc}")

                done_count += 1
                if done_count % 25 == 0 or done_count == total:
                    _report_progress(progress_callback, "fundamentals", done_count, total)

            if batch_idx < len(batches) - 1:
                time.sleep(YF_BATCH_DELAY)

    return phase1_pass_data, failed_tickers


def get_data(
    tickers: list[str],
    pb_threshold: float | None = 3.0,
    loss_type: str | None = "Gross Loss",
    check_issuance: bool = False,
    check_revenue: bool = False,
    max_revenue_growth: float = 0.0,
    check_eps: bool = False,
    max_eps_growth: float = 0.0,
    check_52w_positive: bool = False,
    check_min_drawdown: bool = False,
    min_drawdown_pct: float = 25.0,
    check_max_drawdown: bool = False,
    max_drawdown_pct: float = 60.0,
    check_3m_neg_momentum: bool = False,
    check_2m_neg_rel_momentum: bool = False,
    benchmark_ticker: str = "IWM",
    progress_callback=None,
) -> dict:
    """
    Run the short screen over the provided ticker universe.

    Args:
        tickers:            List of ticker symbols to screen
        pb_threshold:       P/B ratio must exceed this value (3.0 – 5.0)
        loss_type:          "Gross Loss" | "Operating Loss"
        check_issuance:     If True, keep only the top quartile by net equity issuance (SEC EDGAR)
        check_revenue:      If True, filter by max YoY revenue growth (each of last 3 quarters)
        max_revenue_growth: Max allowed YoY revenue growth % (e.g. 0 = flat or declining)
        check_eps:          If True, filter by max avg YoY EPS growth (last 3 quarters)
        max_eps_growth:     Max allowed avg YoY EPS growth % (e.g. 0 = flat or declining)
        check_52w_positive: If True, keep only stocks with positive 52-week return
        check_min_drawdown: If True, keep only stocks at least min_drawdown_pct% below 52w high
        min_drawdown_pct:   Minimum drawdown threshold (e.g. 25 means 25% off highs)
        check_max_drawdown: If True, exclude stocks more than max_drawdown_pct% below 52w high
        max_drawdown_pct:   Maximum drawdown threshold (e.g. 60 means 60% off highs)
        check_3m_neg_momentum:    If True, keep only stocks with negative 3-month return
        check_2m_neg_rel_momentum: If True, keep only stocks underperforming benchmark over 2 months
        benchmark_ticker:   Benchmark for relative momentum (e.g. "IWM", "SPY", "QQQ")
        progress_callback:  Optional callable(done: int, total: int)

    Returns on success:
        {
            "results_df":          pd.DataFrame — one row per candidate, sorted by P/B desc
            "failed_tickers":      List[str]     — tickers that errored in Phase 1
            "phase1_count":        int           — universe size
            "phase1_pass_count":   int           — tickers passing Phase 1
            "phase3_pass_count":   int           — tickers passing price filters (if any enabled)
            "final_count":         int           — rows in results_df
        }

    Returns on hard failure:
        {"error": str}
    """
    universe = tickers

    if not universe:
        return {"error": "No tickers provided"}

    total = len(universe)
    any_price_filter = _any_short_price_filter(
        check_52w_positive=check_52w_positive,
        check_min_drawdown=check_min_drawdown,
        check_max_drawdown=check_max_drawdown,
        check_3m_neg_momentum=check_3m_neg_momentum,
        check_2m_neg_rel_momentum=check_2m_neg_rel_momentum,
    )
    needs_fundamentals = _needs_short_fundamentals(
        pb_threshold=pb_threshold,
        loss_type=loss_type,
        check_revenue=check_revenue,
        check_eps=check_eps,
    )

    # ------------------------------------------------------------------
    # Pre-warm yfinance session so the authentication crumb is fresh
    # before spawning parallel threads.  A stale/missing crumb causes
    # t.info to silently return {} for every ticker, producing 0 results.
    # ------------------------------------------------------------------
    try:
        _throttled_yf_call(f"{universe[0]}.prewarm", lambda: yf.Ticker(universe[0]).fast_info.last_price)  # noqa: B018
    except Exception:
        LOGGER.debug("yfinance session pre-warm failed", exc_info=True)

    # ------------------------------------------------------------------
    # Price prefilter: use one cheap batch chart pass before heavyweight
    # quote-summary/statement calls when issuance does not need the full
    # fundamental-pass universe for its quartile cutoff.
    # ------------------------------------------------------------------
    price_metrics: dict[str, dict] = {}
    phase3_pass_count: int | None = None
    price_prefiltered = False
    candidate_universe = list(universe)

    if any_price_filter and not check_issuance:
        _report_progress(progress_callback, "prices", 0, total)
        price_pass_data, price_metrics = _apply_price_filters(
            [{"ticker": tk} for tk in universe],
            check_52w_positive=check_52w_positive,
            check_min_drawdown=check_min_drawdown,
            min_drawdown_pct=min_drawdown_pct,
            check_max_drawdown=check_max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            check_3m_neg_momentum=check_3m_neg_momentum,
            check_2m_neg_rel_momentum=check_2m_neg_rel_momentum,
            benchmark_ticker=benchmark_ticker,
        )
        price_prefiltered = True
        phase3_pass_count = len(price_pass_data)
        candidate_universe = [d["ticker"] for d in price_pass_data]
        _report_progress(progress_callback, "prices", phase3_pass_count, total)

        if not needs_fundamentals:
            prefilter_rows = [
                _build_result_row({"ticker": d["ticker"]}, price_metrics=price_metrics.get(d["ticker"]))
                for d in price_pass_data
            ]
            results_df = pd.DataFrame(prefilter_rows)
            return {
                "results_df": results_df,
                "failed_tickers": [],
                "phase1_count": total,
                "phase1_pass_count": None,
                "phase3_pass_count": phase3_pass_count,
                "final_count": len(results_df),
            }

    # ------------------------------------------------------------------
    # Phase 1: throttled yfinance fundamentals fetch + P/B + loss filter.
    # ------------------------------------------------------------------
    _report_progress(progress_callback, "fundamentals", 0, len(candidate_universe))
    phase1_pass_data, failed_tickers = _screen_short_fundamentals(
        candidate_universe,
        pb_threshold=pb_threshold,
        loss_type=loss_type,
        check_revenue=check_revenue,
        max_revenue_growth=max_revenue_growth,
        check_eps=check_eps,
        max_eps_growth=max_eps_growth,
        need_market_cap=check_issuance,
        progress_callback=progress_callback,
    )

    phase1_pass_count = len(phase1_pass_data)

    if not phase1_pass_data:
        return {
            "results_df": pd.DataFrame(),
            "failed_tickers": failed_tickers,
            "phase1_count": total,
            "phase1_pass_count": 0,
            **({"phase3_pass_count": phase3_pass_count} if phase3_pass_count is not None else {}),
            "final_count": 0,
        }

    # ------------------------------------------------------------------
    # Phase 2 (optional): Sequential SEC EDGAR issuance check
    # Runs only for Phase 1 passers to minimise SEC API calls.
    # Keeps only stocks in the top quartile of net equity issuance
    # among the Phase 1 passers that have valid SEC data.
    # ------------------------------------------------------------------
    issuance_info: dict[str, dict] = {}  # ticker -> {net, pct}

    if not check_issuance:
        phase2_pass_data = list(phase1_pass_data)
    else:
        _report_progress(progress_callback, "issuance", 0, len(phase1_pass_data))
        phase2_pass_data = []
        issuance_records: list[dict] = []
        for idx, data in enumerate(phase1_pass_data, start=1):
            sec = fetch_sec_issuance(data["ticker"])
            if "error" in sec:
                _report_progress(progress_callback, "issuance", idx, len(phase1_pass_data))
                continue

            net = sec.get("net_issuance", np.nan)
            mktcap = data.get("market_cap", np.nan)

            if (
                (isinstance(net, float) and np.isnan(net))
                or (isinstance(mktcap, float) and np.isnan(mktcap))
                or mktcap <= 0
            ):
                _report_progress(progress_callback, "issuance", idx, len(phase1_pass_data))
                continue

            issuance_records.append({"data": data, "net": net, "pct": net / mktcap})
            _report_progress(progress_callback, "issuance", idx, len(phase1_pass_data))

        if issuance_records:
            net_values = [r["net"] for r in issuance_records]
            cutoff = float(np.percentile(net_values, 75))
            for rec in issuance_records:
                if rec["net"] >= cutoff:
                    phase2_pass_data.append(rec["data"])
                    issuance_info[rec["data"]["ticker"]] = {
                        "net": rec["net"],
                        "pct": rec["pct"],
                    }

    if any_price_filter and not price_prefiltered and phase2_pass_data:
        _report_progress(progress_callback, "prices", 0, len(phase2_pass_data))
        phase2_pass_data, price_metrics = _apply_price_filters(
            phase2_pass_data,
            check_52w_positive=check_52w_positive,
            check_min_drawdown=check_min_drawdown,
            min_drawdown_pct=min_drawdown_pct,
            check_max_drawdown=check_max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            check_3m_neg_momentum=check_3m_neg_momentum,
            check_2m_neg_rel_momentum=check_2m_neg_rel_momentum,
            benchmark_ticker=benchmark_ticker,
        )
        phase3_pass_count = len(phase2_pass_data)
        _report_progress(progress_callback, "prices", phase3_pass_count, len(phase1_pass_data))

    # ------------------------------------------------------------------
    # Build final result rows
    # ------------------------------------------------------------------
    final_rows: list[dict] = []
    for data in phase2_pass_data:
        tk = data["ticker"]
        pm = price_metrics.get(tk) if any_price_filter else None
        row = _build_result_row(data, price_metrics=pm)
        if tk in issuance_info:
            row["Net Issuance ($M)"] = round(issuance_info[tk]["net"] / 1e6, 1)
            row["Issuance % Mkt Cap"] = round(issuance_info[tk]["pct"] * 100, 1)
        final_rows.append(row)

    results_df = pd.DataFrame(final_rows)

    if not results_df.empty:
        results_df = results_df.sort_values("P/B Ratio", ascending=False).reset_index(drop=True)

    result = {
        "results_df": results_df,
        "failed_tickers": failed_tickers,
        "phase1_count": total,
        "phase1_pass_count": phase1_pass_count,
        "final_count": len(results_df),
    }
    if phase3_pass_count is not None:
        result["phase3_pass_count"] = phase3_pass_count
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Short Screen")
    parser.add_argument(
        "universe", nargs="?", default="russell2000", help="Universe: sp500, russell2000, sp400, xlk, etc."
    )
    parser.add_argument("--pb", type=float, default=3.0, help="P/B threshold (default 3.0)")
    parser.add_argument(
        "--loss", choices=["gross", "operating"], default="gross", help="Loss type: gross (default) or operating"
    )
    parser.add_argument(
        "--issuance",
        action="store_true",
        help="Keep only top-quartile net equity issuers among screened stocks (SEC EDGAR)",
    )
    parser.add_argument(
        "--check-revenue", action="store_true", help="Filter by max YoY revenue growth (each of last 3 quarters)"
    )
    parser.add_argument("--max-rev-growth", type=float, default=0.0, help="Max YoY revenue growth %% (default 0)")
    parser.add_argument("--check-eps", action="store_true", help="Filter by max avg YoY EPS growth (last 3 quarters)")
    parser.add_argument("--max-eps-growth", type=float, default=0.0, help="Max avg YoY EPS growth %% (default 0)")
    args = parser.parse_args()

    tickers = load_universe(args.universe)
    if not tickers:
        print(f"ERROR: Failed to load universe '{args.universe}'")
        return

    loss_type = "Gross Loss" if args.loss == "gross" else "Operating Loss"

    def cb(done, total):
        print(f"\rPhase 1: {done}/{total}", end="", flush=True)

    print(
        f"Running short screen: {args.universe} ({len(tickers)} tickers), P/B > {args.pb}, {loss_type}"
        + (", heavy issuance" if args.issuance else "")
    )

    result = get_data(
        tickers=tickers,
        pb_threshold=args.pb,
        loss_type=loss_type,
        check_issuance=args.issuance,
        check_revenue=args.check_revenue,
        max_revenue_growth=args.max_rev_growth,
        check_eps=args.check_eps,
        max_eps_growth=args.max_eps_growth,
        progress_callback=cb,
    )
    print()

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"\nUniverse: {result['phase1_count']} tickers")
    print(f"Phase 1 pass: {result['phase1_pass_count']}")
    if "phase3_pass_count" in result:
        print(f"Phase 3 pass: {result['phase3_pass_count']}")
    print(f"Final candidates: {result['final_count']}")
    print(f"Data errors: {len(result['failed_tickers'])}")

    df = result["results_df"]
    if df.empty:
        print("No candidates found.")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
