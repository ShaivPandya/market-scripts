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

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_universe

# ---------------------------------------------------------------------------
# Module-level SEC caches (survive across Streamlit reruns within a session)
# ---------------------------------------------------------------------------
_cik_map: Dict[str, str] = {}       # ticker.upper() -> zero-padded 10-digit CIK string
_cik_map_loaded: bool = False
_cik_map_lock = threading.Lock()

_edgar_facts_cache: Dict[str, Optional[dict]] = {}   # cik_str -> raw companyfacts JSON or None
_edgar_facts_lock = threading.Lock()

SEC_HEADERS = {"User-Agent": "market-scripts research@example.com"}
SEC_RATE_LIMIT_DELAY = 0.12   # comfortably under SEC's 10 requests/second limit


# ---------------------------------------------------------------------------
# yfinance helpers (pattern from equities/quality/quality_single.py)
# ---------------------------------------------------------------------------

def last_col(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """Return most recent column of a yfinance financial statement."""
    if df is None or df.empty:
        return None
    return df.iloc[:, 0]


def get_item(s: Optional[pd.Series], keys: List[str]) -> float:
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

def fetch_yf_data(ticker: str) -> dict:
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
        try:
            info = t.info or {}
        except Exception:
            info = {}

        fin = t.financials
        inc_last = last_col(fin)

        bal = t.balance_sheet
        bal_last = last_col(bal)

        # P/B from info dict (primary source)
        price_to_book: float = np.nan
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
        try:
            mc_fast = t.fast_info.market_cap
            if mc_fast is not None:
                market_cap = float(mc_fast)
        except Exception:
            pass
        if np.isnan(market_cap):
            mc_raw = info.get("marketCap")
            if mc_raw is not None:
                try:
                    market_cap = float(mc_raw)
                except (TypeError, ValueError):
                    pass

        # Book value from balance sheet
        book_value = get_item(bal_last, [
            "Stockholders Equity",
            "StockholdersEquity",
            "Total Stockholder Equity",
            "TotalStockholderEquity",
            "Common Stock Equity",
        ])

        # P/B fallback: market_cap / book_value when priceToBook is missing
        if np.isnan(price_to_book) and not np.isnan(market_cap) and not np.isnan(book_value) and book_value > 0:
            price_to_book = market_cap / book_value

        # Gross profit (most recent annual)
        gross_profit = get_item(inc_last, ["Gross Profit", "GrossProfit"])

        # Operating income (most recent annual)
        operating_income = get_item(inc_last, [
            "Operating Income",
            "OperatingIncome",
            "EBIT",
            "Ebit",
        ])

        company_name: str = info.get("longName") or info.get("shortName") or ""

        return {
            "ticker": ticker,
            "company_name": company_name,
            "price_to_book": price_to_book,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "market_cap": market_cap,
            "book_value": book_value,
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def screen_ticker(ticker: str, pb_threshold: float, loss_type: str) -> Tuple[bool, dict]:
    """
    Apply Phase 1 criteria (P/B + loss type) using yfinance data.

    Returns:
        (passes: bool, data: dict)
        data["error"] is set if yfinance fetch failed hard.
    """
    data = fetch_yf_data(ticker)

    if "error" in data:
        return False, data

    pb = data.get("price_to_book", np.nan)
    gross = data.get("gross_profit", np.nan)
    operating = data.get("operating_income", np.nan)

    pb_ok = (not (isinstance(pb, float) and np.isnan(pb))) and (pb > pb_threshold)

    if loss_type == "Gross Loss":
        loss_ok = (not (isinstance(gross, float) and np.isnan(gross))) and (gross < 0)
    else:
        loss_ok = (not (isinstance(operating, float) and np.isnan(operating))) and (operating < 0)

    return (pb_ok and loss_ok), data


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
            resp = requests.get(
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
            pass  # _cik_map stays empty; fetch_sec_issuance will return "CIK not found"
        finally:
            _cik_map_loaded = True


def _fetch_edgar_facts(cik_str: str) -> Optional[dict]:
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

    result: Optional[dict] = None
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
        if resp.status_code == 200:
            result = resp.json()
        # 404 → company has no XBRL facts; result stays None
    except Exception:
        pass

    with _edgar_facts_lock:
        _edgar_facts_cache[cik_str] = result

    return result


def _extract_annual_10k(entries: list) -> Optional[float]:
    """
    From a list of XBRL fact entries, return the value from the most recent 10-K
    whose 'end' date falls within the past 18 months.

    18 months is used (rather than 12) because small-caps often have non-December
    fiscal year ends and can file their 10-Ks up to 9 months later.
    """
    cutoff = date.today() - timedelta(days=548)
    best_end: Optional[date] = None
    best_val: Optional[float] = None

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

        proceeds_entries = (
            us_gaap.get("ProceedsFromIssuanceOfCommonStock", {})
                   .get("units", {}).get("USD", [])
        )
        repurchase_entries = (
            us_gaap.get("PaymentsForRepurchaseOfCommonStock", {})
                   .get("units", {}).get("USD", [])
        )

        proceeds = _extract_annual_10k(proceeds_entries)
        repurchases = _extract_annual_10k(repurchase_entries)

        proceeds_f = float(proceeds) if proceeds is not None else np.nan
        repurchases_f = float(repurchases) if repurchases is not None else np.nan

        if not (np.isnan(proceeds_f) and np.isnan(repurchases_f)):
            net = (0.0 if np.isnan(proceeds_f) else proceeds_f) - \
                  (0.0 if np.isnan(repurchases_f) else repurchases_f)
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
# Result row builder
# ---------------------------------------------------------------------------

def _build_result_row(data: dict) -> dict:
    """Convert raw yfinance data dict to a display-ready row (values in $M)."""
    def to_m(val) -> Optional[float]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val) / 1e6, 1)

    def fmt_pb(val) -> Optional[float]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 2)

    return {
        "Ticker": data["ticker"],
        "Company": data.get("company_name") or "",
        "P/B Ratio": fmt_pb(data.get("price_to_book")),
        "Gross Profit ($M)": to_m(data.get("gross_profit")),
        "Operating Income ($M)": to_m(data.get("operating_income")),
        "Market Cap ($M)": to_m(data.get("market_cap")),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_data(
    pb_threshold: float = 3.0,
    loss_type: str = "Gross Loss",
    check_issuance: bool = False,
    progress_callback=None,
) -> dict:
    """
    Run the short screen over the Russell 2000 universe.

    Args:
        pb_threshold:       P/B ratio must exceed this value (3.0 – 5.0)
        loss_type:          "Gross Loss" | "Operating Loss"
        check_issuance:     If True, keep only the top quartile by net equity issuance (SEC EDGAR)
        progress_callback:  Optional callable(done: int, total: int)

    Returns on success:
        {
            "results_df":          pd.DataFrame — one row per candidate, sorted by P/B desc
            "failed_tickers":      List[str]     — tickers that errored in Phase 1
            "phase1_count":        int           — universe size
            "phase1_pass_count":   int           — tickers passing Phase 1
            "final_count":         int           — rows in results_df
        }

    Returns on hard failure:
        {"error": str}
    """
    try:
        universe = load_universe("russell2000")
    except Exception as e:
        return {"error": f"Failed to load Russell 2000 universe: {e}"}

    if not universe:
        return {"error": "Russell 2000 universe is empty"}

    total = len(universe)

    # ------------------------------------------------------------------
    # Pre-warm yfinance session so the authentication crumb is fresh
    # before spawning 8 parallel threads.  A stale/missing crumb causes
    # t.info to silently return {} for every ticker, producing 0 results.
    # ------------------------------------------------------------------
    try:
        yf.Ticker(universe[0]).fast_info.last_price
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Phase 1: Parallel yfinance fetch + P/B + loss filter
    # ------------------------------------------------------------------
    phase1_pass_data: List[dict] = []
    failed_tickers: List[str] = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(screen_ticker, tk, pb_threshold, loss_type): tk
            for tk in universe
        }
        for future in as_completed(futures):
            tk = futures[future]
            try:
                passes, data = future.result()
                if passes:
                    phase1_pass_data.append(data)
                elif "error" in data:
                    failed_tickers.append(tk)
            except Exception:
                failed_tickers.append(tk)

            done_count += 1
            if progress_callback and (done_count % 25 == 0 or done_count == total):
                progress_callback(done_count, total)

    phase1_pass_count = len(phase1_pass_data)

    if not phase1_pass_data:
        return {
            "results_df": pd.DataFrame(),
            "failed_tickers": failed_tickers,
            "phase1_count": total,
            "phase1_pass_count": 0,
            "final_count": 0,
        }

    # ------------------------------------------------------------------
    # Phase 2 (optional): Sequential SEC EDGAR issuance check
    # Runs only for Phase 1 passers to minimise SEC API calls.
    # Keeps only stocks in the top quartile of net equity issuance
    # among the Phase 1 passers that have valid SEC data.
    # ------------------------------------------------------------------
    final_rows: List[dict] = []

    if not check_issuance:
        for data in phase1_pass_data:
            final_rows.append(_build_result_row(data))
    else:
        # Step 1: fetch issuance data for all Phase 1 passers
        issuance_records: List[dict] = []
        for data in phase1_pass_data:
            sec = fetch_sec_issuance(data["ticker"])
            if "error" in sec:
                continue  # SEC data unavailable — exclude (conservative)

            net = sec.get("net_issuance", np.nan)
            mktcap = data.get("market_cap", np.nan)

            if (
                isinstance(net, float) and np.isnan(net)
            ) or (
                isinstance(mktcap, float) and np.isnan(mktcap)
            ) or mktcap <= 0:
                continue  # Cannot compute issuance; exclude

            issuance_records.append({
                "data": data,
                "net": net,
                "pct": net / mktcap,
            })

        if issuance_records:
            # Step 2: top-quartile cutoff (75th percentile of net issuance)
            net_values = [r["net"] for r in issuance_records]
            cutoff = float(np.percentile(net_values, 75))

            # Step 3: keep only stocks at or above the cutoff
            for rec in issuance_records:
                if rec["net"] >= cutoff:
                    row = _build_result_row(rec["data"])
                    row["Net Issuance ($M)"] = round(rec["net"] / 1e6, 1)
                    row["Issuance % Mkt Cap"] = round(rec["pct"] * 100, 1)
                    final_rows.append(row)

    results_df = pd.DataFrame(final_rows)

    if not results_df.empty:
        results_df = results_df.sort_values("P/B Ratio", ascending=False).reset_index(drop=True)

    return {
        "results_df": results_df,
        "failed_tickers": failed_tickers,
        "phase1_count": total,
        "phase1_pass_count": phase1_pass_count,
        "final_count": len(results_df),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Short Screen — Russell 2000")
    parser.add_argument("--pb", type=float, default=3.0, help="P/B threshold (default 3.0)")
    parser.add_argument("--loss", choices=["gross", "operating"], default="gross",
                        help="Loss type: gross (default) or operating")
    parser.add_argument("--issuance", action="store_true",
                        help="Keep only top-quartile net equity issuers among screened stocks (SEC EDGAR)")
    args = parser.parse_args()

    loss_type = "Gross Loss" if args.loss == "gross" else "Operating Loss"

    def cb(done, total):
        print(f"\rPhase 1: {done}/{total}", end="", flush=True)

    print(f"Running short screen: P/B > {args.pb}, {loss_type}"
          + (", heavy issuance" if args.issuance else ""))

    result = get_data(
        pb_threshold=args.pb,
        loss_type=loss_type,
        check_issuance=args.issuance,
        progress_callback=cb,
    )
    print()

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"\nUniverse: {result['phase1_count']} tickers")
    print(f"Phase 1 pass: {result['phase1_pass_count']}")
    print(f"Final candidates: {result['final_count']}")
    print(f"Data errors: {len(result['failed_tickers'])}")

    df = result["results_df"]
    if df.empty:
        print("No candidates found.")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
