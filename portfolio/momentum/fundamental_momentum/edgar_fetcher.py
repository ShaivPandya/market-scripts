"""
SEC EDGAR XBRL fetcher for financial data.

Provides thread-safe, rate-limited access to the SEC EDGAR companyfacts API.
Used for quarterly EPS/revenue momentum, DCF valuation historicals, and other
fundamental analysis requiring deep historical financial data.

Rate limiting: all EDGAR requests are serialised through a global lock with a
0.11 s delay, keeping throughput at ≤ 9 req/s across all threads (SEC limit: 10).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date
from typing import Dict, List, Optional, Tuple  # noqa: UP035

import requests  # type: ignore[import-untyped]

from portfolio.momentum.fundamental_momentum._edgar_periods import (
    _annual_fact_entries,
    _as_float,
    _entries_for,
    _quarterly_average_entries,
    _quarterly_direct_entries,
    _quarterly_flow_entries,
)
from utils.retry import requests_get

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (process lifetime)
# ---------------------------------------------------------------------------
_cik_map: dict[str, str] = {}  # ticker.upper() -> zero-padded 10-digit CIK
_cik_map_loaded: bool = False
_cik_map_lock = threading.Lock()

_edgar_facts_cache: dict[str, dict | None] = {}  # cik_str -> companyfacts JSON or None
_edgar_facts_lock = threading.Lock()

_edgar_submissions_cache: dict[str, dict | None] = {}  # cik_str -> submissions JSON or None
_edgar_submissions_lock = threading.Lock()

# Serialise all HTTP requests through one lock so concurrent threads cannot
# collectively exceed the SEC's 10 req/s rate limit.
_edgar_request_lock = threading.Lock()

SEC_HEADERS = {"User-Agent": "market-scripts research@example.com"}
_SEC_DELAY = 0.11  # seconds between requests; keeps throughput ≤ 9/s


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rate_limited_get(url: str, timeout: int = 20) -> requests.Response | None:
    """Make a GET request, serialised and rate-limited across all threads."""
    with _edgar_request_lock:
        time.sleep(_SEC_DELAY)
        try:
            resp = requests_get(url, headers=SEC_HEADERS, timeout=timeout)
            return resp
        except Exception:
            return None


def _load_cik_map() -> None:
    """Download SEC's company-ticker-CIK mapping once per process lifetime."""
    global _cik_map_loaded
    with _cik_map_lock:
        if _cik_map_loaded:
            return
        try:
            resp = _rate_limited_get("https://www.sec.gov/files/company_tickers.json", timeout=30)
            if resp is not None and resp.status_code == 200:
                for entry in resp.json().values():
                    tk = str(entry.get("ticker", "")).upper()
                    cik_int = entry.get("cik_str", 0)
                    if tk and cik_int:
                        _cik_map[tk] = f"{int(cik_int):010d}"
        except Exception:
            pass
        finally:
            _cik_map_loaded = True


def _fetch_edgar_facts(cik_str: str) -> dict | None:
    """
    Fetch XBRL companyfacts JSON from SEC EDGAR for one CIK.
    Results are cached for the process lifetime.
    """
    with _edgar_facts_lock:
        if cik_str in _edgar_facts_cache:
            return _edgar_facts_cache[cik_str]

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
    resp = _rate_limited_get(url)
    result: dict | None = None
    if resp is not None and resp.status_code == 200:
        try:
            result = resp.json()
        except Exception:
            pass

    with _edgar_facts_lock:
        _edgar_facts_cache[cik_str] = result

    return result


def _fetch_edgar_submissions(cik_str: str) -> dict | None:
    """
    Fetch SEC submissions JSON for one CIK.
    Results are cached for the process lifetime.
    """
    with _edgar_submissions_lock:
        if cik_str in _edgar_submissions_cache:
            return _edgar_submissions_cache[cik_str]

    url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
    resp = _rate_limited_get(url)
    result: dict | None = None
    if resp is not None and resp.status_code == 200:
        try:
            result = resp.json()
        except Exception:
            pass

    with _edgar_submissions_lock:
        _edgar_submissions_cache[cik_str] = result

    return result


def _quarterly_entries_from_concept(us_gaap: dict, concept: str, unit: str) -> list[dict]:
    """
    Return normalized discrete quarterly flow entries for a GAAP concept/unit.

    SEC companyfacts often includes YTD income-statement facts for Q2/Q3.
    This normalizes those rows into quarter-specific values before returning.
    """
    return _quarterly_flow_entries(_entries_for(us_gaap, concept, unit))


def _annual_entries_from_concept(us_gaap: dict, concept: str, unit: str) -> list[dict]:
    """
    Return normalized annual FY fact entries for a given GAAP concept and unit.
    """
    return _annual_fact_entries(_entries_for(us_gaap, concept, unit))


def _instant_entries_from_concept(us_gaap: dict, concept: str, unit: str) -> list[dict]:
    """
    Return point-in-time (instant) fact entries for balance sheet concepts.

    Balance sheet items don't have start/end duration — they are snapshots at
    a point in time.  We keep entries from Q1-Q4 and FY filings,
    deduplicated by period-end date (keeping most recently filed).
    """
    try:
        entries = us_gaap[concept]["units"][unit]
    except (KeyError, TypeError):
        return []

    by_end: dict[str, dict] = {}
    for e in entries:
        fp = e.get("fp", "")
        if fp not in {"Q1", "Q2", "Q3", "Q4", "FY"}:
            continue
        end = e.get("end", "")
        if not end:
            continue
        filed = e.get("filed", "")
        existing = by_end.get(end)
        if existing is None or filed > existing.get("filed", ""):
            by_end[end] = e

    return list(by_end.values())


def _extract_concept(
    us_gaap: dict,
    concepts: tuple[str, ...],
    unit: str,
    mode: str,
    n: int,
) -> list[tuple[date, float]]:
    """
    Generic extraction: try multiple XBRL concept names, return (date, value)
    pairs sorted newest-first.

    mode: "quarterly", "annual", or "instant"
    """
    extractor = {
        "quarterly": _quarterly_entries_from_concept,
        "annual": _annual_entries_from_concept,
        "instant": _instant_entries_from_concept,
    }[mode]

    for concept in concepts:
        entries = extractor(us_gaap, concept, unit)
        if entries:
            result = sorted(
                [(date.fromisoformat(e["end"]), float(e["val"])) for e in entries],
                key=lambda x: x[0],
                reverse=True,
            )
            return result[:n]
    return []


# ---------------------------------------------------------------------------
# XBRL concept registries for DCF-relevant metrics
# ---------------------------------------------------------------------------

REVENUE_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)

OPERATING_INCOME_CONCEPTS = (
    "OperatingIncomeLoss",
    "IncomeLossFromOperations",
)

DA_CONCEPTS = (
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "Depreciation",
)

CAPEX_CONCEPTS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)

CURRENT_ASSETS_CONCEPTS = ("AssetsCurrent",)

CURRENT_LIABILITIES_CONCEPTS = ("LiabilitiesCurrent",)

TOTAL_DEBT_CONCEPTS = (
    "LongTermDebt",
    "LongTermDebtNoncurrent",
)

CURRENT_DEBT_CONCEPTS = (
    "DebtCurrent",
    "ShortTermBorrowings",
    "LongTermDebtCurrent",
)

CASH_CONCEPTS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
    "Cash",
)

INTEREST_EXPENSE_CONCEPTS = (
    "InterestExpense",
    "InterestExpenseDebt",
    "InterestPaid",
)

PRETAX_INCOME_CONCEPTS = (
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
)

TAX_EXPENSE_CONCEPTS = ("IncomeTaxExpenseBenefit",)

SHARES_OUTSTANDING_CONCEPTS = (
    "EntityCommonStockSharesOutstanding",
    "CommonStockSharesOutstanding",
    "CommonStockSharesIssued",
)

EBITDA_CONCEPTS = ("EarningsBeforeInterestTaxesDepreciationAndAmortization",)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_cik_for_ticker(ticker: str) -> str | None:
    """Return zero-padded 10-digit CIK for a ticker, or None if unavailable."""
    _load_cik_map()
    return _cik_map.get(ticker.upper().strip())


def fetch_companyfacts_by_cik(cik_str: str) -> dict | None:
    """Fetch SEC companyfacts payload for a 10-digit CIK string."""
    cik = str(cik_str).strip()
    if not cik:
        return None
    return _fetch_edgar_facts(cik.zfill(10))


def fetch_companyfacts_by_ticker(ticker: str) -> dict | None:
    """Fetch SEC companyfacts payload for a ticker symbol."""
    cik_str = get_cik_for_ticker(ticker)
    if not cik_str:
        return None
    return _fetch_edgar_facts(cik_str)


def fetch_submissions_by_cik(cik_str: str) -> dict | None:
    """Fetch SEC submissions payload for a 10-digit CIK string."""
    cik = str(cik_str).strip()
    if not cik:
        return None
    return _fetch_edgar_submissions(cik.zfill(10))


def fetch_submissions_by_ticker(ticker: str) -> dict | None:
    """Fetch SEC submissions payload for a ticker symbol."""
    cik_str = get_cik_for_ticker(ticker)
    if not cik_str:
        return None
    return _fetch_edgar_submissions(cik_str)


def build_filing_url(
    cik_str: str,
    accession: str,
    submissions: dict | None = None,
) -> str:
    """
    Build a SEC filing URL for an accession.

    Uses submissions metadata to build a direct primary-document URL when
    available; falls back to a browse-edgar URL filtered by accession.
    """
    cik_digits = str(cik_str).strip().zfill(10)
    cik_int = str(int(cik_digits))
    accn = str(accession or "").strip()
    if not accn:
        return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_int}&owner=exclude&count=40"

    primary_doc = ""
    src = submissions or {}
    recent = src.get("filings", {}).get("recent", {}) if isinstance(src, dict) else {}
    accns = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    if isinstance(accns, list) and isinstance(docs, list):
        for i, a in enumerate(accns):
            if str(a) == accn and i < len(docs):
                primary_doc = str(docs[i] or "").strip()
                break

    accn_nodash = accn.replace("-", "")
    if primary_doc:
        return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/{primary_doc}"

    return (
        f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_int}&accno={accn}&owner=exclude&count=40"
    )


def extract_quarterly_eps(facts: dict, n: int = 8) -> list[tuple[date, float]]:
    """
    Extract up to n quarterly EPS values from companyfacts JSON.

    Tries, in order:
      1. EarningsPerShareDiluted  (USD/shares)
      2. EarningsPerShareBasic    (USD/shares)
      3. Derived: NetIncomeLoss / WeightedAverageNumberOfDilutedSharesOutstanding
      4. Derived: NetIncomeLoss / WeightedAverageNumberOfSharesOutstandingBasic

    Returns a list of (period_end_date, eps_value) sorted newest-first, length ≤ n.
    Returns an empty list if no usable data is found.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    # Try direct EPS concepts first
    direct_result: list[tuple[date, float]] = []
    for concept in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
        for unit in ("USD/shares", "USD-per-shares"):
            entries = _quarterly_direct_entries(_entries_for(us_gaap, concept, unit))
            if entries:
                direct_result = sorted(
                    [(date.fromisoformat(e["end"]), float(e["val"])) for e in entries],
                    key=lambda x: x[0],
                    reverse=True,
                )
                break
        if direct_result:
            break

    # Derived: normalized quarterly NetIncomeLoss / normalized quarterly average shares
    ni_entries = _quarterly_flow_entries(_entries_for(us_gaap, "NetIncomeLoss", "USD"))
    if not ni_entries:
        return direct_result[:n]

    ni_by_end: dict[str, float] = {}
    for e in ni_entries:
        val = _as_float(e.get("val"))
        end = str(e.get("end") or "")
        if end and val is not None:
            ni_by_end[end] = val

    for shares_concept in (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ):
        sh_entries = _quarterly_average_entries(_entries_for(us_gaap, shares_concept, "shares"))
        if not sh_entries:
            continue
        sh_by_end: dict[str, float] = {}
        for e in sh_entries:
            val = _as_float(e.get("val"))
            end = str(e.get("end") or "")
            if end and val is not None:
                sh_by_end[end] = val

        common_ends = sorted(
            (e for e in ni_by_end if e in sh_by_end),
            reverse=True,
        )
        if not common_ends:
            continue

        derived_result = []
        for end in common_ends:
            shares = sh_by_end[end]
            if shares and shares != 0:
                derived_result.append((date.fromisoformat(end), ni_by_end[end] / shares))
        if derived_result:
            if not direct_result:
                return derived_result[:n]

            seen = {end for end, _ in direct_result}
            merged = direct_result + [(end, value) for end, value in derived_result if end not in seen]
            return sorted(merged, key=lambda x: x[0], reverse=True)[:n]

    return direct_result[:n]


def extract_quarterly_revenue(facts: dict, n: int = 8) -> list[tuple[date, float]]:
    """
    Extract up to n quarterly revenue values from companyfacts JSON.

    Tries concepts in order:
      1. Revenues
      2. RevenueFromContractWithCustomerExcludingAssessedTax
      3. SalesRevenueNet
      4. SalesRevenueGoodsNet
      5. RevenueFromContractWithCustomerIncludingAssessedTax

    Returns a list of (period_end_date, revenue_value) sorted newest-first, length ≤ n.
    Returns an empty list if no usable data is found.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for concept in (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ):
        entries = _quarterly_flow_entries(_entries_for(us_gaap, concept, "USD"))
        if entries:
            result = sorted(
                [(date.fromisoformat(e["end"]), float(e["val"])) for e in entries],
                key=lambda x: x[0],
                reverse=True,
            )
            return result[:n]

    return []


def fetch_quarterly_eps_edgar(ticker: str, n: int = 8) -> list[tuple[date, float]] | None:
    """
    Fetch quarterly EPS data from SEC EDGAR for one ticker.

    Returns a list of (period_end_date, eps) sorted newest-first (length ≤ n),
    or None if the ticker cannot be found in EDGAR or has no EPS XBRL data.
    """
    _load_cik_map()
    cik_str = _cik_map.get(ticker.upper())
    if not cik_str:
        return None

    facts = _fetch_edgar_facts(cik_str)
    if facts is None:
        return None

    result = extract_quarterly_eps(facts, n=n)
    return result if result else None


def fetch_quarterly_revenue_edgar(ticker: str, n: int = 8) -> list[tuple[date, float]] | None:
    """
    Fetch quarterly revenue data from SEC EDGAR for one ticker.

    Returns a list of (period_end_date, revenue) sorted newest-first (length ≤ n),
    or None if the ticker cannot be found in EDGAR or has no revenue XBRL data.
    """
    _load_cik_map()
    cik_str = _cik_map.get(ticker.upper())
    if not cik_str:
        return None

    facts = _fetch_edgar_facts(cik_str)
    if facts is None:
        return None

    result = extract_quarterly_revenue(facts, n=n)
    return result if result else None


# ---------------------------------------------------------------------------
# DCF-oriented extraction functions
# ---------------------------------------------------------------------------


def extract_dcf_historicals(ticker: str) -> dict | None:
    """
    Extract all DCF-relevant historical data from SEC EDGAR for a ticker.

    Returns a dict with annual and quarterly data, or None if unavailable.
    Each value is a list of (date, float) tuples sorted newest-first.
    """
    facts = fetch_companyfacts_by_ticker(ticker)
    if facts is None:
        return None

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return None

    def _annual(concepts: tuple[str, ...], n: int = 10) -> list[tuple[date, float]]:
        return _extract_concept(us_gaap, concepts, "USD", "annual", n)

    def _quarterly(concepts: tuple[str, ...], n: int = 24) -> list[tuple[date, float]]:
        return _extract_concept(us_gaap, concepts, "USD", "quarterly", n)

    def _instant_q(concepts: tuple[str, ...], n: int = 24) -> list[tuple[date, float]]:
        return _extract_concept(us_gaap, concepts, "USD", "instant", n)

    def _instant_shares(concepts: tuple[str, ...], n: int = 24) -> list[tuple[date, float]]:
        return _extract_concept(us_gaap, concepts, "shares", "instant", n)

    def _instant_annual(concepts: tuple[str, ...], n: int = 10) -> list[tuple[date, float]]:
        """Extract FY-only instant (balance sheet) values."""
        for concept in concepts:
            try:
                entries = us_gaap[concept]["units"]["USD"]
            except (KeyError, TypeError):
                continue
            annual: dict[str, dict] = {}
            for e in entries:
                if e.get("fp") != "FY":
                    continue
                end = e.get("end", "")
                if not end:
                    continue
                filed = e.get("filed", "")
                existing = annual.get(end)
                if existing is None or filed > existing.get("filed", ""):
                    annual[end] = e
            if annual:
                result = sorted(
                    [(date.fromisoformat(e["end"]), float(e["val"])) for e in annual.values()],
                    key=lambda x: x[0],
                    reverse=True,
                )
                return result[:n]
        return []

    return {
        # Annual income statement / cash flow items
        "annual_revenue": _annual(REVENUE_CONCEPTS),
        "annual_operating_income": _annual(OPERATING_INCOME_CONCEPTS),
        "annual_ebitda": _annual(EBITDA_CONCEPTS),
        "annual_da": _annual(DA_CONCEPTS),
        "annual_capex": _annual(CAPEX_CONCEPTS),
        "annual_interest_expense": _annual(INTEREST_EXPENSE_CONCEPTS),
        "annual_pretax_income": _annual(PRETAX_INCOME_CONCEPTS),
        "annual_tax_expense": _annual(TAX_EXPENSE_CONCEPTS),
        # Annual balance sheet items (instant/point-in-time, FY only)
        "annual_current_assets": _instant_annual(CURRENT_ASSETS_CONCEPTS),
        "annual_current_liabilities": _instant_annual(CURRENT_LIABILITIES_CONCEPTS),
        # Quarterly income statement items
        "quarterly_revenue": _quarterly(REVENUE_CONCEPTS),
        "quarterly_operating_income": _quarterly(OPERATING_INCOME_CONCEPTS),
        "quarterly_ebitda": _quarterly(EBITDA_CONCEPTS),
        "quarterly_da": _quarterly(DA_CONCEPTS),
        # Quarterly balance sheet items (instant/point-in-time)
        "quarterly_current_assets": _instant_q(CURRENT_ASSETS_CONCEPTS),
        "quarterly_current_liabilities": _instant_q(CURRENT_LIABILITIES_CONCEPTS),
        "quarterly_total_debt": _instant_q(TOTAL_DEBT_CONCEPTS),
        "quarterly_current_debt": _instant_q(CURRENT_DEBT_CONCEPTS),
        "quarterly_cash": _instant_q(CASH_CONCEPTS),
        "quarterly_shares_outstanding": _instant_shares(SHARES_OUTSTANDING_CONCEPTS),
    }
