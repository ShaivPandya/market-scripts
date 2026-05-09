#!/usr/bin/env python3
"""
Single-company financials from SEC EDGAR companyfacts.

Outputs annual and quarterly Revenue/EPS history with filing links, plus key growth
metrics and latest-filing segment/region revenue breakdown.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Set, SupportsFloat, SupportsIndex, Tuple, TypedDict  # noqa: UP035

from llm_utils import MODEL_LOW, MODEL_MID, call_llm_text, has_llm_api_key, parse_json_text
from portfolio.momentum.fundamental_momentum._edgar_periods import (
    ALLOWED_ANNUAL_FORMS,
    ALLOWED_QUARTERLY_FORMS,
    QUARTER_FPS,
    _annual_fact_entries,
    _as_float,
    _entries_for,
    _is_valid_fact_row,
    _keep_latest_by,
    _parse_iso_date,
    _pick_best_concept_entries,
    _quarterly_average_entries,
    _quarterly_direct_entries,
    _safe_growth,
    _sort_newest,
)
from portfolio.momentum.fundamental_momentum._edgar_periods import (
    _quarterly_flow_entries as _quarterly_fact_entries,
)
from portfolio.momentum.fundamental_momentum.edgar_fetcher import (
    INTEREST_EXPENSE_CONCEPTS as EDGAR_INTEREST_EXPENSE_CONCEPTS,
)
from portfolio.momentum.fundamental_momentum.edgar_fetcher import (
    build_filing_url,
    fetch_companyfacts_by_cik,
    fetch_submissions_by_cik,
    get_cik_for_ticker,
)
from utils.retry import requests_get

LOGGER = logging.getLogger(__name__)

REVENUE_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)

EPS_CONCEPTS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")
EPS_UNITS = ("USD/shares", "USD-per-shares")
OPERATING_INCOME_CONCEPTS = ("OperatingIncomeLoss",)
NET_INCOME_CONCEPTS = ("NetIncomeLoss",)
INTEREST_EXPENSE_RATIO_CONCEPTS = (*EDGAR_INTEREST_EXPENSE_CONCEPTS, "InterestAndDebtExpense")
INTEREST_COVERAGE_WARNING_THRESHOLD = 4.0

YF_REVENUE_KEYS = (
    "Total Revenue",
    "TotalRevenue",
    "Revenue",
    "Net Sales",
    "NetSales",
    "Sales",
)
YF_EPS_KEYS = (
    "Diluted EPS",
    "DilutedEPS",
    "EPS Diluted",
    "Earnings Per Share Diluted",
    "Basic EPS",
    "BasicEPS",
    "EPS Basic",
    "Earnings Per Share Basic",
)
YF_OPERATING_INCOME_KEYS = (
    "Operating Income",
    "OperatingIncome",
    "Operating Income or Loss",
    "OperatingIncomeLoss",
)
YF_NET_INCOME_KEYS = (
    "Net Income",
    "NetIncome",
    "Net Income Common Stockholders",
    "NetIncomeCommonStockholders",
    "Net Income Applicable To Common Shares",
    "NetIncomeApplicableToCommonShares",
)
YF_INTEREST_EXPENSE_KEYS = (
    "Interest Expense",
    "InterestExpense",
    "Interest Expense Debt",
    "InterestExpenseDebt",
    "Interest Paid",
    "InterestPaid",
    "Interest And Debt Expense",
    "InterestAndDebtExpense",
)
YF_DILUTED_SHARES_KEYS = (
    "Diluted Average Shares",
    "DilutedAverageShares",
    "Weighted Average Shares Diluted",
    "WeightedAverageSharesDiluted",
    "Diluted Shares",
    "DilutedShares",
)
YF_BASIC_SHARES_KEYS = (
    "Basic Average Shares",
    "BasicAverageShares",
    "Weighted Average Shares Basic",
    "WeightedAverageSharesBasic",
    "Basic Shares",
    "BasicShares",
)

ANNUAL_DISPLAY_LIMIT = 5
QUARTERLY_DISPLAY_LIMIT = 20
ANNUAL_YOY_STEP = 1
QUARTERLY_YOY_STEP = 4


class _RatioResult(TypedDict):
    value: float | None
    basis: str | None
    period_end: str | None


def _period_label(e: dict, frequency: str) -> str:
    end = str(e.get("end") or "")
    fy = e.get("fy")
    fp = str(e.get("fp") or "")

    if frequency == "annual":
        if fy is not None:
            return f"FY{fy}"
        if len(end) >= 4:
            return f"FY{end[:4]}"
        return "FY"

    if fp in QUARTER_FPS and fy is not None:
        return f"{fp} {fy}"

    d = _parse_iso_date(end)
    if d is not None:
        q = ((d.month - 1) // 3) + 1
        return f"Q{q} {d.year}"
    return fp or "Quarter"


def _rows_from_entries(
    entries: list[dict],
    *,
    frequency: str,
    limit: int,
    cik_str: str,
    submissions: dict | None,
    yoy_step: int,
    yoy_abs_denom: bool,
) -> list[dict]:
    rows: list[dict] = []
    for e in entries[:limit]:
        val = _as_float(e.get("val"))
        if val is None:
            continue
        accn = str(e.get("accn") or "")
        rows.append(
            {
                "period_label": _period_label(e, frequency),
                "period_end": str(e.get("end") or ""),
                "value": val,
                "yoy_growth": None,
                "form": str(e.get("form") or ""),
                "filed": str(e.get("filed") or ""),
                "accn": accn,
                "filing_url": build_filing_url(cik_str, accn, submissions=submissions),
            }
        )

    for i, row in enumerate(rows):
        j = i + yoy_step
        if j >= len(rows):
            continue
        curr = _as_float(row.get("value"))
        prev = _as_float(rows[j].get("value"))
        if curr is None or prev is None:
            continue
        row["yoy_growth"] = _safe_growth(curr - prev, prev, denom_abs=yoy_abs_denom)

    return rows


def _build_revenue_rows(us_gaap: dict, cik_str: str, submissions: dict | None) -> tuple[list[dict], list[dict]]:
    annual_entries = _pick_best_concept_entries(
        us_gaap,
        REVENUE_CONCEPTS,
        "USD",
        _annual_fact_entries,
    )
    quarterly_entries = _pick_best_concept_entries(
        us_gaap,
        REVENUE_CONCEPTS,
        "USD",
        _quarterly_fact_entries,
    )

    annual_rows_full = _rows_from_entries(
        annual_entries,
        frequency="annual",
        # Pull one extra year so oldest displayed row still has YoY.
        limit=ANNUAL_DISPLAY_LIMIT + ANNUAL_YOY_STEP,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=ANNUAL_YOY_STEP,
        yoy_abs_denom=False,
    )
    quarterly_rows_full = _rows_from_entries(
        quarterly_entries,
        frequency="quarterly",
        # Pull four extra quarters so oldest displayed row still has YoY.
        limit=QUARTERLY_DISPLAY_LIMIT + QUARTERLY_YOY_STEP,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=QUARTERLY_YOY_STEP,
        yoy_abs_denom=False,
    )
    return annual_rows_full[:ANNUAL_DISPLAY_LIMIT], quarterly_rows_full[:QUARTERLY_DISPLAY_LIMIT]


def _pick_first_concept_entries(
    us_gaap: dict,
    concepts: tuple[str, ...],
    extractor: Callable[[list[dict]], list[dict]],
) -> list[dict]:
    for concept in concepts:
        raw = _entries_for(us_gaap, concept, "USD")
        if not raw:
            continue
        candidate = extractor(raw)
        if candidate:
            return candidate
    return []


def _build_flow_rows(
    us_gaap: dict,
    concepts: tuple[str, ...],
    cik_str: str,
    submissions: dict | None,
    *,
    preserve_concept_order: bool = False,
) -> tuple[list[dict], list[dict]]:
    if preserve_concept_order:
        annual_entries = _pick_first_concept_entries(us_gaap, concepts, _annual_fact_entries)
        quarterly_entries = _pick_first_concept_entries(us_gaap, concepts, _quarterly_fact_entries)
    else:
        annual_entries = _pick_best_concept_entries(us_gaap, concepts, "USD", _annual_fact_entries)
        quarterly_entries = _pick_best_concept_entries(us_gaap, concepts, "USD", _quarterly_fact_entries)

    annual_rows = _rows_from_entries(
        annual_entries,
        frequency="annual",
        limit=ANNUAL_DISPLAY_LIMIT,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=ANNUAL_YOY_STEP,
        yoy_abs_denom=False,
    )
    quarterly_rows = _rows_from_entries(
        quarterly_entries,
        frequency="quarterly",
        limit=QUARTERLY_DISPLAY_LIMIT,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=QUARTERLY_YOY_STEP,
        yoy_abs_denom=False,
    )
    return annual_rows, quarterly_rows


def _derived_eps_entries(us_gaap: dict, frequency: str) -> list[dict]:
    period_key = lambda e: f"END:{e.get('end') or ''}"  # noqa: E731

    ni_raw = _entries_for(us_gaap, "NetIncomeLoss", "USD")
    if frequency == "annual":
        ni = _annual_fact_entries(ni_raw)
    else:
        ni = _quarterly_fact_entries(ni_raw)

    shares = []
    for concept in (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ):
        sh_raw = _entries_for(us_gaap, concept, "shares")
        if frequency == "annual":
            sh = _annual_fact_entries(sh_raw)
        else:
            sh = _quarterly_average_entries(sh_raw)
        if sh:
            shares = sh
            break

    if not ni or not shares:
        return []

    shares_by_period = {period_key(e): e for e in _keep_latest_by(shares, period_key)}

    derived: list[dict] = []
    for ni_row in _keep_latest_by(ni, period_key):
        key = period_key(ni_row)
        sh_row = shares_by_period.get(key)
        if not sh_row:
            continue
        ni_val = _as_float(ni_row.get("val"))
        sh_val = _as_float(sh_row.get("val"))
        eps = _safe_growth(ni_val, sh_val, denom_abs=False)
        if eps is None:
            continue
        clone = dict(ni_row)
        clone["val"] = eps
        derived.append(clone)

    return _sort_newest(derived)


def _fill_missing_period_entries(primary: list[dict], fallback: list[dict]) -> list[dict]:
    if not primary:
        return fallback

    seen_ends = {str(e.get("end") or "") for e in primary if e.get("end")}
    merged = list(primary)
    for e in fallback:
        end = str(e.get("end") or "")
        if not end or end in seen_ends:
            continue
        merged.append(e)
        seen_ends.add(end)

    return _sort_newest(merged)


def _build_eps_rows(us_gaap: dict, cik_str: str, submissions: dict | None) -> tuple[list[dict], list[dict]]:
    annual_entries: list[dict] = []
    quarterly_entries: list[dict] = []

    for concept in EPS_CONCEPTS:
        for unit in EPS_UNITS:
            raw = _entries_for(us_gaap, concept, unit)
            if not raw:
                continue
            a = _annual_fact_entries(raw)
            q = _quarterly_direct_entries(raw)
            if a and not annual_entries:
                annual_entries = a
            if q and not quarterly_entries:
                quarterly_entries = q
            if annual_entries and quarterly_entries:
                break
        if annual_entries and quarterly_entries:
            break

    derived_annual_entries = _derived_eps_entries(us_gaap, "annual")
    derived_quarterly_entries = _derived_eps_entries(us_gaap, "quarterly")

    annual_entries = _fill_missing_period_entries(annual_entries, derived_annual_entries)
    quarterly_entries = _fill_missing_period_entries(quarterly_entries, derived_quarterly_entries)

    annual_rows_full = _rows_from_entries(
        annual_entries,
        frequency="annual",
        limit=ANNUAL_DISPLAY_LIMIT + ANNUAL_YOY_STEP,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=ANNUAL_YOY_STEP,
        yoy_abs_denom=True,
    )
    quarterly_rows_full = _rows_from_entries(
        quarterly_entries,
        frequency="quarterly",
        limit=QUARTERLY_DISPLAY_LIMIT + QUARTERLY_YOY_STEP,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=QUARTERLY_YOY_STEP,
        yoy_abs_denom=True,
    )
    return annual_rows_full[:ANNUAL_DISPLAY_LIMIT], quarterly_rows_full[:QUARTERLY_DISPLAY_LIMIT]


def _calc_cagr(rows: list[dict], years: int = 3, *, abs_fallback: bool = False) -> float | None:
    values = [_as_float(r.get("value")) for r in rows]
    clean: list[float] = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    n = min(years, len(clean) - 1)
    if n < 1:
        return None
    latest = clean[0]
    prior = clean[n]
    if latest > 0 and prior > 0:
        return float((latest / prior) ** (1.0 / n) - 1.0)

    # EPS can cross zero (loss to profit), where strict CAGR is undefined.
    # Fallback to CAGR on absolute magnitude so the card remains informative.
    if not abs_fallback or latest == 0 or prior == 0:
        return None
    return float((abs(latest) / abs(prior)) ** (1.0 / n) - 1.0)


def _calc_avg_3q_yoy(rows: list[dict], denom_abs: bool) -> float | None:
    values = [_as_float(r.get("value")) for r in rows]
    if len(values) < 7:
        return None
    changes: list[float] = []
    for i in (0, 1, 2):
        a = values[i]
        b = values[i + 4]
        if a is None or b is None:
            continue
        ch = _safe_growth(a - b, b, denom_abs=denom_abs)
        if ch is not None:
            changes.append(ch)
    if not changes:
        return None
    return sum(changes) / len(changes)


def _values_by_period(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        period_end = str(row.get("period_end") or "")
        value = _as_float(row.get("value"))
        if period_end and value is not None:
            out[period_end] = value
    return out


def _empty_ratio_result() -> _RatioResult:
    return {"value": None, "basis": None, "period_end": None}


def _ratio_value(numerator: float | None, denominator: float | None, *, denominator_abs: bool = False) -> float | None:
    if numerator is None or denominator is None:
        return None
    adjusted_denominator = abs(denominator) if denominator_abs else denominator
    if adjusted_denominator == 0:
        return None
    return numerator / adjusted_denominator


def _sum_periods(values: dict[str, float], periods: list[str]) -> float | None:
    total = 0.0
    for period in periods:
        value = values.get(period)
        if value is None:
            return None
        total += value
    return total


def _calc_aligned_ratio(
    numerator_annual: list[dict],
    numerator_quarterly: list[dict],
    denominator_annual: list[dict],
    denominator_quarterly: list[dict],
    *,
    denominator_abs: bool = False,
) -> _RatioResult:
    quarterly_numerator = _values_by_period(numerator_quarterly)
    quarterly_denominator = _values_by_period(denominator_quarterly)
    common_quarterly = sorted(set(quarterly_numerator) & set(quarterly_denominator), reverse=True)
    if len(common_quarterly) >= 4:
        periods = common_quarterly[:4]
        value = _ratio_value(
            _sum_periods(quarterly_numerator, periods),
            _sum_periods(quarterly_denominator, periods),
            denominator_abs=denominator_abs,
        )
        if value is None:
            return _empty_ratio_result()
        return {"value": value, "basis": "ttm", "period_end": periods[0]}

    annual_numerator = _values_by_period(numerator_annual)
    annual_denominator = _values_by_period(denominator_annual)
    common_annual = sorted(set(annual_numerator) & set(annual_denominator), reverse=True)
    if not common_annual:
        return _empty_ratio_result()

    period = common_annual[0]
    value = _ratio_value(annual_numerator[period], annual_denominator[period], denominator_abs=denominator_abs)
    if value is None:
        return _empty_ratio_result()
    return {"value": value, "basis": "annual", "period_end": period}


def _build_profitability_metrics(
    annual_revenue: list[dict],
    quarterly_revenue: list[dict],
    annual_operating_income: list[dict],
    quarterly_operating_income: list[dict],
    annual_net_income: list[dict],
    quarterly_net_income: list[dict],
    annual_interest_expense: list[dict],
    quarterly_interest_expense: list[dict],
) -> dict[str, object | None]:
    operating_margin = _calc_aligned_ratio(
        annual_operating_income,
        quarterly_operating_income,
        annual_revenue,
        quarterly_revenue,
    )
    net_income_margin = _calc_aligned_ratio(
        annual_net_income,
        quarterly_net_income,
        annual_revenue,
        quarterly_revenue,
    )
    interest_coverage = _calc_aligned_ratio(
        annual_operating_income,
        quarterly_operating_income,
        annual_interest_expense,
        quarterly_interest_expense,
        denominator_abs=True,
    )
    interest_coverage_value = interest_coverage["value"]

    return {
        "interest_coverage": interest_coverage_value,
        "interest_coverage_flag": (
            interest_coverage_value is not None and float(interest_coverage_value) < INTEREST_COVERAGE_WARNING_THRESHOLD
        ),
        "interest_coverage_basis": interest_coverage["basis"],
        "interest_coverage_period_end": interest_coverage["period_end"],
        "interest_coverage_warning_threshold": INTEREST_COVERAGE_WARNING_THRESHOLD,
        "operating_margin": operating_margin["value"],
        "operating_margin_basis": operating_margin["basis"],
        "operating_margin_period_end": operating_margin["period_end"],
        "net_income_margin": net_income_margin["value"],
        "net_income_margin_basis": net_income_margin["basis"],
        "net_income_margin_period_end": net_income_margin["period_end"],
    }


def _yf_numeric(v: object) -> float | None:
    if not isinstance(v, (str, bytes, SupportsFloat, SupportsIndex)):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def _yf_statement(ticker_obj: Any, attrs: tuple[str, ...]):
    for attr in attrs:
        try:
            obj = getattr(ticker_obj, attr)
        except Exception:
            continue

        try:
            candidate = obj() if callable(obj) else obj
        except TypeError:
            try:
                candidate = obj(freq="annual")
            except Exception:
                continue
        except Exception:
            continue

        if candidate is not None and not getattr(candidate, "empty", True):
            return candidate

    for attr in ("get_income_stmt", "get_financials"):
        try:
            obj = getattr(ticker_obj, attr)
        except Exception:
            continue
        if not callable(obj):
            continue
        for kwargs in ({"freq": "annual"}, {}):
            try:
                candidate = obj(**kwargs)
            except Exception:
                continue
            if candidate is not None and not getattr(candidate, "empty", True):
                return candidate
    return None


def _yf_info(ticker_obj: Any) -> dict:
    for attr in ("info", "get_info"):
        try:
            obj = getattr(ticker_obj, attr)
            candidate = obj() if callable(obj) else obj
        except Exception:
            continue
        if isinstance(candidate, dict):
            return candidate
    return {}


def _yf_index_lookup(statement: Any, names: tuple[str, ...]) -> object | None:
    if statement is None or getattr(statement, "empty", True):
        return None
    index_values: list[object] = list(getattr(statement, "index", []))
    exact = set(index_values)
    for name in names:
        if name in exact:
            return name

    normalized = {re.sub(r"[^a-z0-9]", "", str(idx).lower()): idx for idx in index_values}
    for name in names:
        found = normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if found is not None:
            return found
    return None


def _yf_line_value(statement: Any, column: object, names: tuple[str, ...]) -> float | None:
    row = _yf_index_lookup(statement, names)
    if row is None:
        return None
    try:
        return _yf_numeric(statement.at[row, column])
    except Exception:
        return None


def _yf_eps_value(statement: Any, column: object) -> float | None:
    direct = _yf_line_value(statement, column, YF_EPS_KEYS)
    if direct is not None:
        return direct

    net_income = _yf_line_value(statement, column, YF_NET_INCOME_KEYS)
    shares = _yf_line_value(statement, column, YF_DILUTED_SHARES_KEYS)
    if shares is None:
        shares = _yf_line_value(statement, column, YF_BASIC_SHARES_KEYS)
    return _safe_growth(net_income, shares, denom_abs=False)


def _yf_column_date(column: object) -> str:
    try:
        if hasattr(column, "date"):
            return str(column.date().isoformat())
    except Exception:
        pass
    raw = str(column or "")
    return raw[:10] if len(raw) >= 10 else raw


def _yf_column_sort_key(column: object) -> str:
    return _yf_column_date(column)


def _yf_period_label(column: object) -> str:
    period_end = _yf_column_date(column)
    if len(period_end) >= 4 and period_end[:4].isdigit():
        return f"FY{period_end[:4]}"
    return "FY"


def _yf_rows_from_statement(
    statement: Any,
    *,
    value_getter: Any,
    yoy_abs_denom: bool,
) -> list[dict]:
    if statement is None or getattr(statement, "empty", True):
        return []

    columns = sorted(list(getattr(statement, "columns", [])), key=_yf_column_sort_key, reverse=True)
    rows: list[dict] = []
    for column in columns[: ANNUAL_DISPLAY_LIMIT + ANNUAL_YOY_STEP]:
        value = value_getter(statement, column)
        if value is None:
            continue
        rows.append(
            {
                "period_label": _yf_period_label(column),
                "period_end": _yf_column_date(column),
                "value": value,
                "yoy_growth": None,
                "form": "Yahoo Finance",
                "filed": "",
                "accn": "",
                "filing_url": "",
                "source": "yfinance",
            }
        )

    for i, row in enumerate(rows):
        j = i + ANNUAL_YOY_STEP
        if j >= len(rows):
            continue
        curr = _yf_numeric(row.get("value"))
        prev = _yf_numeric(rows[j].get("value"))
        if curr is None or prev is None:
            continue
        row["yoy_growth"] = _safe_growth(curr - prev, prev, denom_abs=yoy_abs_denom)

    return rows[:ANNUAL_DISPLAY_LIMIT]


def _empty_breakdown() -> dict:
    return {
        "source_filing": None,
        "by_segment": [],
        "by_region": [],
        "extraction_meta": {
            "segment": {"status": "unavailable", "source": "none"},
            "region": {"status": "unavailable", "source": "none"},
            "ai_fallback_attempted": False,
        },
    }


def _build_yfinance_fallback(ticker: str, reason: str) -> dict:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ValueError("Yahoo Finance fallback unavailable: missing yfinance dependency") from exc

    ticker_obj = yf.Ticker(ticker)
    info = _yf_info(ticker_obj)
    annual_income = _yf_statement(ticker_obj, ("income_stmt", "financials"))

    annual_revenue = _yf_rows_from_statement(
        annual_income,
        value_getter=lambda statement, column: _yf_line_value(statement, column, YF_REVENUE_KEYS),
        yoy_abs_denom=False,
    )
    annual_eps = _yf_rows_from_statement(
        annual_income,
        value_getter=_yf_eps_value,
        yoy_abs_denom=True,
    )
    annual_operating_income = _yf_rows_from_statement(
        annual_income,
        value_getter=lambda statement, column: _yf_line_value(statement, column, YF_OPERATING_INCOME_KEYS),
        yoy_abs_denom=False,
    )
    annual_net_income = _yf_rows_from_statement(
        annual_income,
        value_getter=lambda statement, column: _yf_line_value(statement, column, YF_NET_INCOME_KEYS),
        yoy_abs_denom=False,
    )
    annual_interest_expense = _yf_rows_from_statement(
        annual_income,
        value_getter=lambda statement, column: _yf_line_value(statement, column, YF_INTEREST_EXPENSE_KEYS),
        yoy_abs_denom=False,
    )

    if not annual_revenue and not annual_eps:
        raise ValueError(f"No Revenue or EPS history found in Yahoo Finance for ticker: {ticker}")

    profitability_metrics = _build_profitability_metrics(
        annual_revenue,
        [],
        annual_operating_income,
        [],
        annual_net_income,
        [],
        annual_interest_expense,
        [],
    )

    return {
        "ticker": ticker,
        "company_name": str(info.get("longName") or info.get("shortName") or ticker),
        "cik": None,
        "data_source": "yfinance",
        "fallback_reason": reason,
        "financial_currency": str(info.get("financialCurrency") or info.get("currency") or "USD"),
        "metrics": {
            "revenue_cagr_3y": _calc_cagr(annual_revenue, years=3, abs_fallback=False),
            "eps_cagr_3y": _calc_cagr(annual_eps, years=3, abs_fallback=True),
            "avg_yoy_eps_growth_3q": None,
            "avg_yoy_revenue_growth_3q": None,
            **profitability_metrics,
        },
        "annual": {
            "revenue": annual_revenue,
            "eps": annual_eps,
        },
        "quarterly": {
            "revenue": [],
            "eps": [],
        },
        "breakdown": _empty_breakdown(),
    }


def _yfinance_fallback_or_raise(ticker: str, reason: str) -> dict:
    try:
        return _build_yfinance_fallback(ticker, reason)
    except Exception as exc:
        raise ValueError(f"{reason}; yfinance fallback failed: {exc}") from exc


def _iter_segment_pairs(segment_obj: object) -> list[tuple[str, str]]:
    if isinstance(segment_obj, dict):
        if "dimension" in segment_obj and "value" in segment_obj:
            dim = str(segment_obj.get("dimension") or "").strip()
            val = str(segment_obj.get("value") or "").strip()
            if dim and val:
                return [(dim, val)]
        out: list[tuple[str, str]] = []
        for k, v in segment_obj.items():
            if isinstance(v, str):
                out.append((str(k), v))
            elif isinstance(v, dict):
                out.extend(_iter_segment_pairs(v))
            elif isinstance(v, list):
                for item in v:
                    out.extend(_iter_segment_pairs(item))
        return out
    if isinstance(segment_obj, list):
        out_list: list[tuple[str, str]] = []
        for item in segment_obj:
            out_list.extend(_iter_segment_pairs(item))
        return out_list
    return []


def _classify_dimension(dimension: str) -> str | None:
    d = dimension.lower()
    if any(k in d for k in ("geo", "geograph", "region", "country", "market", "area", "location", "territor")):
        return "region"
    if any(
        k in d
        for k in (
            "product",
            "service",
            "segment",
            "lineofbusiness",
            "business",
            "operatingsegment",
            "reportablesegment",
        )
    ):
        return "segment"
    return None


def _normalize_member_label(raw_member: str) -> str:
    m = str(raw_member or "")
    if ":" in m:
        m = m.split(":", 1)[1]
    m = re.sub(r"Member$", "", m)
    m = re.sub(r"Segment$", "", m)
    m = re.sub(r"Region$", "", m)
    m = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", m)
    m = m.replace("_", " ").strip()
    return m


def _is_total_like_member(member: str) -> bool:
    s = member.lower()
    return (
        s in {"total", "all"}
        or s.startswith("total ")
        or s in {"worldwide", "global"}
        or s.startswith("worldwide ")
        or s.startswith("global ")
        or "all geograph" in s
        or "all region" in s
        or "all countr" in s
        or "all market" in s
        or "entire company" in s
        or "whole company" in s
        or "consolidated" in s
        or "elimination" in s
        or "all other" in s
        or s.startswith("all ")
        or " total " in s
        or s.endswith(" total")
        or s.endswith(" totals")
    )


def _classified_row_members(segment_obj: object) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set = set()
    for dim, member in _iter_segment_pairs(segment_obj):
        kind = _classify_dimension(dim)
        if kind is None:
            continue
        label = _normalize_member_label(member)
        if not label or _is_total_like_member(label):
            continue
        key = (kind, label)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _row_has_only_total_context(row: dict) -> bool:
    seg = row.get("segment")
    if not seg:
        return True

    for dim, member in _iter_segment_pairs(seg):
        kind = _classify_dimension(dim)
        if kind is None:
            continue
        label = _normalize_member_label(member)
        if label and not _is_total_like_member(label):
            return False
    return True


def _pick_best_period_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("fp") or ""), str(row.get("start") or ""))
        grouped[key].append(row)

    def _score(group_rows: list[dict]) -> tuple[int, int, int, str]:
        informative = 0
        total_like = 0
        for row in group_rows:
            if _classified_row_members(row.get("segment")):
                informative += 1
            if _row_has_only_total_context(row):
                total_like += 1
        latest_filed = max(str(r.get("filed") or "") for r in group_rows)
        return informative, total_like, len(group_rows), latest_filed

    return max(grouped.values(), key=_score)


def _pct_rows(values_by_label: dict[str, float], total: float | None) -> list[dict]:
    rows = []
    for label, value in sorted(values_by_label.items(), key=lambda kv: kv[1], reverse=True):
        pct = None
        if total is not None and total != 0:
            pct = value / total
        rows.append({"label": label, "value": value, "pct_of_total": pct})
    return rows


def _parse_money_like(v: object) -> float | None:
    if isinstance(v, (int, float)):
        return _as_float(v)
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    if s.startswith("+"):
        s = s[1:]
    if s.startswith("-"):
        neg = True
        s = s[1:]
    try:
        out = float(s)
    except Exception:
        return None
    return -out if neg else out


def _rows_to_value_map(rows: object) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        if not label or _is_total_like_member(label):
            continue
        val = _parse_money_like(row.get("value"))
        if val is None:
            continue
        cur = out.get(label)
        if cur is None or abs(val) > abs(cur):
            out[label] = val
    return out


def _parse_optional_bool(v: object) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "y", "1"}:
            return True
        if s in {"false", "no", "n", "0"}:
            return False
    return None


def _filing_context_for_nlp(html: str) -> str:
    try:
        from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
    except Exception:
        return ""

    head = html.lstrip()[:5000].lower()
    parser = "lxml"
    if head.startswith("<?xml") or "<xbrl" in head or "<ix:" in head:
        parser = "xml"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, parser)
    blocks: list[str] = []
    keywords = ("revenue", "segment", "geograph", "region", "united states", "foreign")

    for table in soup.find_all("table"):
        txt = " ".join(table.get_text(" ", strip=True).split())
        if not txt:
            continue
        lower = txt.lower()
        if "revenue" not in lower:
            continue
        if not any(k in lower for k in keywords[1:]):
            continue
        blocks.append(txt[:2500])
        if len(blocks) >= 8:
            break

    if not blocks:
        body = " ".join(soup.get_text(" ", strip=True).split())
        if body:
            lower = body.lower()
            idx = lower.find("revenue")
            if idx >= 0:
                start = max(0, idx - 5000)
                end = min(len(body), idx + 15000)
                blocks.append(body[start:end])
            else:
                blocks.append(body[:12000])

    joined = "\n\n".join(blocks).strip()
    return joined[:18000]


# ---------------------------------------------------------------------------
# Heading keywords for classifying HTML revenue-breakdown tables
# ---------------------------------------------------------------------------
_SEGMENT_HEADING_KW = (
    "products and services",
    "by category",
    "by product",
    "product and service",
    "business segment",
    "operating segment",
    "reportable segment",
    "line of business",
)
_REGION_HEADING_KW = (
    "geographic",
    "by region",
    "by country",
    "by market",
    "by area",
)

# If the heading contains "segment" but also one of these, it's region.
_REGION_CONTEXT_KW = (
    "geograph",
    "americas",
    "europe",
    "greater china",
    "japan",
    "asia",
    "country",
    "region",
    "market",
    "domestic",
    "international",
    "united states",
    "foreign",
)


def _classify_table(heading_text: str, table_text: str) -> str | None:
    """Classify a table as 'segment', 'region', or None based on heading and content."""
    h = heading_text.lower()
    t = table_text.lower()

    # Check region first — Apple uses "Segment Operating Performance" for
    # geographic data, so we need region-context keywords from both heading AND table body.
    if any(k in h for k in _REGION_HEADING_KW):
        return "region"

    if any(k in h for k in _SEGMENT_HEADING_KW):
        # Disambiguate by checking if the table contains obvious region labels
        if any(k in t for k in _REGION_CONTEXT_KW):
            return "region"
        return "segment"

    if "segment" in h:
        if any(k in h for k in _REGION_CONTEXT_KW) or any(k in t for k in _REGION_CONTEXT_KW):
            return "region"
        return "segment"

    return None


def _detect_unit_scale(text: str) -> float:
    """Detect 'in millions', 'in billions', etc. from surrounding text."""
    t = text.lower()
    if "in billion" in t:
        return 1_000_000_000.0
    if "in million" in t:
        return 1_000_000.0
    if "in thousand" in t:
        return 1_000.0
    return 1.0


def _extract_number_from_cell(cell_text: str) -> float | None:
    """Extract a numeric value from a table cell like '$209,586' or '(1,234)'."""
    s = cell_text.strip()
    if not s or s in ("—", "–", "-", "—", "N/A", "n/a"):
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    s = s.replace("$", "").replace(",", "").replace(" ", "").replace("\xa0", "")
    # Remove percentage signs
    if s.endswith("%"):
        return None  # skip percentage columns

    if s.startswith("+"):
        s = s[1:]
    if s.startswith("-") or s.startswith("−"):
        neg = True
        s = s[1:]

    # Strip footnote markers like (1)
    s = re.sub(r"\(\d+\)$", "", s).strip()

    try:
        val = float(s)
    except (ValueError, TypeError):
        return None
    return -val if neg else val


def _extract_breakdown_from_html(
    *,
    cik_str: str,
    accn: str,
    form: str,
    filed: str,
    submissions: dict | None,
    wanted_axes: set[str],
) -> dict | None:
    """
    Fetch the filing HTML and parse revenue-breakdown tables directly.

    Returns a dict with by_segment/by_region lists, or None on failure.
    """
    from portfolio.momentum.fundamental_momentum.edgar_fetcher import build_filing_url

    filing_url = build_filing_url(cik_str, accn, submissions=submissions)
    if not filing_url:
        return None

    try:
        resp = requests_get(
            filing_url,
            headers={"User-Agent": "market-scripts research@example.com"},
            timeout=25,
        )
        if resp.status_code != 200 or not resp.text:
            return None
    except Exception:
        return None

    try:
        from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
    except Exception:
        return None

    head = resp.text.lstrip()[:5000].lower()
    parser = "lxml" if not (head.startswith("<?xml") or "<xbrl" in head or "<ix:" in head) else "lxml-xml"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(resp.text, parser)

    # ── Detect unit scale from the document ──────────────────────────────
    body_text = soup.get_text(" ", strip=True)[:30000]
    default_scale = _detect_unit_scale(body_text)

    # ── Collect all tables with their preceding text context ─────────────
    tables = [table for table in soup.find_all("table") if isinstance(table, Tag)]
    breakdowns: dict[str, dict[str, float]] = {}  # kind -> { label: value }

    for table in tables:
        # Build heading context: walk backwards from the table to find
        # descriptive text (headings, bold spans, or regular paragraphs).
        heading_parts: list[str] = []
        node = table
        for _ in range(15):  # look at up to 15 preceding siblings/parents
            prev_node = node.find_previous(
                ["p", "div", "span", "b", "strong", "h1", "h2", "h3", "h4", "h5", "h6", "td"]
            )
            if prev_node is None:
                break
            if not isinstance(prev_node, Tag):
                break
            node = prev_node
            txt = " ".join(node.get_text(" ", strip=True).split())
            if txt:
                heading_parts.append(txt)
                # Stop if heading already long enough or we found a table break
                if len(" ".join(heading_parts)) > 600:
                    break
                # If we found a strong classification, stop early
                combined = " ".join(heading_parts).lower()
                if any(k in combined for k in _SEGMENT_HEADING_KW + _REGION_HEADING_KW):
                    break

        heading_context = " ".join(reversed(heading_parts))

        # Get just the first few rows of the table for context checking
        preview_rows = [row for row in table.find_all("tr")[:10] if isinstance(row, Tag)]
        table_text = "\n".join(row.get_text(" ", strip=True) for row in preview_rows)

        kind = _classify_table(heading_context, table_text)
        if kind is None or kind not in wanted_axes:
            continue

        # Skip if already found this axis
        if kind in breakdowns and breakdowns[kind]:
            continue

        # ── Detect per-table scale ───────────────────────────────────────
        table_scale = _detect_unit_scale(heading_context)
        if table_scale == 1.0:
            table_scale = default_scale

        # ── Extract rows from the table ──────────────────────────────────
        rows = [row for row in table.find_all("tr") if isinstance(row, Tag)]
        if not rows:
            continue

        value_map: dict[str, float] = {}
        for row in rows:
            cells = [cell for cell in row.find_all(["td", "th"]) if isinstance(cell, Tag)]
            if len(cells) < 2:
                continue

            # First cell is the label
            label_raw = cells[0].get_text(" ", strip=True)
            label = re.sub(r"\(\d+\)\s*$", "", label_raw).strip()
            label = re.sub(r"^\(\d+\)\s+", "", label).strip()
            if not label:
                continue
            if _is_total_like_member(label):
                continue
            if label.lower() in ("", "change", "%", "change %"):
                continue

            # Find the first numeric value cell (the latest year column).
            # Skip header-like rows where all cells parse as None.
            val = None
            for cell in cells[1:]:
                parsed = _extract_number_from_cell(cell.get_text(" ", strip=True))
                if parsed is not None:
                    val = parsed
                    break  # take the first (leftmost) numeric = latest period

            if val is not None:
                val *= table_scale
                prev = value_map.get(label)
                if prev is None or abs(val) > abs(prev):
                    value_map[label] = val

        if value_map:
            breakdowns[kind] = value_map

    if not breakdowns:
        return None

    by_segment_map = breakdowns.get("segment", {})
    by_region_map = breakdowns.get("region", {})

    total_val: float | None = None
    seg_sum = sum(abs(v) for v in by_segment_map.values()) if by_segment_map else 0.0
    reg_sum = sum(abs(v) for v in by_region_map.values()) if by_region_map else 0.0
    total_val = max(seg_sum, reg_sum) if max(seg_sum, reg_sum) > 0 else None

    return {
        "accn": accn,
        "filed": filed,
        "form": form,
        "by_segment": _pct_rows(by_segment_map, total_val) if by_segment_map else [],
        "by_region": _pct_rows(by_region_map, total_val) if by_region_map else [],
    }


def _extract_breakdown_via_nlp(
    *,
    cik_str: str,
    accn: str,
    form: str,
    filed: str,
    submissions: dict | None,
    wanted_axes: set[str],
) -> dict | None:
    if not has_llm_api_key():
        return None

    filing_url = build_filing_url(cik_str, accn, submissions=submissions)
    if not filing_url:
        return None

    try:
        resp = requests_get(
            filing_url,
            headers={"User-Agent": "market-scripts research@example.com"},
            timeout=25,
        )
        if resp.status_code != 200 or not resp.text:
            return None
    except Exception:
        return None

    context = _filing_context_for_nlp(resp.text)
    if not context:
        return None

    want_segment = "segment" in wanted_axes
    want_region = "region" in wanted_axes
    wanted_axes_str = ", ".join(sorted(wanted_axes)) if wanted_axes else "none"

    prompt = (
        "Extract ONLY the latest-period revenue breakdown from this SEC filing excerpt.\n"
        f"Only populate requested axes: {wanted_axes_str}.\n"
        "Return strict JSON with this schema:\n"
        "{\n"
        '  "period_end": "YYYY-MM-DD or empty",\n'
        '  "total_revenue": number or null,\n'
        '  "unit_scale": "ones" | "thousands" | "millions" | "billions",\n'
        '  "segment_disclosed": true | false | null,\n'
        '  "region_disclosed": true | false | null,\n'
        '  "by_segment": [{"label": string, "value": number}],\n'
        '  "by_region": [{"label": string, "value": number}]\n'
        "}\n"
        "Rules: include only revenue rows, exclude totals/eliminations, keep latest quarter in this filing.\n"
        "If an axis is not requested, return [] for that axis and null for its *_disclosed flag.\n"
        "If an axis is requested but not disclosed, return [] and set *_disclosed to false.\n"
        "If an axis is requested and disclosed, set *_disclosed to true.\n"
        "No markdown, no explanation, JSON only.\n\n"
        f"FORM: {form}\nFILED: {filed}\nACCN: {accn}\nURL: {filing_url}\n\n"
        f"EXCERPT:\n{context}"
    )

    payload: dict | None = None
    for model in (MODEL_LOW, MODEL_MID):
        try:
            txt, _citations, _resp = call_llm_text(
                prompt=prompt,
                model=model,
                api_key=None,
                max_tokens=2048,
            )
        except Exception:
            continue
        if not txt:
            continue
        parsed = parse_json_text(txt)
        if isinstance(parsed, dict):
            payload = parsed
            break

    if payload is None:
        return None

    by_segment = _rows_to_value_map(payload.get("by_segment")) if want_segment else {}
    by_region = _rows_to_value_map(payload.get("by_region")) if want_region else {}
    segment_disclosed = _parse_optional_bool(payload.get("segment_disclosed")) if want_segment else None
    region_disclosed = _parse_optional_bool(payload.get("region_disclosed")) if want_region else None

    scale_map = {
        "ones": 1.0,
        "thousands": 1_000.0,
        "millions": 1_000_000.0,
        "billions": 1_000_000_000.0,
    }
    scale_key = str(payload.get("unit_scale") or "ones").strip().lower()
    scale = scale_map.get(scale_key, 1.0)
    if scale != 1.0:
        by_segment = {k: v * scale for k, v in by_segment.items()}
        by_region = {k: v * scale for k, v in by_region.items()}

    total_val = _parse_money_like(payload.get("total_revenue"))
    if total_val is not None:
        total_val *= scale
    else:
        seg_sum = sum(abs(v) for v in by_segment.values())
        reg_sum = sum(abs(v) for v in by_region.values())
        total_val = seg_sum if seg_sum >= reg_sum and seg_sum > 0 else (reg_sum if reg_sum > 0 else None)

    return {
        "accn": accn,
        "period_end": str(payload.get("period_end") or ""),
        "filed": filed,
        "form": form,
        "by_segment": _pct_rows(by_segment, total_val),
        "by_region": _pct_rows(by_region, total_val),
        "segment_disclosed": segment_disclosed,
        "region_disclosed": region_disclosed,
    }


def _extract_breakdown_for_filing(us_gaap: dict, accn: str) -> dict:
    for concept in REVENUE_CONCEPTS:
        all_rows = _entries_for(us_gaap, concept, "USD")
        if not all_rows:
            continue

        in_filing = [
            e
            for e in all_rows
            if str(e.get("accn") or "") == accn
            and str(e.get("form") or "") in ALLOWED_QUARTERLY_FORMS
            and _is_valid_fact_row(e)
        ]
        if not in_filing:
            continue

        target_end = max(str(e.get("end") or "") for e in in_filing)
        period_rows = [e for e in in_filing if str(e.get("end") or "") == target_end]
        if not period_rows:
            continue

        period_rows = _pick_best_period_rows(period_rows)

        total_candidates = [e for e in period_rows if _row_has_only_total_context(e)]
        total_val: float | None = None
        if total_candidates:
            total_row = max(
                total_candidates,
                key=lambda e: (abs(_as_float(e.get("val")) or 0.0), str(e.get("filed") or "")),
            )
            total_val = _as_float(total_row.get("val"))

        by_segment: dict[str, float] = {}
        by_region: dict[str, float] = {}

        for row in period_rows:
            classified = _classified_row_members(row.get("segment"))
            if not classified:
                continue
            if len(classified) != 1:
                # Cross-dimensional intersections are ambiguous for a simple
                # one-axis breakdown; skip to avoid double counting.
                continue
            kind, label = classified[0]
            if not kind or not label:
                continue

            val = _as_float(row.get("val"))
            if val is None:
                continue

            bucket = by_region if kind == "region" else by_segment
            prev = bucket.get(label)
            if prev is None or abs(val) > abs(prev):
                bucket[label] = val

        if by_segment or by_region:
            best = max(period_rows, key=lambda e: str(e.get("filed") or ""))
            return {
                "accn": accn,
                "period_end": target_end,
                "filed": str(best.get("filed") or ""),
                "form": str(best.get("form") or ""),
                "by_segment": _pct_rows(by_segment, total_val),
                "by_region": _pct_rows(by_region, total_val),
            }

    return {"accn": accn, "by_segment": [], "by_region": []}


def _candidate_revenue_filings(us_gaap: dict) -> list[dict]:
    by_accn: dict[str, dict] = {}
    for concept in REVENUE_CONCEPTS:
        for e in _entries_for(us_gaap, concept, "USD"):
            form = str(e.get("form") or "")
            accn = str(e.get("accn") or "")
            filed = str(e.get("filed") or "")
            if form not in ALLOWED_QUARTERLY_FORMS or not accn or not filed:
                continue
            cur = by_accn.get(accn)
            if cur is None or filed > cur["filed"]:
                by_accn[accn] = {"accn": accn, "form": form, "filed": filed}
    return sorted(by_accn.values(), key=lambda x: x["filed"], reverse=True)


def _build_breakdown(us_gaap: dict, cik_str: str, submissions: dict | None) -> dict:
    def _axis_meta(status: str, source: str) -> dict:
        return {"status": status, "source": source}

    filings = _candidate_revenue_filings(us_gaap)
    if not filings:
        return {
            "source_filing": None,
            "by_segment": [],
            "by_region": [],
            "extraction_meta": {
                "segment": _axis_meta("unavailable", "none"),
                "region": _axis_meta("unavailable", "none"),
                "ai_fallback_attempted": False,
            },
        }

    annual_filings = [f for f in filings if str(f.get("form") or "") in ALLOWED_ANNUAL_FORMS]
    search_filings = annual_filings if annual_filings else filings
    latest_filing = search_filings[0]
    xbrl_choice: dict | None = None
    for f in search_filings:
        candidate = _extract_breakdown_for_filing(us_gaap, f["accn"])
        if candidate.get("by_segment") or candidate.get("by_region"):
            xbrl_choice = {**f, **candidate}
            break

    xbrl_segment = (xbrl_choice or {}).get("by_segment") or []
    xbrl_region = (xbrl_choice or {}).get("by_region") or []

    missing_axes: set[str] = set()
    if not xbrl_segment:
        missing_axes.add("segment")
    if not xbrl_region:
        missing_axes.add("region")

    html_candidate: dict | None = None
    if missing_axes:
        html_candidate = _extract_breakdown_from_html(
            cik_str=cik_str,
            accn=str(latest_filing.get("accn") or ""),
            form=str(latest_filing.get("form") or ""),
            filed=str(latest_filing.get("filed") or ""),
            submissions=submissions,
            wanted_axes=missing_axes,
        )

    html_segment = (html_candidate or {}).get("by_segment") or []
    html_region = (html_candidate or {}).get("by_region") or []

    if html_segment:
        missing_axes.discard("segment")
    if html_region:
        missing_axes.discard("region")

    ai_fallback_attempted = False
    nlp_candidate: dict | None = None
    if missing_axes and has_llm_api_key():
        ai_fallback_attempted = True
        nlp_candidate = _extract_breakdown_via_nlp(
            cik_str=cik_str,
            accn=str(latest_filing.get("accn") or ""),
            form=str(latest_filing.get("form") or ""),
            filed=str(latest_filing.get("filed") or ""),
            submissions=submissions,
            wanted_axes=missing_axes,
        )

    ai_segment = (nlp_candidate or {}).get("by_segment") or []
    ai_region = (nlp_candidate or {}).get("by_region") or []
    segment_disclosed = _parse_optional_bool((nlp_candidate or {}).get("segment_disclosed"))
    region_disclosed = _parse_optional_bool((nlp_candidate or {}).get("region_disclosed"))

    if xbrl_segment:
        by_segment = xbrl_segment
        segment_meta = _axis_meta("found", "xbrl")
    elif html_segment:
        by_segment = html_segment
        segment_meta = _axis_meta("found", "html")
    elif ai_segment:
        by_segment = ai_segment
        segment_meta = _axis_meta("found", "ai")
    elif "segment" in missing_axes and segment_disclosed is False:
        by_segment = []
        segment_meta = _axis_meta("not_disclosed", "none")
    else:
        by_segment = []
        segment_meta = _axis_meta("unavailable", "none")

    if xbrl_region:
        by_region = xbrl_region
        region_meta = _axis_meta("found", "xbrl")
    elif html_region:
        by_region = html_region
        region_meta = _axis_meta("found", "html")
    elif ai_region:
        by_region = ai_region
        region_meta = _axis_meta("found", "ai")
    elif "region" in missing_axes and region_disclosed is False:
        by_region = []
        region_meta = _axis_meta("not_disclosed", "none")
    else:
        by_region = []
        region_meta = _axis_meta("unavailable", "none")

    if nlp_candidate:
        chosen = xbrl_choice or {**latest_filing, **nlp_candidate}
    elif html_candidate:
        chosen = xbrl_choice or {**latest_filing, **html_candidate}
    else:
        chosen = xbrl_choice or latest_filing

    accn = str(chosen.get("accn") or "")
    source_filing = {
        "form": str(chosen.get("form") or ""),
        "filed": str(chosen.get("filed") or ""),
        "accn": accn,
        "period_end": str(chosen.get("period_end") or ""),
        "filing_url": build_filing_url(cik_str, accn, submissions=submissions),
    }

    return {
        "source_filing": source_filing,
        "by_segment": by_segment,
        "by_region": by_region,
        "extraction_meta": {
            "segment": segment_meta,
            "region": region_meta,
            "ai_fallback_attempted": ai_fallback_attempted,
        },
    }


def get_data(ticker: str) -> dict:
    tk = str(ticker or "").strip().upper()
    if not tk:
        raise ValueError("Ticker is required")

    cik_str = get_cik_for_ticker(tk)
    if not cik_str:
        return _yfinance_fallback_or_raise(tk, f"CIK not found for ticker: {tk}")

    facts = fetch_companyfacts_by_cik(cik_str)
    if facts is None:
        return _yfinance_fallback_or_raise(tk, f"No SEC EDGAR companyfacts available for ticker: {tk}")

    submissions = fetch_submissions_by_cik(cik_str)
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    annual_revenue, quarterly_revenue = _build_revenue_rows(us_gaap, cik_str, submissions)
    annual_eps, quarterly_eps = _build_eps_rows(us_gaap, cik_str, submissions)
    annual_operating_income, quarterly_operating_income = _build_flow_rows(
        us_gaap,
        OPERATING_INCOME_CONCEPTS,
        cik_str,
        submissions,
    )
    annual_net_income, quarterly_net_income = _build_flow_rows(
        us_gaap,
        NET_INCOME_CONCEPTS,
        cik_str,
        submissions,
    )
    annual_interest_expense, quarterly_interest_expense = _build_flow_rows(
        us_gaap,
        INTEREST_EXPENSE_RATIO_CONCEPTS,
        cik_str,
        submissions,
        preserve_concept_order=True,
    )

    if not annual_revenue and not quarterly_revenue and not annual_eps and not quarterly_eps:
        return _yfinance_fallback_or_raise(
            tk,
            f"No Revenue or EPS history found in EDGAR companyfacts for ticker: {tk}",
        )

    profitability_metrics = _build_profitability_metrics(
        annual_revenue,
        quarterly_revenue,
        annual_operating_income,
        quarterly_operating_income,
        annual_net_income,
        quarterly_net_income,
        annual_interest_expense,
        quarterly_interest_expense,
    )

    metrics = {
        "revenue_cagr_3y": _calc_cagr(annual_revenue, years=3, abs_fallback=False),
        "eps_cagr_3y": _calc_cagr(annual_eps, years=3, abs_fallback=True),
        "avg_yoy_eps_growth_3q": _calc_avg_3q_yoy(quarterly_eps, denom_abs=True),
        "avg_yoy_revenue_growth_3q": _calc_avg_3q_yoy(quarterly_revenue, denom_abs=False),
        **profitability_metrics,
    }

    breakdown = _build_breakdown(us_gaap, cik_str, submissions)

    return {
        "ticker": tk,
        "company_name": str(facts.get("entityName") or tk),
        "cik": cik_str,
        "data_source": "sec_edgar",
        "fallback_reason": None,
        "financial_currency": "USD",
        "metrics": metrics,
        "annual": {
            "revenue": annual_revenue,
            "eps": annual_eps,
        },
        "quarterly": {
            "revenue": quarterly_revenue,
            "eps": quarterly_eps,
        },
        "breakdown": breakdown,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("Starting script execution: %s", __file__)
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fetch SEC EDGAR financials for one ticker")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    args = parser.parse_args()

    print(json.dumps(get_data(args.ticker), indent=2))
