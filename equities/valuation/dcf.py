"""
Discounted Cash Flow (DCF) valuation model.

Provides:
  - get_historical_data(ticker) → historical financials for the Historical tab
  - run_valuation(ticker, assumptions) → N-year projection + multi-method valuations

Data sourcing: SEC EDGAR (primary, deeper history) with yfinance fallback.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date as _date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from portfolio.momentum.fundamental_momentum.edgar_fetcher import (  # noqa: E402
    extract_dcf_historicals,
)
from utils.retry import yf_download, yf_ticker_info  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EQUITY_RISK_PREMIUM = 0.055  # 5.5%
DEFAULT_COST_OF_DEBT = 0.05
DEFAULT_BETA = 1.0
DEFAULT_TAX_RATE = 0.21


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------


def _fetch_yfinance_data(ticker: str) -> dict[str, Any]:
    """Fetch yfinance data (used for info, prices, and fallback financials)."""
    t = yf.Ticker(ticker)
    info = yf_ticker_info(ticker)
    if not info:
        raise ValueError(f"Could not fetch data for ticker '{ticker}'")

    prices = yf_download(ticker, period="6y", interval="1d")

    return {
        "info": info,
        "income_stmt": t.income_stmt,
        "quarterly_income_stmt": t.quarterly_income_stmt,
        "balance_sheet": t.balance_sheet,
        "quarterly_balance_sheet": t.quarterly_balance_sheet,
        "cashflow": t.cashflow,
        "quarterly_cashflow": t.quarterly_cashflow,
        "prices": prices,
    }


def _get_row(df: pd.DataFrame, *names: str) -> pd.Series | None:
    """Try multiple row names in a yfinance DataFrame."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _normalize_numeric_series(
    value: Any,
    key: str,
    years: int,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> list[float]:
    """Normalize a scalar or series assumption to one float per projection year."""
    if isinstance(value, pd.Series):
        raw_values = value.tolist()
    elif isinstance(value, np.ndarray):
        raw_values = value.tolist()
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = [value] * years

    if len(raw_values) != years:
        raise ValueError(f"{key} must have {years} values")

    values: list[float] = []
    for raw in raw_values:
        f = _safe_float(raw)
        if f is None:
            raise ValueError(f"{key} must contain only finite numbers")
        if min_value is not None:
            if min_inclusive and f < min_value:
                raise ValueError(f"{key} values must be >= {min_value}")
            if not min_inclusive and f <= min_value:
                raise ValueError(f"{key} values must be > {min_value}")
        if max_value is not None:
            if max_inclusive and f > max_value:
                raise ValueError(f"{key} values must be <= {max_value}")
            if not max_inclusive and f >= max_value:
                raise ValueError(f"{key} values must be < {max_value}")
        values.append(f)
    return values


def _fiscal_bucket(d: _date) -> tuple[int, int]:
    """Convert date to a (year, month) bucket for fuzzy fiscal period matching.

    EDGAR reports slightly different exact dates for the same fiscal period
    across different XBRL concepts. Dates near month boundaries can straddle
    months (e.g. Aug-28 vs Sep-01 for Micron's FY end). We snap dates in the
    first 5 days of a month back to the previous month so they land in the
    same bucket.
    """
    if d.day <= 5 and d.month > 1:
        return (d.year, d.month - 1)
    if d.day <= 5 and d.month == 1:
        return (d.year - 1, 12)
    return (d.year, d.month)


# ---------------------------------------------------------------------------
# EDGAR → list-of-dicts table builders
# ---------------------------------------------------------------------------


def _edgar_to_annual_map(pairs: list[tuple[_date, float]]) -> dict[str, float]:
    """Convert EDGAR (date, value) pairs to {date_label: value} dict."""
    return {d.strftime("%b-%y"): v for d, v in pairs}


def _build_annual_table_edgar(
    revenue_pairs: list[tuple[_date, float]],
    metric_pairs: list[tuple[_date, float]],
    metric_key: str,
    pct_key: str,
    n: int = 5,
) -> list[dict]:
    """Build an annual table (EBITDA, D&A, or CapEx) from EDGAR data."""
    rev_map = {_fiscal_bucket(d): v for d, v in revenue_pairs}
    metric_map = {_fiscal_bucket(d): v for d, v in metric_pairs}
    # Use metric dates where we also have revenue in the same (year, month)
    common_yms = sorted([_fiscal_bucket(d) for d, _ in metric_pairs if _fiscal_bucket(d) in rev_map])[-n:]

    rows = []
    for ym in common_yms:
        rev = rev_map[ym]
        metric = metric_map.get(ym)
        if metric is not None and metric_key in ("capex", "da"):
            metric = abs(metric)
        pct = (metric / rev * 100) if rev and metric and rev != 0 else None
        label = _date(ym[0], ym[1], 1).strftime("%b-%y")
        rows.append(
            {
                "fiscal_year": label,
                "revenue": rev,
                metric_key: metric,
                pct_key: round(pct, 1) if pct is not None else None,
            }
        )

    pcts = [float(value) for r in rows if isinstance(value := r.get(pct_key), (int, float))]
    avg = round(sum(pcts) / len(pcts), 1) if pcts else None
    for r in rows:
        r["avg"] = avg
    return rows


def _build_nwc_table_edgar(
    annual_revenue: list[tuple[_date, float]],
    annual_ca: list[tuple[_date, float]],
    annual_cl: list[tuple[_date, float]],
    n: int = 5,
) -> list[dict]:
    """Build NWC table from EDGAR annual balance sheet data."""
    rev_map = {_fiscal_bucket(d): v for d, v in annual_revenue}
    ca_map = {_fiscal_bucket(d): v for d, v in annual_ca}
    cl_map = {_fiscal_bucket(d): v for d, v in annual_cl}

    common_yms = sorted(ym for ym in ca_map if ym in cl_map)[-n:]
    rows = []
    for ym in common_yms:
        ca = ca_map[ym]
        cl = cl_map[ym]
        nwc = ca - cl
        rev = rev_map.get(ym)
        pct = (nwc / rev * 100) if nwc is not None and rev and rev != 0 else None
        label = _date(ym[0], ym[1], 1).strftime("%b-%y")
        rows.append(
            {
                "fiscal_year": label,
                "revenue": rev,
                "nwc": nwc,
                "nwc_pct_rev": round(pct, 1) if pct is not None else None,
            }
        )

    pcts = [float(value) for r in rows if isinstance(value := r.get("nwc_pct_rev"), (int, float))]
    avg = round(sum(pcts) / len(pcts), 1) if pcts else None
    for r in rows:
        r["avg"] = avg
    return rows


def _build_quarterly_multiples_edgar(
    quarterly_revenue: list[tuple[_date, float]],
    quarterly_ebitda: list[tuple[_date, float]] | None,
    quarterly_op_income: list[tuple[_date, float]] | None,
    quarterly_da: list[tuple[_date, float]] | None,
    quarterly_debt: list[tuple[_date, float]],
    quarterly_current_debt: list[tuple[_date, float]],
    quarterly_cash: list[tuple[_date, float]],
    prices: pd.DataFrame,
    shares_outstanding: int | None,
) -> tuple[list[dict], list[dict]]:
    """Build EV/EBITDA and EV/Revenue tables from EDGAR quarterly data."""
    if not shares_outstanding or not quarterly_revenue:
        return [], []

    # Build EBITDA from direct or OpIncome + D&A (keyed by year-month)
    ebitda_map: dict[tuple[int, int], float] = {}
    if quarterly_ebitda:
        ebitda_map = {_fiscal_bucket(d): v for d, v in quarterly_ebitda}
    elif quarterly_op_income and quarterly_da:
        oi_map = {_fiscal_bucket(d): v for d, v in quarterly_op_income}
        da_map = {_fiscal_bucket(d): v for d, v in quarterly_da}
        for ym in oi_map:
            if ym in da_map:
                ebitda_map[ym] = oi_map[ym] + abs(da_map[ym])

    rev_map = {_fiscal_bucket(d): v for d, v in quarterly_revenue}
    debt_map = {_fiscal_bucket(d): v for d, v in quarterly_debt}
    cd_map = {_fiscal_bucket(d): v for d, v in quarterly_current_debt}
    cash_map = {_fiscal_bucket(d): v for d, v in quarterly_cash}

    # Get price series
    price_series = prices
    if isinstance(prices.columns, pd.MultiIndex):
        if "Close" in prices.columns.get_level_values(0):
            price_series = prices["Close"]
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]
        else:
            price_series = prices.iloc[:, 0]
    elif "Close" in prices.columns:
        price_series = prices["Close"]
    else:
        price_series = prices.iloc[:, 0]

    # Get all quarter-end (year, month) keys from revenue, sorted chronologically
    all_yms = sorted(rev_map.keys())[-24:]

    ev_ebitda_rows: list[dict] = []
    ev_rev_rows: list[dict] = []

    for i, ym in enumerate(all_yms):
        # Use ~end of month for price lookup
        d_ts = pd.Timestamp(_date(ym[0], ym[1], 28))
        mask = price_series.index <= d_ts
        if not mask.any():
            continue
        close_price = float(price_series[mask].iloc[-1])
        market_cap = close_price * shares_outstanding

        total_debt = (debt_map.get(ym) or 0) + (cd_map.get(ym) or 0)
        cash = cash_map.get(ym) or 0
        net_debt = total_debt - cash
        ev = market_cap + net_debt

        # TTM = sum of last 4 quarters
        start_idx = max(0, i - 3)
        ttm_yms = all_yms[start_idx : i + 1]
        if len(ttm_yms) < 4:
            continue

        ttm_rev = sum(rev_map.get(qym, 0) for qym in ttm_yms)
        ttm_ebitda = sum(ebitda_map.get(qym, 0) for qym in ttm_yms) if ebitda_map else 0

        label = _date(ym[0], ym[1], 1).strftime("%b-%y")

        if ttm_ebitda and ttm_ebitda > 0:
            ev_ebitda_rows.append(
                {
                    "quarter_end": label,
                    "ev": ev,
                    "ebitda_ttm": ttm_ebitda,
                    "ev_ebitda": round(ev / ttm_ebitda, 1),
                }
            )

        if ttm_rev and ttm_rev > 0:
            ev_rev_rows.append(
                {
                    "quarter_end": label,
                    "ev": ev,
                    "revenue_ttm": ttm_rev,
                    "ev_revenue": round(ev / ttm_rev, 1),
                }
            )

    # Keep last 20 quarters max
    ev_ebitda_rows = ev_ebitda_rows[-20:]
    ev_rev_rows = ev_rev_rows[-20:]

    if ev_ebitda_rows:
        avg = round(sum(r["ev_ebitda"] for r in ev_ebitda_rows) / len(ev_ebitda_rows), 1)
        for r in ev_ebitda_rows:
            r["avg"] = avg

    if ev_rev_rows:
        avg = round(sum(r["ev_revenue"] for r in ev_rev_rows) / len(ev_rev_rows), 1)
        for r in ev_rev_rows:
            r["avg"] = avg

    return ev_ebitda_rows, ev_rev_rows


# ---------------------------------------------------------------------------
# yfinance fallback table builders (unchanged logic from original)
# ---------------------------------------------------------------------------


def _compute_ebitda_table_yf(income_stmt: pd.DataFrame, cashflow: pd.DataFrame) -> list[dict]:
    revenue_row = _get_row(income_stmt, "Total Revenue", "Operating Revenue")
    ebitda_row = _get_row(income_stmt, "EBITDA", "Normalized EBITDA")
    if ebitda_row is None:
        op_income = _get_row(income_stmt, "Operating Income", "EBIT")
        da = _get_row(cashflow, "Depreciation And Amortization", "Depreciation & Amortization")
        if da is None:
            da = _get_row(income_stmt, "Reconciled Depreciation")
        if op_income is not None and da is not None:
            ebitda_row = op_income + da.reindex(op_income.index, fill_value=0)
    if revenue_row is None or ebitda_row is None:
        return []

    dates = sorted(revenue_row.index, reverse=False)[-5:]
    rows = []
    for d in dates:
        rev = _safe_float(revenue_row.get(d))
        ebitda = _safe_float(ebitda_row.get(d))
        pct = (ebitda / rev * 100) if rev and ebitda and rev != 0 else None
        rows.append(
            {
                "fiscal_year": d.strftime("%b-%y") if hasattr(d, "strftime") else str(d),
                "revenue": rev,
                "ebitda": ebitda,
                "ebitda_margin": round(pct, 1) if pct is not None else None,
            }
        )
    margins = [float(value) for r in rows if isinstance(value := r.get("ebitda_margin"), (int, float))]
    avg = round(sum(margins) / len(margins), 1) if margins else None
    for r in rows:
        r["avg"] = avg
    return rows


def _compute_da_table_yf(income_stmt: pd.DataFrame, cashflow: pd.DataFrame) -> list[dict]:
    revenue_row = _get_row(income_stmt, "Total Revenue", "Operating Revenue")
    da_row = _get_row(cashflow, "Depreciation And Amortization", "Depreciation & Amortization")
    if da_row is None:
        da_row = _get_row(income_stmt, "Reconciled Depreciation")
    if revenue_row is None or da_row is None:
        return []

    dates = sorted(revenue_row.index, reverse=False)[-5:]
    rows = []
    for d in dates:
        rev = _safe_float(revenue_row.get(d))
        da = _safe_float(da_row.get(d))
        if da is not None:
            da = abs(da)
        pct = (da / rev * 100) if rev and da and rev != 0 else None
        rows.append(
            {
                "fiscal_year": d.strftime("%b-%y") if hasattr(d, "strftime") else str(d),
                "revenue": rev,
                "da": da,
                "da_pct_rev": round(pct, 1) if pct is not None else None,
            }
        )
    pcts = [float(value) for r in rows if isinstance(value := r.get("da_pct_rev"), (int, float))]
    avg = round(sum(pcts) / len(pcts), 1) if pcts else None
    for r in rows:
        r["avg"] = avg
    return rows


def _compute_capex_table_yf(income_stmt: pd.DataFrame, cashflow: pd.DataFrame) -> list[dict]:
    revenue_row = _get_row(income_stmt, "Total Revenue", "Operating Revenue")
    capex_row = _get_row(cashflow, "Capital Expenditure", "Capital Expenditures")
    if revenue_row is None or capex_row is None:
        return []

    dates = sorted(revenue_row.index, reverse=False)[-5:]
    rows = []
    for d in dates:
        rev = _safe_float(revenue_row.get(d))
        capex = _safe_float(capex_row.get(d))
        if capex is not None:
            capex = abs(capex)
        pct = (capex / rev * 100) if rev and capex and rev != 0 else None
        rows.append(
            {
                "fiscal_year": d.strftime("%b-%y") if hasattr(d, "strftime") else str(d),
                "revenue": rev,
                "capex": capex,
                "capex_pct_rev": round(pct, 1) if pct is not None else None,
            }
        )
    pcts = [float(value) for r in rows if isinstance(value := r.get("capex_pct_rev"), (int, float))]
    avg = round(sum(pcts) / len(pcts), 1) if pcts else None
    for r in rows:
        r["avg"] = avg
    return rows


def _compute_nwc_table_yf(
    balance_sheet: pd.DataFrame,
    income_stmt: pd.DataFrame,
) -> list[dict]:
    ca_row = _get_row(balance_sheet, "Current Assets")
    cl_row = _get_row(balance_sheet, "Current Liabilities")
    rev_row = _get_row(income_stmt, "Total Revenue", "Operating Revenue")
    if ca_row is None or cl_row is None:
        return []

    dates = sorted(ca_row.index, reverse=False)[-5:]
    rows = []
    for d in dates:
        ca = _safe_float(ca_row.get(d))
        cl = _safe_float(cl_row.get(d))
        nwc = (ca - cl) if ca is not None and cl is not None else None
        rev = _safe_float(rev_row.get(d)) if rev_row is not None else None
        pct = (nwc / rev * 100) if nwc is not None and rev and rev != 0 else None
        rows.append(
            {
                "fiscal_year": d.strftime("%b-%y") if hasattr(d, "strftime") else str(d),
                "revenue": rev,
                "nwc": nwc,
                "nwc_pct_rev": round(pct, 1) if pct is not None else None,
            }
        )
    pcts = [float(value) for r in rows if isinstance(value := r.get("nwc_pct_rev"), (int, float))]
    avg = round(sum(pcts) / len(pcts), 1) if pcts else None
    for r in rows:
        r["avg"] = avg
    return rows


def _compute_multiples_yf(
    quarterly_income_stmt: pd.DataFrame,
    quarterly_balance_sheet: pd.DataFrame,
    prices: pd.DataFrame,
    info: dict,
) -> tuple[list[dict], list[dict]]:
    """Fallback: compute multiples from yfinance quarterly data."""
    shares = info.get("sharesOutstanding")
    if not shares:
        return [], []

    rev_row = _get_row(quarterly_income_stmt, "Total Revenue", "Operating Revenue")
    ebitda_row = _get_row(quarterly_income_stmt, "EBITDA", "Normalized EBITDA")
    if ebitda_row is None:
        op_income = _get_row(quarterly_income_stmt, "Operating Income", "EBIT")
        da_q = _get_row(quarterly_income_stmt, "Reconciled Depreciation")
        if op_income is not None and da_q is not None:
            ebitda_row = op_income + da_q.reindex(op_income.index, fill_value=0)

    debt_row = _get_row(quarterly_balance_sheet, "Total Debt", "Long Term Debt")
    cash_row = _get_row(
        quarterly_balance_sheet,
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash Financial",
    )

    if rev_row is None:
        return [], []

    price_series = prices
    if isinstance(prices.columns, pd.MultiIndex):
        if "Close" in prices.columns.get_level_values(0):
            price_series = prices["Close"]
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]
        else:
            price_series = prices.iloc[:, 0]
    elif "Close" in prices.columns:
        price_series = prices["Close"]
    else:
        price_series = prices.iloc[:, 0]

    quarter_dates = sorted(rev_row.index, reverse=False)[-20:]
    ev_ebitda_rows: list[dict] = []
    ev_rev_rows: list[dict] = []

    for i, d in enumerate(quarter_dates):
        d_ts = pd.Timestamp(d)
        mask = price_series.index <= d_ts
        if not mask.any():
            continue
        close_price = float(price_series[mask].iloc[-1])
        market_cap = close_price * shares

        total_debt = _safe_float(debt_row.get(d)) if debt_row is not None else 0
        cash = _safe_float(cash_row.get(d)) if cash_row is not None else 0
        net_debt = (total_debt or 0) - (cash or 0)
        ev = market_cap + net_debt

        start_idx = max(0, i - 3)
        ttm_dates = quarter_dates[start_idx : i + 1]
        if len(ttm_dates) < 4:
            continue

        ttm_rev = sum(_safe_float(rev_row.get(qd)) or 0 for qd in ttm_dates)
        ttm_ebitda = sum(_safe_float(ebitda_row.get(qd)) or 0 for qd in ttm_dates) if ebitda_row is not None else 0

        label = d.strftime("%b-%y") if hasattr(d, "strftime") else str(d)

        if ttm_ebitda and ttm_ebitda > 0:
            ev_ebitda_rows.append(
                {
                    "quarter_end": label,
                    "ev": ev,
                    "ebitda_ttm": ttm_ebitda,
                    "ev_ebitda": round(ev / ttm_ebitda, 1),
                }
            )
        if ttm_rev and ttm_rev > 0:
            ev_rev_rows.append(
                {
                    "quarter_end": label,
                    "ev": ev,
                    "revenue_ttm": ttm_rev,
                    "ev_revenue": round(ev / ttm_rev, 1),
                }
            )

    if ev_ebitda_rows:
        avg = round(sum(r["ev_ebitda"] for r in ev_ebitda_rows) / len(ev_ebitda_rows), 1)
        for r in ev_ebitda_rows:
            r["avg"] = avg
    if ev_rev_rows:
        avg = round(sum(r["ev_revenue"] for r in ev_rev_rows) / len(ev_rev_rows), 1)
        for r in ev_rev_rows:
            r["avg"] = avg

    return ev_ebitda_rows, ev_rev_rows


# ---------------------------------------------------------------------------
# WACC
# ---------------------------------------------------------------------------


def _get_risk_free_rate() -> float:
    """Fetch 10Y Treasury yield. Try FRED first, then yfinance ^TNX."""
    try:
        fred_key = os.environ.get("FRED_API_KEY")
        if fred_key:
            from fredapi import Fred

            from utils.retry import fred_get_series

            fred = Fred(api_key=fred_key)
            s = fred_get_series(fred, "DGS10")
            if not s.empty:
                return float(s.dropna().iloc[-1]) / 100.0
    except Exception:
        logger.warning("FRED DGS10 fetch failed, falling back to ^TNX")

    try:
        tnx = yf_download("^TNX", period="5d", interval="1d")
        if isinstance(tnx.columns, pd.MultiIndex):
            close = tnx["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        elif "Close" in tnx.columns:
            close = tnx["Close"]
        else:
            close = tnx.iloc[:, 0]
        if not close.empty:
            return float(close.dropna().iloc[-1]) / 100.0
    except Exception:
        logger.warning("^TNX fetch failed, using 4.0%% default")

    return 0.04


def _compute_wacc(
    info: dict,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
) -> dict[str, Any]:
    """Compute WACC via CAPM."""
    beta = info.get("beta")
    if beta is None or np.isnan(beta):
        beta = DEFAULT_BETA
        beta_warning = True
    else:
        beta_warning = False

    rf = _get_risk_free_rate()
    cost_of_equity = rf + beta * EQUITY_RISK_PREMIUM

    # Cost of debt
    interest = _get_row(income_stmt, "Interest Expense", "Interest Expense Non Operating")
    total_debt_row = _get_row(balance_sheet, "Total Debt", "Long Term Debt")

    cost_of_debt = DEFAULT_COST_OF_DEBT
    debt_warning = True
    if interest is not None and total_debt_row is not None:
        latest_date = sorted(interest.index, reverse=True)[0] if len(interest.index) > 0 else None
        if latest_date is not None:
            ie = _safe_float(interest.get(latest_date))
            td = _safe_float(total_debt_row.get(latest_date))
            if ie is not None and td is not None and td > 0:
                cost_of_debt = abs(ie) / td
                debt_warning = False

    # Tax rate
    tax_provision = _get_row(income_stmt, "Tax Provision", "Income Tax Expense")
    pretax_income = _get_row(income_stmt, "Pretax Income", "Income Before Tax")
    tax_rate = DEFAULT_TAX_RATE
    if tax_provision is not None and pretax_income is not None:
        latest_date = sorted(tax_provision.index, reverse=True)[0]
        tp = _safe_float(tax_provision.get(latest_date))
        pi = _safe_float(pretax_income.get(latest_date))
        if tp is not None and pi is not None and pi > 0:
            tax_rate = tp / pi

    # Capital structure
    market_cap = info.get("marketCap", 0) or 0
    total_debt_val = 0.0
    if total_debt_row is not None:
        latest_date = sorted(total_debt_row.index, reverse=True)[0]
        td = _safe_float(total_debt_row.get(latest_date))
        total_debt_val = td if td else 0.0

    total_capital = market_cap + total_debt_val
    equity_weight = market_cap / total_capital if total_capital > 0 else 1.0
    debt_weight = total_debt_val / total_capital if total_capital > 0 else 0.0

    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)

    return {
        "beta": round(beta, 2),
        "beta_warning": beta_warning,
        "risk_free_rate": round(rf * 100, 2),
        "erp": round(EQUITY_RISK_PREMIUM * 100, 1),
        "cost_of_equity": round(cost_of_equity * 100, 2),
        "cost_of_debt": round(cost_of_debt * 100, 2),
        "debt_warning": debt_warning,
        "tax_rate": round(tax_rate * 100, 1),
        "equity_weight": round(equity_weight * 100, 1),
        "debt_weight": round(debt_weight * 100, 1),
        "wacc": round(wacc * 100, 2),
    }


# ---------------------------------------------------------------------------
# Historical averages (defaults for DCF tab)
# ---------------------------------------------------------------------------


def _compute_historical_averages(
    ebitda_table: list[dict],
    da_table: list[dict],
    nwc_table: list[dict],
    capex_table: list[dict],
) -> dict[str, float | None]:
    return {
        "ebitda_margin_avg": ebitda_table[0]["avg"] if ebitda_table else None,
        "da_pct_avg": da_table[0]["avg"] if da_table else None,
        "nwc_pct_avg": nwc_table[0]["avg"] if nwc_table else None,
        "capex_pct_avg": capex_table[0]["avg"] if capex_table else None,
    }


# ---------------------------------------------------------------------------
# Public API: get_historical_data
# ---------------------------------------------------------------------------


def get_historical_data(ticker: str) -> dict[str, Any]:
    """Fetch all historical data for the DCF Historical tab.

    Strategy: try EDGAR first for deeper history, fall back to yfinance.
    Always use yfinance for: info, prices, WACC inputs.
    """
    ticker = ticker.strip().upper()
    yf_data = _fetch_yfinance_data(ticker)
    info = yf_data["info"]

    # --- Annual tables: yfinance (reliable for ~5 years) ---
    ebitda_table = _compute_ebitda_table_yf(yf_data["income_stmt"], yf_data["cashflow"])
    da_table = _compute_da_table_yf(yf_data["income_stmt"], yf_data["cashflow"])
    capex_table = _compute_capex_table_yf(yf_data["income_stmt"], yf_data["cashflow"])
    nwc_table = _compute_nwc_table_yf(yf_data["balance_sheet"], yf_data["income_stmt"])

    # --- Quarterly multiples: EDGAR for 20Q depth, yfinance fallback ---
    data_source = "yfinance"
    ev_ebitda: list[dict] = []
    ev_revenue: list[dict] = []
    try:
        edgar = extract_dcf_historicals(ticker)
        if edgar and edgar.get("quarterly_revenue"):
            data_source = "edgar"
            q_rev = edgar["quarterly_revenue"]
            q_ebitda = edgar.get("quarterly_ebitda") or []
            q_oi = edgar.get("quarterly_operating_income") or []
            q_da = edgar.get("quarterly_da") or []
            q_debt = edgar.get("quarterly_total_debt") or []
            q_cd = edgar.get("quarterly_current_debt") or []
            q_cash = edgar.get("quarterly_cash") or []

            ev_ebitda, ev_revenue = _build_quarterly_multiples_edgar(
                q_rev,
                q_ebitda or None,
                q_oi or None,
                q_da or None,
                q_debt,
                q_cd,
                q_cash,
                yf_data["prices"],
                info.get("sharesOutstanding"),
            )
    except Exception:
        logger.warning("EDGAR fetch failed for %s, using yfinance for multiples", ticker)

    if data_source == "yfinance" or not ev_ebitda or not ev_revenue:
        yf_ev_ebitda, yf_ev_revenue = _compute_multiples_yf(
            yf_data["quarterly_income_stmt"],
            yf_data["quarterly_balance_sheet"],
            yf_data["prices"],
            info,
        )
        if data_source == "yfinance":
            ev_ebitda, ev_revenue = yf_ev_ebitda, yf_ev_revenue
        else:
            if not ev_ebitda:
                ev_ebitda = yf_ev_ebitda
            if not ev_revenue:
                ev_revenue = yf_ev_revenue
            if not ev_ebitda and not ev_revenue:
                data_source = "yfinance"

    # WACC (always from yfinance — needs beta, market cap, etc.)
    wacc_inputs = _compute_wacc(info, yf_data["income_stmt"], yf_data["balance_sheet"])
    historical_averages = _compute_historical_averages(ebitda_table, da_table, nwc_table, capex_table)

    # Net debt (from yfinance for most current values)
    debt_row = _get_row(yf_data["balance_sheet"], "Total Debt", "Long Term Debt")
    cash_row = _get_row(
        yf_data["balance_sheet"],
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash Financial",
    )
    net_debt = 0.0
    if debt_row is not None and cash_row is not None:
        latest = sorted(debt_row.index, reverse=True)[0]
        d = _safe_float(debt_row.get(latest)) or 0
        c = _safe_float(cash_row.get(latest)) or 0
        net_debt = d - c

    # Base revenue (most recent annual)
    rev_row = _get_row(yf_data["income_stmt"], "Total Revenue", "Operating Revenue")
    base_revenue = None
    if rev_row is not None:
        latest = sorted(rev_row.index, reverse=True)[0]
        base_revenue = _safe_float(rev_row.get(latest))

    return {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "net_debt": net_debt,
        "base_revenue": base_revenue,
        "data_source": data_source,
        "ebitda": ebitda_table,
        "depreciation": da_table,
        "capex": capex_table,
        "nwc": nwc_table,
        "ev_ebitda": ev_ebitda,
        "rev_multiple": ev_revenue,
        "wacc_inputs": wacc_inputs,
        "historical_averages": historical_averages,
    }


# ---------------------------------------------------------------------------
# Public API: run_valuation
# ---------------------------------------------------------------------------


def run_valuation(ticker: str, assumptions: dict[str, Any]) -> dict[str, Any]:
    """Run full DCF valuation with user-provided assumptions."""
    ticker = ticker.strip().upper()
    yf_data = _fetch_yfinance_data(ticker)
    info = yf_data["info"]

    # Base revenue
    rev_row = _get_row(yf_data["income_stmt"], "Total Revenue", "Operating Revenue")
    if rev_row is None:
        raise ValueError(f"No revenue data available for {ticker}")
    latest = sorted(rev_row.index, reverse=True)[0]
    base_revenue = _safe_float(rev_row.get(latest))
    if not base_revenue:
        raise ValueError(f"No revenue data available for {ticker}")

    base_year_label = latest.strftime("%b-%y") if hasattr(latest, "strftime") else str(latest)

    raw_growth_rates = assumptions["revenue_growth_rates"]
    if not isinstance(raw_growth_rates, (list, tuple, pd.Series, np.ndarray)):
        raise ValueError("revenue_growth_rates must be a list")
    projection_years = len(raw_growth_rates)
    growth_rates = _normalize_numeric_series(raw_growth_rates, "revenue_growth_rates", projection_years)
    projection_years = len(growth_rates)
    if projection_years < 5 or projection_years > 8:
        raise ValueError("revenue_growth_rates must contain between 5 and 8 values")

    ebitda_margins = _normalize_numeric_series(
        assumptions["ebitda_margin"],
        "ebitda_margin",
        projection_years,
        min_value=0,
        max_value=1,
        min_inclusive=False,
        max_inclusive=False,
    )
    tax_rates = _normalize_numeric_series(
        assumptions["tax_rate"],
        "tax_rate",
        projection_years,
        min_value=0,
        max_value=1,
        max_inclusive=False,
    )
    da_pcts = _normalize_numeric_series(
        assumptions["da_pct_revenue"],
        "da_pct_revenue",
        projection_years,
        min_value=0,
        max_value=1,
        max_inclusive=False,
    )
    nwc_pcts = _normalize_numeric_series(
        assumptions["nwc_pct_revenue"],
        "nwc_pct_revenue",
        projection_years,
        min_value=-1,
        max_value=1,
    )
    capex_pcts = _normalize_numeric_series(
        assumptions["capex_pct_revenue"],
        "capex_pct_revenue",
        projection_years,
        min_value=0,
        max_value=1,
        max_inclusive=False,
    )
    wacc = _safe_float(assumptions["wacc"])
    if wacc is None or wacc <= 0 or wacc >= 1:
        raise ValueError("wacc must be > 0 and < 1")

    # Build projection
    projection = []
    prev_revenue = base_revenue
    prev_nwc = prev_revenue * nwc_pcts[0]

    for year in range(projection_years):
        growth_rate = growth_rates[year]
        ebitda_margin = ebitda_margins[year]
        tax_rate = tax_rates[year]
        da_pct = da_pcts[year]
        nwc_pct = nwc_pcts[year]
        capex_pct = capex_pcts[year]

        revenue = prev_revenue * (1 + growth_rate)
        ebitda = revenue * ebitda_margin
        da = revenue * da_pct
        ebit = ebitda - da
        taxes = ebit * tax_rate
        nopat = ebit - taxes
        capex = revenue * capex_pct
        current_nwc = revenue * nwc_pct
        delta_nwc = current_nwc - prev_nwc
        ufcf = nopat + da - capex - delta_nwc
        discount_factor = 1 / ((1 + wacc) ** (year + 1))
        pv_ufcf = ufcf * discount_factor

        projection.append(
            {
                "year": year + 1,
                "year_label": f"Year {year + 1}",
                "revenue": revenue,
                "revenue_growth": growth_rate * 100,
                "ebitda": ebitda,
                "ebitda_margin": ebitda_margin * 100,
                "da": da,
                "ebit": ebit,
                "tax_rate": tax_rate * 100,
                "nopat": nopat,
                "capex": capex,
                "nwc": current_nwc,
                "delta_nwc": delta_nwc,
                "ufcf": ufcf,
                "discount_rate": wacc * 100,
                "pv_ufcf": pv_ufcf,
            }
        )

        prev_revenue = revenue
        prev_nwc = current_nwc

    # Sum of PV of UFCFs
    pv_fcfs = sum(p["pv_ufcf"] for p in projection)
    terminal_year_ufcf = projection[-1]["ufcf"]
    terminal_year_ebitda = projection[-1]["ebitda"]
    terminal_year_revenue = projection[-1]["revenue"]

    # Net debt & shares
    debt_row = _get_row(yf_data["balance_sheet"], "Total Debt", "Long Term Debt")
    cash_row = _get_row(
        yf_data["balance_sheet"],
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash Financial",
    )
    net_debt = 0.0
    if debt_row is not None and cash_row is not None:
        ld = sorted(debt_row.index, reverse=True)[0]
        d = _safe_float(debt_row.get(ld)) or 0
        c = _safe_float(cash_row.get(ld)) or 0
        net_debt = d - c

    shares = info.get("sharesOutstanding", 1) or 1
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    # Terminal growth rates
    tgr = assumptions.get("terminal_growth_rates", {})
    gordon_rates = {
        "bear": tgr.get("bear", 0.02),
        "base": tgr.get("base", 0.03),
        "bull": tgr.get("bull", 0.04),
    }

    exit_ebitda = assumptions.get("exit_ev_ebitda", {})
    exit_rev = assumptions.get("exit_ev_revenue", {})

    def _gordon(g: float) -> dict | None:
        if wacc <= g:
            return {"error": f"WACC ({wacc * 100:.1f}%) must exceed growth rate ({g * 100:.1f}%)"}
        tv = terminal_year_ufcf * (1 + g) / (wacc - g)
        pv_tv = tv / ((1 + wacc) ** projection_years)
        ev = pv_fcfs + pv_tv
        equity = ev - net_debt
        per_share = equity / shares
        upside = ((per_share / current_price) - 1) * 100 if current_price else None
        return {
            "terminal_value": tv,
            "pv_terminal_value": pv_tv,
            "enterprise_value": ev,
            "equity_value": equity,
            "per_share": round(per_share, 2),
            "upside": round(upside, 1) if upside is not None else None,
        }

    def _exit_multiple(terminal_metric: float, multiple: float) -> dict:
        tv = terminal_metric * multiple
        pv_tv = tv / ((1 + wacc) ** projection_years)
        ev = pv_fcfs + pv_tv
        equity = ev - net_debt
        per_share = equity / shares
        upside = ((per_share / current_price) - 1) * 100 if current_price else None
        return {
            "terminal_value": tv,
            "pv_terminal_value": pv_tv,
            "enterprise_value": ev,
            "equity_value": equity,
            "per_share": round(per_share, 2),
            "upside": round(upside, 1) if upside is not None else None,
        }

    valuations = {
        "gordon_growth": {scenario: _gordon(rate) for scenario, rate in gordon_rates.items()},
        "ev_ebitda_exit": {
            scenario: _exit_multiple(terminal_year_ebitda, mult) for scenario, mult in exit_ebitda.items()
        },
        "ev_revenue_exit": {
            scenario: _exit_multiple(terminal_year_revenue, mult) for scenario, mult in exit_rev.items()
        },
    }

    return {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "current_price": current_price,
        "shares_outstanding": shares,
        "net_debt": net_debt,
        "base_revenue": base_revenue,
        "base_year": base_year_label,
        "pv_fcfs": pv_fcfs,
        "projection": projection,
        "valuations": valuations,
        "assumptions_used": assumptions,
    }
