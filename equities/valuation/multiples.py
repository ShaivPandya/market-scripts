"""Shared equity valuation multiples service.

This module intentionally returns snapshot-shaped dictionaries but does not
persist metric snapshots. The only persisted state is the optional per-ticker
profile override.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from utils.retry import yf_download, yf_ticker_info

LOGGER = logging.getLogger(__name__)

VALUATION_COLUMNS = (
    "price_sales",
    "price_operating_income",
    "price_fcf",
    "price_earnings",
    "price_book",
)
ENTERPRISE_VALUE_METRICS = {"price_sales", "price_operating_income", "price_fcf"}

VALUATION_LABELS = {
    "price_sales": "EV/S",
    "price_operating_income": "EV/EBIT",
    "price_fcf": "EV/FCF",
    "price_earnings": "P/E",
    "price_book": "P/B",
}

VALUATION_PERIODS = {
    "price_sales": "TTM",
    "price_operating_income": "TTM",
    "price_fcf": "TTM",
    "price_earnings": "TTM",
    "price_book": "MRQ/latest",
}

DENOMINATOR_LABELS = {
    "price_sales": "TTM revenue",
    "price_operating_income": "TTM EBIT",
    "price_fcf": "TTM free cash flow",
    "price_earnings": "TTM net income",
    "price_book": "latest book value",
}
NUMERATOR_LABELS = {
    "price_sales": "enterprise value",
    "price_operating_income": "enterprise value",
    "price_fcf": "enterprise value",
    "price_earnings": "market capitalization",
    "price_book": "market capitalization",
}

DEFAULT_PROFILE_ID = "general_equity"
AUTO_PROFILE_ID = "auto"


@dataclass(frozen=True)
class ValuationProfile:
    id: str
    label: str
    weights: dict[str, float]
    rationale: str


VALUATION_PROFILES: dict[str, ValuationProfile] = {
    "general_equity": ValuationProfile(
        id="general_equity",
        label="General Equity",
        weights={
            "price_sales": 0.15,
            "price_operating_income": 0.25,
            "price_fcf": 0.30,
            "price_earnings": 0.20,
            "price_book": 0.10,
        },
        rationale="Balanced operating, cash-flow, earnings, sales, and balance-sheet valuation.",
    ),
    "bank_financial": ValuationProfile(
        id="bank_financial",
        label="Bank / Financial",
        weights={
            "price_sales": 0.05,
            "price_operating_income": 0.10,
            "price_fcf": 0.05,
            "price_earnings": 0.30,
            "price_book": 0.50,
        },
        rationale="Banks and balance-sheet lenders are commonly judged against book value and earnings power.",
    ),
    "insurance_asset_manager": ValuationProfile(
        id="insurance_asset_manager",
        label="Insurance / Asset Manager",
        weights={
            "price_sales": 0.05,
            "price_operating_income": 0.20,
            "price_fcf": 0.15,
            "price_earnings": 0.35,
            "price_book": 0.25,
        },
        rationale="Financial franchises need earnings and book-value context, with operating income as a cross-check.",
    ),
    "high_growth_software_saas": ValuationProfile(
        id="high_growth_software_saas",
        label="High-Growth Software / SaaS",
        weights={
            "price_sales": 0.45,
            "price_operating_income": 0.10,
            "price_fcf": 0.35,
            "price_earnings": 0.05,
            "price_book": 0.05,
        },
        rationale="High-growth software often has reinvestment-suppressed earnings, so sales and free cash flow matter more.",
    ),
    "mature_software": ValuationProfile(
        id="mature_software",
        label="Mature Software",
        weights={
            "price_sales": 0.20,
            "price_operating_income": 0.25,
            "price_fcf": 0.35,
            "price_earnings": 0.15,
            "price_book": 0.05,
        },
        rationale="Mature software is best cross-checked with cash-flow conversion and operating profitability.",
    ),
    "capital_intensive_cyclical": ValuationProfile(
        id="capital_intensive_cyclical",
        label="Capital-Intensive / Cyclical",
        weights={
            "price_sales": 0.10,
            "price_operating_income": 0.30,
            "price_fcf": 0.25,
            "price_earnings": 0.25,
            "price_book": 0.10,
        },
        rationale="Cyclical and capital-intensive businesses need operating-profit and cash-flow valuation across the cycle.",
    ),
}

SECTOR_ETFS = {
    "basic materials": "XLB",
    "communication services": "XLC",
    "consumer cyclical": "XLY",
    "consumer defensive": "XLP",
    "consumer discretionary": "XLY",
    "consumer staples": "XLP",
    "energy": "XLE",
    "financial services": "XLF",
    "financials": "XLF",
    "healthcare": "XLV",
    "health care": "XLV",
    "industrials": "XLI",
    "real estate": "XLRE",
    "technology": "XLK",
    "information technology": "XLK",
    "utilities": "XLU",
}

PROFILE_OVERRIDE_LOCAL_PATH = Path("data_cache/valuation/profile_overrides.json")
PROFILE_OVERRIDE_GCS_KEY = "live/valuation/profile_overrides.json"

REVENUE_KEYS = (
    "Total Revenue",
    "TotalRevenue",
    "Operating Revenue",
    "OperatingRevenue",
    "Revenue",
    "Revenues",
)
OPERATING_INCOME_KEYS = (
    "EBIT",
    "Operating Income",
    "OperatingIncome",
    "Operating Income Loss",
    "OperatingIncomeLoss",
    "Income From Operations",
    "IncomeLossFromOperations",
)
NET_INCOME_KEYS = (
    "Net Income",
    "NetIncome",
    "Net Income Common Stockholders",
    "NetIncomeCommonStockholders",
    "Net Income Continuous Operations",
    "NetIncomeContinuousOperations",
    "NetIncomeLoss",
)
OPERATING_CASH_FLOW_KEYS = (
    "Operating Cash Flow",
    "OperatingCashFlow",
    "Total Cash From Operating Activities",
    "Net Cash Provided By Operating Activities",
)
CAPEX_KEYS = (
    "Capital Expenditure",
    "CapitalExpenditure",
    "Capital Expenditures",
    "PaymentsToAcquirePropertyPlantAndEquipment",
)
BOOK_VALUE_KEYS = (
    "Stockholders Equity",
    "StockholdersEquity",
    "Total Stockholder Equity",
    "Total Equity Gross Minority Interest",
    "Common Stock Equity",
    "CommonStocksIncludingAdditionalPaidInCapital",
)
TOTAL_DEBT_KEYS = (
    "Total Debt",
    "TotalDebt",
    "Long Term Debt And Capital Lease Obligation",
    "LongTermDebtAndCapitalLeaseObligation",
    "Long Term Debt",
    "LongTermDebt",
)
CASH_KEYS = (
    "Cash And Cash Equivalents",
    "CashAndCashEquivalents",
    "Cash Cash Equivalents And Short Term Investments",
    "CashCashEquivalentsAndShortTermInvestments",
    "Cash Financial",
    "Cash",
)


def profile_options() -> list[dict[str, str]]:
    return [{"id": key, "label": value.label} for key, value in VALUATION_PROFILES.items()]


def normalize_profile_id(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text == AUTO_PROFILE_ID:
        return None
    return text if text in VALUATION_PROFILES else None


def resolve_profile(info: Mapping[str, Any] | None, override_profile_id: str | None = None) -> dict[str, Any]:
    override = normalize_profile_id(override_profile_id)
    if override:
        profile = VALUATION_PROFILES[override]
        return _profile_payload(profile, selection_mode="override")

    sector = str((info or {}).get("sector") or "").strip().lower()
    industry = str((info or {}).get("industry") or "").strip().lower()
    combined = f"{sector} {industry}"
    revenue_growth = _safe_float((info or {}).get("revenueGrowth"))

    profile_id = DEFAULT_PROFILE_ID
    if "bank" in combined or "credit" in combined or "mortgage" in combined:
        profile_id = "bank_financial"
    elif any(token in combined for token in ("insurance", "asset management", "capital markets", "broker")):
        profile_id = "insurance_asset_manager"
    elif any(token in combined for token in ("software", "saas", "cloud", "application")):
        profile_id = (
            "high_growth_software_saas" if revenue_growth is not None and revenue_growth >= 0.15 else "mature_software"
        )
    elif sector in {"energy", "utilities", "basic materials", "industrials"} or any(
        token in combined
        for token in (
            "airline",
            "automobile",
            "steel",
            "chemical",
            "semiconductor",
            "manufacturing",
            "construction",
            "machinery",
            "mining",
        )
    ):
        profile_id = "capital_intensive_cyclical"

    return _profile_payload(VALUATION_PROFILES[profile_id], selection_mode="auto")


def _profile_payload(profile: ValuationProfile, *, selection_mode: str) -> dict[str, Any]:
    return {
        "id": profile.id,
        "label": profile.label,
        "selection_mode": selection_mode,
        "weights": dict(profile.weights),
        "rationale": profile.rationale,
    }


def read_profile_override(ticker: str) -> str | None:
    overrides = _read_profile_overrides()
    return normalize_profile_id(overrides.get(_clean_ticker(ticker)))


def write_profile_override(ticker: str, profile_id: str | None) -> dict[str, Any]:
    normalized_ticker = _clean_ticker(ticker)
    if not normalized_ticker:
        raise ValueError("Ticker is required")

    normalized_profile = normalize_profile_id(profile_id)
    if profile_id and not normalized_profile:
        raise ValueError(f"Unsupported valuation profile: {profile_id}")

    overrides = _read_profile_overrides()
    if normalized_profile:
        overrides[normalized_ticker] = normalized_profile
    else:
        overrides.pop(normalized_ticker, None)
    _write_profile_overrides(overrides)
    return {"ticker": normalized_ticker, "profile_override": normalized_profile}


def _read_profile_overrides() -> dict[str, str]:
    try:
        from api.state_storage import exists_text, read_text
        from paths import PROJECT_ROOT

        path = PROJECT_ROOT / PROFILE_OVERRIDE_LOCAL_PATH
        if not exists_text(path, PROFILE_OVERRIDE_GCS_KEY):
            return {}
        raw = json.loads(read_text(path, PROFILE_OVERRIDE_GCS_KEY, encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {
            _clean_ticker(key): profile
            for key, value in raw.items()
            if (profile := normalize_profile_id(value)) and _clean_ticker(key)
        }
    except Exception:
        LOGGER.debug("Failed to read valuation profile overrides", exc_info=True)
        return {}


def _write_profile_overrides(overrides: Mapping[str, str]) -> None:
    from api.state_storage import write_text
    from paths import PROJECT_ROOT

    clean = {
        ticker: profile
        for key, value in overrides.items()
        if (ticker := _clean_ticker(key)) and (profile := normalize_profile_id(value))
    }
    path = PROJECT_ROOT / PROFILE_OVERRIDE_LOCAL_PATH
    write_text(
        path,
        PROFILE_OVERRIDE_GCS_KEY,
        json.dumps(dict(sorted(clean.items())), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        content_type="application/json",
    )


def get_position_valuation(ticker: str, *, include_peers: bool = True, include_history: bool = True) -> dict[str, Any]:
    normalized = _clean_ticker(ticker)
    if not normalized:
        raise ValueError("Ticker is required")

    info = _fetch_info(normalized)
    override = read_profile_override(normalized)
    current = fetch_current_valuation(normalized, info=info)
    profile = resolve_profile(info, override)
    effective_weights = effective_profile_weights(profile["weights"], current["metrics"])
    peers = peer_context(normalized, info, current["metrics"]) if include_peers else _empty_peer_context()
    history = historical_bands(normalized, current["metrics"]) if include_history else {}
    composite_score = composite_valuation_score(current["metrics"], peers, effective_weights)

    return {
        "ticker": normalized,
        "company_name": info.get("longName") or info.get("shortName") or normalized,
        "as_of": datetime.now(UTC).isoformat(),
        "source_policy": "free_providers",
        "market_data": {
            "market_cap": current.get("market_cap"),
            "enterprise_value": current.get("enterprise_value"),
            "net_debt": current.get("net_debt"),
            "currency": info.get("currency"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        },
        "profile": {
            **profile,
            "override_profile_id": override,
            "effective_weights": effective_weights,
            "options": profile_options(),
        },
        "metrics": current["metrics"],
        "peer_context": peers,
        "historical_bands": history,
        "composite_score": composite_score,
        "data_quality": valuation_data_quality(current["metrics"], peers),
    }


def fetch_current_valuation(ticker: str, *, info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized = _clean_ticker(ticker)
    info_dict = dict(info or _fetch_info(normalized))
    ticker_obj = yf.Ticker(normalized)

    quarterly_income = _get_yf_statement(ticker_obj, ("quarterly_income_stmt", "quarterly_financials"))
    annual_income = _get_yf_statement(ticker_obj, ("income_stmt", "financials"))
    quarterly_cashflow = _get_yf_statement(ticker_obj, ("quarterly_cashflow", "quarterly_cash_flow"))
    annual_cashflow = _get_yf_statement(ticker_obj, ("cashflow", "cash_flow"))
    quarterly_balance = _get_yf_statement(ticker_obj, ("quarterly_balance_sheet", "quarterly_balancesheet"))
    annual_balance = _get_yf_statement(ticker_obj, ("balance_sheet", "balancesheet"))

    return compute_current_multiples_from_statements(
        info_dict,
        quarterly_income=quarterly_income,
        annual_income=annual_income,
        quarterly_cashflow=quarterly_cashflow,
        annual_cashflow=annual_cashflow,
        quarterly_balance=quarterly_balance,
        annual_balance=annual_balance,
    )


def compute_current_multiples_from_statements(
    info: Mapping[str, Any],
    *,
    quarterly_income: pd.DataFrame | None = None,
    annual_income: pd.DataFrame | None = None,
    quarterly_cashflow: pd.DataFrame | None = None,
    annual_cashflow: pd.DataFrame | None = None,
    quarterly_balance: pd.DataFrame | None = None,
    annual_balance: pd.DataFrame | None = None,
) -> dict[str, Any]:
    market_cap = _market_cap(info)
    enterprise_value_payload = _enterprise_value(
        info,
        market_cap=market_cap,
        quarterly_balance=quarterly_balance,
        annual_balance=annual_balance,
    )
    enterprise_value = enterprise_value_payload["value"]
    revenue_ttm = _ttm_or_latest(quarterly_income, annual_income, REVENUE_KEYS)
    operating_income_ttm = _ttm_or_latest(quarterly_income, annual_income, OPERATING_INCOME_KEYS)
    net_income_ttm = _ttm_or_latest(quarterly_income, annual_income, NET_INCOME_KEYS)
    operating_cash_flow_ttm = _ttm_or_latest(quarterly_cashflow, annual_cashflow, OPERATING_CASH_FLOW_KEYS)
    capex_ttm = _ttm_or_latest(quarterly_cashflow, annual_cashflow, CAPEX_KEYS)
    fcf_ttm = _free_cash_flow(operating_cash_flow_ttm, capex_ttm)
    book_value = _latest_statement_value(quarterly_balance, BOOK_VALUE_KEYS)
    if book_value is None:
        book_value = _latest_statement_value(annual_balance, BOOK_VALUE_KEYS)

    metrics = {
        "price_sales": _metric_payload(
            "price_sales",
            enterprise_value,
            revenue_ttm,
            source="yfinance_statements",
            numerator_source=enterprise_value_payload["source"],
            numerator_degraded=enterprise_value_payload["degraded"],
            numerator_reason=enterprise_value_payload["reason"],
        ),
        "price_operating_income": _metric_payload(
            "price_operating_income",
            enterprise_value,
            operating_income_ttm,
            source="yfinance_statements",
            numerator_source=enterprise_value_payload["source"],
            numerator_degraded=enterprise_value_payload["degraded"],
            numerator_reason=enterprise_value_payload["reason"],
        ),
        "price_fcf": _metric_payload(
            "price_fcf",
            enterprise_value,
            fcf_ttm,
            source="yfinance_statements",
            numerator_source=enterprise_value_payload["source"],
            numerator_degraded=enterprise_value_payload["degraded"],
            numerator_reason=enterprise_value_payload["reason"],
        ),
        "price_earnings": _metric_payload(
            "price_earnings",
            market_cap,
            net_income_ttm,
            source="yfinance_statements",
            numerator_source="market_cap",
            fallback_value=_positive_float(info.get("trailingPE")),
            fallback_source="yfinance_info.trailingPE",
        ),
        "price_book": _metric_payload(
            "price_book",
            market_cap,
            book_value,
            source="yfinance_balance_sheet",
            numerator_source="market_cap",
            fallback_value=_positive_float(info.get("priceToBook")),
            fallback_source="yfinance_info.priceToBook",
        ),
    }
    return {
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "net_debt": enterprise_value_payload["net_debt"],
        "metrics": metrics,
    }


def _metric_payload(
    key: str,
    numerator: float | None,
    denominator: float | None,
    *,
    source: str,
    numerator_source: str | None = None,
    numerator_degraded: bool = False,
    numerator_reason: str | None = None,
    fallback_value: float | None = None,
    fallback_source: str | None = None,
) -> dict[str, Any]:
    value = _multiple(numerator, denominator)
    status = "ok" if value is not None else "missing"
    reason = None

    if numerator is None or numerator <= 0:
        status = "missing"
        reason = "missing_enterprise_value" if key in ENTERPRISE_VALUE_METRICS else "missing_market_cap"
    elif denominator is None:
        status = "missing"
        reason = "missing_denominator"
    elif denominator <= 0:
        status = "not_meaningful"
        reason = "non_positive_denominator"
    elif numerator_degraded:
        status = "degraded"
        reason = numerator_reason or "using_degraded_numerator"

    if value is None and fallback_value is not None and fallback_value > 0:
        value = fallback_value
        status = "degraded"
        reason = "using_provider_ratio_fallback"
        source = fallback_source or source

    return {
        "key": key,
        "label": VALUATION_LABELS[key],
        "value": value,
        "period": VALUATION_PERIODS[key],
        "numerator": numerator,
        "numerator_label": NUMERATOR_LABELS[key],
        "numerator_source": numerator_source,
        "denominator": denominator,
        "denominator_label": DENOMINATOR_LABELS[key],
        "status": status,
        "reason": reason,
        "source": source,
    }


def fetch_valuation_metrics_batch(tickers: Sequence[str], *, max_workers: int | None = None) -> pd.DataFrame:
    clean_tickers = list(dict.fromkeys(_clean_ticker(ticker) for ticker in tickers if _clean_ticker(ticker)))
    columns = [
        *VALUATION_COLUMNS,
        *(f"{key}_profile_weight" for key in VALUATION_COLUMNS),
        "valuation_profile_id",
        "sector",
        "industry",
    ]
    if not clean_tickers:
        return pd.DataFrame(columns=columns)

    rows: dict[str, dict[str, Any]] = {}
    workers = max_workers or min(6, len(clean_tickers))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_batch_row, ticker): ticker for ticker in clean_tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                rows[ticker] = future.result()
            except Exception as exc:
                LOGGER.warning("%s: valuation batch fetch failed (%s)", ticker, exc)

    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=clean_tickers, columns=columns)


def _batch_row(ticker: str) -> dict[str, Any]:
    info = _fetch_info(ticker)
    current = fetch_current_valuation(ticker, info=info)
    profile = resolve_profile(info, read_profile_override(ticker))
    effective = effective_profile_weights(profile["weights"], current["metrics"])
    row = {key: current["metrics"].get(key, {}).get("value") for key in VALUATION_COLUMNS}
    row.update({f"{key}_profile_weight": effective.get(key, 0.0) for key in VALUATION_COLUMNS})
    row["valuation_profile_id"] = profile["id"]
    row["sector"] = info.get("sector")
    row["industry"] = info.get("industry")
    return row


def effective_profile_weights(
    profile_weights: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for key in VALUATION_COLUMNS:
        metric = metrics.get(key) or {}
        if metric.get("status") not in {"ok", "degraded"}:
            raw[key] = 0.0
            continue
        value = _safe_float(metric.get("value"))
        if value is None or value <= 0:
            raw[key] = 0.0
            continue
        raw[key] = max(0.0, _safe_float(profile_weights.get(key)) or 0.0)
    total = sum(raw.values())
    if total <= 0:
        return {key: 0.0 for key in VALUATION_COLUMNS}
    return {key: round(value / total, 6) for key, value in raw.items()}


def peer_context(
    ticker: str,
    info: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    min_valid_peers: int = 5,
    min_industry_peers: int = 8,
    max_peer_fetch: int = 30,
) -> dict[str, Any]:
    peers, peer_source = resolve_peer_universe(
        ticker,
        info,
        min_industry_peers=min_industry_peers,
        max_peer_fetch=max_peer_fetch,
    )
    if not peers:
        return _empty_peer_context(source=peer_source)

    peer_df = fetch_valuation_metrics_batch(peers, max_workers=min(6, len(peers)))
    metric_stats: dict[str, Any] = {}
    for key in VALUATION_COLUMNS:
        current_value = _safe_float((metrics.get(key) or {}).get("value"))
        values = pd.to_numeric(peer_df.get(key), errors="coerce").dropna()
        values = values[values > 0].astype(float)
        if current_value is None or current_value <= 0 or len(values) < min_valid_peers:
            metric_stats[key] = {
                "status": "insufficient_peers",
                "sample_size": int(len(values)),
                "percentile": None,
                "rank": None,
                "median": None,
                "q1": None,
                "q3": None,
            }
            continue

        sorted_values = values.sort_values()
        rank = int((sorted_values < current_value).sum()) + 1
        percentile = 100.0 * (1.0 - ((rank - 1) / max(len(sorted_values) - 1, 1)))
        metric_stats[key] = {
            "status": "ok",
            "sample_size": int(len(sorted_values)),
            "percentile": round(percentile, 1),
            "rank": rank,
            "median": _round_float(sorted_values.median()),
            "q1": _round_float(sorted_values.quantile(0.25)),
            "q3": _round_float(sorted_values.quantile(0.75)),
        }

    return {
        "source": peer_source,
        "peer_count": len(peers),
        "peers": peers,
        "metric_stats": metric_stats,
    }


def resolve_peer_universe(
    ticker: str,
    info: Mapping[str, Any],
    *,
    min_industry_peers: int = 8,
    max_peer_fetch: int = 30,
) -> tuple[list[str], str]:
    normalized = _clean_ticker(ticker)
    sector = str(info.get("sector") or "").strip().lower()
    industry = str(info.get("industry") or "").strip().lower()
    etf = SECTOR_ETFS.get(sector)
    sector_peers: list[str] = []

    if etf:
        try:
            from equities.common.universe_loader import get_etf_holdings

            sector_peers = [
                peer
                for peer in get_etf_holdings(etf)
                if peer and _clean_ticker(peer) != normalized and not _clean_ticker(peer).endswith("=F")
            ][: max_peer_fetch * 2]
        except Exception:
            LOGGER.debug("%s: failed to load sector ETF peers", normalized, exc_info=True)

    if sector_peers and industry:
        industry_peers = []
        for peer in sector_peers[: max_peer_fetch * 2]:
            peer_info = _fetch_info(peer)
            peer_industry = str(peer_info.get("industry") or "").strip().lower()
            if peer_industry and peer_industry == industry:
                industry_peers.append(_clean_ticker(peer))
            if len(industry_peers) >= max_peer_fetch:
                break
        if len(industry_peers) >= min_industry_peers:
            return industry_peers[:max_peer_fetch], "same_industry_sector_etf"

    if sector_peers:
        return [_clean_ticker(peer) for peer in sector_peers[:max_peer_fetch]], "sector_etf"

    try:
        from ontology.runtime_read_service import OntologyRuntimeReadService

        portfolio_peers = []
        for row in OntologyRuntimeReadService().positions():
            peer = _clean_ticker(row.get("ticker"))
            if peer and peer != normalized and str(row.get("asset") or "").lower() == "equity":
                portfolio_peers.append(peer)
        return portfolio_peers[:max_peer_fetch], "portfolio_equity"
    except Exception:
        return [], "unavailable"


def historical_bands(ticker: str, metrics: Mapping[str, Mapping[str, Any]], *, min_periods: int = 6) -> dict[str, Any]:
    normalized = _clean_ticker(ticker)
    try:
        ticker_obj = yf.Ticker(normalized)
        info = _fetch_info(normalized)
        shares = _positive_float(info.get("sharesOutstanding"))
        if shares is None:
            return {}
        prices = yf_download(normalized, period="6y", interval="1d", progress=False)
        price_series = _close_series(prices, normalized)
        if price_series is None or price_series.dropna().empty:
            return {}
        quarterly_income = _get_yf_statement(ticker_obj, ("quarterly_income_stmt", "quarterly_financials"))
        quarterly_cashflow = _get_yf_statement(ticker_obj, ("quarterly_cashflow", "quarterly_cash_flow"))
        quarterly_balance = _get_yf_statement(ticker_obj, ("quarterly_balance_sheet", "quarterly_balancesheet"))
        return _historical_bands_from_statements(
            metrics,
            prices=price_series,
            shares=shares,
            quarterly_income=quarterly_income,
            quarterly_cashflow=quarterly_cashflow,
            quarterly_balance=quarterly_balance,
            min_periods=min_periods,
        )
    except Exception:
        LOGGER.debug("%s: failed to compute valuation historical bands", normalized, exc_info=True)
        return {}


def _historical_bands_from_statements(
    current_metrics: Mapping[str, Mapping[str, Any]],
    *,
    prices: pd.Series,
    shares: float,
    quarterly_income: pd.DataFrame | None,
    quarterly_cashflow: pd.DataFrame | None,
    quarterly_balance: pd.DataFrame | None,
    min_periods: int,
) -> dict[str, Any]:
    dates = _statement_dates(quarterly_income)
    if len(dates) < 4:
        return {}

    series_by_metric: dict[str, list[float]] = {key: [] for key in VALUATION_COLUMNS}
    for idx, period_date in enumerate(dates):
        if idx + 4 > len(dates):
            continue
        px = _price_on_or_before(prices, period_date)
        if px is None or px <= 0:
            continue
        market_cap = px * shares
        revenue = _rolling_statement_sum(quarterly_income, REVENUE_KEYS, idx, 4)
        operating_income = _rolling_statement_sum(quarterly_income, OPERATING_INCOME_KEYS, idx, 4)
        net_income = _rolling_statement_sum(quarterly_income, NET_INCOME_KEYS, idx, 4)
        ocf = _rolling_statement_sum(quarterly_cashflow, OPERATING_CASH_FLOW_KEYS, idx, 4)
        capex = _rolling_statement_sum(quarterly_cashflow, CAPEX_KEYS, idx, 4)
        fcf = _free_cash_flow(ocf, capex)
        book = _statement_value_at(quarterly_balance, BOOK_VALUE_KEYS, idx)
        debt = _non_negative_float(_statement_value_at(quarterly_balance, TOTAL_DEBT_KEYS, idx))
        cash = _non_negative_float(_statement_value_at(quarterly_balance, CASH_KEYS, idx))
        enterprise_value = _enterprise_value_from_parts(market_cap, debt, cash) or market_cap
        denominators = {
            "price_sales": revenue,
            "price_operating_income": operating_income,
            "price_fcf": fcf,
            "price_earnings": net_income,
            "price_book": book,
        }
        for key, denominator in denominators.items():
            numerator = enterprise_value if key in ENTERPRISE_VALUE_METRICS else market_cap
            multiple = _multiple(numerator, denominator)
            if multiple is not None:
                series_by_metric[key].append(multiple)

    out: dict[str, Any] = {}
    for key, values in series_by_metric.items():
        clean = pd.Series([value for value in values if value > 0], dtype="float64")
        current = _safe_float((current_metrics.get(key) or {}).get("value"))
        if len(clean) < min_periods:
            out[key] = {"status": "insufficient_history", "periods": int(len(clean))}
            continue
        percentile = None
        if current is not None and current > 0:
            rank = int((clean.sort_values() < current).sum()) + 1
            percentile = 100.0 * (1.0 - ((rank - 1) / max(len(clean) - 1, 1)))
        out[key] = {
            "status": "ok",
            "periods": int(len(clean)),
            "median": _round_float(clean.median()),
            "q1": _round_float(clean.quantile(0.25)),
            "q3": _round_float(clean.quantile(0.75)),
            "min": _round_float(clean.min()),
            "max": _round_float(clean.max()),
            "percentile": round(percentile, 1) if percentile is not None else None,
            "source": "yfinance_quarterly",
        }
    return out


def composite_valuation_score(
    metrics: Mapping[str, Mapping[str, Any]],
    peer_context_payload: Mapping[str, Any],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    metric_stats = peer_context_payload.get("metric_stats") if isinstance(peer_context_payload, Mapping) else None
    if not isinstance(metric_stats, Mapping):
        return {"value": None, "status": "missing_peer_context", "components": {}}

    components: dict[str, Any] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    for key in VALUATION_COLUMNS:
        weight = max(0.0, _safe_float(weights.get(key)) or 0.0)
        percentile = (
            _safe_float((metric_stats.get(key) or {}).get("percentile"))
            if isinstance(metric_stats.get(key), Mapping)
            else None
        )
        if weight <= 0 or percentile is None:
            components[key] = {"weight": weight, "percentile": percentile, "contribution": None}
            continue
        weighted_sum += weight * percentile
        weight_sum += weight
        components[key] = {"weight": weight, "percentile": percentile, "contribution": round(weight * percentile, 2)}

    if weight_sum <= 0:
        return {"value": None, "status": "insufficient_peer_metrics", "components": components}
    return {"value": round(weighted_sum / weight_sum, 1), "status": "ok", "components": components}


def valuation_data_quality(
    metrics: Mapping[str, Mapping[str, Any]],
    peers: Mapping[str, Any],
) -> dict[str, Any]:
    metric_statuses = {key: (metric or {}).get("status") for key, metric in metrics.items()}
    usable = [key for key, status in metric_statuses.items() if status in {"ok", "degraded"}]
    warnings = []
    if len(usable) < 2:
        warnings.append("Few valuation metrics are meaningful for this position.")
    peer_stats = peers.get("metric_stats") if isinstance(peers, Mapping) else {}
    if not any(isinstance(row, Mapping) and row.get("status") == "ok" for row in (peer_stats or {}).values()):
        warnings.append("Peer-relative valuation context is unavailable or thin.")
    if any(status == "degraded" for status in metric_statuses.values()):
        warnings.append("Some multiples rely on fallback or partial provider data.")
    return {
        "status": "ok" if not warnings else "degraded",
        "usable_metric_count": len(usable),
        "metric_statuses": metric_statuses,
        "warnings": warnings,
    }


def _empty_peer_context(*, source: str = "unavailable") -> dict[str, Any]:
    return {
        "source": source,
        "peer_count": 0,
        "peers": [],
        "metric_stats": {
            key: {"status": "insufficient_peers", "sample_size": 0, "percentile": None, "rank": None}
            for key in VALUATION_COLUMNS
        },
    }


def _fetch_info(ticker: str) -> dict[str, Any]:
    try:
        value = yf_ticker_info(_clean_ticker(ticker), max_retries=1)
        return dict(value or {})
    except TypeError:
        value = yf_ticker_info(_clean_ticker(ticker))
        return dict(value or {})
    except Exception:
        LOGGER.debug("%s: yfinance info fetch failed", ticker, exc_info=True)
        return {}


def _get_yf_statement(ticker_obj: yf.Ticker, attr_names: tuple[str, ...]) -> pd.DataFrame | None:
    for attr in attr_names:
        try:
            value = getattr(ticker_obj, attr)
            df = value() if callable(value) else value
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return None


def _market_cap(info: Mapping[str, Any]) -> float | None:
    market_cap = _positive_float(info.get("marketCap"))
    if market_cap is not None:
        return market_cap
    price = _positive_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    shares = _positive_float(info.get("sharesOutstanding"))
    if price is None or shares is None:
        return None
    return price * shares


def _enterprise_value(
    info: Mapping[str, Any],
    *,
    market_cap: float | None,
    quarterly_balance: pd.DataFrame | None,
    annual_balance: pd.DataFrame | None,
) -> dict[str, Any]:
    debt = _first_not_none(
        _non_negative_float(info.get("totalDebt")),
        _non_negative_float(_latest_statement_value(quarterly_balance, TOTAL_DEBT_KEYS)),
        _non_negative_float(_latest_statement_value(annual_balance, TOTAL_DEBT_KEYS)),
    )
    cash = _first_not_none(
        _non_negative_float(info.get("totalCash")),
        _non_negative_float(_latest_statement_value(quarterly_balance, CASH_KEYS)),
        _non_negative_float(_latest_statement_value(annual_balance, CASH_KEYS)),
    )
    net_debt = debt - cash if debt is not None and cash is not None else None

    provider_ev = _positive_float(info.get("enterpriseValue"))
    if provider_ev is not None:
        return {
            "value": provider_ev,
            "source": "yfinance_info.enterpriseValue",
            "degraded": False,
            "reason": None,
            "net_debt": net_debt,
        }

    from_parts = _enterprise_value_from_parts(market_cap, debt, cash)
    if from_parts is not None:
        complete = debt is not None and cash is not None
        return {
            "value": from_parts,
            "source": "yfinance_balance_sheet" if complete else "yfinance_balance_sheet_partial",
            "degraded": not complete,
            "reason": None if complete else "partial_enterprise_value_inputs",
            "net_debt": net_debt,
        }

    if market_cap is not None and market_cap > 0:
        return {
            "value": market_cap,
            "source": "market_cap_proxy",
            "degraded": True,
            "reason": "using_market_cap_enterprise_value_proxy",
            "net_debt": net_debt,
        }

    return {
        "value": None,
        "source": None,
        "degraded": False,
        "reason": "missing_enterprise_value",
        "net_debt": net_debt,
    }


def _enterprise_value_from_parts(market_cap: float | None, debt: float | None, cash: float | None) -> float | None:
    if market_cap is None or market_cap <= 0:
        return None
    if debt is None and cash is None:
        return None
    value = market_cap + (debt or 0.0) - (cash or 0.0)
    return float(value) if math.isfinite(value) and value > 0 else None


def _first_not_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _ttm_or_latest(
    quarterly_stmt: pd.DataFrame | None,
    annual_stmt: pd.DataFrame | None,
    keys: tuple[str, ...],
) -> float | None:
    quarterly = _statement_row_sum(quarterly_stmt, keys, 4)
    if quarterly is not None:
        return quarterly
    return _statement_row_sum(annual_stmt, keys, 1)


def _statement_row_sum(stmt: pd.DataFrame | None, keys: tuple[str, ...], periods: int) -> float | None:
    row = _statement_row(stmt, keys)
    if row is None:
        return None
    values = row.dropna().iloc[:periods]
    if len(values) < periods:
        return None
    total = float(values.sum())
    return total if math.isfinite(total) else None


def _latest_statement_value(stmt: pd.DataFrame | None, keys: tuple[str, ...]) -> float | None:
    row = _statement_row(stmt, keys)
    if row is None:
        return None
    values = row.dropna()
    if values.empty:
        return None
    value = float(values.iloc[0])
    return value if math.isfinite(value) else None


def _statement_row(stmt: pd.DataFrame | None, keys: tuple[str, ...]) -> pd.Series | None:
    if stmt is None or stmt.empty:
        return None
    for key in keys:
        if key not in stmt.index:
            continue
        row = pd.to_numeric(stmt.loc[key], errors="coerce")
        if row.dropna().empty:
            continue
        return _series_newest_first(row)
    return None


def _series_newest_first(row: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(row.index, errors="coerce")
        if parsed.notna().any():
            order = np.argsort(parsed.to_numpy())[::-1]
            return row.iloc[order]
    except Exception:
        pass
    return row


def _statement_dates(stmt: pd.DataFrame | None) -> list[pd.Timestamp]:
    if stmt is None or stmt.empty:
        return []
    parsed = pd.to_datetime(stmt.columns, errors="coerce")
    dates = [pd.Timestamp(date) for date in parsed if pd.notna(date)]
    return sorted(dates, reverse=True)


def _rolling_statement_sum(
    stmt: pd.DataFrame | None, keys: tuple[str, ...], start_idx: int, periods: int
) -> float | None:
    row = _statement_row(stmt, keys)
    if row is None:
        return None
    values = row.dropna().iloc[start_idx : start_idx + periods]
    if len(values) < periods:
        return None
    value = float(values.sum())
    return value if math.isfinite(value) else None


def _statement_value_at(stmt: pd.DataFrame | None, keys: tuple[str, ...], idx: int) -> float | None:
    row = _statement_row(stmt, keys)
    if row is None:
        return None
    values = row.dropna()
    if idx >= len(values):
        return None
    value = float(values.iloc[idx])
    return value if math.isfinite(value) else None


def _free_cash_flow(operating_cash_flow: float | None, capex: float | None) -> float | None:
    if operating_cash_flow is None or capex is None:
        return None
    return operating_cash_flow + capex if capex < 0 else operating_cash_flow - capex


def _multiple(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or numerator <= 0:
        return None
    if denominator is None or denominator <= 0:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def _price_multiple(market_cap: float | None, denominator: float | None) -> float | None:
    return _multiple(market_cap, denominator)


def _close_series(prices: pd.DataFrame | pd.Series, symbol: str) -> pd.Series | None:
    if isinstance(prices, pd.Series):
        return prices
    if prices is None or prices.empty:
        return None
    columns = prices.columns
    if isinstance(columns, pd.MultiIndex):
        level0 = set(str(item) for item in columns.get_level_values(0))
        level1 = set(str(item) for item in columns.get_level_values(1))
        if "Close" in level0 and symbol in level1:
            return prices["Close"][symbol]
        if symbol in level0 and "Close" in level1:
            return prices[symbol]["Close"]
        if "Close" in level0:
            close_df = prices["Close"]
            return close_df.iloc[:, 0] if isinstance(close_df, pd.DataFrame) and not close_df.empty else None
        return None
    if symbol in columns:
        return prices[symbol]
    if "Close" in columns:
        return prices["Close"]
    return prices.iloc[:, 0] if len(prices.columns) else None


def _price_on_or_before(prices: pd.Series, period_date: pd.Timestamp) -> float | None:
    clean = pd.to_numeric(prices, errors="coerce").dropna()
    if clean.empty:
        return None
    idx = pd.to_datetime(clean.index, errors="coerce")
    series = pd.Series(clean.to_numpy(), index=idx).dropna()
    series = series[series.index <= period_date]
    if series.empty:
        return None
    return _safe_float(series.iloc[-1])


def _positive_float(value: Any) -> float | None:
    out = _safe_float(value)
    return out if out is not None and out > 0 else None


def _non_negative_float(value: Any) -> float | None:
    out = _safe_float(value)
    return out if out is not None and out >= 0 else None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _round_float(value: Any, digits: int = 2) -> float | None:
    out = _safe_float(value)
    return round(out, digits) if out is not None else None


def _clean_ticker(value: Any) -> str:
    return str(value or "").strip().upper()
