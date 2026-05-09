"""Shared equity valuation multiples service.

This module intentionally returns snapshot-shaped dictionaries but does not
persist metric snapshots. The persisted state is limited to per-ticker user
assumptions, such as profile overrides and value-range scenarios.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yfinance as yf

from utils.fx import clean_currency as _clean_currency
from utils.fx import currency_lookup_and_unit_scale as _currency_lookup_and_unit_scale
from utils.fx import fx_rate_to_base
from utils.retry import yf_ticker_info

LOGGER = logging.getLogger(__name__)
VALUATION_CURRENT_CACHE_VERSION = "v2"
VALUATION_PEER_ROW_CACHE_VERSION = "v2"
VALUATION_COLUMNS = (
    "price_sales",
    "price_operating_income",
    "price_fcf",
    "price_earnings",
    "price_book",
)
ENTERPRISE_VALUE_METRICS = {"price_sales", "price_operating_income", "price_fcf"}
VALUE_RANGE_SCENARIOS = ("bear", "base", "bull")

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
VALUE_RANGE_LOCAL_PATH = Path("data_cache/valuation/value_ranges.json")
VALUE_RANGE_GCS_KEY = "live/valuation/value_ranges.json"


def _valuation_current_cache_key(ticker: str) -> str:
    return f"valuation_current:{VALUATION_CURRENT_CACHE_VERSION}:{_clean_ticker(ticker)}"


def _valuation_peer_row_cache_key(ticker: str) -> str:
    return f"valuation_peer_row:{VALUATION_PEER_ROW_CACHE_VERSION}:{_clean_ticker(ticker)}"


def _without_cache_meta(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    clean = dict(value)
    clean.pop("_meta", None)
    return clean


def _get_or_set_daily_cache[T](key: str, loader: Callable[[], T]) -> T:
    from api.cache import daily_cache, get_or_set_cached
    from api.serializers import serialize_value

    return cast(T, _without_cache_meta(get_or_set_cached(daily_cache, key, lambda: serialize_value(loader()))))


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


def currency_context_from_info(info: Mapping[str, Any]) -> dict[str, Any]:
    price_currency = _clean_currency(info.get("currency")) or "USD"
    financial_currency = _clean_currency(info.get("financialCurrency")) or price_currency

    if _same_currency(financial_currency, price_currency):
        return {
            "price_currency": price_currency,
            "financial_currency": financial_currency,
            "financial_to_price_fx_rate": 1.0,
            "fx_rate_as_of": None,
            "conversion_status": "same_currency",
        }

    fx = fx_rate_to_base(financial_currency, price_currency)
    if not isinstance(fx, Mapping):
        return {
            "price_currency": price_currency,
            "financial_currency": financial_currency,
            "financial_to_price_fx_rate": None,
            "fx_rate_as_of": None,
            "conversion_status": "missing_fx_rate",
        }
    rate = _positive_float(fx.get("rate"))
    if rate is None:
        return {
            "price_currency": price_currency,
            "financial_currency": financial_currency,
            "financial_to_price_fx_rate": None,
            "fx_rate_as_of": None,
            "conversion_status": "missing_fx_rate",
        }
    return {
        "price_currency": price_currency,
        "financial_currency": financial_currency,
        "financial_to_price_fx_rate": rate,
        "fx_rate_as_of": fx.get("as_of"),
        "conversion_status": "ok",
    }


def _same_currency(a: Any, b: Any) -> bool:
    left = _currency_lookup_and_unit_scale(a)
    right = _currency_lookup_and_unit_scale(b)
    return bool(left and right and left == right)


def _financial_to_price_rate(currency_context: Mapping[str, Any]) -> float | None:
    return _positive_float(currency_context.get("financial_to_price_fx_rate"))


def _convert_financial_value(value: Any, currency_context: Mapping[str, Any]) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    rate = _financial_to_price_rate(currency_context)
    if rate is None:
        return None
    converted = parsed * rate
    return converted if math.isfinite(converted) else None


def _conversion_rate(
    source_currency: Any,
    target_currency: Any,
    currency_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = _clean_currency(source_currency)
    target = _clean_currency(target_currency)
    if not source or not target:
        return {"rate": None, "as_of": None, "status": "missing_currency"}
    if _same_currency(source, target):
        return {"rate": 1.0, "as_of": None, "status": "same_currency"}

    if isinstance(currency_context, Mapping):
        financial = currency_context.get("financial_currency")
        price = currency_context.get("price_currency")
        context_rate = _financial_to_price_rate(currency_context)
        if context_rate is not None and _same_currency(source, financial) and _same_currency(target, price):
            return {"rate": context_rate, "as_of": currency_context.get("fx_rate_as_of"), "status": "ok"}
        if context_rate is not None and _same_currency(source, price) and _same_currency(target, financial):
            return {"rate": 1.0 / context_rate, "as_of": currency_context.get("fx_rate_as_of"), "status": "ok"}

    fx = fx_rate_to_base(source, target)
    if not isinstance(fx, Mapping):
        return {"rate": None, "as_of": None, "status": "missing_fx_rate"}
    rate = _positive_float(fx.get("rate"))
    if rate is None:
        return {"rate": None, "as_of": None, "status": "missing_fx_rate"}
    return {"rate": rate, "as_of": fx.get("as_of"), "status": "ok"}


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


def normalize_value_range_metric(value: Any) -> str:
    text = str(value or "").strip()
    if text not in VALUATION_COLUMNS:
        raise ValueError(f"Unsupported valuation metric: {value}")
    return text


def read_value_range_assumption(ticker: str) -> dict[str, Any] | None:
    ranges = _read_value_ranges()
    return ranges.get(_clean_ticker(ticker))


def write_value_range_assumption(ticker: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized_ticker = _clean_ticker(ticker)
    if not normalized_ticker:
        raise ValueError("Ticker is required")

    ranges = _read_value_ranges()
    current = ranges.get(normalized_ticker) or _empty_value_range_storage()
    metric, assumption = _normalize_value_range_update_payload(payload)
    assumptions = dict(current.get("metric_assumptions") or {})
    assumptions[metric] = assumption
    ranges[normalized_ticker] = {"selected_metric": metric, "metric_assumptions": assumptions}
    _write_value_ranges(ranges)
    return {"ticker": normalized_ticker, "value_range": ranges[normalized_ticker]}


def delete_value_range_assumption(ticker: str, metric: str) -> dict[str, Any]:
    normalized_ticker = _clean_ticker(ticker)
    if not normalized_ticker:
        raise ValueError("Ticker is required")

    normalized_metric = normalize_value_range_metric(metric)
    ranges = _read_value_ranges()
    current = ranges.get(normalized_ticker) or _empty_value_range_storage()
    assumptions = dict(current.get("metric_assumptions") or {})
    assumptions.pop(normalized_metric, None)

    selected_metric = current.get("selected_metric")
    if selected_metric == normalized_metric:
        selected_metric = _first_value_range_metric(assumptions)

    if assumptions:
        ranges[normalized_ticker] = {
            "selected_metric": selected_metric or _first_value_range_metric(assumptions) or "price_sales",
            "metric_assumptions": assumptions,
        }
    else:
        ranges.pop(normalized_ticker, None)

    _write_value_ranges(ranges)
    return {
        "ticker": normalized_ticker,
        "value_range": ranges.get(normalized_ticker) or _empty_value_range_storage(),
    }


def _empty_value_range_storage() -> dict[str, Any]:
    return {"selected_metric": "price_sales", "metric_assumptions": {}}


def _first_value_range_metric(metric_assumptions: Mapping[str, Any]) -> str | None:
    for metric in VALUATION_COLUMNS:
        if metric in metric_assumptions:
            return metric
    return None


def _normalize_value_range_scenarios(raw_scenarios: Any, *, require_complete: bool) -> dict[str, dict[str, float]]:
    if not isinstance(raw_scenarios, Mapping):
        raise ValueError("Value range scenarios are required")

    scenarios: dict[str, dict[str, float]] = {}
    for scenario in VALUE_RANGE_SCENARIOS:
        raw = raw_scenarios.get(scenario)
        if not isinstance(raw, Mapping):
            if require_complete:
                raise ValueError(f"Missing {scenario} scenario")
            continue
        multiple = _positive_float(raw.get("multiple"))
        denominator = _positive_float(raw.get("denominator"))
        if multiple is None or denominator is None:
            if require_complete:
                raise ValueError(f"{scenario.title()} scenario requires positive multiple and denominator")
            continue
        scenarios[scenario] = {"multiple": multiple, "denominator": denominator}

    if require_complete and set(scenarios) != set(VALUE_RANGE_SCENARIOS):
        raise ValueError("Bear, base, and bull scenarios are required")
    return scenarios


def _normalize_value_range_metric_assumption(
    payload: Mapping[str, Any] | None,
    *,
    require_complete: bool,
    legacy_without_currency: bool,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Value range metric assumption is required")

    denominator_currency = _clean_currency(payload.get("denominator_currency"))
    scenarios = _normalize_value_range_scenarios(payload.get("scenarios"), require_complete=require_complete)
    legacy_denominator_currency = bool(payload.get("legacy_denominator_currency")) or (
        legacy_without_currency and not denominator_currency
    )

    out: dict[str, Any] = {
        "scenarios": scenarios,
        "legacy_denominator_currency": legacy_denominator_currency,
    }
    if denominator_currency:
        out["denominator_currency"] = denominator_currency
    return out


def _normalize_value_range_update_payload(payload: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("Value range payload is required")
    metric = normalize_value_range_metric(payload.get("metric"))
    return metric, _normalize_value_range_metric_assumption(
        payload,
        require_complete=True,
        legacy_without_currency=False,
    )


def _normalize_value_range_payload(payload: Mapping[str, Any] | None, *, require_complete: bool) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Value range payload is required")

    raw_assumptions = payload.get("metric_assumptions")
    if isinstance(raw_assumptions, Mapping):
        metric_assumptions: dict[str, dict[str, Any]] = {}
        for key, value in raw_assumptions.items():
            try:
                metric = normalize_value_range_metric(key)
                metric_assumptions[metric] = _normalize_value_range_metric_assumption(
                    value if isinstance(value, Mapping) else None,
                    require_complete=require_complete,
                    legacy_without_currency=False,
                )
            except ValueError:
                if require_complete:
                    raise
                continue

        selected_metric: str | None = None
        raw_selected = payload.get("selected_metric")
        if raw_selected:
            try:
                selected_metric = normalize_value_range_metric(raw_selected)
            except ValueError:
                if require_complete:
                    raise
        selected_metric = selected_metric or _first_value_range_metric(metric_assumptions) or "price_sales"
        return {"selected_metric": selected_metric, "metric_assumptions": metric_assumptions}

    metric = normalize_value_range_metric(payload.get("metric"))
    assumption = _normalize_value_range_metric_assumption(
        payload,
        require_complete=require_complete,
        legacy_without_currency=True,
    )
    return {"selected_metric": metric, "metric_assumptions": {metric: assumption}}


def _read_value_ranges() -> dict[str, dict[str, Any]]:
    try:
        from api.state_storage import exists_text, read_text
        from paths import PROJECT_ROOT

        path = PROJECT_ROOT / VALUE_RANGE_LOCAL_PATH
        if not exists_text(path, VALUE_RANGE_GCS_KEY):
            return {}
        raw = json.loads(read_text(path, VALUE_RANGE_GCS_KEY, encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}

        out: dict[str, dict[str, Any]] = {}
        for key, value in raw.items():
            ticker = _clean_ticker(key)
            if not ticker or not isinstance(value, Mapping):
                continue
            try:
                out[ticker] = _normalize_value_range_payload(value, require_complete=True)
            except ValueError:
                LOGGER.debug("Ignoring invalid value-range assumption for %s", ticker, exc_info=True)
        return out
    except Exception:
        LOGGER.debug("Failed to read valuation value ranges", exc_info=True)
        return {}


def _write_value_ranges(ranges: Mapping[str, Mapping[str, Any]]) -> None:
    from api.state_storage import write_text
    from paths import PROJECT_ROOT

    clean: dict[str, dict[str, Any]] = {}
    for key, value in ranges.items():
        ticker = _clean_ticker(key)
        if not ticker or not isinstance(value, Mapping):
            continue
        try:
            clean[ticker] = _normalize_value_range_payload(value, require_complete=True)
        except ValueError:
            LOGGER.debug("Skipping invalid value-range assumption for %s", ticker, exc_info=True)

    path = PROJECT_ROOT / VALUE_RANGE_LOCAL_PATH
    write_text(
        path,
        VALUE_RANGE_GCS_KEY,
        json.dumps(dict(sorted(clean.items())), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        content_type="application/json",
    )


def get_position_valuation(ticker: str, *, include_peers: bool = True) -> dict[str, Any]:
    normalized = _clean_ticker(ticker)
    if not normalized:
        raise ValueError("Ticker is required")

    info = _fetch_info(normalized)
    override = read_profile_override(normalized)
    current = fetch_current_valuation(normalized, info=info)
    raw_currency_context = current.get("currency_context")
    currency_context: Mapping[str, Any] = (
        raw_currency_context if isinstance(raw_currency_context, Mapping) else currency_context_from_info(info)
    )
    market_cap = _safe_float(current.get("market_cap"))
    enterprise_value = _safe_float(current.get("enterprise_value"))
    current_price = _current_price(info)
    net_debt = _net_debt_or_ev_spread(current.get("net_debt"), enterprise_value, market_cap)
    shares = _shares_outstanding(info, market_cap=market_cap, current_price=current_price)
    profile = resolve_profile(info, override)
    effective_weights = effective_profile_weights(profile["weights"], current["metrics"])
    peers = peer_context(normalized, info, current["metrics"]) if include_peers else _empty_peer_context()
    composite_score = composite_valuation_score(current["metrics"], peers, effective_weights)
    value_range = value_range_payload(
        saved_assumption=read_value_range_assumption(normalized),
        metrics=current["metrics"],
        peers=peers,
        effective_weights=effective_weights,
        currency_context=currency_context,
        market_data={
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "net_debt": net_debt,
            "current_price": current_price,
            "shares": shares,
            "currency": currency_context.get("price_currency"),
        },
    )

    return {
        "ticker": normalized,
        "company_name": info.get("longName") or info.get("shortName") or normalized,
        "as_of": datetime.now(UTC).isoformat(),
        "source_policy": "free_providers",
        "currency_context": currency_context,
        "market_data": {
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "net_debt": net_debt,
            "net_debt_financial": (current.get("financial_data") or {}).get("net_debt")
            if isinstance(current.get("financial_data"), Mapping)
            else None,
            "shares_outstanding": shares,
            "currency": currency_context.get("price_currency"),
            "price_currency": currency_context.get("price_currency"),
            "financial_currency": currency_context.get("financial_currency"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "current_price": current_price,
            "fifty_two_week_high": _fifty_two_week_high(info),
            "fifty_two_week_low": _fifty_two_week_low(info),
        },
        "financial_data": current.get("financial_data") if isinstance(current.get("financial_data"), Mapping) else {},
        "profile": {
            **profile,
            "override_profile_id": override,
            "effective_weights": effective_weights,
            "options": profile_options(),
        },
        "metrics": current["metrics"],
        "peer_context": peers,
        "composite_score": composite_score,
        "data_quality": valuation_data_quality(current["metrics"], peers),
        "value_range": value_range,
    }


def value_range_payload(
    *,
    saved_assumption: Mapping[str, Any] | None,
    metrics: Mapping[str, Mapping[str, Any]],
    peers: Mapping[str, Any],
    effective_weights: Mapping[str, Any],
    currency_context: Mapping[str, Any],
    market_data: Mapping[str, Any],
) -> dict[str, Any]:
    stored = (
        _normalize_value_range_payload(saved_assumption, require_complete=True)
        if saved_assumption is not None
        else {"metric_assumptions": {}}
    )
    raw_metric_assumptions = stored.get("metric_assumptions")
    metric_assumptions: dict[str, Any] = (
        dict(cast(Mapping[str, Any], raw_metric_assumptions)) if isinstance(raw_metric_assumptions, Mapping) else {}
    )
    selected_metric = stored.get("selected_metric")
    metric = (
        normalize_value_range_metric(selected_metric)
        if selected_metric in VALUATION_COLUMNS
        else (_first_value_range_metric(metric_assumptions) or _default_value_range_metric(metrics, effective_weights))
    )
    raw_assumption = metric_assumptions.get(metric)
    assumption = cast(Mapping[str, Any], raw_assumption) if isinstance(raw_assumption, Mapping) else None
    saved = assumption is not None
    calculation = _value_range_metric_calculation(
        metric,
        assumption,
        currency_context=currency_context,
        market_data=market_data,
    )
    computed_metric_assumptions: dict[str, Any] = {}
    for saved_metric in VALUATION_COLUMNS:
        raw_saved_assumption = metric_assumptions.get(saved_metric)
        if not isinstance(raw_saved_assumption, Mapping):
            continue
        saved_calculation = _value_range_metric_calculation(
            saved_metric,
            raw_saved_assumption,
            currency_context=currency_context,
            market_data=market_data,
        )
        computed_metric_assumptions[saved_metric] = {
            **dict(raw_saved_assumption),
            "computed_scenarios": saved_calculation["scenarios"],
        }

    return {
        "saved": saved,
        "source": "saved_assumptions" if saved else "blank",
        "selected_metric": metric,
        "metric_assumptions": computed_metric_assumptions,
        "metric": metric,
        "metric_label": VALUATION_LABELS[metric],
        "denominator_label": DENOMINATOR_LABELS[metric],
        "denominator_currency": calculation["denominator_currency"],
        "stored_denominator_currency": calculation["stored_denominator_currency"],
        "legacy_denominator_currency": calculation["legacy_denominator_currency"],
        "denominator_to_price_fx_rate": calculation["denominator_to_price_fx_rate"],
        "fx_rate_as_of": calculation["fx_rate_as_of"],
        "calculation_method": "enterprise_value_to_equity" if metric in ENTERPRISE_VALUE_METRICS else "equity_value",
        "current_price": _safe_float(market_data.get("current_price")),
        "shares": _safe_float(market_data.get("shares")),
        "net_debt": _safe_float(market_data.get("net_debt")),
        "currency": calculation["output_currency"],
        "output_currency": calculation["output_currency"],
        "scenarios": calculation["scenarios"],
    }


def _value_range_metric_calculation(
    metric: str,
    assumption: Mapping[str, Any] | None,
    *,
    currency_context: Mapping[str, Any],
    market_data: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_metric = normalize_value_range_metric(metric)
    price_currency = (
        _clean_currency(currency_context.get("price_currency")) or _clean_currency(market_data.get("currency")) or "USD"
    )
    financial_currency = _clean_currency(currency_context.get("financial_currency")) or price_currency
    assumption_currency = _clean_currency(assumption.get("denominator_currency")) if assumption else None
    legacy_denominator_currency = bool(assumption.get("legacy_denominator_currency")) if assumption else False
    stored_denominator_currency = assumption_currency or (
        price_currency if legacy_denominator_currency else financial_currency
    )
    display_denominator_currency = financial_currency

    display_conversion = _conversion_rate(stored_denominator_currency, display_denominator_currency, currency_context)
    display_rate = _positive_float(display_conversion.get("rate"))
    if display_rate is None:
        display_denominator_currency = stored_denominator_currency
        display_rate = 1.0

    denominator_to_price = _conversion_rate(display_denominator_currency, price_currency, currency_context)
    denominator_to_price_rate = _positive_float(denominator_to_price.get("rate"))

    def _display_scenario(row: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(row)
        denominator = _safe_float(out.get("denominator"))
        if denominator is not None:
            out["denominator"] = denominator * display_rate
        return out

    raw_scenarios = assumption.get("scenarios") if assumption is not None else None
    scenario_rows = cast(Mapping[str, Any], raw_scenarios) if isinstance(raw_scenarios, Mapping) else {}
    scenarios = {
        scenario: compute_value_range_scenario(
            normalized_metric,
            _display_scenario(row),
            current_price=market_data.get("current_price"),
            shares=market_data.get("shares"),
            net_debt=market_data.get("net_debt"),
            denominator_currency=display_denominator_currency,
            output_currency=price_currency,
            denominator_to_output_fx_rate=denominator_to_price_rate,
            fx_rate_as_of=denominator_to_price.get("as_of"),
        )
        for scenario, row in scenario_rows.items()
        if scenario in VALUE_RANGE_SCENARIOS and isinstance(row, Mapping)
    }

    for scenario in VALUE_RANGE_SCENARIOS:
        scenarios.setdefault(
            scenario,
            compute_value_range_scenario(
                normalized_metric,
                {"multiple": None, "denominator": None},
                current_price=market_data.get("current_price"),
                shares=market_data.get("shares"),
                net_debt=market_data.get("net_debt"),
                denominator_currency=display_denominator_currency,
                output_currency=price_currency,
                denominator_to_output_fx_rate=denominator_to_price_rate,
                fx_rate_as_of=denominator_to_price.get("as_of"),
            ),
        )

    return {
        "denominator_currency": display_denominator_currency,
        "stored_denominator_currency": stored_denominator_currency,
        "legacy_denominator_currency": legacy_denominator_currency,
        "denominator_to_price_fx_rate": denominator_to_price_rate,
        "fx_rate_as_of": denominator_to_price.get("as_of"),
        "output_currency": price_currency,
        "scenarios": {scenario: scenarios[scenario] for scenario in VALUE_RANGE_SCENARIOS},
    }


def default_value_range_assumption(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    peers: Mapping[str, Any],
    effective_weights: Mapping[str, Any],
) -> dict[str, Any]:
    metric = _default_value_range_metric(metrics, effective_weights)
    denominator = _positive_float((metrics.get(metric) or {}).get("denominator"))
    bear, base, bull = _default_value_range_multiples(metric, metrics, peers=peers)
    return {
        "metric": metric,
        "scenarios": {
            "bear": {"multiple": bear, "denominator": denominator},
            "base": {"multiple": base, "denominator": denominator},
            "bull": {"multiple": bull, "denominator": denominator},
        },
    }


def compute_value_range_scenario(
    metric: str,
    scenario: Mapping[str, Any],
    *,
    current_price: Any,
    shares: Any,
    net_debt: Any,
    denominator_currency: Any = None,
    output_currency: Any = None,
    denominator_to_output_fx_rate: Any = 1.0,
    fx_rate_as_of: Any = None,
) -> dict[str, Any]:
    normalized_metric = normalize_value_range_metric(metric)
    multiple = _positive_float(scenario.get("multiple"))
    denominator = _positive_float(scenario.get("denominator"))
    denominator_fx_rate = _positive_float(denominator_to_output_fx_rate)
    denominator_converted = (
        denominator * denominator_fx_rate if denominator is not None and denominator_fx_rate is not None else None
    )
    current_price_value = _positive_float(current_price)
    share_count = _positive_float(shares)
    net_debt_value = _safe_float(net_debt)

    status = "ok"
    reason = None
    equity_value = None
    expected_price = None
    percent_change = None

    if multiple is None:
        status = "missing"
        reason = "missing_multiple"
    elif denominator is None:
        status = "missing"
        reason = "missing_denominator"
    elif denominator_fx_rate is None:
        status = "missing"
        reason = "missing_fx_rate"
    elif share_count is None:
        status = "missing"
        reason = "missing_shares"
    elif normalized_metric in ENTERPRISE_VALUE_METRICS and net_debt_value is None:
        status = "missing"
        reason = "missing_net_debt"
    else:
        enterprise_or_equity_value = multiple * (denominator_converted or 0.0)
        equity_value = (
            enterprise_or_equity_value - (net_debt_value or 0.0)
            if normalized_metric in ENTERPRISE_VALUE_METRICS
            else enterprise_or_equity_value
        )
        if equity_value <= 0:
            status = "not_meaningful"
            reason = "non_positive_equity_value"
        else:
            expected_price = equity_value / share_count
            if current_price_value is not None:
                percent_change = (expected_price / current_price_value - 1.0) * 100.0

    return {
        "multiple": multiple,
        "denominator": denominator,
        "denominator_currency": _clean_currency(denominator_currency),
        "denominator_converted": _round_float(denominator_converted),
        "denominator_converted_currency": _clean_currency(output_currency),
        "denominator_to_output_fx_rate": _round_float(denominator_fx_rate, 10),
        "fx_rate_as_of": fx_rate_as_of,
        "equity_value": _round_float(equity_value),
        "expected_price": _round_float(expected_price, 4),
        "percent_change": _round_float(percent_change, 2),
        "output_currency": _clean_currency(output_currency),
        "status": status,
        "reason": reason,
    }


def _default_value_range_metric(
    metrics: Mapping[str, Mapping[str, Any]],
    effective_weights: Mapping[str, Any],
) -> str:
    weighted: list[tuple[float, int, str]] = []
    for idx, metric in enumerate(VALUATION_COLUMNS):
        weight = _positive_float(effective_weights.get(metric))
        denominator = _positive_float((metrics.get(metric) or {}).get("denominator"))
        multiple = _positive_float((metrics.get(metric) or {}).get("value"))
        if weight is not None and denominator is not None and multiple is not None:
            weighted.append((weight, -idx, metric))
    if weighted:
        return max(weighted)[2]

    for metric in VALUATION_COLUMNS:
        denominator = _positive_float((metrics.get(metric) or {}).get("denominator"))
        multiple = _positive_float((metrics.get(metric) or {}).get("value"))
        if denominator is not None and multiple is not None:
            return metric
    return "price_sales"


def _default_value_range_multiples(
    metric: str,
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    peers: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None]:
    peer_stats = peers.get("metric_stats") if isinstance(peers, Mapping) else None
    peer = peer_stats.get(metric) if isinstance(peer_stats, Mapping) else None
    if isinstance(peer, Mapping) and peer.get("status") == "ok":
        values = (
            _positive_float(peer.get("q1")),
            _positive_float(peer.get("median")),
            _positive_float(peer.get("q3")),
        )
        if all(value is not None for value in values):
            return values

    current = _positive_float((metrics.get(metric) or {}).get("value"))
    if current is None:
        return None, None, None
    return current * 0.8, current, current * 1.2


def fetch_current_valuation(ticker: str, *, info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized = _clean_ticker(ticker)
    return _get_or_set_daily_cache(
        _valuation_current_cache_key(normalized),
        lambda: _fetch_current_valuation_uncached(normalized, info=info),
    )


def _fetch_current_valuation_uncached(ticker: str, *, info: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
    currency_context = currency_context_from_info(info)
    market_cap = _market_cap(info)
    revenue_ttm = _ttm_or_latest(quarterly_income, annual_income, REVENUE_KEYS)
    operating_income_ttm = _ttm_or_latest(quarterly_income, annual_income, OPERATING_INCOME_KEYS)
    net_income_ttm = _ttm_or_latest(quarterly_income, annual_income, NET_INCOME_KEYS)
    operating_cash_flow_ttm = _ttm_or_latest(quarterly_cashflow, annual_cashflow, OPERATING_CASH_FLOW_KEYS)
    capex_ttm = _ttm_or_latest(quarterly_cashflow, annual_cashflow, CAPEX_KEYS)
    fcf_ttm = _free_cash_flow(operating_cash_flow_ttm, capex_ttm)
    book_value = _latest_statement_value(quarterly_balance, BOOK_VALUE_KEYS)
    if book_value is None:
        book_value = _latest_statement_value(annual_balance, BOOK_VALUE_KEYS)
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
    net_debt_financial = debt - cash if debt is not None and cash is not None else None
    debt_converted = _convert_financial_value(debt, currency_context)
    cash_converted = _convert_financial_value(cash, currency_context)
    net_debt_converted = _convert_financial_value(net_debt_financial, currency_context)
    enterprise_value_payload = _enterprise_value(
        info,
        market_cap=market_cap,
        debt=debt_converted,
        cash=cash_converted,
        net_debt=net_debt_converted,
        conversion_status=str(currency_context.get("conversion_status") or ""),
    )
    enterprise_value = enterprise_value_payload["value"]

    metrics = {
        "price_sales": _metric_payload(
            "price_sales",
            enterprise_value,
            revenue_ttm,
            _convert_financial_value(revenue_ttm, currency_context),
            source="yfinance_statements",
            numerator_source=enterprise_value_payload["source"],
            numerator_degraded=enterprise_value_payload["degraded"],
            numerator_reason=enterprise_value_payload["reason"],
            denominator_currency=currency_context.get("financial_currency"),
            denominator_converted_currency=currency_context.get("price_currency"),
        ),
        "price_operating_income": _metric_payload(
            "price_operating_income",
            enterprise_value,
            operating_income_ttm,
            _convert_financial_value(operating_income_ttm, currency_context),
            source="yfinance_statements",
            numerator_source=enterprise_value_payload["source"],
            numerator_degraded=enterprise_value_payload["degraded"],
            numerator_reason=enterprise_value_payload["reason"],
            denominator_currency=currency_context.get("financial_currency"),
            denominator_converted_currency=currency_context.get("price_currency"),
        ),
        "price_fcf": _metric_payload(
            "price_fcf",
            enterprise_value,
            fcf_ttm,
            _convert_financial_value(fcf_ttm, currency_context),
            source="yfinance_statements",
            numerator_source=enterprise_value_payload["source"],
            numerator_degraded=enterprise_value_payload["degraded"],
            numerator_reason=enterprise_value_payload["reason"],
            denominator_currency=currency_context.get("financial_currency"),
            denominator_converted_currency=currency_context.get("price_currency"),
        ),
        "price_earnings": _metric_payload(
            "price_earnings",
            market_cap,
            net_income_ttm,
            _convert_financial_value(net_income_ttm, currency_context),
            source="yfinance_statements",
            numerator_source="market_cap",
            denominator_currency=currency_context.get("financial_currency"),
            denominator_converted_currency=currency_context.get("price_currency"),
            fallback_value=_positive_float(info.get("trailingPE")),
            fallback_source="yfinance_info.trailingPE",
        ),
        "price_book": _metric_payload(
            "price_book",
            market_cap,
            book_value,
            _convert_financial_value(book_value, currency_context),
            source="yfinance_balance_sheet",
            numerator_source="market_cap",
            denominator_currency=currency_context.get("financial_currency"),
            denominator_converted_currency=currency_context.get("price_currency"),
            fallback_value=_positive_float(info.get("priceToBook")),
            fallback_source="yfinance_info.priceToBook",
        ),
    }
    return {
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "net_debt": enterprise_value_payload["net_debt"],
        "currency_context": currency_context,
        "financial_data": {
            "revenue_ttm": revenue_ttm,
            "operating_income_ttm": operating_income_ttm,
            "net_income_ttm": net_income_ttm,
            "operating_cash_flow_ttm": operating_cash_flow_ttm,
            "capex_ttm": capex_ttm,
            "fcf_ttm": fcf_ttm,
            "book_value": book_value,
            "debt": debt,
            "cash": cash,
            "net_debt": net_debt_financial,
            "currency": currency_context.get("financial_currency"),
        },
        "metrics": metrics,
    }


def _metric_payload(
    key: str,
    numerator: float | None,
    denominator: float | None,
    denominator_converted: float | None,
    *,
    source: str,
    numerator_source: str | None = None,
    numerator_degraded: bool = False,
    numerator_reason: str | None = None,
    denominator_currency: Any = None,
    denominator_converted_currency: Any = None,
    fallback_value: float | None = None,
    fallback_source: str | None = None,
) -> dict[str, Any]:
    value = _multiple(numerator, denominator_converted)
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
    elif denominator_converted is None:
        status = "missing"
        reason = "missing_fx_rate"
    elif denominator_converted <= 0:
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
        "denominator_currency": _clean_currency(denominator_currency),
        "denominator_converted": denominator_converted,
        "denominator_converted_currency": _clean_currency(denominator_converted_currency),
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
    normalized = _clean_ticker(ticker)
    raw = _get_or_set_daily_cache(_valuation_peer_row_cache_key(normalized), lambda: _batch_row_uncached(normalized))
    info = dict(raw.get("info") or {})
    metrics = raw.get("metrics") or {}
    profile = resolve_profile(info, read_profile_override(ticker))
    effective = effective_profile_weights(profile["weights"], metrics)
    row = {key: metrics.get(key, {}).get("value") for key in VALUATION_COLUMNS}
    row.update({f"{key}_profile_weight": effective.get(key, 0.0) for key in VALUATION_COLUMNS})
    row["valuation_profile_id"] = profile["id"]
    row["sector"] = raw.get("sector") or info.get("sector")
    row["industry"] = raw.get("industry") or info.get("industry")
    return row


def _batch_row_uncached(ticker: str) -> dict[str, Any]:
    info = _fetch_info(ticker)
    current = fetch_current_valuation(ticker, info=info)
    return {
        "info": info,
        "metrics": current["metrics"],
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


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
    if any((metric or {}).get("reason") == "missing_fx_rate" for metric in metrics.values()):
        warnings.append("FX conversion is unavailable for mixed-currency valuation inputs.")
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
    price = _current_price(info)
    shares = _positive_float(info.get("sharesOutstanding"))
    if price is None or shares is None:
        return None
    return price * shares


def _current_price(info: Mapping[str, Any]) -> float | None:
    return _positive_float(info.get("currentPrice") or info.get("regularMarketPrice"))


def _fifty_two_week_high(info: Mapping[str, Any]) -> float | None:
    return _positive_float(info.get("fiftyTwoWeekHigh") or info.get("52WeekHigh"))


def _fifty_two_week_low(info: Mapping[str, Any]) -> float | None:
    return _positive_float(info.get("fiftyTwoWeekLow") or info.get("52WeekLow"))


def _shares_outstanding(
    info: Mapping[str, Any],
    *,
    market_cap: float | None,
    current_price: float | None,
) -> float | None:
    shares = _positive_float(info.get("sharesOutstanding"))
    if shares is not None:
        return shares
    if market_cap is None or current_price is None or current_price <= 0:
        return None
    value = market_cap / current_price
    return value if math.isfinite(value) and value > 0 else None


def _net_debt_or_ev_spread(net_debt: Any, enterprise_value: float | None, market_cap: float | None) -> float | None:
    value = _safe_float(net_debt)
    if value is not None:
        return value
    if enterprise_value is None or market_cap is None:
        return None
    spread = enterprise_value - market_cap
    return spread if math.isfinite(spread) else None


def _enterprise_value(
    info: Mapping[str, Any],
    *,
    market_cap: float | None,
    debt: float | None,
    cash: float | None,
    net_debt: float | None,
    conversion_status: str,
) -> dict[str, Any]:
    provider_ev = _positive_float(info.get("enterpriseValue"))
    if provider_ev is not None:
        return {
            "value": provider_ev,
            "source": "yfinance_info.enterpriseValue",
            "degraded": False,
            "reason": None,
            "net_debt": net_debt,
        }

    from_parts = (
        None if conversion_status == "missing_fx_rate" else _enterprise_value_from_parts(market_cap, debt, cash)
    )
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
        reason = "using_market_cap_enterprise_value_proxy"
        if conversion_status == "missing_fx_rate":
            reason = "missing_fx_rate"
        return {
            "value": market_cap,
            "source": "market_cap_proxy",
            "degraded": True,
            "reason": reason,
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
            date_index = pd.Series(parsed, index=row.index).dropna().sort_values(ascending=False).index
            return row.loc[date_index]
    except Exception:
        pass
    return row


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
