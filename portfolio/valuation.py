"""Currency detection and base-currency valuation for portfolio rows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import yfinance as yf

from portfolio.instruments import normalize_spot_fx_symbol, spot_fx_currencies
from utils.fx import fx_rate_to_base as _shared_fx_rate_to_base
from utils.retry import yf_ticker_info

DEFAULT_BASE_CURRENCY = "USD"

VALUATION_FIELDS = (
    "currency",
    "country",
    "exchange",
    "base_currency",
    "fx_rate_to_base",
    "fx_rate_as_of",
    "cost_basis_base",
    "notional_base",
    "valuation_status",
    "fx_base_currency",
    "fx_quote_currency",
)

_MINOR_UNIT_CURRENCIES: dict[str, tuple[str, float]] = {
    "GBP": ("GBP", 1.0),
    "GBX": ("GBP", 0.01),
    "GBp": ("GBP", 0.01),
    "ZAc": ("ZAR", 0.01),
    "ILA": ("ILS", 0.01),
    "IEp": ("IEP", 0.01),
}

_SUFFIX_METADATA: tuple[tuple[str, dict[str, str]], ...] = (
    (".TWO", {"currency": "TWD", "country": "Taiwan", "exchange": "Taipei Exchange"}),
    (".T", {"currency": "JPY", "country": "Japan", "exchange": "Tokyo Stock Exchange"}),
    (".L", {"currency": "GBp", "country": "United Kingdom", "exchange": "London Stock Exchange"}),
    (".HE", {"currency": "EUR", "country": "Finland", "exchange": "Nasdaq Helsinki"}),
    (".PA", {"currency": "EUR", "country": "France", "exchange": "Euronext Paris"}),
    (".DE", {"currency": "EUR", "country": "Germany", "exchange": "XETRA"}),
    (".F", {"currency": "EUR", "country": "Germany", "exchange": "Frankfurt Stock Exchange"}),
    (".AS", {"currency": "EUR", "country": "Netherlands", "exchange": "Euronext Amsterdam"}),
    (".MI", {"currency": "EUR", "country": "Italy", "exchange": "Borsa Italiana"}),
    (".SW", {"currency": "CHF", "country": "Switzerland", "exchange": "SIX Swiss Exchange"}),
    (".TO", {"currency": "CAD", "country": "Canada", "exchange": "Toronto Stock Exchange"}),
    (".V", {"currency": "CAD", "country": "Canada", "exchange": "TSX Venture Exchange"}),
    (".AX", {"currency": "AUD", "country": "Australia", "exchange": "Australian Securities Exchange"}),
    (".HK", {"currency": "HKD", "country": "Hong Kong", "exchange": "Hong Kong Stock Exchange"}),
    (".SS", {"currency": "CNY", "country": "China", "exchange": "Shanghai Stock Exchange"}),
    (".SZ", {"currency": "CNY", "country": "China", "exchange": "Shenzhen Stock Exchange"}),
    (".KS", {"currency": "KRW", "country": "South Korea", "exchange": "Korea Exchange"}),
    (".TW", {"currency": "TWD", "country": "Taiwan", "exchange": "Taiwan Stock Exchange"}),
    (".SI", {"currency": "SGD", "country": "Singapore", "exchange": "Singapore Exchange"}),
    (".SA", {"currency": "BRL", "country": "Brazil", "exchange": "B3"}),
    (".MX", {"currency": "MXN", "country": "Mexico", "exchange": "Mexican Stock Exchange"}),
    (".JO", {"currency": "ZAc", "country": "South Africa", "exchange": "Johannesburg Stock Exchange"}),
    (".OL", {"currency": "NOK", "country": "Norway", "exchange": "Oslo Stock Exchange"}),
    (".ST", {"currency": "SEK", "country": "Sweden", "exchange": "Nasdaq Stockholm"}),
    (".CO", {"currency": "DKK", "country": "Denmark", "exchange": "Nasdaq Copenhagen"}),
    (".IR", {"currency": "EUR", "country": "Ireland", "exchange": "Euronext Dublin"}),
)


def enrich_position_valuations(
    positions: Sequence[Mapping[str, Any]],
    *,
    base_currency: str = DEFAULT_BASE_CURRENCY,
    preserve_existing: bool = False,
) -> list[dict[str, Any]]:
    """Return copies of position rows with detected metadata and base valuation."""

    return [
        enrich_position_valuation(row, base_currency=base_currency, preserve_existing=preserve_existing)
        for row in positions
    ]


def enrich_position_valuation(
    row: Mapping[str, Any],
    *,
    base_currency: str = DEFAULT_BASE_CURRENCY,
    preserve_existing: bool = False,
) -> dict[str, Any]:
    out = dict(row)
    ticker = _clean_symbol(out.get("ticker"))
    price_symbol = _clean_symbol(out.get("price_symbol")) or ticker
    base = _clean_currency(base_currency) or DEFAULT_BASE_CURRENCY
    instrument_type = str(out.get("instrument_type") or "").strip().lower()
    if instrument_type == "spot_fx":
        return _enrich_spot_fx_valuation(out, base_currency=base, preserve_existing=preserve_existing)

    metadata = detect_market_metadata(price_symbol or ticker, overrides=out)

    currency = _clean_currency(out.get("currency")) or metadata.get("currency")
    country = _clean_text(out.get("country")) or metadata.get("country")
    exchange = _clean_text(out.get("exchange")) or metadata.get("exchange")

    out["base_currency"] = base
    out["currency"] = currency
    out["country"] = country
    out["exchange"] = exchange

    cost_basis = _to_float(out.get("cost_basis"))
    quantity = _to_float(out.get("quantity") if out.get("quantity") is not None else out.get("shares"))
    multiplier = _to_float(out.get("contract_multiplier")) or 1.0

    if cost_basis is None or quantity is None or multiplier <= 0:
        return _set_valuation_missing(out, "missing_position_inputs")
    if not currency:
        return _set_valuation_missing(out, "missing_currency")

    if preserve_existing:
        existing_rate = _to_float(out.get("fx_rate_to_base"))
        if existing_rate is not None and existing_rate > 0:
            return _set_base_valuation(out, cost_basis, quantity, multiplier, existing_rate, out.get("fx_rate_as_of"))

    fx = fx_rate_to_base(currency, base)
    if fx is None:
        return _set_valuation_missing(out, "missing_fx_rate")

    rate = fx["rate"]
    if rate is None or rate <= 0 or not math.isfinite(rate):
        return _set_valuation_missing(out, "missing_fx_rate")

    return _set_base_valuation(out, cost_basis, quantity, multiplier, rate, fx.get("as_of"))


def _enrich_spot_fx_valuation(
    row: dict[str, Any],
    *,
    base_currency: str,
    preserve_existing: bool,
) -> dict[str, Any]:
    symbol = normalize_spot_fx_symbol(row.get("price_symbol") or row.get("ticker"), field_name="price_symbol")
    fx_base, fx_quote = spot_fx_currencies(symbol)
    portfolio_base = _clean_currency(row.get("base_currency")) or base_currency or DEFAULT_BASE_CURRENCY

    row["ticker"] = symbol
    row["price_symbol"] = symbol
    row["instrument_type"] = "spot_fx"
    row["asset"] = "fx"
    row["contract_multiplier"] = 1.0
    row["fx_base_currency"] = fx_base
    row["fx_quote_currency"] = fx_quote
    row["base_currency"] = portfolio_base
    row["currency"] = fx_quote
    row["country"] = _clean_text(row.get("country"))
    row["exchange"] = _clean_text(row.get("exchange")) or "FX"

    cost_basis = _to_float(row.get("cost_basis"))
    quantity = _to_float(row.get("quantity") if row.get("quantity") is not None else row.get("shares"))
    if cost_basis is None or cost_basis <= 0 or quantity is None:
        return _set_valuation_missing(row, "missing_position_inputs")

    if preserve_existing:
        existing_rate = _to_float(row.get("fx_rate_to_base"))
        if existing_rate is not None and existing_rate > 0:
            return _set_base_valuation(row, cost_basis, quantity, 1.0, existing_rate, row.get("fx_rate_as_of"))

    rate: float | None
    as_of: Any
    if fx_quote == portfolio_base:
        rate = 1.0
        as_of = date.today().isoformat()
    elif fx_base == portfolio_base:
        rate = 1.0 / cost_basis
        as_of = date.today().isoformat()
    else:
        fx = fx_rate_to_base(fx_quote, portfolio_base)
        if fx is None:
            return _set_valuation_missing(row, "missing_fx_rate")
        rate = _to_float(fx.get("rate"))
        as_of = fx.get("as_of")

    if rate is None or rate <= 0 or not math.isfinite(rate):
        return _set_valuation_missing(row, "missing_fx_rate")
    return _set_base_valuation(row, cost_basis, quantity, 1.0, rate, as_of)


def detect_market_metadata(symbol: str, *, overrides: Mapping[str, Any] | None = None) -> dict[str, str | None]:
    """Detect market metadata without requiring a live lookup for common symbols."""

    overrides = overrides or {}
    explicit = {
        "currency": _clean_currency(overrides.get("currency")),
        "country": _clean_text(overrides.get("country")),
        "exchange": _clean_text(overrides.get("exchange")),
    }
    fallback = fallback_market_metadata(symbol)
    metadata = {
        "currency": explicit["currency"] or fallback.get("currency"),
        "country": explicit["country"] or fallback.get("country"),
        "exchange": explicit["exchange"] or fallback.get("exchange"),
    }
    if all(metadata.values()):
        return metadata

    live = _cached_yfinance_metadata(symbol)
    for key in ("currency", "country", "exchange"):
        if not metadata.get(key):
            metadata[key] = live.get(key)
    return metadata


def fallback_market_metadata(symbol: str) -> dict[str, str | None]:
    normalized = _clean_symbol(symbol)
    if not normalized:
        return {"currency": None, "country": None, "exchange": None}
    if normalized.endswith("=F"):
        return {"currency": "USD", "country": "United States", "exchange": "CME"}
    if normalized.endswith("=X"):
        return {"currency": normalized[3:6] or "USD", "country": None, "exchange": "FX"}
    for suffix, metadata in _SUFFIX_METADATA:
        if normalized.endswith(suffix):
            return dict(metadata)
    if "." not in normalized:
        return {"currency": "USD", "country": "United States", "exchange": None}
    return {"currency": None, "country": None, "exchange": None}


def fx_rate_to_base(currency: str, base_currency: str = DEFAULT_BASE_CURRENCY) -> dict[str, Any] | None:
    return _shared_fx_rate_to_base(currency, base_currency)


def _cached_yfinance_metadata(symbol: str) -> dict[str, str | None]:
    normalized = _clean_symbol(symbol)
    if not normalized:
        return {"currency": None, "country": None, "exchange": None}
    key = f"portfolio_market_metadata:{normalized}"

    def _loader() -> dict[str, str | None]:
        return _fetch_yfinance_metadata_uncached(normalized)

    try:
        from api.cache import get_or_set_cached, long_cache

        value = get_or_set_cached(long_cache, key, _loader)
    except Exception:
        value = _loader()
    return dict(value) if isinstance(value, Mapping) else {"currency": None, "country": None, "exchange": None}


def _fetch_yfinance_metadata_uncached(symbol: str) -> dict[str, str | None]:
    currency = None
    exchange = None
    country = None
    try:
        fast_info = yf.Ticker(symbol).fast_info
        currency = _clean_currency(_lookup(fast_info, "currency"))
        exchange = _clean_text(_lookup(fast_info, "exchange"))
    except Exception:
        pass
    if not currency or not country or not exchange:
        info = yf_ticker_info(symbol, max_retries=1)
        currency = currency or _clean_currency(info.get("currency"))
        country = country or _clean_text(info.get("country"))
        exchange = exchange or _clean_text(info.get("exchange") or info.get("fullExchangeName"))
    return {"currency": currency, "country": country, "exchange": exchange}


def _set_valuation_missing(row: dict[str, Any], status: str) -> dict[str, Any]:
    row["fx_rate_to_base"] = None
    row["fx_rate_as_of"] = None
    row["cost_basis_base"] = None
    row["notional_base"] = None
    row["valuation_status"] = status
    return row


def _set_base_valuation(
    row: dict[str, Any],
    cost_basis: float,
    quantity: float,
    multiplier: float,
    rate: float,
    as_of: Any,
) -> dict[str, Any]:
    cost_basis_base = abs(cost_basis * rate)
    notional_base = abs(quantity * cost_basis * multiplier * rate)
    row["fx_rate_to_base"] = round(rate, 10)
    row["fx_rate_as_of"] = _clean_text(as_of)
    row["cost_basis_base"] = round(cost_basis_base, 6)
    row["notional_base"] = round(notional_base, 4)
    row["valuation_status"] = "ok"
    return row


def _lookup(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    try:
        return getattr(value, key)
    except Exception:
        return None


def _clean_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _clean_currency(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text in _MINOR_UNIT_CURRENCIES:
        return text
    return text.upper()


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
