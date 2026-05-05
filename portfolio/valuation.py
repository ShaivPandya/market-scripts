"""Currency detection and base-currency valuation for portfolio rows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date
from typing import Any

import pandas as pd
import yfinance as yf

from utils.retry import yf_download, yf_ticker_info

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
    raw = _clean_currency(currency)
    base = _clean_currency(base_currency) or DEFAULT_BASE_CURRENCY
    if not raw:
        return None

    lookup, unit_scale = _currency_lookup_and_unit_scale(raw)
    if lookup == base:
        return {"rate": unit_scale, "as_of": date.today().isoformat()}

    key = f"portfolio_fx_rate:{lookup}:{base}"

    def _loader() -> dict[str, Any] | None:
        return _fetch_fx_rate_uncached(lookup, base)

    try:
        from api.cache import get_or_set_cached, short_cache

        quote = get_or_set_cached(short_cache, key, _loader)
    except Exception:
        quote = _loader()
    if not isinstance(quote, Mapping) or quote.get("rate") is None:
        return None
    rate = _to_float(quote.get("rate"))
    if rate is None:
        return None
    return {"rate": rate * unit_scale, "as_of": quote.get("as_of")}


def _fetch_fx_rate_uncached(currency: str, base_currency: str) -> dict[str, Any] | None:
    direct = f"{currency}{base_currency}=X"
    inverse = f"{base_currency}{currency}=X"
    try:
        prices = yf_download(
            [direct, inverse],
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            max_retries=1,
        )
    except Exception:
        return None
    direct_close = _latest_close(prices, direct)
    if direct_close is not None and direct_close > 0:
        return {"rate": direct_close, "as_of": _latest_as_of(prices, direct)}
    inverse_close = _latest_close(prices, inverse)
    if inverse_close is not None and inverse_close > 0:
        return {"rate": 1.0 / inverse_close, "as_of": _latest_as_of(prices, inverse)}
    return None


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


def _currency_lookup_and_unit_scale(currency: str) -> tuple[str, float]:
    if currency in _MINOR_UNIT_CURRENCIES:
        return _MINOR_UNIT_CURRENCIES[currency]
    upper = currency.upper()
    if upper in _MINOR_UNIT_CURRENCIES:
        return _MINOR_UNIT_CURRENCIES[upper]
    return upper, 1.0


def _latest_close(prices: pd.DataFrame, symbol: str) -> float | None:
    series = _close_series(prices, symbol)
    if series is None:
        return None
    series = series.dropna()
    if series.empty:
        return None
    return _to_float(series.iloc[-1])


def _latest_as_of(prices: pd.DataFrame, symbol: str) -> str | None:
    series = _close_series(prices, symbol)
    if series is None:
        return None
    series = series.dropna()
    if series.empty:
        return None
    value = series.index[-1]
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return str(dt.isoformat())
    return str(value)


def _close_series(prices: pd.DataFrame, symbol: str) -> pd.Series | None:
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
        return None
    if symbol in columns:
        return prices[symbol]
    if "Close" in columns:
        return prices["Close"]
    return None


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
