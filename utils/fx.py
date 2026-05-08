"""Shared FX conversion helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, date
from typing import Any

import pandas as pd

from utils.retry import yf_download

DEFAULT_BASE_CURRENCY = "USD"

_MINOR_UNIT_CURRENCIES: dict[str, tuple[str, float]] = {
    "GBP": ("GBP", 1.0),
    "GBX": ("GBP", 0.01),
    "GBp": ("GBP", 0.01),
    "ZAc": ("ZAR", 0.01),
    "ILA": ("ILS", 0.01),
    "IEp": ("IEP", 0.01),
}


def clean_currency(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if text else None


def fx_rate_to_base(currency: str, base_currency: str = DEFAULT_BASE_CURRENCY) -> dict[str, Any] | None:
    raw = clean_currency(currency)
    base = clean_currency(base_currency) or DEFAULT_BASE_CURRENCY
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
    if isinstance(prices.columns, pd.MultiIndex):
        for key in (("Close", symbol), (symbol, "Close")):
            if key in prices.columns:
                return prices[key]
        return None
    if symbol in prices:
        return prices[symbol]
    if "Close" in prices:
        return prices["Close"]
    return None


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
