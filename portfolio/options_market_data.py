"""Delayed option quote fetch via yfinance option chains."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import yfinance as yf

from portfolio.instruments import (
    DEFAULT_OPTION_MULTIPLIER,
    build_option_contract_symbol,
    normalize_option_type,
    parse_occ_symbol,
)

LOGGER = logging.getLogger(__name__)

_OPTION_QUOTE_CACHE: dict[str, dict[str, Any]] = {}


def _cache_key(
    underlying: str,
    expiration: str,
    option_type: str,
    strike: float,
    contract_symbol: str | None,
) -> str:
    return "|".join(
        [
            underlying.strip().upper(),
            expiration.strip(),
            normalize_option_type(option_type) or "",
            f"{float(strike):.4f}",
            (contract_symbol or "").strip().upper(),
        ]
    )


def _pick_price(row: Any) -> float | None:
    for field in ("lastPrice", "regularMarketPrice", "bid", "ask"):
        try:
            value = float(getattr(row, field, None) if hasattr(row, field) else row.get(field))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    try:
        bid = float(row.get("bid") if hasattr(row, "get") else getattr(row, "bid", None) or 0)
        ask = float(row.get("ask") if hasattr(row, "get") else getattr(row, "ask", None) or 0)
    except (TypeError, ValueError):
        return None
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return None


def _match_chain_row(frame, *, contract_symbol: str | None, strike: float):
    if frame is None or getattr(frame, "empty", True):
        return None
    symbol = (contract_symbol or "").strip().upper()
    for _, row in frame.iterrows():
        row_symbol = str(row.get("contractSymbol") or "").strip().upper()
        if symbol and row_symbol == symbol:
            return row
    for _, row in frame.iterrows():
        try:
            row_strike = float(row.get("strike"))
        except (TypeError, ValueError):
            continue
        if abs(row_strike - float(strike)) <= 0.001:
            return row
    return None


def fetch_option_quote(
    *,
    underlying_ticker: str,
    option_expiration: str,
    option_strike: float,
    option_type: str,
    option_contract_symbol: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Return delayed quote metadata for one listed option contract."""
    underlying = str(underlying_ticker or "").strip().upper()
    expiration = str(option_expiration or "").strip()
    normalized_type = normalize_option_type(option_type)
    if not underlying or not expiration or normalized_type is None:
        return {"status": "missing_option_inputs"}
    try:
        strike = float(option_strike)
    except (TypeError, ValueError):
        return {"status": "missing_option_inputs"}
    if strike <= 0:
        return {"status": "missing_option_inputs"}

    contract_symbol = str(option_contract_symbol or "").strip().upper() or build_option_contract_symbol(
        underlying,
        expiration,
        normalized_type,
        strike,
    )
    key = _cache_key(underlying, expiration, normalized_type, strike, contract_symbol)
    if use_cache and key in _OPTION_QUOTE_CACHE:
        return dict(_OPTION_QUOTE_CACHE[key])

    out: dict[str, Any] = {
        "underlying_ticker": underlying,
        "option_expiration": expiration,
        "option_strike": strike,
        "option_type": normalized_type,
        "option_contract_symbol": contract_symbol,
        "contract_multiplier": DEFAULT_OPTION_MULTIPLIER,
        "status": "missing_option_quote",
        "quote_as_of": datetime.now().isoformat(timespec="seconds"),
        "quote_source": "yfinance_delayed",
    }

    try:
        ticker = yf.Ticker(underlying)
        expiries = list(getattr(ticker, "options", ()) or ())
        if not expiries:
            out["status"] = "missing_option_chain"
            if use_cache:
                _OPTION_QUOTE_CACHE[key] = dict(out)
            return out

        exp_key = expiration
        if exp_key not in expiries:
            # Accept YYYY-MM-DD when chain uses same format; otherwise pick nearest exact date match.
            candidates = [exp for exp in expiries if exp.startswith(exp_key[:10])]
            exp_key = candidates[0] if candidates else expiration
        chain = ticker.option_chain(exp_key)
        side = chain.calls if normalized_type == "call" else chain.puts
        row = _match_chain_row(
            side,
            contract_symbol=contract_symbol,
            strike=strike,
        )
        if row is None:
            out["status"] = "missing_option_quote"
            if use_cache:
                _OPTION_QUOTE_CACHE[key] = dict(out)
            return out

        price = _pick_price(row)
        out.update(
            {
                "option_contract_symbol": str(row.get("contractSymbol") or contract_symbol).upper(),
                "last_price": price,
                "bid": _safe_float(row.get("bid")),
                "ask": _safe_float(row.get("ask")),
                "volume": _safe_int(row.get("volume")),
                "open_interest": _safe_int(row.get("openInterest")),
                "implied_volatility": _safe_float(row.get("impliedVolatility")),
            }
        )
        if price is not None and price > 0:
            out["status"] = "ok"
            out["price"] = price
        else:
            out["status"] = "missing_option_quote"
    except Exception as exc:
        LOGGER.warning("Option quote fetch failed for %s: %s", contract_symbol, exc)
        out["status"] = "missing_option_quote"

    if use_cache:
        _OPTION_QUOTE_CACHE[key] = dict(out)
    return out


def parse_and_fetch_option_quote(value: str) -> dict[str, Any]:
    parsed = parse_occ_symbol(value)
    if parsed is None:
        return {"status": "invalid_option_symbol"}
    return fetch_option_quote(
        underlying_ticker=parsed.underlying_ticker,
        option_expiration=parsed.option_expiration,
        option_strike=parsed.option_strike,
        option_type=parsed.option_type,
        option_contract_symbol=parsed.option_contract_symbol,
    )


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # NaN check


def _safe_int(value: Any) -> int | None:
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return None
    return out


def clear_option_quote_cache() -> None:
    _OPTION_QUOTE_CACHE.clear()
