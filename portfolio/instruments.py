"""Instrument metadata and normalization helpers for portfolio positions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

AssetClass = Literal["equity", "commodity", "fx", "bond"]
InstrumentType = Literal["security", "future"]

ASSET_CLASSES: set[str] = {"equity", "commodity", "fx", "bond"}
INSTRUMENT_TYPES: set[str] = {"security", "future"}

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.=-]{0,31}$")
_CONTINUOUS_FUTURE_RE = re.compile(r"^[A-Z0-9]{1,8}=F$")


@dataclass(frozen=True, slots=True)
class FuturesSpec:
    root: str
    asset: AssetClass
    contract_multiplier: float
    label: str


FUTURES_SPECS: dict[str, FuturesSpec] = {
    # Equity index futures
    "ES": FuturesSpec("ES", "equity", 50.0, "E-mini S&P 500"),
    "MES": FuturesSpec("MES", "equity", 5.0, "Micro E-mini S&P 500"),
    "NQ": FuturesSpec("NQ", "equity", 20.0, "E-mini Nasdaq-100"),
    "MNQ": FuturesSpec("MNQ", "equity", 2.0, "Micro E-mini Nasdaq-100"),
    "RTY": FuturesSpec("RTY", "equity", 50.0, "E-mini Russell 2000"),
    "M2K": FuturesSpec("M2K", "equity", 5.0, "Micro E-mini Russell 2000"),
    "YM": FuturesSpec("YM", "equity", 5.0, "E-mini Dow"),
    "MYM": FuturesSpec("MYM", "equity", 0.5, "Micro E-mini Dow"),
    # Energy and metals
    "CL": FuturesSpec("CL", "commodity", 1000.0, "WTI Crude Oil"),
    "MCL": FuturesSpec("MCL", "commodity", 100.0, "Micro WTI Crude Oil"),
    "BZ": FuturesSpec("BZ", "commodity", 1000.0, "Brent Crude Oil"),
    "NG": FuturesSpec("NG", "commodity", 10000.0, "Natural Gas"),
    "GC": FuturesSpec("GC", "commodity", 100.0, "Gold"),
    "MGC": FuturesSpec("MGC", "commodity", 10.0, "Micro Gold"),
    "SI": FuturesSpec("SI", "commodity", 5000.0, "Silver"),
    "SIL": FuturesSpec("SIL", "commodity", 1000.0, "Micro Silver"),
    "HG": FuturesSpec("HG", "commodity", 25000.0, "Copper"),
    "PL": FuturesSpec("PL", "commodity", 50.0, "Platinum"),
    "PA": FuturesSpec("PA", "commodity", 100.0, "Palladium"),
    # Treasury futures. Prices are quoted in points, so multiplier is dollars per point.
    "ZT": FuturesSpec("ZT", "bond", 2000.0, "2-Year T-Note"),
    "ZF": FuturesSpec("ZF", "bond", 1000.0, "5-Year T-Note"),
    "ZN": FuturesSpec("ZN", "bond", 1000.0, "10-Year T-Note"),
    "ZB": FuturesSpec("ZB", "bond", 1000.0, "30-Year Treasury Bond"),
    "UB": FuturesSpec("UB", "bond", 1000.0, "Ultra Treasury Bond"),
    # Major currency futures
    "6E": FuturesSpec("6E", "fx", 125000.0, "Euro FX"),
    "6B": FuturesSpec("6B", "fx", 62500.0, "British Pound"),
    "6A": FuturesSpec("6A", "fx", 100000.0, "Australian Dollar"),
    "6C": FuturesSpec("6C", "fx", 100000.0, "Canadian Dollar"),
    "6S": FuturesSpec("6S", "fx", 125000.0, "Swiss Franc"),
    "6J": FuturesSpec("6J", "fx", 12500000.0, "Japanese Yen"),
}


def normalize_symbol(value: Any, *, field_name: str = "ticker") -> str:
    symbol = str(value or "").strip().upper()
    if not symbol:
        raise ValueError(f"{field_name} cannot be empty.")
    if any(ch.isspace() for ch in symbol) or "/" in symbol or "\\" in symbol or ":" in symbol:
        raise ValueError(f"Invalid {field_name} format: '{symbol}'.")
    if not _SYMBOL_RE.match(symbol):
        raise ValueError(
            f"Invalid {field_name} format: '{symbol}'. Only letters, digits, dots, dashes, and '=' are allowed."
        )
    return symbol


def is_continuous_future_symbol(symbol: str) -> bool:
    return bool(_CONTINUOUS_FUTURE_RE.match(symbol.strip().upper()))


def futures_root(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    return normalized[:-2] if normalized.endswith("=F") else normalized


def futures_spec(symbol: str) -> FuturesSpec | None:
    return FUTURES_SPECS.get(futures_root(symbol))


def normalize_instrument_type(value: Any, *, ticker: str, price_symbol: str | None = None) -> InstrumentType:
    raw = str(value or "").strip().lower()
    if raw:
        if raw not in INSTRUMENT_TYPES:
            raise ValueError(f"Invalid instrument_type: {raw!r}.")
        return raw  # type: ignore[return-value]
    symbol = price_symbol or ticker
    return "future" if is_continuous_future_symbol(symbol) else "security"


def normalize_asset(value: Any, *, instrument_type: InstrumentType, symbol: str) -> AssetClass:
    raw = str(value or "").strip().lower()
    spec = futures_spec(symbol) if instrument_type == "future" else None
    asset = raw or (spec.asset if spec else "equity")
    if asset not in ASSET_CLASSES:
        raise ValueError(f"Invalid asset: {asset!r}.")
    return asset  # type: ignore[return-value]


def default_contract_multiplier(
    *,
    instrument_type: InstrumentType,
    symbol: str,
    override: Any = None,
) -> float:
    has_override = override is not None and str(override).strip() != ""
    if has_override:
        try:
            multiplier = float(override)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid contract_multiplier: {override!r}.") from None
        if math.isfinite(multiplier):
            if multiplier <= 0:
                raise ValueError("contract_multiplier must be positive.")
            return multiplier
        if str(override).lower() != "nan":
            raise ValueError("contract_multiplier must be positive.")
    if instrument_type == "security":
        return 1.0
    spec = futures_spec(symbol)
    if spec is None:
        raise ValueError(
            f"Unknown futures contract multiplier for {symbol}. "
            "Provide contract_multiplier explicitly for unsupported futures."
        )
    return float(spec.contract_multiplier)


def normalize_quantity(*, quantity: Any = None, shares: Any = None, allow_negative: bool = False) -> float | None:
    raw = quantity if quantity is not None else shares
    if raw is None or str(raw).strip() == "":
        return None
    try:
        out = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out if allow_negative or out >= 0 else abs(out)


def notional_value(quantity: Any, price: Any, contract_multiplier: Any = 1.0) -> float | None:
    try:
        qty = float(quantity)
        px = float(price)
        multiplier = float(contract_multiplier)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(qty) and math.isfinite(px) and math.isfinite(multiplier)):
        return None
    if qty <= 0 or px <= 0 or multiplier <= 0:
        return None
    return abs(qty * px * multiplier)
