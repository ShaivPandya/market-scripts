"""Economic exposure resolution for leveraged and inverse ETFs.

Traded symbols (e.g. METU) keep their own price/P&L identity; risk rollups use
the underlying ticker with a signed leverage factor (e.g. METU -> META x2).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from portfolio.instruments import direction_sign, display_ticker, normalize_symbol, signed_notional

ExposureSource = Literal["static", "metadata", "identity"]

# Curated single-name leveraged / inverse ETFs: traded -> (underlying, factor).
# Factor is signed: positive = long leveraged exposure, negative = inverse/short underlying.
STATIC_ECONOMIC_EXPOSURE: dict[str, tuple[str, float]] = {
    "METU": ("META", 2.0),
    "METD": ("META", -1.0),
    "NVDU": ("NVDA", 2.0),
    "NVDD": ("NVDA", -1.0),
    "AMZU": ("AMZN", 2.0),
    "AMZD": ("AMZN", -1.0),
    "MSFU": ("MSFT", 2.0),
    "MSFD": ("MSFT", -1.0),
    "AAPU": ("AAPL", 2.0),
    "AAPD": ("AAPL", -1.0),
    "TSLU": ("TSLA", 2.0),
    "TSLD": ("TSLA", -1.0),
    "GOOU": ("GOOGL", 2.0),
    "GOOD": ("GOOGL", -1.0),
}

_LEVERAGE_FACTOR_RE = re.compile(
    r"(?:(-?\d+(?:\.\d+)?)\s*[x×])|(?:([x×])\s*(-?\d+(?:\.\d+)?))",
    re.IGNORECASE,
)
_INVERSE_WORDS = frozenset({"inverse", "short", "bear", "ultrashort", "ultra-short"})


@dataclass(frozen=True, slots=True)
class EconomicExposure:
    traded_ticker: str
    underlying_ticker: str
    factor: float
    source: ExposureSource

    @property
    def is_mapped(self) -> bool:
        return self.source != "identity" or self.factor != 1.0

    def exposure_key(self) -> str:
        """Ticker used to group economic risk (options use display_ticker elsewhere)."""
        return self.underlying_ticker


def _normalize_traded_ticker(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    try:
        return normalize_symbol(raw)
    except ValueError:
        return raw


def _parse_leverage_from_text(*texts: str | None) -> float | None:
    combined = " ".join(str(t or "").strip() for t in texts if t).strip()
    if not combined:
        return None
    lower = combined.lower()
    inverse = any(word in lower for word in _INVERSE_WORDS)
    match = _LEVERAGE_FACTOR_RE.search(combined)
    if match:
        g1, _x, g3 = match.groups()
        raw = g1 if g1 is not None else g3
        if raw is None:
            return None
        try:
            magnitude = abs(float(raw))
        except (TypeError, ValueError):
            return None
        if magnitude <= 0:
            return None
        if inverse and magnitude > 0:
            return -magnitude
        return magnitude if not inverse else -magnitude
    if inverse:
        return -1.0
    return None


def _underlying_from_metadata(metadata: Mapping[str, Any] | None, traded: str) -> str | None:
    if not metadata:
        return None
    for key in ("underlying_ticker", "underlying_symbol", "target_ticker", "benchmark"):
        candidate = str(metadata.get(key) or "").strip().upper()
        if candidate and candidate != traded:
            try:
                return normalize_symbol(candidate)
            except ValueError:
                if re.match(r"^[A-Z0-9][A-Z0-9.=-]{0,31}$", candidate):
                    return candidate
    long_name = str(metadata.get("longName") or metadata.get("shortName") or "").strip()
    if not long_name:
        return None
    # e.g. "Direxion Daily META Bull 2X Shares" -> try to find META-like token
    tokens: list[str] = re.findall(r"\b([A-Z]{1,5})\b", long_name.upper())
    for token in tokens:
        if token == traded or token in {"ETF", "ETN", "THE", "AND", "FOR", "DAILY", "SHARES", "FUND"}:
            continue
        if len(token) >= 2:
            return str(token)
    return None


def resolve_economic_exposure(
    row: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> EconomicExposure:
    """Resolve economic underlying and signed leverage factor for a position row."""
    instrument_type = str(row.get("instrument_type") or "security").strip().lower()
    if instrument_type == "option":
        underlying = _normalize_traded_ticker(row.get("underlying_ticker") or display_ticker(row))
        traded = underlying or _normalize_traded_ticker(row.get("ticker"))
        return EconomicExposure(
            traded_ticker=traded,
            underlying_ticker=underlying or traded,
            factor=1.0,
            source="identity",
        )

    traded = _normalize_traded_ticker(row.get("ticker") or row.get("price_symbol"))
    if not traded:
        return EconomicExposure("", "", 1.0, "identity")

    static = STATIC_ECONOMIC_EXPOSURE.get(traded)
    if static is not None:
        underlying, factor = static
        return EconomicExposure(
            traded_ticker=traded,
            underlying_ticker=underlying,
            factor=float(factor),
            source="static",
        )

    explicit_underlying = str(row.get("economic_underlying_ticker") or "").strip().upper()
    explicit_factor = row.get("exposure_multiplier") or row.get("economic_exposure_factor")
    if explicit_underlying:
        try:
            underlying = normalize_symbol(explicit_underlying)
        except ValueError:
            underlying = explicit_underlying
        try:
            factor = float(explicit_factor) if explicit_factor is not None else 1.0
        except (TypeError, ValueError):
            factor = 1.0
        if factor != 0:
            return EconomicExposure(traded, underlying, factor, "static")

    meta = metadata if metadata is not None else {}
    if not meta:
        for key in ("longName", "shortName", "category", "quoteType"):
            if row.get(key):
                meta = {**meta, key: row.get(key)}

    underlying_meta = _underlying_from_metadata(meta, traded)
    factor_meta = _parse_leverage_from_text(
        meta.get("longName"),
        meta.get("shortName"),
        meta.get("category"),
        row.get("longName"),
        row.get("shortName"),
    )
    if underlying_meta and factor_meta is not None and factor_meta != 0:
        return EconomicExposure(traded, underlying_meta, float(factor_meta), "metadata")

    return EconomicExposure(traded, traded, 1.0, "identity")


def exposure_group_key(row: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> str:
    """Group key for risk: options use display_ticker; leveraged ETFs use economic underlying."""
    instrument_type = str(row.get("instrument_type") or "security").strip().lower()
    if instrument_type == "option":
        return display_ticker(row)
    exposure = resolve_economic_exposure(row, metadata=metadata)
    return exposure.exposure_key()


def economic_exposure_fields(
    row: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    exposure = resolve_economic_exposure(row, metadata=metadata)
    return {
        "traded_ticker": exposure.traded_ticker,
        "economic_underlying_ticker": exposure.underlying_ticker,
        "exposure_multiplier": exposure.factor,
        "economic_exposure_source": exposure.source,
    }


def scale_signed_notional_for_exposure(
    base_notional: float | None,
    row: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> float | None:
    """Apply direction sign and economic leverage factor to a gross notional."""
    if base_notional is None:
        return None
    exposure = resolve_economic_exposure(row, metadata=metadata)
    signed = signed_notional(base_notional, row)
    if signed is None:
        return None
    instrument_type = str(row.get("instrument_type") or "security").strip().lower()
    if instrument_type == "option":
        return signed
    if exposure.source == "identity" and exposure.factor == 1.0:
        return signed
    return signed * exposure.factor


def scale_gross_notional_for_exposure(
    base_notional: float | None,
    row: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> float | None:
    """Scale absolute notional by |factor| for weight denominators."""
    if base_notional is None:
        return None
    exposure = resolve_economic_exposure(row, metadata=metadata)
    instrument_type = str(row.get("instrument_type") or "security").strip().lower()
    if instrument_type == "option":
        return base_notional
    if exposure.source == "identity" and abs(exposure.factor) == 1.0:
        return base_notional
    return base_notional * abs(exposure.factor)
