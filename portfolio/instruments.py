"""Instrument metadata and normalization helpers for portfolio positions."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, cast

AssetClass = Literal["equity", "commodity", "fx", "bond"]
InstrumentType = Literal["security", "future", "spot_fx", "option"]
OptionType = Literal["call", "put"]

ASSET_CLASSES: set[str] = {"equity", "commodity", "fx", "bond"}
INSTRUMENT_TYPES: set[str] = {"security", "future", "spot_fx", "option"}
DEFAULT_OPTION_MULTIPLIER = 100.0

# When |net option exposure| is within this fraction of gross premium, the legs
# are treated as nearly fully offsetting (e.g. a balanced straddle) and reported
# as a near-zero net so they are not sized as a tiny ghost position.
NEAR_ZERO_NET_RATIO = 0.02

_OCC_SYMBOL_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.=-]{0,31}$")
_CONTINUOUS_FUTURE_RE = re.compile(r"^[A-Z0-9]{1,8}=F$")
_SPOT_FX_PAIR_RE = re.compile(r"^([A-Z]{3})([A-Z]{3})(?:=X)?$")


@dataclass(frozen=True, slots=True)
class ParsedOptionContract:
    underlying_ticker: str
    option_expiration: str
    option_type: OptionType
    option_strike: float
    option_contract_symbol: str


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


def normalize_spot_fx_symbol(value: Any, *, field_name: str = "ticker") -> str:
    symbol = str(value or "").strip().upper()
    if not symbol:
        raise ValueError(f"{field_name} cannot be empty.")
    if "\\" in symbol or ":" in symbol or any(ch.isspace() for ch in symbol):
        raise ValueError(f"Invalid {field_name} format: '{symbol}'.")
    compact = symbol.replace("/", "").replace("-", "")
    match = _SPOT_FX_PAIR_RE.match(compact)
    if not match:
        raise ValueError(
            f"Invalid {field_name} format: '{symbol}'. Spot FX pairs must look like EURUSD=X, EURUSD, EUR/USD, or EUR-USD."
        )
    base, quote = match.groups()
    if base == quote:
        raise ValueError("Spot FX base and quote currencies must be different.")
    return f"{base}{quote}=X"


def is_spot_fx_symbol(value: Any) -> bool:
    try:
        normalize_spot_fx_symbol(value)
        return True
    except ValueError:
        return False


def spot_fx_currencies(symbol: Any) -> tuple[str, str]:
    normalized = normalize_spot_fx_symbol(symbol)
    return normalized[:3], normalized[3:6]


def futures_root(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    return normalized[:-2] if normalized.endswith("=F") else normalized


def futures_spec(symbol: str) -> FuturesSpec | None:
    return FUTURES_SPECS.get(futures_root(symbol))


def normalize_option_type(value: Any) -> OptionType | None:
    raw = str(value or "").strip().lower()
    if raw in {"call", "c"}:
        return "call"
    if raw in {"put", "p"}:
        return "put"
    return None


def _occ_expiration_to_iso(raw: str) -> str:
    text = str(raw or "").strip()
    if len(text) == 6 and text.isdigit():
        year = 2000 + int(text[:2])
        month = int(text[2:4])
        day = int(text[4:6])
        return date(year, month, day).isoformat()
    return text


def parse_occ_symbol(value: Any) -> ParsedOptionContract | None:
    symbol = str(value or "").strip().upper().replace(" ", "")
    if not symbol:
        return None
    match = _OCC_SYMBOL_RE.match(symbol)
    if not match:
        return None
    underlying, exp_raw, cp_flag, strike_raw = match.groups()
    option_type = "call" if cp_flag == "C" else "put"
    strike = int(strike_raw) / 1000.0
    return ParsedOptionContract(
        underlying_ticker=underlying,
        option_expiration=_occ_expiration_to_iso(exp_raw),
        option_type=option_type,  # type: ignore[arg-type]
        option_strike=strike,
        option_contract_symbol=symbol,
    )


def build_option_contract_symbol(
    underlying_ticker: str,
    option_expiration: str,
    option_type: str,
    option_strike: float,
) -> str:
    underlying = str(underlying_ticker or "").strip().upper()
    normalized_type = normalize_option_type(option_type)
    if not underlying or normalized_type is None:
        raise ValueError("Option contract requires underlying ticker and call/put type.")
    exp_text = str(option_expiration or "").strip()
    if len(exp_text) == 10 and exp_text[4] == "-":
        year, month, day = exp_text.split("-")
        exp_raw = f"{int(year) % 100:02d}{int(month):02d}{int(day):02d}"
    elif len(exp_text) == 6 and exp_text.isdigit():
        exp_raw = exp_text
    else:
        raise ValueError(f"Invalid option expiration: {exp_text!r}.")
    strike_raw = f"{int(round(float(option_strike) * 1000)):08d}"
    return f"{underlying}{exp_raw}{'C' if normalized_type == 'call' else 'P'}{strike_raw}"


def position_row_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("position_id") or "").strip().upper()
    if explicit:
        return explicit
    instrument_type = str(row.get("instrument_type") or "").strip().lower()
    if instrument_type == "option":
        contract = str(row.get("option_contract_symbol") or row.get("price_symbol") or "").strip().upper()
        if contract:
            return contract
    return str(row.get("ticker") or "").strip().upper()


def display_ticker(row: Mapping[str, Any]) -> str:
    instrument_type = str(row.get("instrument_type") or "").strip().lower()
    if instrument_type == "option":
        underlying = str(row.get("underlying_ticker") or row.get("ticker") or "").strip().upper()
        if underlying:
            return underlying
    return str(row.get("ticker") or "").strip().upper()


def chart_price_symbol(row: Mapping[str, Any]) -> str:
    instrument_type = str(row.get("instrument_type") or "").strip().lower()
    if instrument_type == "option":
        underlying = str(row.get("underlying_ticker") or row.get("ticker") or "").strip().upper()
        if underlying:
            return underlying
    return str(row.get("price_symbol") or row.get("ticker") or "").strip().upper()


def normalize_option_position_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    instrument_type = str(out.get("instrument_type") or "").strip().lower()
    if instrument_type != "option" and not any(
        out.get(key) for key in ("option_contract_symbol", "option_expiration", "option_strike", "option_type")
    ):
        return out

    parsed = parse_occ_symbol(out.get("option_contract_symbol") or out.get("price_symbol"))
    if parsed is None and out.get("underlying_ticker") and out.get("option_expiration") and out.get("option_strike"):
        normalized_type = normalize_option_type(out.get("option_type"))
        if normalized_type is None:
            raise ValueError("Option positions require option_type call or put.")
        strike_raw = out.get("option_strike")
        if strike_raw is None:
            raise ValueError("Option positions require option_strike.")
        option_strike = float(strike_raw)
        parsed = ParsedOptionContract(
            underlying_ticker=str(out.get("underlying_ticker")).strip().upper(),
            option_expiration=str(out.get("option_expiration")).strip(),
            option_type=normalized_type,
            option_strike=option_strike,
            option_contract_symbol=build_option_contract_symbol(
                str(out.get("underlying_ticker")),
                str(out.get("option_expiration")),
                normalized_type,
                option_strike,
            ),
        )
    if parsed is None:
        raise ValueError(
            "Option positions require underlying_ticker, expiration, strike, and type, or a valid OCC contract symbol."
        )

    out["instrument_type"] = "option"
    out["underlying_ticker"] = parsed.underlying_ticker
    out["option_expiration"] = parsed.option_expiration
    out["option_strike"] = parsed.option_strike
    out["option_type"] = parsed.option_type
    out["option_contract_symbol"] = parsed.option_contract_symbol
    out["price_symbol"] = parsed.option_contract_symbol
    out["ticker"] = parsed.underlying_ticker
    out["position_id"] = str(out.get("position_id") or parsed.option_contract_symbol).strip().upper()
    if out.get("contract_multiplier") in (None, ""):
        out["contract_multiplier"] = DEFAULT_OPTION_MULTIPLIER
    return out


def normalize_instrument_type(value: Any, *, ticker: str, price_symbol: str | None = None) -> InstrumentType:
    raw = str(value or "").strip().lower()
    if raw:
        if raw not in INSTRUMENT_TYPES:
            raise ValueError(f"Invalid instrument_type: {raw!r}.")
        return raw  # type: ignore[return-value]
    symbol = price_symbol or ticker
    if parse_occ_symbol(symbol) is not None:
        return "option"
    if is_continuous_future_symbol(symbol):
        return "future"
    if str(symbol or "").strip().upper().endswith("=X") and is_spot_fx_symbol(symbol):
        return "spot_fx"
    return "security"


def normalize_asset(value: Any, *, instrument_type: InstrumentType, symbol: str) -> AssetClass:
    raw = str(value or "").strip().lower()
    if instrument_type == "spot_fx":
        return "fx"
    if instrument_type == "option":
        return cast(AssetClass, raw or "equity")
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
    if instrument_type in {"security", "spot_fx"}:
        return 1.0
    if instrument_type == "option":
        return DEFAULT_OPTION_MULTIPLIER
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


def normalize_portfolio_instrument_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize instrument fields for portfolio/hedge position rows."""
    out = dict(row)
    ticker_raw = str(out.get("ticker") or "").strip()
    price_raw = str(out.get("price_symbol") or ticker_raw).strip()

    instrument_type = normalize_instrument_type(
        out.get("instrument_type"),
        ticker=ticker_raw,
        price_symbol=price_raw,
    )
    out["instrument_type"] = instrument_type

    if instrument_type == "spot_fx":
        out["price_symbol"] = normalize_spot_fx_symbol(out.get("price_symbol") or ticker_raw, field_name="price_symbol")
        out["ticker"] = out["price_symbol"]
        fx_base, fx_quote = spot_fx_currencies(out["price_symbol"])
        out["fx_base_currency"] = fx_base
        out["fx_quote_currency"] = fx_quote
        out["asset"] = "fx"
        out["currency"] = fx_quote
        out["exchange"] = out.get("exchange") or "FX"
    elif instrument_type == "option":
        out = normalize_option_position_fields(
            {
                **out,
                "underlying_ticker": out.get("underlying_ticker") or ticker_raw,
                "option_contract_symbol": out.get("option_contract_symbol") or price_raw,
            }
        )
    else:
        out["ticker"] = normalize_symbol(ticker_raw)
        out["price_symbol"] = normalize_symbol(out.get("price_symbol") or out["ticker"], field_name="price_symbol")
        if instrument_type == "future" and not is_continuous_future_symbol(out["price_symbol"]):
            raise ValueError("Futures positions require a continuous '=F' price_symbol.")
        out["position_id"] = str(out.get("position_id") or out["ticker"]).strip().upper()

    out["asset"] = normalize_asset(
        out.get("asset"), instrument_type=instrument_type, symbol=str(out.get("price_symbol") or out["ticker"])
    )
    out["contract_multiplier"] = default_contract_multiplier(
        instrument_type=instrument_type,
        symbol=str(out.get("price_symbol") or out["ticker"]),
        override=out.get("contract_multiplier"),
    )
    quantity = normalize_quantity(quantity=out.get("quantity"), shares=out.get("shares"), allow_negative=True)
    out["quantity"] = quantity
    out["shares"] = quantity
    if not out.get("position_id"):
        out["position_id"] = position_row_id(out)
    return out


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


def direction_sign(row: Mapping[str, Any]) -> int:
    """Return +1 for long exposure and -1 for short, defaulting to long."""
    return -1 if str(row.get("direction") or "long").strip().lower() == "short" else 1


def option_exposure_sign(row: Mapping[str, Any]) -> int | None:
    """Signed directional polarity for an option leg.

    Combines option polarity (call +1, put -1) with leg direction (long +1,
    short -1) so that a long call and a short put both return +1 (bullish/long
    exposure) while a long put and a short call return -1 (bearish/short
    exposure). The sign comes solely from ``option_type`` and ``direction``;
    the quantity sign is intentionally ignored to avoid double-counting shorts.

    Returns ``None`` when the row is not a recognizable option.
    """
    option_type = normalize_option_type(row.get("option_type"))
    if option_type is None:
        return None
    polarity = 1 if option_type == "call" else -1
    return polarity * direction_sign(row)


def signed_notional(base_notional: float | None, row: Mapping[str, Any]) -> float | None:
    """Apply the directional sign to a non-negative base notional.

    Options use :func:`option_exposure_sign`; all other instruments use plain
    long/short direction. Returns ``None`` when ``base_notional`` is ``None``.
    """
    if base_notional is None:
        return None
    sign = option_exposure_sign(row)
    if sign is None:
        sign = direction_sign(row)
    return base_notional * sign


def infer_underlying_direction(legs: Iterable[Mapping[str, Any]]) -> tuple[str | None, bool]:
    """Infer one ``(direction, near_zero)`` for an underlying from its legs.

    A real non-option (share/equity) leg dictates direction. Otherwise the sign
    of net option exposure is used, with ``|quantity| * cost_basis *
    contract_multiplier`` as the per-leg magnitude and option polarity x leg
    direction as the sign. Near-perfect offsets (e.g. a balanced straddle) are
    reported as ``("neutral", True)``. Returns ``(None, False)`` when there are
    no legs with a determinable exposure.
    """
    option_legs: list[Mapping[str, Any]] = []
    share_direction: str | None = None
    for row in legs:
        if str(row.get("instrument_type") or "security").strip().lower() == "option":
            option_legs.append(row)
        elif share_direction is None:
            share_direction = "short" if str(row.get("direction") or "long").strip().lower() == "short" else "long"
    if share_direction is not None:
        return share_direction, False
    if not option_legs:
        return None, False

    gross = 0.0
    net = 0.0
    for row in option_legs:
        sign = option_exposure_sign(row)
        if sign is None:
            continue
        quantity = row.get("quantity")
        if quantity is None:
            quantity = row.get("shares")
        try:
            qty = abs(float(quantity))
        except (TypeError, ValueError):
            qty = None
        magnitude = notional_value(qty, row.get("cost_basis"), row.get("contract_multiplier") or 1.0)
        if magnitude is None:
            magnitude = 1.0
        gross += magnitude
        net += magnitude * sign

    if gross <= 0:
        return None, False
    if abs(net) <= NEAR_ZERO_NET_RATIO * gross:
        return "neutral", True
    return ("long" if net > 0 else "short"), False
