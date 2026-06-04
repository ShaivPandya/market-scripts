"""Parse IBKR Flex Query Open Positions XML into portfolio position rows."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date
from typing import Any

from portfolio.instruments import (
    build_option_contract_symbol,
    normalize_option_type,
    normalize_portfolio_instrument_row,
    parse_occ_symbol,
    position_row_id,
)

SUPPORTED_ASSET_CATEGORIES = {"STK", "OPT"}
PRESERVED_METADATA_FIELDS = ("conviction", "contrarian", "group_name", "group_conviction")
DEFAULT_CONVICTION = 3
IBKR_FLEX_INDEX_HEDGE_TICKERS = frozenset({"SPY", "IWM", "QQQ"})


def _attr(row: ET.Element, name: str) -> str:
    return str(row.attrib.get(name) or "").strip()


def _optional_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if parsed == parsed else None


def _ibkr_expiry_to_iso(value: str) -> str | None:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        year = int(text[:4])
        month = int(text[4:6])
        day = int(text[6:8])
        return date(year, month, day).isoformat()
    if len(text) == 10 and text[4] == "-":
        return text
    return None


def _normalize_direction(side: str, quantity: float | None) -> str:
    side_text = side.strip().lower()
    if side_text == "short":
        return "short"
    if side_text == "long":
        return "long"
    if quantity is not None and quantity < 0:
        return "short"
    return "long"


def _abs_quantity(value: float | None) -> float | None:
    if value is None:
        return None
    return abs(value)


def _compact_ibkr_option_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace(" ", "")


def _option_type_from_put_call(value: str) -> str | None:
    raw = str(value or "").strip().upper()
    if raw in {"C", "CALL"}:
        return "call"
    if raw in {"P", "PUT"}:
        return "put"
    return normalize_option_type(raw)


def _parse_open_position_row(element: ET.Element) -> dict[str, Any] | None:
    asset_category = _attr(element, "assetCategory").upper()
    if asset_category not in SUPPORTED_ASSET_CATEGORIES:
        return None

    raw_quantity = _optional_float(_attr(element, "position"))
    direction = _normalize_direction(_attr(element, "side"), raw_quantity)
    quantity = _abs_quantity(raw_quantity)
    currency = _attr(element, "currency") or None
    exchange = _attr(element, "listingExchange") or None
    fx_rate = _optional_float(_attr(element, "fxRateToBase"))
    cost_basis = _optional_float(_attr(element, "costBasisPrice"))
    cost_basis_base = _optional_float(_attr(element, "costBasisMoney"))
    notional_base = _optional_float(_attr(element, "positionValue"))
    report_date = _attr(element, "reportDate") or None

    base_row: dict[str, Any] = {
        "asset": "equity",
        "direction": direction,
        "contrarian": False,
        "conviction": DEFAULT_CONVICTION,
        "cost_basis": cost_basis,
        "shares": quantity,
        "quantity": quantity,
        "currency": currency,
        "exchange": exchange,
        "base_currency": currency,
        "fx_rate_to_base": fx_rate,
        "fx_rate_as_of": report_date,
        "cost_basis_base": cost_basis_base,
        "notional_base": notional_base,
    }

    if asset_category == "STK":
        ticker = _attr(element, "symbol").upper()
        if not ticker:
            raise ValueError("STK row missing symbol.")
        base_row.update(
            {
                "ticker": ticker,
                "instrument_type": "security",
                "price_symbol": ticker,
                "contract_multiplier": 1,
            }
        )
        return base_row

    underlying = _attr(element, "underlyingSymbol").upper()
    expiry = _ibkr_expiry_to_iso(_attr(element, "expiry"))
    strike = _optional_float(_attr(element, "strike"))
    option_type = _option_type_from_put_call(_attr(element, "putCall"))
    multiplier = _optional_float(_attr(element, "multiplier")) or 100.0
    compact_symbol = _compact_ibkr_option_symbol(_attr(element, "symbol"))
    parsed_occ = parse_occ_symbol(compact_symbol)

    if not underlying and parsed_occ is not None:
        underlying = parsed_occ.underlying_ticker
    if expiry is None and parsed_occ is not None:
        expiry = parsed_occ.option_expiration
    if strike is None and parsed_occ is not None:
        strike = parsed_occ.option_strike
    if option_type is None and parsed_occ is not None:
        option_type = parsed_occ.option_type

    if not underlying or expiry is None or strike is None or option_type is None:
        raise ValueError(f"Option row missing required fields: symbol={_attr(element, 'symbol')!r}.")

    contract_symbol = build_option_contract_symbol(underlying, expiry, option_type, strike)
    base_row.update(
        {
            "ticker": underlying,
            "underlying_ticker": underlying,
            "instrument_type": "option",
            "price_symbol": contract_symbol,
            "option_contract_symbol": contract_symbol,
            "option_expiration": expiry,
            "option_strike": strike,
            "option_type": option_type,
            "contract_multiplier": multiplier,
        }
    )
    return base_row


def parse_ibkr_flex_open_positions_xml(payload: bytes | str) -> list[dict[str, Any]]:
    """Parse IBKR Flex Open Positions XML into normalized portfolio rows."""
    if isinstance(payload, bytes):
        text = payload.decode("utf-8-sig")
    else:
        text = str(payload)
    if not text.strip():
        raise ValueError("Uploaded Flex XML is empty.")

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid Flex XML: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for element in root.iter("OpenPosition"):
        parsed = _parse_open_position_row(element)
        if parsed is None:
            continue
        normalized = normalize_portfolio_instrument_row(parsed)
        normalized["position_id"] = position_row_id(normalized)
        rows.append(normalized)

    if not rows:
        raise ValueError("No supported STK/OPT open positions found in Flex XML.")

    position_ids = [position_row_id(row) for row in rows]
    if len(set(position_ids)) != len(position_ids):
        duplicate = next(pid for pid in position_ids if position_ids.count(pid) > 1)
        raise ValueError(f"Duplicate position_id in Flex import: {duplicate}.")

    return rows


def merge_preserved_portfolio_metadata(
    imported_rows: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve app-only metadata from existing rows when position_id matches."""
    existing_by_id: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        if str(row.get("role") or "position").lower() == "hedge":
            continue
        key = position_row_id(row)
        if key:
            existing_by_id[key] = row

    merged: list[dict[str, Any]] = []
    for row in imported_rows:
        out = dict(row)
        existing = existing_by_id.get(position_row_id(out))
        if existing:
            for field in PRESERVED_METADATA_FIELDS:
                if field in existing and existing.get(field) is not None:
                    out[field] = existing[field]
        else:
            out.setdefault("conviction", DEFAULT_CONVICTION)
            out.setdefault("contrarian", False)
            out.setdefault("group_name", None)
            out.setdefault("group_conviction", None)
        merged.append(out)
    return merged


def is_ibkr_flex_index_hedge_row(row: dict[str, Any]) -> bool:
    """True when a Flex row should be staged as a hedge (short index ETF security)."""
    ticker = str(row.get("ticker") or "").strip().upper()
    direction = str(row.get("direction") or "").strip().lower()
    instrument_type = str(row.get("instrument_type") or "security").strip().lower()
    return instrument_type == "security" and direction == "short" and ticker in IBKR_FLEX_INDEX_HEDGE_TICKERS


def split_ibkr_flex_import_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split parsed Flex rows into portfolio positions and hedge positions."""
    portfolio_rows: list[dict[str, Any]] = []
    hedge_rows: list[dict[str, Any]] = []
    for row in rows:
        if is_ibkr_flex_index_hedge_row(row):
            hedge_rows.append(row)
        else:
            portfolio_rows.append(row)
    return portfolio_rows, hedge_rows


def merge_ibkr_flex_hedge_replacement(
    imported_hedge_rows: list[dict[str, Any]],
    existing_hedge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build full hedge replacement payload: imported rows replace matches; others preserved."""
    imported_by_id: dict[str, dict[str, Any]] = {}
    for row in imported_hedge_rows:
        key = position_row_id(row)
        if key:
            imported_by_id[key] = row

    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for existing in existing_hedge_rows:
        key = position_row_id(existing)
        if not key or key in seen_ids:
            continue
        if key in imported_by_id:
            merged.append(dict(imported_by_id[key]))
            seen_ids.add(key)
        else:
            merged.append(dict(existing))
            seen_ids.add(key)

    for key, row in imported_by_id.items():
        if key not in seen_ids:
            merged.append(dict(row))
            seen_ids.add(key)

    return merged
