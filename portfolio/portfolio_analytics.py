"""
portfolio_analytics.py — PnL, drawdown, and attribution from portfolio positions.

Consumes price series and ontology-backed position metadata to compute
per-position and portfolio-level metrics.

Attribution is weighted by notional value when quantity and current price are
available. Futures use quantity x price x contract multiplier; securities use
the same formula with multiplier 1.0. Missing quantities fall back to equal
weight.
"""

from __future__ import annotations

import math

import pandas as pd

from portfolio.economic_exposure import (
    economic_exposure_fields,
    exposure_group_key,
    scale_gross_notional_for_exposure,
)
from portfolio.instruments import (
    NEAR_ZERO_NET_RATIO,
    chart_price_symbol,
    display_ticker,
    notional_value,
    position_row_id,
    signed_notional,
)


def _position_pnl(
    cost_basis: float | None,
    current_price: float,
    direction: str,
    quantity: float | None = None,
    contract_multiplier: float = 1.0,
    quote_to_base_rate: float | None = 1.0,
) -> tuple[float | None, float | None, float | None]:
    """Unrealized PnL % plus total and per-unit dollars, direction-adjusted."""
    if cost_basis is None or cost_basis == 0:
        return None, None, None
    if direction == "short":
        pnl_per_unit = cost_basis - current_price
    else:
        pnl_per_unit = current_price - cost_basis
    pnl_pct = (pnl_per_unit / cost_basis) * 100
    pnl_dollar = None
    if quantity is not None:
        pnl_dollar = pnl_per_unit * abs(quantity) * contract_multiplier
        if quote_to_base_rate is not None:
            pnl_dollar *= quote_to_base_rate
    return round(pnl_pct, 2), round(pnl_dollar, 4) if pnl_dollar is not None else None, round(pnl_per_unit, 4)


def _drawdown_52w(
    price_series: pd.Series,
    direction: str,
) -> tuple[float | None, float | None]:
    """52-week peak and drawdown from it.

    For longs: peak = 52w high, dd = (current - peak) / peak.
    For shorts: peak = 52w low (best PnL point), dd = (peak - current) / peak.
    Both return <= 0 when in drawdown.
    """
    if price_series.empty or len(price_series) < 2:
        return None, None
    latest_date = price_series.index[-1]
    cutoff = latest_date - pd.Timedelta(days=365)
    window = price_series[price_series.index >= cutoff]
    if window.empty or len(window) < 2:
        window = price_series
    current = float(window.iloc[-1])

    if direction == "short":
        peak = float(window.min())
        if peak == 0:
            return peak, None
        dd_pct = ((peak - current) / peak) * 100
    else:
        peak = float(window.max())
        if peak == 0:
            return peak, None
        dd_pct = ((current - peak) / peak) * 100
    return round(peak, 4), round(dd_pct, 2)


def _period_return(
    price_series: pd.Series,
    calendar_days: int,
    direction: str,
) -> float | None:
    """Return over the last N calendar days (direction-adjusted)."""
    if price_series.empty or len(price_series) < 2:
        return None
    latest_date = price_series.index[-1]
    cutoff = latest_date - pd.Timedelta(days=calendar_days)
    older = price_series[price_series.index <= cutoff]
    if older.empty:
        old_price = float(price_series.iloc[0])
    else:
        old_price = float(older.iloc[-1])
    new_price = float(price_series.iloc[-1])
    if old_price == 0:
        return None
    if direction == "short":
        ret = ((old_price - new_price) / old_price) * 100
    else:
        ret = ((new_price - old_price) / old_price) * 100
    return round(ret, 2)


def _portfolio_summary(per_position: dict) -> dict:
    """Aggregate per-position metrics into portfolio-level summary."""
    pnl_values: list[tuple[str, float]] = []
    drawdowns: list[tuple[str, float]] = []
    weekly_contribs: list[float] = []
    monthly_contribs: list[float] = []
    weekly_returns: list[tuple[str, float]] = []

    for ticker, m in per_position.items():
        if m["unrealized_pnl_pct"] is not None:
            pnl_values.append((ticker, m["unrealized_pnl_pct"]))
        if m["drawdown_from_52w_pct"] is not None:
            drawdowns.append((ticker, m["drawdown_from_52w_pct"]))
        if m["weekly_contribution_pct"] is not None:
            weekly_contribs.append(m["weekly_contribution_pct"])
        if m["monthly_contribution_pct"] is not None:
            monthly_contribs.append(m["monthly_contribution_pct"])
        if m["weekly_return_pct"] is not None:
            weekly_returns.append((ticker, m["weekly_return_pct"]))

    profitable = sum(1 for _, pnl in pnl_values if pnl > 0)
    losing = sum(1 for _, pnl in pnl_values if pnl < 0)
    avg_pnl = round(sum(p for _, p in pnl_values) / len(pnl_values), 2) if pnl_values else None

    worst_dd = min(drawdowns, key=lambda x: x[1]) if drawdowns else None
    best_wk = max(weekly_returns, key=lambda x: x[1]) if weekly_returns else None
    worst_wk = min(weekly_returns, key=lambda x: x[1]) if weekly_returns else None

    return {
        "total_unrealized_pnl_pct": avg_pnl,
        "positions_profitable": profitable,
        "positions_losing": losing,
        "worst_drawdown": {"ticker": worst_dd[0], "pct": worst_dd[1]} if worst_dd else None,
        "best_performer": {"ticker": best_wk[0], "weekly_pct": best_wk[1]} if best_wk else None,
        "worst_performer": {"ticker": worst_wk[0], "weekly_pct": worst_wk[1]} if worst_wk else None,
        "weekly_portfolio_return_pct": round(sum(weekly_contribs), 2) if weekly_contribs else None,
        "monthly_portfolio_return_pct": round(sum(monthly_contribs), 2) if monthly_contribs else None,
    }


def _option_current_price(pos: dict) -> float | None:
    from portfolio.options_market_data import fetch_option_quote

    quote = fetch_option_quote(
        underlying_ticker=str(pos.get("underlying_ticker") or pos.get("ticker") or ""),
        option_expiration=str(pos.get("option_expiration") or ""),
        option_strike=float(pos.get("option_strike") or 0),
        option_type=str(pos.get("option_type") or ""),
        option_contract_symbol=pos.get("option_contract_symbol"),
    )
    price = quote.get("price")
    try:
        return float(price) if price is not None else None
    except (TypeError, ValueError):
        return None


def _sum_present(values: dict[str, float | None], leg_ids: list[str]) -> float:
    """Sum the non-None values for the given leg ids."""
    total = 0.0
    for leg_id in leg_ids:
        value = values.get(leg_id)
        if value is not None:
            total += value
    return total


def _is_near_zero_net(gross_current: float, net_current: float, gross_cost: float, net_cost: float) -> bool:
    """True when option legs nearly fully offset (real gross, ~zero net)."""
    if gross_current > 0:
        gross, net = gross_current, net_current
    elif gross_cost > 0:
        gross, net = gross_cost, net_cost
    else:
        return False
    return abs(net) <= NEAR_ZERO_NET_RATIO * gross


def _net_exposure_direction(
    pos_lookup: dict[str, dict],
    other_leg_ids: list[str],
    net_current: float,
    net_cost: float,
    gross_current: float,
    gross_cost: float,
) -> str | None:
    """Direction for an underlying's net exposure.

    A real share/equity (non-option) leg dictates direction; otherwise the sign
    of net option exposure is used, with near-perfect offsets reported as
    ``"neutral"``.
    """
    for leg_id in other_leg_ids:
        direction = str(pos_lookup[leg_id].get("direction") or "long").strip().lower()
        return "short" if direction == "short" else "long"
    if gross_current > 0:
        gross, net = gross_current, net_current
    elif gross_cost > 0:
        gross, net = gross_cost, net_cost
    else:
        return None
    if abs(net) <= NEAR_ZERO_NET_RATIO * gross:
        return "neutral"
    return "long" if net > 0 else "short"


def compute_analytics(
    prices: dict[str, pd.Series],
    positions: list[dict],
) -> dict:
    """Compute per-position PnL, drawdown, and attribution.

    Attribution weights are based on current notional value. Positions without
    quantity fall back to equal-weight.
    """
    n_positions = len(positions)
    if n_positions == 0:
        return {"per_position": {}, "portfolio": {}}

    pos_lookup = {position_row_id(p): p for p in positions}

    # Pass 1: per-leg current price plus gross (base-adjusted) and signed notionals.
    current_prices: dict[str, float | None] = {}
    leg_current_notional: dict[str, float | None] = {}
    leg_cost_notional: dict[str, float | None] = {}
    leg_signed_current: dict[str, float | None] = {}
    leg_signed_cost: dict[str, float | None] = {}
    for leg_id, pos in pos_lookup.items():
        instrument_type = str(pos.get("instrument_type") or "security").strip().lower()
        display = display_ticker(pos)
        price_key = chart_price_symbol(pos) if instrument_type != "option" else display
        series = _price_series(prices, display, price_key)
        if instrument_type == "option":
            cp = _option_current_price(pos)
        else:
            cp = float(series.iloc[-1]) if series is not None and not series.empty else None
        current_prices[leg_id] = cp
        quantity = pos.get("quantity", pos.get("shares"))
        contract_multiplier = pos.get("contract_multiplier") or 1.0
        notional_base = _float_or_none(pos.get("notional_base"))
        current_notional = _position_base_notional_value(pos, quantity, cp, contract_multiplier)
        cost_notional = (
            notional_base
            if notional_base is not None
            else _position_base_notional_value(pos, quantity, pos.get("cost_basis"), contract_multiplier)
        )
        leg_current_notional[leg_id] = current_notional
        leg_cost_notional[leg_id] = cost_notional
        if instrument_type == "option":
            leg_signed_current[leg_id] = signed_notional(current_notional, pos)
            leg_signed_cost[leg_id] = signed_notional(cost_notional, pos)
        else:
            leg_signed_current[leg_id] = None
            leg_signed_cost[leg_id] = None

    # Group legs by underlying exposure key (display ticker) and aggregate option
    # legs into a single net directional exposure per underlying.
    exposure_groups: dict[str, dict[str, list[str]]] = {}
    for leg_id, pos in pos_lookup.items():
        key = exposure_group_key(pos)
        is_option = str(pos.get("instrument_type") or "security").strip().lower() == "option"
        group = exposure_groups.setdefault(key, {"option_legs": [], "other_legs": []})
        (group["option_legs"] if is_option else group["other_legs"]).append(leg_id)

    underlying_exposure: dict[str, dict] = {}
    for key, group in exposure_groups.items():
        option_legs = group["option_legs"]
        other_legs = group["other_legs"]
        has_options = bool(option_legs)
        gross_current = _sum_present(leg_current_notional, option_legs)
        net_current = _sum_present(leg_signed_current, option_legs)
        gross_cost = _sum_present(leg_cost_notional, option_legs)
        net_cost = _sum_present(leg_signed_cost, option_legs)
        # Size used for weighting is the absolute net option exposure (net debit).
        option_size_current = abs(net_current) if has_options else 0.0
        net_direction = _net_exposure_direction(
            pos_lookup, other_legs, net_current, net_cost, gross_current, gross_cost
        )
        near_zero = (
            _is_near_zero_net(gross_current, net_current, gross_cost, net_cost)
            if has_options and not other_legs
            else False
        )
        underlying_exposure[key] = {
            "has_options": has_options,
            "option_gross_current": gross_current,
            "option_net_current": net_current,
            "option_gross_cost": gross_cost,
            "option_net_cost": net_cost,
            "option_size_current": option_size_current,
            "net_direction": net_direction,
            "near_zero_net": near_zero,
        }

    # Weights: non-option legs are weighted on their own gross notional; option
    # legs of an underlying share that underlying's absolute net size, split
    # pro-rata by each leg's gross premium so summed leg weights equal the net.
    total_notional = 0.0
    for leg_id, pos in pos_lookup.items():
        if str(pos.get("instrument_type") or "security").strip().lower() == "option":
            continue
        value = leg_current_notional.get(leg_id)
        scaled = scale_gross_notional_for_exposure(value, pos)
        if scaled is not None:
            total_notional += scaled
    for agg in underlying_exposure.values():
        total_notional += agg["option_size_current"]

    weights: dict[str, float] = {}
    if total_notional > 0:
        for leg_id, pos in pos_lookup.items():
            key = exposure_group_key(pos)
            is_option = str(pos.get("instrument_type") or "security").strip().lower() == "option"
            if is_option:
                agg = underlying_exposure.get(key, {})
                gross = agg.get("option_gross_current") or 0.0
                size = agg.get("option_size_current") or 0.0
                leg_cur = leg_current_notional.get(leg_id)
                if gross > 0 and leg_cur is not None:
                    weights[leg_id] = (size / total_notional) * (leg_cur / gross)
                else:
                    weights[leg_id] = 0.0
            else:
                leg_cur = scale_gross_notional_for_exposure(leg_current_notional.get(leg_id), pos)
                weights[leg_id] = (leg_cur / total_notional) if leg_cur is not None else 0.0
    else:
        eq = 1.0 / n_positions
        weights = {leg_id: eq for leg_id in pos_lookup}

    per_position: dict[str, dict] = {}
    for leg_id, pos in pos_lookup.items():
        instrument_type = str(pos.get("instrument_type") or "security").strip().lower()
        display = display_ticker(pos)
        group_key = exposure_group_key(pos)
        price_key = chart_price_symbol(pos) if instrument_type != "option" else display
        series = _price_series(prices, display, price_key)
        direction = pos.get("direction", "long")
        cost_basis = pos.get("cost_basis")
        quantity = pos.get("quantity", pos.get("shares"))
        contract_multiplier = pos.get("contract_multiplier") or 1.0
        cp = current_prices[leg_id]
        pnl_quote_to_base = _spot_fx_quote_to_base_rate(pos, cp) if instrument_type == "spot_fx" else 1.0

        pnl_pct, pnl_dollar, pnl_per_unit = (
            _position_pnl(cost_basis, cp, direction, quantity, contract_multiplier, pnl_quote_to_base)
            if cp is not None
            else (None, None, None)
        )
        notional_base = _float_or_none(pos.get("notional_base"))
        cost_basis_base = _float_or_none(pos.get("cost_basis_base"))
        current_notional = leg_current_notional[leg_id]
        cost_notional = leg_cost_notional[leg_id]
        signed_current = leg_signed_current[leg_id]
        signed_cost = leg_signed_cost[leg_id]
        agg = underlying_exposure.get(group_key, {})
        underlying_has_options = bool(agg.get("has_options"))
        high_52w, dd_pct = (
            (None, None)
            if instrument_type == "option"
            else (_drawdown_52w(series, direction) if series is not None else (None, None))
        )
        weekly_ret = (
            None
            if instrument_type == "option"
            else (_period_return(series, 7, direction) if series is not None else None)
        )
        monthly_ret = (
            None
            if instrument_type == "option"
            else (_period_return(series, 30, direction) if series is not None else None)
        )

        w = weights.get(leg_id, 0.0)
        weekly_contrib = round(weekly_ret * w, 4) if weekly_ret is not None else None
        monthly_contrib = round(monthly_ret * w, 4) if monthly_ret is not None else None

        exposure_meta = economic_exposure_fields(pos)
        per_position[leg_id] = {
            "ticker": pos.get("ticker"),
            "display_ticker": display,
            "exposure_group_key": group_key,
            **exposure_meta,
            "position_id": leg_id,
            "cost_basis": cost_basis,
            "current_price": cp,
            "shares": quantity,
            "quantity": quantity,
            "instrument_type": instrument_type,
            "contract_multiplier": contract_multiplier,
            "underlying_ticker": pos.get("underlying_ticker"),
            "option_contract_symbol": pos.get("option_contract_symbol"),
            "option_expiration": pos.get("option_expiration"),
            "option_strike": pos.get("option_strike"),
            "option_type": pos.get("option_type"),
            "current_notional": round(current_notional, 2) if current_notional is not None else None,
            "cost_notional": round(cost_notional, 2) if cost_notional is not None else None,
            "signed_current_notional": round(signed_current, 2) if signed_current is not None else None,
            "signed_cost_notional": round(signed_cost, 2) if signed_cost is not None else None,
            "net_current_notional": (round(agg["option_net_current"], 2) if underlying_has_options else None),
            "net_cost_notional": round(agg["option_net_cost"], 2) if underlying_has_options else None,
            "net_direction": agg.get("net_direction") if underlying_has_options else None,
            "near_zero_net": bool(agg.get("near_zero_net")) if underlying_has_options else None,
            "notional_base": round(notional_base, 2) if notional_base is not None else None,
            "cost_basis_base": round(cost_basis_base, 4) if cost_basis_base is not None else None,
            "currency": pos.get("currency"),
            "base_currency": pos.get("base_currency"),
            "valuation_status": pos.get("valuation_status"),
            "direction": direction,
            "unrealized_pnl_pct": pnl_pct,
            "unrealized_pnl_dollar": pnl_dollar,
            "unrealized_pnl_per_unit": pnl_per_unit,
            "high_52w": high_52w,
            "drawdown_from_52w_pct": dd_pct,
            "weekly_return_pct": weekly_ret,
            "monthly_return_pct": monthly_ret,
            "weight": round(w, 4),
            "weekly_contribution_pct": weekly_contrib,
            "monthly_contribution_pct": monthly_contrib,
        }

    portfolio = _portfolio_summary(per_position)
    return {"per_position": per_position, "portfolio": portfolio}


def _price_series(prices: dict, display: str, price_key: str) -> pd.Series | None:
    for key in (display, price_key):
        series = prices.get(key)
        if series is not None and not getattr(series, "empty", True):
            return series
    return None


def _float_or_none(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) and math.isfinite(out) else None


def _base_notional_value(
    quantity,
    price,
    contract_multiplier,
    fx_rate_to_base,
) -> float | None:
    local_notional = notional_value(quantity, price, contract_multiplier)
    if local_notional is None:
        return None
    fx_rate = _float_or_none(fx_rate_to_base)
    if fx_rate is None:
        return local_notional
    if fx_rate <= 0:
        return None
    return local_notional * fx_rate


def _position_base_notional_value(
    position: dict,
    quantity,
    price,
    contract_multiplier,
) -> float | None:
    if str(position.get("instrument_type") or "").strip().lower() == "spot_fx":
        return _spot_fx_base_notional_value(position, quantity, price)
    return _base_notional_value(quantity, price, contract_multiplier, position.get("fx_rate_to_base"))


def _spot_fx_base_notional_value(position: dict, quantity, price) -> float | None:
    try:
        qty = abs(float(quantity))
        px = float(price)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(qty) and math.isfinite(px)) or qty <= 0 or px <= 0:
        return None
    rate = _spot_fx_quote_to_base_rate(position, px)
    if rate is None or rate <= 0:
        return None
    return qty * px * rate


def _spot_fx_quote_to_base_rate(position: dict, price) -> float | None:
    fx_base = str(position.get("fx_base_currency") or "").strip().upper()
    fx_quote = str(position.get("fx_quote_currency") or position.get("currency") or "").strip().upper()
    portfolio_base = str(position.get("base_currency") or "USD").strip().upper()
    px = _float_or_none(price)
    if not fx_quote:
        return _float_or_none(position.get("fx_rate_to_base"))
    if fx_quote == portfolio_base:
        return 1.0
    if fx_base == portfolio_base and px is not None and px > 0:
        return 1.0 / px
    return _float_or_none(position.get("fx_rate_to_base"))
