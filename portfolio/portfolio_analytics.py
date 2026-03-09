"""
portfolio_analytics.py — PnL, drawdown, and attribution from portfolio positions.

Consumes price series (from portfolio_dashboard) and position metadata
(from portfolio_db) to compute per-position and portfolio-level metrics.

Attribution is weighted by notional value (shares x current price) when
shares are available, falling back to equal-weight otherwise.
"""

from __future__ import annotations

import pandas as pd


def _position_pnl(
    cost_basis: float | None,
    current_price: float,
    direction: str,
) -> tuple[float | None, float | None]:
    """Unrealized PnL % and $ (per-share, direction-adjusted)."""
    if cost_basis is None or cost_basis == 0:
        return None, None
    if direction == "short":
        pnl_dollar = cost_basis - current_price
    else:
        pnl_dollar = current_price - cost_basis
    pnl_pct = (pnl_dollar / cost_basis) * 100
    return round(pnl_pct, 2), round(pnl_dollar, 4)


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


def compute_analytics(
    prices: dict[str, pd.Series],
    positions: list[dict],
) -> dict:
    """Compute per-position PnL, drawdown, and attribution.

    Attribution weights are based on notional value (shares x current price).
    Positions without shares fall back to equal-weight.
    """
    n_positions = len(positions)
    if n_positions == 0:
        return {"per_position": {}, "portfolio": {}}

    pos_lookup = {p["ticker"]: p for p in positions}

    # First pass: compute current prices and notional values for weighting
    notionals: dict[str, float] = {}
    current_prices: dict[str, float | None] = {}
    for ticker, pos in pos_lookup.items():
        series = prices.get(ticker)
        cp = float(series.iloc[-1]) if series is not None and not series.empty else None
        current_prices[ticker] = cp
        shares = pos.get("shares")
        if cp is not None and shares is not None and shares > 0:
            notionals[ticker] = shares * cp

    # Build weights: notional-weighted if any shares exist, else equal-weight
    weights: dict[str, float] = {}
    if notionals:
        total_notional = sum(notionals.values())
        for ticker in pos_lookup:
            if ticker in notionals and total_notional > 0:
                weights[ticker] = notionals[ticker] / total_notional
            else:
                weights[ticker] = 0.0
    else:
        eq = 1.0 / n_positions
        weights = {t: eq for t in pos_lookup}

    # Second pass: compute per-position metrics
    per_position: dict[str, dict] = {}
    for ticker, pos in pos_lookup.items():
        series = prices.get(ticker)
        direction = pos.get("direction", "long")
        cost_basis = pos.get("cost_basis")
        shares = pos.get("shares")
        cp = current_prices[ticker]

        pnl_pct, pnl_dollar = _position_pnl(cost_basis, cp, direction) if cp is not None else (None, None)
        high_52w, dd_pct = _drawdown_52w(series, direction) if series is not None else (None, None)
        weekly_ret = _period_return(series, 7, direction) if series is not None else None
        monthly_ret = _period_return(series, 30, direction) if series is not None else None

        w = weights.get(ticker, 0.0)
        weekly_contrib = round(weekly_ret * w, 4) if weekly_ret is not None else None
        monthly_contrib = round(monthly_ret * w, 4) if monthly_ret is not None else None

        per_position[ticker] = {
            "cost_basis": cost_basis,
            "current_price": cp,
            "shares": shares,
            "direction": direction,
            "unrealized_pnl_pct": pnl_pct,
            "unrealized_pnl_dollar": pnl_dollar,
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
