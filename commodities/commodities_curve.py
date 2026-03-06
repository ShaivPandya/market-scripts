#!/usr/bin/env python3
"""
Commodities forward curve (term structure) for the web dashboard.

Provides current and historical (N-day lookback) futures curves for:
- WTI Crude Oil (CL)
- Brent Crude Oil (BZ)
- Natural Gas (NG)

Terminal:
  python commodities/commodities_curve.py

GUI/API:
  Accessed via api/routers/commodities_curve.py
"""

from __future__ import annotations

import warnings
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Month code mapping: 1-based month index -> Yahoo Finance futures month code
MONTH_CODES = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

COMMODITIES = [
    ("CL", "WTI Crude Oil", "$/bbl"),
    ("BZ", "Brent Crude Oil", "$/bbl"),
    ("NG", "Natural Gas", "$/MMBtu"),
]

VALID_CODES = {c[0] for c in COMMODITIES}


def _build_futures_tickers(
    base: str,
    num_months: int = 12,
    ref_date: Optional[date] = None,
) -> list[dict]:
    """
    Build Yahoo Finance futures ticker symbols for consecutive delivery months
    starting from next month (current month contracts are typically expired).
    """
    if ref_date is None:
        ref_date = date.today()

    contracts: list[dict] = []
    # Start from next month — current month's contract is typically expired
    month = ref_date.month + 1
    year = ref_date.year
    if month > 12:
        month = 1
        year += 1

    for _ in range(num_months):
        code = MONTH_CODES[month]
        yy = year % 100
        ticker = f"{base}{code}{yy:02d}.NYM"
        label = f"{MONTH_NAMES[month]} {year}"
        contracts.append({
            "ticker": ticker,
            "month": month,
            "year": year,
            "label": label,
        })
        month += 1
        if month > 12:
            month = 1
            year += 1

    return contracts


def _empty_point(contract: dict) -> dict:
    return {
        "ticker": contract["ticker"],
        "label": contract["label"],
        "month": contract["month"],
        "year": contract["year"],
        "current": None,
        "historical": None,
        "change": None,
        "change_pct": None,
        "current_date": None,
        "historical_date": None,
    }


def _fetch_curve_prices(
    contracts: list[dict],
    lookback_days: int = 30,
) -> tuple[list[dict], list[str]]:
    """Fetch current and historical closing prices for a list of futures contracts."""
    warn: list[str] = []
    tickers = [c["ticker"] for c in contracts]

    fetch_days = lookback_days + 14  # buffer for weekends/holidays
    try:
        raw = yf.download(
            tickers=tickers,
            period=f"{fetch_days}d",
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as e:
        return [], [f"yfinance download failed: {e}"]

    if raw is None or raw.empty:
        return [], ["No data returned from yfinance"]

    is_multi = isinstance(raw.columns, pd.MultiIndex)

    points: list[dict] = []
    for contract in contracts:
        tk = contract["ticker"]
        try:
            if is_multi:
                if tk not in raw.columns.get_level_values(0):
                    warn.append(f"{tk}: no data returned")
                    points.append(_empty_point(contract))
                    continue
                series = raw[tk]["Close"].dropna()
            else:
                series = raw["Close"].dropna()

            if series.empty:
                warn.append(f"{tk}: empty series")
                points.append(_empty_point(contract))
                continue

            if hasattr(series.index, "tz") and series.index.tz is not None:
                series.index = series.index.tz_localize(None)

            current_price = float(series.iloc[-1])
            current_date = series.index[-1]

            target_date = pd.Timestamp(date.today()) - pd.Timedelta(days=lookback_days)
            hist_eligible = series[series.index <= target_date]

            historical_price: Optional[float] = None
            historical_date_str: Optional[str] = None
            if not hist_eligible.empty:
                historical_price = float(hist_eligible.iloc[-1])
                historical_date_str = hist_eligible.index[-1].date().isoformat()

            change: Optional[float] = None
            change_pct: Optional[float] = None
            if historical_price is not None:
                change = round(current_price - historical_price, 4)
                if historical_price != 0:
                    change_pct = round(
                        (current_price - historical_price) / historical_price * 100, 2
                    )

            points.append({
                "ticker": tk,
                "label": contract["label"],
                "month": contract["month"],
                "year": contract["year"],
                "current": round(current_price, 4),
                "historical": round(historical_price, 4) if historical_price is not None else None,
                "change": change,
                "change_pct": change_pct,
                "current_date": current_date.date().isoformat(),
                "historical_date": historical_date_str,
            })

        except Exception as exc:
            warn.append(f"{tk}: {exc}")
            points.append(_empty_point(contract))

    return points, warn


def _analyze_curve(points: list[dict]) -> dict:
    """Determine contango vs backwardation and compute spread metrics."""
    valid = [p for p in points if p["current"] is not None]
    total = len(points)
    available = len(valid)

    if available < 2:
        return {
            "front_month_price": valid[0]["current"] if available == 1 else None,
            "back_month_price": None,
            "spread": None,
            "spread_pct": None,
            "shape": "N/A",
            "contracts_available": available,
            "contracts_total": total,
        }

    front = valid[0]["current"]
    back = valid[-1]["current"]
    spread = round(back - front, 4)
    spread_pct = round((back - front) / front * 100, 2) if front != 0 else None

    if spread > 0.01:
        shape = "Contango"
    elif spread < -0.01:
        shape = "Backwardation"
    else:
        shape = "Flat"

    return {
        "front_month_price": front,
        "back_month_price": back,
        "spread": spread,
        "spread_pct": spread_pct,
        "shape": shape,
        "contracts_available": available,
        "contracts_total": total,
    }


def get_data(commodity: str = "CL", lookback_days: int = 30) -> dict:
    """
    Build commodities forward curve snapshot.

    Args:
        commodity: One of "CL" (WTI), "BZ" (Brent), "NG" (Natural Gas)
        lookback_days: Days back for historical comparison curve

    Returns:
        JSON-serializable dict with curve data.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")

    if commodity not in VALID_CODES:
        raise ValueError(f"Invalid commodity: {commodity}. Must be one of {VALID_CODES}")

    commodity_info = next(c for c in COMMODITIES if c[0] == commodity)
    code, name, unit = commodity_info

    # Fetch extra contracts to compensate for expired front-month contracts
    contracts = _build_futures_tickers(base=code, num_months=15)
    points, curve_warnings = _fetch_curve_prices(contracts, lookback_days=lookback_days)

    # Strip leading empty points (expired contracts) — not worth warning about
    while points and points[0]["current"] is None:
        expired_tk = points.pop(0)["ticker"]
        curve_warnings = [w for w in curve_warnings if not w.startswith(expired_tk)]

    # Keep only 12 contracts for the curve
    trimmed = points[12:]
    points = points[:12]

    # Strip warnings for contracts that were trimmed (overflow buffer)
    trimmed_tickers = {p["ticker"] for p in trimmed}
    curve_warnings = [w for w in curve_warnings if not any(w.startswith(tk) for tk in trimmed_tickers)]

    analysis = _analyze_curve(points)

    return {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "commodity_code": code,
        "commodity_name": name,
        "unit": unit,
        "lookback_days": lookback_days,
        "commodities": [{"code": c[0], "name": c[1]} for c in COMMODITIES],
        "analysis": analysis,
        "points": points,
        "warnings": curve_warnings,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(get_data(), indent=2))
