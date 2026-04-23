"""
Commodity Research — proxy-based idea scoring engine.

Ranks each commodity in the universe by a deterministic composite score
built from momentum, curve shape, macro regime, relative performance,
and price velocity.  All supply/demand signals are heuristic proxies
and are labelled accordingly.

Entry point:
    build_commodity_research() -> dict
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from commodities.commodities_dashboard import COMMODITIES, fetch_commodities_data

logger = logging.getLogger("api.commodity_research")

# -- Factor weights (must sum to 1.0) ----------------------------------------

FACTOR_WEIGHTS: dict[str, float] = {
    "momentum": 0.30,
    "relative_value": 0.20,
    "macro": 0.20,
    "supply_demand": 0.20,
    "velocity": 0.10,
}

# Commodity -> sector classification
SECTOR_MAP: dict[str, str] = {
    "Gold": "metals",
    "Silver": "metals",
    "Copper": "metals",
    "Platinum": "metals",
    "Palladium": "metals",
    "Aluminum": "metals",
    "WTI Crude Oil": "energy",
    "Brent Crude Oil": "energy",
    "Natural Gas": "energy",
    "Dutch TTF Gas": "energy",
}

# Commodity name -> curve code (only energy has forward curves)
CURVE_CODES: dict[str, str] = {
    "WTI Crude Oil": "CL",
    "Brent Crude Oil": "BZ",
    "Natural Gas": "NG",
    "Dutch TTF Gas": "TTF",
}

STALE_THRESHOLD_DAYS = 7


# -- Utility helpers ----------------------------------------------------------


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _date_return(series: pd.Series | None, days_back: int) -> float | None:
    """Return over approximately *days_back* calendar days, date-based."""
    if series is None or len(series) < 2:
        return None
    end_val = _safe_float(series.iloc[-1])
    if end_val is None or end_val == 0:
        return None

    target = series.index[-1] - pd.Timedelta(days=days_back)
    earlier = series[series.index <= target]
    if earlier.empty:
        # Target is before all data — use first point if coverage is ≥ 80%
        actual_days = (series.index[-1] - series.index[0]).days
        if actual_days < days_back * 0.8:
            return None
        start_val = _safe_float(series.iloc[0])
    else:
        start_val = _safe_float(earlier.iloc[-1])

    if start_val is None or start_val == 0:
        return None
    return (end_val / start_val - 1) * 100


# -- Data fetching (parallel) ------------------------------------------------


def _fetch_daily_prices() -> dict | None:
    try:
        data = fetch_commodities_data("Daily")
        if "error" in data:
            logger.warning("daily prices error: %s", data["error"])
            return None
        return data
    except Exception:
        logger.exception("daily prices fetch failed")
        return None


def _fetch_monthly_prices() -> dict | None:
    try:
        data = fetch_commodities_data("Monthly")
        if "error" in data:
            logger.warning("monthly prices error: %s", data["error"])
            return None
        return data
    except Exception:
        logger.exception("monthly prices fetch failed")
        return None


def _fetch_curves() -> dict[str, dict]:
    results: dict[str, dict] = {}
    for name, code in CURVE_CODES.items():
        try:
            from commodities.commodities_curve import get_data as get_curve

            results[name] = get_curve(commodity=code)
        except Exception:
            logger.exception("curve fetch failed for %s", code)
    return results


def _fetch_macro() -> dict | None:
    try:
        from api.signal_aggregator import build_signal_aggregator

        return build_signal_aggregator()
    except Exception:
        logger.exception("signal aggregator fetch failed")
        return None


def _fetch_prices_sequential() -> tuple[dict | None, dict | None]:
    # Serialize daily + monthly to avoid nested concurrency with yfinance's
    # internal threadpool (threads=True in fetch_commodities_data), which
    # otherwise corrupts per-ticker results when fired in parallel.
    return _fetch_daily_prices(), _fetch_monthly_prices()


def _fetch_all() -> tuple[dict | None, dict | None, dict[str, dict], dict | None]:
    daily: dict | None = None
    monthly: dict | None = None
    curves: dict[str, dict] = {}
    macro: dict | None = None

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures: dict[Future[Any], str] = {
            pool.submit(_fetch_prices_sequential): "prices",
            pool.submit(_fetch_curves): "curves",
            pool.submit(_fetch_macro): "macro",
        }
        for fut in as_completed(futures, timeout=180):
            key = futures[fut]
            try:
                result = fut.result()
                if key == "prices":
                    daily, monthly = result
                elif key == "curves":
                    curves = result or {}
                elif key == "macro":
                    macro = result
            except Exception:
                logger.exception("fetch failed for %s", key)

    return daily, monthly, curves, macro


# -- Scoring functions --------------------------------------------------------


def _score_momentum(
    ret_1m: float | None,
    ret_3m: float | None,
    ret_12m: float | None,
) -> tuple[float | None, str]:
    vals = {"1m": ret_1m, "3m": ret_3m, "12m": ret_12m}
    weights = {"1m": 0.20, "3m": 0.35, "12m": 0.45}

    total_w = 0.0
    weighted_sum = 0.0
    for k in ("1m", "3m", "12m"):
        v = vals[k]
        if v is not None:
            normed = clamp01((v + 30) / 60)
            weighted_sum += weights[k] * normed
            total_w += weights[k]

    if total_w == 0:
        return None, "no_data"

    score = weighted_sum / total_w

    if score > 0.65:
        label = "strong_up"
    elif score > 0.50:
        label = "moderate_up"
    elif score >= 0.35:
        label = "neutral"
    elif score >= 0.20:
        label = "moderate_down"
    else:
        label = "strong_down"

    return round(score, 4), label


def _score_curve(shape: str | None, spread_pct: float | None) -> float | None:
    if shape is None or shape == "N/A":
        return None
    sp = abs(spread_pct) if spread_pct is not None else 0.0
    if shape == "Backwardation":
        return round(clamp01(0.6 + sp / 30), 4)
    elif shape == "Contango":
        return round(clamp01(0.4 - sp / 30), 4)
    return 0.5  # Flat


def _score_relative_value(
    commodity_3m: float | None,
    median_3m: float,
) -> float:
    if commodity_3m is None:
        return 0.5
    return round(clamp01(0.5 + (commodity_3m - median_3m) / 40), 4)


def _score_macro(regime_score: float | None) -> float | None:
    if regime_score is None:
        return None
    return round(clamp01(regime_score / 100), 4)


def _score_supply_demand(
    trend: float | None,
    curve: float | None,
    cross_rank: float | None,
    macro: float | None,
) -> float | None:
    parts = {
        "trend": (0.30, trend),
        "curve": (0.25, curve),
        "cross_rank": (0.20, cross_rank),
        "macro": (0.25, macro),
    }
    total_w = 0.0
    weighted = 0.0
    for w, v in parts.values():
        if v is not None:
            weighted += w * v
            total_w += w
    if total_w == 0:
        return None
    return round(weighted / total_w, 4)


def _score_velocity(ret_1m: float | None, ret_3m: float | None) -> float | None:
    if ret_1m is None or ret_3m is None:
        return None
    annualized_3m = ret_3m * 4
    acceleration = ret_1m - annualized_3m / 12
    return round(clamp01(0.5 + acceleration / 20), 4)


# -- Composite ----------------------------------------------------------------


def _compute_composite(
    scores: dict[str, float | None],
) -> tuple[float | None, dict[str, float]]:
    valid = {k: v for k, v in scores.items() if v is not None}
    total_available = sum(FACTOR_WEIGHTS[k] for k in valid)
    if total_available <= 0:
        return None, {}

    effective_weights: dict[str, float] = {}
    for k in valid:
        effective_weights[k] = FACTOR_WEIGHTS[k] / total_available

    composite = sum(effective_weights[k] * valid[k] for k in valid)
    return round(composite, 4), effective_weights


# -- Direction and confidence -------------------------------------------------


def assign_direction(composite: float, trend_label: str) -> str:
    composite_100 = composite * 100
    if composite_100 >= 60 and trend_label in ("strong_up", "moderate_up"):
        return "long"
    if composite_100 <= 40 and trend_label in ("strong_down", "moderate_down"):
        return "short"
    return "watchlist"


def assign_confidence(composite: float, data_quality: dict[str, str]) -> str:
    quality_ok = sum(1 for v in data_quality.values() if v == "ok")
    quality_ratio = quality_ok / max(len(data_quality), 1)
    distance = abs(composite - 0.5)

    if distance > 0.20 and quality_ratio >= 0.75:
        return "high"
    if distance > 0.10 and quality_ratio >= 0.50:
        return "medium"
    return "low"


# -- Rationale bullets --------------------------------------------------------


def _generate_rationale(
    commodity: str,
    returns: dict[str, float | None],
    trend_label: str,
    direction: str,
    curve_shape: str | None,
    cross_rank: float | None,
    macro_label: str | None,
    macro_outlook: str | None,
) -> list[str]:
    bullets: list[str] = []

    r3m = returns.get("3m")
    if r3m is not None:
        sign = "+" if r3m >= 0 else ""
        bullets.append(f"{sign}{r3m:.1f}% over 3M; trend regime is {trend_label}")

    if curve_shape and curve_shape not in ("N/A", None):
        interp = "supply tightness" if curve_shape == "Backwardation" else "oversupply signals"
        bullets.append(f"Forward curve in {curve_shape.lower()}, suggesting {interp}")
    elif cross_rank is not None:
        pos = "above" if cross_rank > 0.5 else "below"
        bullets.append(f"Relative performance ranks {pos} commodity complex median")

    if macro_label is not None:
        bullets.append(f"Macro regime: {macro_label}; forward outlook: {macro_outlook or 'n/a'}")

    if direction != "watchlist":
        bullets.append(f"Composite score supports {direction} bias (proxy-based)")

    return bullets[:4]


# -- Data quality -------------------------------------------------------------


def _check_staleness(series: pd.Series | None) -> str:
    if series is None or series.empty:
        return "missing"
    try:
        last_date = pd.Timestamp(series.index[-1])
        if last_date.tzinfo is not None:
            last_date = last_date.tz_localize(None)
        age = (pd.Timestamp(datetime.now()) - last_date).days
        return "stale" if age > STALE_THRESHOLD_DAYS else "ok"
    except Exception:
        return "missing"


# -- Main entry point ---------------------------------------------------------


def build_commodity_research() -> dict[str, Any]:
    daily, monthly, curves, macro = _fetch_all()

    daily_prices = daily.get("commodities", {}) if daily else {}
    monthly_prices = monthly.get("commodities", {}) if monthly else {}

    # Extract macro regime
    macro_regime: dict[str, Any] = {"label": None, "score": None, "forward_outlook": None}
    macro_score_raw: float | None = None
    if macro is not None:
        regime = macro.get("regime", {})
        macro_regime = {
            "label": regime.get("label"),
            "score": _safe_float(regime.get("composite")),
            "forward_outlook": macro.get("forward_outlook", {}).get("label"),
        }
        macro_score_raw = macro_regime["score"]

    macro_factor = _score_macro(macro_score_raw)

    # Compute cross-sectional stats for relative value
    all_3m: dict[str, float] = {}
    for name in COMMODITIES:
        series = daily_prices.get(name)
        r = _date_return(series, 90)
        if r is not None:
            all_3m[name] = r
    median_3m = float(pd.Series(list(all_3m.values())).median()) if all_3m else 0.0

    # Cross-sectional rank (percentile)
    sorted_3m = sorted(all_3m.items(), key=lambda x: x[1])
    rank_map: dict[str, float] = {}
    n = len(sorted_3m)
    for i, (name, _) in enumerate(sorted_3m):
        rank_map[name] = i / max(n - 1, 1)

    # Score each commodity
    ideas: list[dict[str, Any]] = []
    any_degraded = False

    for name, ticker in COMMODITIES.items():
        sector = SECTOR_MAP.get(name, "other")
        daily_s = daily_prices.get(name)
        monthly_s = monthly_prices.get(name)

        # Data quality
        dq: dict[str, str] = {
            "prices_daily": _check_staleness(daily_s),
            "prices_monthly": _check_staleness(monthly_s),
            "curve": "n/a" if name not in CURVE_CODES else "ok",
            "macro": "ok" if macro is not None else "error",
        }

        # Returns
        ret_1m = _date_return(daily_s, 30)
        ret_3m = _date_return(daily_s, 90)
        ret_12m = _date_return(monthly_s, 365)

        # Spot price
        spot = _safe_float(daily_s.iloc[-1]) if daily_s is not None and len(daily_s) > 0 else None

        # Factor 1: Momentum
        mom_score, trend_label = _score_momentum(ret_1m, ret_3m, ret_12m)

        # Factor 2: Relative value / Curve
        curve_shape: str | None = None
        curve_spread: float | None = None
        rv_score: float | None = None

        if name in CURVE_CODES:
            curve_data = curves.get(name)
            if curve_data is not None:
                analysis = curve_data.get("analysis", {})
                curve_shape = analysis.get("shape")
                curve_spread = _safe_float(analysis.get("spread_pct"))
                rv_score = _score_curve(curve_shape, curve_spread)
            else:
                dq["curve"] = "error"
                rv_score = _score_relative_value(all_3m.get(name), median_3m)
        else:
            rv_score = _score_relative_value(all_3m.get(name), median_3m)

        # Factor 3: Macro alignment
        # (macro_factor computed once, same for all)

        # Factor 4: Supply/demand proxy
        cross_rank = rank_map.get(name)
        sd_score = _score_supply_demand(mom_score, rv_score, cross_rank, macro_factor)

        # Factor 5: Velocity
        vel_score = _score_velocity(ret_1m, ret_3m)

        # Composite
        factor_scores = {
            "momentum": mom_score,
            "relative_value": rv_score,
            "macro": macro_factor,
            "supply_demand": sd_score,
            "velocity": vel_score,
        }
        composite, effective_weights = _compute_composite(factor_scores)

        if composite is None:
            direction = "watchlist"
            confidence = "low"
            any_degraded = True
        else:
            direction = assign_direction(composite, trend_label)
            confidence = assign_confidence(composite, dq)

        # Check if any data quality issue
        if any(v in ("error", "missing", "stale") for v in dq.values()):
            any_degraded = True

        # Build factors detail
        factors_detail: dict[str, dict[str, Any]] = {}
        for fkey in FACTOR_WEIGHTS:
            sc = factor_scores.get(fkey)
            ew = effective_weights.get(fkey, 0.0)
            contrib = ew * sc if sc is not None else 0.0
            entry: dict[str, Any] = {
                "score": sc,
                "weight": round(ew, 4),
                "contribution": round(contrib, 4),
            }
            if fkey == "momentum":
                entry["label"] = trend_label
            if fkey == "relative_value":
                entry["source"] = (
                    "curve" if (name in CURVE_CODES and curve_shape not in (None, "N/A")) else "cross_section"
                )
            if fkey == "supply_demand":
                entry["proxy"] = True
            factors_detail[fkey] = entry

        # Rationale
        rationale = _generate_rationale(
            commodity=name,
            returns={"1m": ret_1m, "3m": ret_3m, "12m": ret_12m},
            trend_label=trend_label,
            direction=direction,
            curve_shape=curve_shape,
            cross_rank=cross_rank,
            macro_label=macro_regime["label"],
            macro_outlook=macro_regime["forward_outlook"],
        )

        # Price series for chart
        price_series: list[dict[str, Any]] = []
        if daily_s is not None and not daily_s.empty:
            for idx, val in daily_s.items():
                fv = _safe_float(val)
                if fv is not None:
                    dt = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
                    price_series.append({"date": dt, "value": fv})

        ideas.append(
            {
                "commodity": name,
                "ticker": ticker,
                "sector": sector,
                "spot_price": spot,
                "returns": {
                    "1m": round(ret_1m, 2) if ret_1m is not None else None,
                    "3m": round(ret_3m, 2) if ret_3m is not None else None,
                    "12m": round(ret_12m, 2) if ret_12m is not None else None,
                },
                "factors": factors_detail,
                "composite_score": round(composite * 100, 1) if composite is not None else None,
                "direction": direction,
                "confidence": confidence,
                "rationale": rationale,
                "data_quality": dq,
                "price_series": price_series,
            }
        )

    # Sort by composite score descending (None → bottom)
    ideas.sort(key=lambda x: x["composite_score"] if x["composite_score"] is not None else -1, reverse=True)

    # Summary
    longs = [i for i in ideas if i["direction"] == "long" and i["composite_score"] is not None]
    shorts = [i for i in ideas if i["direction"] == "short" and i["composite_score"] is not None]

    top_long = {"commodity": longs[0]["commodity"], "score": longs[0]["composite_score"]} if longs else None
    top_short = {"commodity": shorts[0]["commodity"], "score": shorts[0]["composite_score"]} if shorts else None

    # Macro tailwind/headwind — use the macro factor contribution
    with_macro = [i for i in ideas if i["factors"]["macro"]["score"] is not None]
    strongest_tailwind = None
    strongest_headwind = None
    if with_macro:
        best = max(with_macro, key=lambda x: x["factors"]["macro"]["score"])
        worst = min(with_macro, key=lambda x: x["factors"]["macro"]["score"])
        strongest_tailwind = {"commodity": best["commodity"], "macro_score": best["factors"]["macro"]["score"]}
        strongest_headwind = {"commodity": worst["commodity"], "macro_score": worst["factors"]["macro"]["score"]}

    ok_count = sum(1 for i in ideas if all(v == "ok" or v == "n/a" for v in i["data_quality"].values()))
    degraded_count = sum(1 for i in ideas if any(v in ("stale", "error") for v in i["data_quality"].values()))
    missing_count = sum(1 for i in ideas if any(v == "missing" for v in i["data_quality"].values()))

    return {
        "status": "degraded" if any_degraded else "ok",
        "timestamp": datetime.now().isoformat(),
        "macro_regime": macro_regime,
        "ideas": ideas,
        "summary": {
            "top_long": top_long,
            "top_short": top_short,
            "strongest_tailwind": strongest_tailwind,
            "strongest_headwind": strongest_headwind,
            "data_health": {
                "ok": ok_count,
                "degraded": degraded_count,
                "missing": missing_count,
            },
        },
        "methodology_note": (
            "Scores are proxy-based composites derived from price momentum, "
            "curve shape, macro regime, and cross-sectional rank. "
            "Supply/demand estimates are heuristic. Not investment advice."
        ),
    }
