"""
Commodity Proxy Screener.

Ranks commodities with a proxy-based composite built from orthogonal
technical and curve-structure factors. Market stress is shown as a shared
overlay, not as a ranked input.

Entry point:
    build_commodity_research() -> dict
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

import pandas as pd

from commodities.commodities_dashboard import COMMODITIES, fetch_commodities_data

logger = logging.getLogger("api.commodity_research")

SCHEMA_VERSION = 2
OWN_HISTORY_MIN_DAYS = 730
MIN_PERCENTILE_SAMPLES = 60
MIN_RELATIVE_STRENGTH_PEERS = 3
MIN_VOL_OBSERVATIONS = 40
DAILY_STALE_DAYS = 5
MONTHLY_STALE_DAYS = 45
CURVE_STALE_DAYS = 5
MACRO_STALE_DAYS = 7
QUALITY_SCORES = {
    "ok": 1.0,
    "degraded": 0.5,
    "stale": 0.5,
    "missing": 0.0,
    "error": 0.0,
}

FACTOR_CONFIG: dict[str, dict[str, Any]] = {
    "trend": {
        "configured_weight": 0.40,
        "display_label": "Trend",
        "category": "technical",
        "source": "own-history",
        "ranked": True,
    },
    "relative_strength": {
        "configured_weight": 0.30,
        "display_label": "Relative Strength",
        "category": "technical",
        "source": "family rank",
        "ranked": True,
    },
    "acceleration": {
        "configured_weight": 0.15,
        "display_label": "Acceleration",
        "category": "technical",
        "source": "own-history",
        "ranked": True,
    },
    "curve_structure": {
        "configured_weight": 0.15,
        "display_label": "Curve Structure",
        "category": "market_structure",
        "source": "curve",
        "ranked": True,
    },
    "market_stress_overlay": {
        "configured_weight": 0.00,
        "display_label": "Market Stress Overlay",
        "category": "overlay",
        "source": "overlay",
        "ranked": False,
    },
}

RANK_FACTOR_KEYS = tuple(key for key, meta in FACTOR_CONFIG.items() if meta["ranked"])

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


def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _normalize_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(s.index, errors="coerce")
    s.index = idx
    s = s[~s.index.isna()].sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s


def _series_span_days(series: pd.Series | None) -> int:
    s = _normalize_series(series)
    if len(s) < 2:
        return 0
    return int((s.index[-1] - s.index[0]).days)


def _age_days_from_iso(value: Any) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return int((pd.Timestamp(datetime.now()) - ts).days)
    except Exception:
        return None


def _check_series_quality(series: pd.Series | None, stale_days: int) -> str:
    s = _normalize_series(series)
    if s.empty:
        return "missing"
    try:
        age = int((pd.Timestamp(datetime.now()) - pd.Timestamp(s.index[-1])).days)
        return "stale" if age > stale_days else "ok"
    except Exception:
        return "missing"


def _date_return(series: pd.Series | None, days_back: int) -> float | None:
    """Return over approximately *days_back* calendar days, date-based."""
    s = _normalize_series(series)
    if len(s) < 2:
        return None
    end_val = _safe_float(s.iloc[-1])
    if end_val is None or end_val == 0:
        return None

    target = s.index[-1] - pd.Timedelta(days=days_back)
    earlier = s[s.index <= target]
    if earlier.empty:
        actual_days = (s.index[-1] - s.index[0]).days
        if actual_days < days_back * 0.8:
            return None
        start_val = _safe_float(s.iloc[0])
    else:
        start_val = _safe_float(earlier.iloc[-1])

    if start_val is None or start_val == 0:
        return None
    return (end_val / start_val - 1.0) * 100.0


def _rolling_date_returns(series: pd.Series | None, days_back: int) -> pd.Series:
    s = _normalize_series(series)
    if len(s) < 2:
        return pd.Series(dtype=float)

    idx = s.index
    vals = s.to_numpy(dtype=float)
    out_values: list[float] = []
    out_index: list[pd.Timestamp] = []

    for i, end_date in enumerate(idx):
        target = end_date - pd.Timedelta(days=days_back)
        pos = idx.searchsorted(target, side="right") - 1
        if pos < 0:
            actual_days = (end_date - idx[0]).days
            if actual_days < days_back * 0.8:
                continue
            pos = 0
        if pos >= i:
            continue
        start = vals[pos]
        end = vals[i]
        if not math.isfinite(start) or not math.isfinite(end) or start == 0:
            continue
        out_values.append((end / start - 1.0) * 100.0)
        out_index.append(pd.Timestamp(end_date))

    return pd.Series(out_values, index=out_index, dtype=float)


def _empirical_percentile(history: pd.Series | None, current: float | None) -> float | None:
    if current is None:
        return None
    h = pd.to_numeric(history, errors="coerce").dropna() if history is not None else pd.Series(dtype=float)
    if len(h) < MIN_PERCENTILE_SAMPLES:
        return None
    less = float((h < current).sum())
    equal = float((h == current).sum())
    percentile = ((less + 0.5 * equal) / len(h)) * 100.0
    return round(max(0.0, min(100.0, percentile)), 1)


def _cross_sectional_percentile(history: pd.Series | None, current: float | None) -> float | None:
    if current is None:
        return None
    h = pd.to_numeric(history, errors="coerce").dropna() if history is not None else pd.Series(dtype=float)
    if len(h) < MIN_RELATIVE_STRENGTH_PEERS:
        return None
    less = float((h < current).sum())
    equal = float((h == current).sum())
    percentile = ((less + 0.5 * equal) / len(h)) * 100.0
    return round(max(0.0, min(100.0, percentile)), 1)


def _trend_label(score: float | None) -> str:
    if score is None:
        return "no_data"
    if score >= 70.0:
        return "strong_up"
    if score >= 55.0:
        return "moderate_up"
    if score <= 30.0:
        return "strong_down"
    if score <= 45.0:
        return "moderate_down"
    return "neutral"


def _score_trend(series: pd.Series | None) -> tuple[float | None, str]:
    if _series_span_days(series) < OWN_HISTORY_MIN_DAYS:
        return None, "no_data"

    horizons = {
        "1m": (30, 0.20),
        "3m": (90, 0.35),
        "12m": (365, 0.45),
    }
    weighted = 0.0
    total_w = 0.0

    for days_back, weight in horizons.values():
        current = _date_return(series, days_back)
        history = _rolling_date_returns(series, days_back)
        percentile = _empirical_percentile(history.iloc[:-1], current)
        if percentile is None:
            continue
        weighted += weight * percentile
        total_w += weight

    if total_w <= 0:
        return None, "no_data"

    score = round(weighted / total_w, 1)
    return score, _trend_label(score)


def _rolling_acceleration(series: pd.Series | None) -> pd.Series:
    ret_1m = _rolling_date_returns(series, 30).rename("ret_1m")
    ret_3m = _rolling_date_returns(series, 90).rename("ret_3m")
    merged = pd.concat([ret_1m, ret_3m], axis=1).dropna()
    if merged.empty:
        return pd.Series(dtype=float)
    return (merged["ret_1m"] - merged["ret_3m"] / 3.0).rename("acceleration")


def _score_acceleration(series: pd.Series | None) -> float | None:
    if _series_span_days(series) < OWN_HISTORY_MIN_DAYS:
        return None
    history = _rolling_acceleration(series)
    if len(history) < MIN_PERCENTILE_SAMPLES + 1:
        return None
    current = _safe_float(history.iloc[-1])
    return _empirical_percentile(history.iloc[:-1], current)


def _risk_adjusted_strength(series: pd.Series | None) -> float | None:
    s = _normalize_series(series)
    if len(s) < 2:
        return None
    ret_3m = _date_return(s, 90)
    if ret_3m is None:
        return None
    cutoff = s.index[-1] - pd.Timedelta(days=120)
    recent = s[s.index >= cutoff]
    daily_ret = recent.pct_change().dropna()
    if len(daily_ret) < MIN_VOL_OBSERVATIONS:
        return None
    vol = _safe_float(daily_ret.std())
    if vol is None or vol <= 0:
        return None
    annualized_vol_pct = vol * math.sqrt(252.0) * 100.0
    if annualized_vol_pct <= 0:
        return None
    return ret_3m / annualized_vol_pct


def _family_relative_strength_scores(values: dict[str, float]) -> dict[str, float | None]:
    if len(values) < MIN_RELATIVE_STRENGTH_PEERS:
        return {name: None for name in values}
    history = pd.Series(list(values.values()), dtype=float)
    out: dict[str, float | None] = {}
    for name, value in values.items():
        out[name] = _cross_sectional_percentile(history, value)
    return out


def _curve_quality(curve_data: dict | None) -> str:
    if curve_data is None:
        return "error"

    analysis = curve_data.get("analysis", {})
    valid_contracts = (
        _safe_int(analysis.get("valid_contract_count")) or _safe_int(analysis.get("contracts_available")) or 0
    )
    warning_count = _safe_int(analysis.get("warning_count"))
    if warning_count is None:
        warning_count = len(curve_data.get("warnings", []))
    newest = analysis.get("newest_valid_contract_date")
    age = _age_days_from_iso(newest)

    if valid_contracts < 6:
        return "missing"
    if age is not None and age > CURVE_STALE_DAYS:
        return "stale"
    if valid_contracts >= 9 and warning_count == 0:
        return "ok"
    return "degraded"


def _curve_spread_score(spread_pct: float | None) -> float | None:
    if spread_pct is None:
        return None
    return round(max(0.0, min(100.0, 50.0 - spread_pct * 5.0)), 1)


def _score_curve_structure(curve_data: dict | None) -> float | None:
    if curve_data is None:
        return None

    analysis = curve_data.get("analysis", {})
    parts: list[tuple[float, float]] = []
    for key, weight in (
        ("prompt_to_m3_spread_pct", 0.50),
        ("prompt_to_m6_spread_pct", 0.30),
        ("prompt_to_m12_spread_pct", 0.20),
    ):
        score = _curve_spread_score(_safe_float(analysis.get(key)))
        if score is not None:
            parts.append((weight, score))

    if not parts:
        return None

    total_w = sum(weight for weight, _ in parts)
    return round(sum(weight * score for weight, score in parts) / total_w, 1)


def _macro_overlay_quality(macro: dict | None) -> str:
    if macro is None:
        return "error"
    status = str(macro.get("status") or "ok").lower()
    as_of = macro.get("as_of")
    age = _age_days_from_iso(as_of)
    if age is None:
        return "degraded"
    if age > MACRO_STALE_DAYS:
        return "stale"
    if status != "ok":
        return "degraded"
    return "ok"


def _extract_macro_overlay(macro: dict | None) -> dict[str, Any]:
    quality = _macro_overlay_quality(macro)
    overlay: dict[str, Any] = {
        "label": None,
        "score": None,
        "forward_outlook": None,
        "as_of": None,
        "status": "error" if macro is None else str(macro.get("status") or "ok"),
        "quality": quality,
    }
    if macro is None:
        return overlay

    regime = macro.get("regime", {})
    overlay["label"] = regime.get("label")
    overlay["score"] = _safe_float(regime.get("score"))
    overlay["forward_outlook"] = macro.get("forward_outlook", {}).get("label")
    overlay["as_of"] = macro.get("as_of")
    overlay["confidence"] = _safe_float(regime.get("confidence"))
    overlay["history_percentile"] = _safe_float(regime.get("history_percentile"))
    return overlay


def _quality_ratio(data_quality: dict[str, str]) -> float:
    relevant = [
        QUALITY_SCORES[value] for source, value in data_quality.items() if source != "macro_overlay" and value != "n/a"
    ]
    if not relevant:
        return 1.0
    return sum(relevant) / len(relevant)


def _compute_composite(
    scores: dict[str, float | None],
    eligible_factor_keys: list[str],
) -> tuple[float | None, float, dict[str, float], float | None]:
    eligible_weight = sum(FACTOR_CONFIG[key]["configured_weight"] for key in eligible_factor_keys)
    if eligible_weight <= 0:
        return None, 1.0, {}, None

    available_scores = {
        key: score for key, score in scores.items() if key in eligible_factor_keys and score is not None
    }
    available_weight = sum(FACTOR_CONFIG[key]["configured_weight"] for key in available_scores)
    coverage_ratio = available_weight / eligible_weight if eligible_weight > 0 else 1.0

    if available_weight <= 0:
        return None, round(coverage_ratio, 4), {}, None

    effective_weights = {key: FACTOR_CONFIG[key]["configured_weight"] / available_weight for key in available_scores}
    observed = sum(effective_weights[key] * available_scores[key] for key in available_scores)
    composite = observed * coverage_ratio + 50.0 * (1.0 - coverage_ratio)
    rounded_weights = {key: round(value, 4) for key, value in effective_weights.items()}
    return round(composite, 1), round(coverage_ratio, 4), rounded_weights, round(observed, 1)


def assign_direction(composite_score: float | None, trend_score: float | None) -> str:
    if composite_score is None or trend_score is None:
        return "watchlist"
    if composite_score >= 60.0 and trend_score >= 55.0:
        return "long"
    if composite_score <= 40.0 and trend_score <= 45.0:
        return "short"
    return "watchlist"


def assign_confidence(
    composite_score: float | None,
    coverage_ratio: float,
    data_quality: dict[str, str],
    curve_quality: str,
) -> str:
    if composite_score is None:
        return "low"

    distance = abs(composite_score - 50.0)
    quality = _quality_ratio(data_quality)

    if distance >= 15.0 and coverage_ratio >= 0.90 and quality >= 0.85:
        confidence = "high"
    elif distance >= 8.0 and coverage_ratio >= 0.70 and quality >= 0.60:
        confidence = "medium"
    else:
        confidence = "low"

    if curve_quality == "degraded" and confidence == "high":
        return "medium"
    return confidence


def _generate_rationale(
    sector: str,
    returns: dict[str, float | None],
    trend_label: str,
    direction: str,
    curve_analysis: dict[str, Any] | None,
    relative_strength_score: float | None,
    macro_overlay: dict[str, Any],
) -> list[str]:
    bullets: list[str] = []

    ret_3m = returns.get("3m")
    if ret_3m is not None:
        sign = "+" if ret_3m >= 0 else ""
        bullets.append(f"{sign}{ret_3m:.1f}% over 3M; trend regime reads {trend_label.replace('_', ' ')}")

    if relative_strength_score is not None:
        position = "upper" if relative_strength_score >= 50.0 else "lower"
        bullets.append(
            f"Relative strength sits in the {position} half of the {sector} family on 3M risk-adjusted return"
        )

    if curve_analysis is not None and curve_analysis.get("shape") not in (None, "N/A"):
        spread = _safe_float(curve_analysis.get("prompt_to_m3_spread_pct"))
        if spread is not None:
            bullets.append(
                f"Front curve is {str(curve_analysis.get('shape')).lower()} with prompt-to-M3 spread at {spread:.2f}%"
            )
        else:
            bullets.append(f"Front curve is {str(curve_analysis.get('shape')).lower()}")

    if macro_overlay.get("label") is not None:
        bullets.append(
            f"Market stress overlay is {macro_overlay['label']} with {macro_overlay.get('forward_outlook') or 'n/a'} outlook; overlay is informational only"
        )

    if direction != "watchlist":
        bullets.append(f"Composite proxy score supports a {direction} bias")

    return bullets[:4]


def _build_price_series(series: pd.Series | None, days_back: int = 90) -> list[dict[str, Any]]:
    s = _normalize_series(series)
    if s.empty:
        return []
    cutoff = s.index[-1] - pd.Timedelta(days=days_back)
    recent = s[s.index >= cutoff]
    points: list[dict[str, Any]] = []
    for idx, value in recent.items():
        fv = _safe_float(value)
        if fv is None:
            continue
        points.append({"date": idx.isoformat(), "value": fv})
    return points


def _factor_entry(
    key: str,
    score: float | None,
    effective_weights: dict[str, float],
    quality: str,
) -> dict[str, Any]:
    meta = FACTOR_CONFIG[key]
    included = key in effective_weights
    effective_weight = effective_weights.get(key, 0.0)
    contribution = round(score * effective_weight, 2) if score is not None and included else 0.0
    return {
        "score": score,
        "contribution": contribution,
        "display_label": meta["display_label"],
        "category": meta["category"],
        "source": meta["source"],
        "included_in_composite": included,
        "configured_weight": round(float(meta["configured_weight"]), 4),
        "effective_weight": round(float(effective_weight), 4),
        "quality": quality,
    }


def _fetch_daily_prices() -> dict | None:
    try:
        data = fetch_commodities_data("ResearchDaily")
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


def _fetch_all() -> tuple[dict | None, dict | None, dict[str, dict], dict | None]:
    # All four fetches are serialized. Prices, curves, and macro all hit
    # yfinance, which maintains a single shared crumb/session globally; running
    # them in parallel produces "HTTP 401 Invalid Crumb" failures that drop
    # random subsets of tickers silently.
    daily = _fetch_daily_prices()
    monthly = _fetch_monthly_prices()
    curves = _fetch_curves()
    macro = _fetch_macro()
    return daily, monthly, curves, macro


def build_commodity_research() -> dict[str, Any]:
    daily, monthly, curves, macro = _fetch_all()

    daily_prices = daily.get("commodities", {}) if daily else {}
    monthly_prices = monthly.get("commodities", {}) if monthly else {}
    macro_overlay = _extract_macro_overlay(macro)

    risk_adjusted_by_family: dict[str, dict[str, float]] = {"metals": {}, "energy": {}}
    returns_cache: dict[str, dict[str, float | None]] = {}

    for name in COMMODITIES:
        series = daily_prices.get(name)
        sector = SECTOR_MAP.get(name, "other")
        returns_cache[name] = {
            "1m": _date_return(series, 30),
            "3m": _date_return(series, 90),
            "12m": _date_return(series, 365),
        }
        risk_adjusted = _risk_adjusted_strength(series)
        if risk_adjusted is not None and sector in risk_adjusted_by_family:
            risk_adjusted_by_family[sector][name] = risk_adjusted

    relative_strength_map: dict[str, float | None] = {}
    for sector, values in risk_adjusted_by_family.items():
        family_scores = _family_relative_strength_scores(values)
        for name, score in family_scores.items():
            relative_strength_map[name] = score
        if len(values) < MIN_RELATIVE_STRENGTH_PEERS:
            for name, name_sector in SECTOR_MAP.items():
                if name_sector == sector:
                    relative_strength_map.setdefault(name, None)

    ideas: list[dict[str, Any]] = []

    for name, ticker in COMMODITIES.items():
        sector = SECTOR_MAP.get(name, "other")
        daily_s = daily_prices.get(name)
        monthly_s = monthly_prices.get(name)
        curve_data = curves.get(name) if name in CURVE_CODES else None
        curve_analysis = curve_data.get("analysis", {}) if curve_data is not None else None
        curve_quality = "n/a" if name not in CURVE_CODES else _curve_quality(curve_data)

        data_quality: dict[str, str] = {
            "prices_daily": _check_series_quality(daily_s, DAILY_STALE_DAYS),
            "prices_monthly": _check_series_quality(monthly_s, MONTHLY_STALE_DAYS),
            "curve": curve_quality,
            "macro_overlay": macro_overlay["quality"],
        }

        returns = returns_cache[name]
        spot = None
        normalized_daily = _normalize_series(daily_s)
        if not normalized_daily.empty:
            spot = _safe_float(normalized_daily.iloc[-1])

        trend_score, trend_label = _score_trend(daily_s)
        acceleration_score = _score_acceleration(daily_s)
        relative_strength_score = relative_strength_map.get(name)
        curve_score = _score_curve_structure(curve_data) if name in CURVE_CODES else None
        overlay_score = macro_overlay["score"]

        factor_scores: dict[str, float | None] = {
            "trend": trend_score,
            "relative_strength": relative_strength_score,
            "acceleration": acceleration_score,
            "curve_structure": curve_score,
            "market_stress_overlay": overlay_score,
        }

        eligible_factor_keys = ["trend", "relative_strength", "acceleration"]
        if name in CURVE_CODES:
            eligible_factor_keys.append("curve_structure")

        score_for_composite = dict(factor_scores)
        if curve_quality in ("missing", "error"):
            score_for_composite["curve_structure"] = None

        composite_score, coverage_ratio, effective_weights, observed_composite = _compute_composite(
            score_for_composite,
            eligible_factor_keys,
        )
        direction = assign_direction(composite_score, trend_score)
        confidence = assign_confidence(composite_score, coverage_ratio, data_quality, curve_quality)

        factor_quality = {
            "trend": data_quality["prices_daily"] if trend_score is not None else "missing",
            "relative_strength": data_quality["prices_daily"] if relative_strength_score is not None else "missing",
            "acceleration": data_quality["prices_daily"] if acceleration_score is not None else "missing",
            "curve_structure": curve_quality,
            "market_stress_overlay": data_quality["macro_overlay"] if overlay_score is not None else "error",
        }

        factors = {
            key: _factor_entry(key, factor_scores.get(key), effective_weights, factor_quality[key])
            for key in FACTOR_CONFIG
        }

        rationale = _generate_rationale(
            sector=sector,
            returns=returns,
            trend_label=trend_label,
            direction=direction,
            curve_analysis=curve_analysis,
            relative_strength_score=relative_strength_score,
            macro_overlay=macro_overlay,
        )

        ideas.append(
            {
                "commodity": name,
                "ticker": ticker,
                "sector": sector,
                "spot_price": spot,
                "returns": {
                    "1m": round(returns["1m"], 2) if returns["1m"] is not None else None,
                    "3m": round(returns["3m"], 2) if returns["3m"] is not None else None,
                    "12m": round(returns["12m"], 2) if returns["12m"] is not None else None,
                },
                "factors": factors,
                "composite_score": composite_score,
                "observed_composite_score": observed_composite,
                "coverage_ratio": coverage_ratio,
                "direction": direction,
                "confidence": confidence,
                "rationale": rationale,
                "data_quality": data_quality,
                "price_series": _build_price_series(daily_s, days_back=90),
            }
        )

    ideas.sort(key=lambda item: item["composite_score"] if item["composite_score"] is not None else -1.0, reverse=True)

    longs = [idea for idea in ideas if idea["direction"] == "long" and idea["composite_score"] is not None]
    shorts = [idea for idea in ideas if idea["direction"] == "short" and idea["composite_score"] is not None]
    top_long = {"commodity": longs[0]["commodity"], "score": longs[0]["composite_score"]} if longs else None
    worst_short = min(shorts, key=lambda idea: idea["composite_score"]) if shorts else None
    top_short = (
        {"commodity": worst_short["commodity"], "score": worst_short["composite_score"]}
        if worst_short is not None
        else None
    )

    ok_count = 0
    degraded_count = 0
    missing_count = 0
    for idea in ideas:
        statuses = [status for status in idea["data_quality"].values()]
        if any(status in ("missing", "error") for status in statuses):
            missing_count += 1
        elif any(status in ("degraded", "stale") for status in statuses):
            degraded_count += 1
        else:
            ok_count += 1

    overall_status = "ok"
    if degraded_count > 0 or missing_count > 0 or macro_overlay["quality"] != "ok":
        overall_status = "degraded"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "methodology": {
            "name": "Commodity Proxy Screener",
            "note": (
                "Rankings combine trend, family-relative strength, acceleration, and front-curve structure. "
                "Market stress is shown as an informational overlay and is not part of the ranked composite."
            ),
            "ranking_mode": "proxy_rank_v2",
        },
        "macro_overlay": macro_overlay,
        "ideas": ideas,
        "summary": {
            "top_long": top_long,
            "top_short": top_short,
            "data_health": {
                "ok": ok_count,
                "degraded": degraded_count,
                "missing": missing_count,
            },
        },
    }
