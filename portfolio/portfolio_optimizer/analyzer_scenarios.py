from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from typing import Any, TypedDict


class _ScenarioDriver(TypedDict):
    key: str
    detail: str
    value: float


SCENARIO_FACTOR_DEFAULTS = {
    "quality": 0.20,
    "price_momentum": 0.30,
    "fundamental_momentum": 0.21,
    "valuation": 0.09,
    "qualitative": 0.20,
}
SCENARIO_FUNDAMENTAL_DEFAULTS = {
    "revenue": 0.60,
    "eps": 0.40,
}
SCENARIO_VALUATION_DEFAULTS = {
    "price_sales": 0.10,
    "price_operating_income": 0.30,
    "price_fcf": 0.40,
    "price_earnings": 0.10,
    "price_book": 0.10,
}
SCENARIO_QUALITATIVE_DEFAULTS = {
    "business_quality_qualitative": 0.40,
    "industry_quality": 0.30,
    "management_quality": 0.30,
}
SCENARIO_METRIC_SCORE_DEFAULTS = {
    "quality": 0.0,
    "price_momentum": 0.0,
    "revenue": 0.0,
    "eps": 0.0,
    "price_sales": 0.0,
    "price_operating_income": 0.0,
    "price_fcf": 0.0,
    "price_earnings": 0.0,
    "price_book": 0.0,
    "business_quality_qualitative": 0.0,
    "industry_quality": 0.0,
    "management_quality": 0.0,
}
SCENARIO_BRAKE_DEFAULTS = {
    "drawdown_sensitivity": 0.0,
    "contrarian_penalty": 0.0,
    "short_squeeze_brake": 0.0,
}
VALUATION_COLUMNS = (
    "price_sales",
    "price_operating_income",
    "price_fcf",
    "price_earnings",
    "price_book",
)
VALUATION_LABELS = {
    "price_sales": "EV/S",
    "price_operating_income": "EV/Operating Income",
    "price_fcf": "EV/FCF",
    "price_earnings": "P/E",
    "price_book": "P/B",
}
QUALITATIVE_COLUMNS = (
    "business_quality_qualitative",
    "industry_quality",
    "management_quality",
)
QUALITATIVE_LABELS = {
    "business_quality_qualitative": "Business quality (qualitative)",
    "industry_quality": "Industry quality",
    "management_quality": "Management quality",
}

# Scenario UI controls use 0-100 "importance" scores. They are normalized below
# into relative factor/metric weights before touching z-score-like alpha signals.
# Brake scores are also normalized to 0.0-1.0; portfolio_analyzer.py then maps
# them into bounded risk magnitudes around the action thresholds near +/-0.75.
SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "preset": "balanced",
        "metric_scores": {
            "quality": 20,
            "price_momentum": 30,
            "revenue": 13,
            "eps": 8,
            "price_sales": 1,
            "price_operating_income": 2,
            "price_fcf": 4,
            "price_earnings": 1,
            "price_book": 1,
            "business_quality_qualitative": 8,
            "industry_quality": 6,
            "management_quality": 6,
        },
        "brakes": {
            "drawdown_sensitivity": 10,
            "contrarian_penalty": 5,
            "short_squeeze_brake": 10,
        },
    },
    "capital_preservation": {
        "preset": "capital_preservation",
        "metric_scores": {
            "quality": 30,
            "price_momentum": 15,
            "revenue": 8,
            "eps": 5,
            "price_sales": 2,
            "price_operating_income": 5,
            "price_fcf": 8,
            "price_earnings": 3,
            "price_book": 2,
            "business_quality_qualitative": 8,
            "industry_quality": 6,
            "management_quality": 8,
        },
        "brakes": {
            "drawdown_sensitivity": 70,
            "contrarian_penalty": 60,
            "short_squeeze_brake": 60,
        },
    },
    "momentum_exploitation": {
        "preset": "momentum_exploitation",
        "metric_scores": {
            "quality": 15,
            "price_momentum": 50,
            "revenue": 13,
            "eps": 9,
            "price_sales": 0,
            "price_operating_income": 0,
            "price_fcf": 0,
            "price_earnings": 0,
            "price_book": 0,
            "business_quality_qualitative": 5,
            "industry_quality": 4,
            "management_quality": 4,
        },
        "brakes": {
            "drawdown_sensitivity": 10,
            "contrarian_penalty": 10,
            "short_squeeze_brake": 25,
        },
    },
    "value_dislocation": {
        "preset": "value_dislocation",
        "metric_scores": {
            "quality": 18,
            "price_momentum": 8,
            "revenue": 8,
            "eps": 4,
            "price_sales": 8,
            "price_operating_income": 10,
            "price_fcf": 17,
            "price_earnings": 10,
            "price_book": 5,
            "business_quality_qualitative": 5,
            "industry_quality": 4,
            "management_quality": 3,
        },
        "brakes": {
            "drawdown_sensitivity": 30,
            "contrarian_penalty": 30,
            "short_squeeze_brake": 35,
        },
    },
}

AI_RECOMMENDED_PRESET = "ai_recommended"
REMOVED_SCENARIO_PRESETS = {"short_defense"}

LEGACY_BALANCED_FACTOR_WEIGHTS = {
    "quality": 0.30,
    "price_momentum": 0.40,
    "fundamental_momentum": 0.30,
    "valuation": 0.0,
}

EXPLICIT_LEGACY_FACTOR_DEFAULTS = {
    **SCENARIO_FACTOR_DEFAULTS,
    "qualitative": 0.0,
}


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if isfinite(out) else float("nan")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def quantize_slider(value: float) -> int:
    return int(min(100, max(0, round(float(value) / 10.0) * 10)))


def factor_score(payload: Mapping[str, Any], key: str) -> float | None:
    factors = payload.get("factors")
    if not isinstance(factors, list):
        return None
    for factor in factors:
        if not isinstance(factor, Mapping):
            continue
        if factor.get("key") != key or factor.get("status") != "ok":
            continue
        value = _safe_float(factor.get("score"))
        if not isfinite(value):
            return None
        return max(0.0, min(100.0, value))
    return None


def _ramp_up(value: float, low: float, high: float) -> float:
    return clamp01((value - low) / (high - low))


def _ramp_down(value: float, low: float, high: float) -> float:
    return clamp01((high - value) / (high - low))


def _normalized_factor(payload: Mapping[str, Any], key: str) -> float | None:
    score = factor_score(payload, key)
    if score is None:
        return None
    return clamp01(score / 100.0)


def _factor_inputs(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in ("vix", "breadth", "liquidity", "sector", "momentum"):
        score = factor_score(payload, key)
        out[key] = {
            "score": score,
            "normalized": None if score is None else clamp01(score / 100.0),
            "status": "ok" if score is not None else "unavailable",
        }
    return out


def _regime_score(payload: Mapping[str, Any]) -> float | None:
    regime = payload.get("regime")
    if not isinstance(regime, Mapping):
        return None
    score = _safe_float(regime.get("score"))
    if not isfinite(score):
        return None
    return max(0.0, min(100.0, score))


def build_ai_recommended_scenario(signal_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic analyzer scenario from Signal Aggregator output.

    Raw signal scores are first transformed into regime variables with ramps.
    The raw 0-100 values are not treated as if 50 were a neutral point.
    """
    vix = _normalized_factor(signal_payload, "vix")
    breadth_stress = _normalized_factor(signal_payload, "breadth")
    liquidity_score = _normalized_factor(signal_payload, "liquidity")
    sector_stress = _normalized_factor(signal_payload, "sector")
    momentum_stress = _normalized_factor(signal_payload, "momentum")
    regime_score = _regime_score(signal_payload)
    composite = None if regime_score is None else clamp01(regime_score / 100.0)

    risk_on = 0.0 if composite is None else _ramp_down(composite, 0.15, 0.40)
    risk_off = 0.0 if composite is None else _ramp_up(composite, 0.40, 0.65)
    vix_shock = 0.0 if vix is None else _ramp_up(vix, 0.35, 0.80)
    breadth_damage = 0.0 if breadth_stress is None else _ramp_up(breadth_stress, 0.35, 0.75)
    liquidity_tight = 0.0 if liquidity_score is None else _ramp_up(liquidity_score, 0.45, 0.75)
    sector_damage = 0.0 if sector_stress is None else _ramp_up(sector_stress, 0.30, 0.70)
    momentum_damage = 0.0 if momentum_stress is None else _ramp_up(momentum_stress, 0.35, 0.75)
    momentum_health = 0.0 if momentum_stress is None else _ramp_down(momentum_stress, 0.25, 0.55)

    market_damage = 0.30 * vix_shock + 0.30 * breadth_damage + 0.20 * momentum_damage + 0.20 * sector_damage

    metric_scores = {
        "quality": quantize_slider(20 + 25 * liquidity_tight + 15 * risk_off + 10 * market_damage),
        "price_momentum": quantize_slider(
            30 + 20 * momentum_health + 10 * risk_on - 20 * liquidity_tight - 10 * breadth_damage
        ),
        "revenue": quantize_slider(13 + 12 * risk_on + 8 * momentum_health - 8 * liquidity_tight),
        "eps": quantize_slider(8 + 10 * risk_on + 6 * momentum_health - 6 * liquidity_tight),
        "price_sales": quantize_slider(1 + 6 * breadth_damage + 4 * risk_off),
        "price_operating_income": quantize_slider(2 + 10 * liquidity_tight + 6 * market_damage),
        "price_fcf": quantize_slider(4 + 18 * liquidity_tight + 10 * risk_off + 8 * market_damage),
        "price_earnings": quantize_slider(1 + 5 * market_damage + 3 * risk_off),
        "price_book": quantize_slider(1 + 6 * breadth_damage + 5 * sector_damage),
        "business_quality_qualitative": quantize_slider(8 + 10 * liquidity_tight + 6 * risk_off),
        "industry_quality": quantize_slider(6 + 8 * sector_damage + 6 * breadth_damage),
        "management_quality": quantize_slider(6 + 8 * liquidity_tight + 4 * risk_off),
    }
    brakes = {
        "drawdown_sensitivity": quantize_slider(10 + 55 * liquidity_tight + 20 * vix_shock + 15 * risk_off),
        "contrarian_penalty": quantize_slider(5 + 35 * breadth_damage + 20 * sector_damage + 15 * market_damage),
        "short_squeeze_brake": quantize_slider(10 + 40 * momentum_health + 20 * risk_on + 10 * (1.0 - breadth_damage)),
    }
    scenario = {
        "preset": AI_RECOMMENDED_PRESET,
        "metric_scores": metric_scores,
        "brakes": brakes,
    }

    transforms = {
        "risk_on": round(risk_on, 4),
        "risk_off": round(risk_off, 4),
        "vix_shock": round(vix_shock, 4),
        "breadth_damage": round(breadth_damage, 4),
        "liquidity_tight": round(liquidity_tight, 4),
        "sector_damage": round(sector_damage, 4),
        "momentum_damage": round(momentum_damage, 4),
        "momentum_health": round(momentum_health, 4),
        "market_damage": round(market_damage, 4),
    }
    driver_candidates = [
        (
            "liquidity_tight",
            "Tight liquidity increased quality, cash-flow valuation, and drawdown brakes.",
            liquidity_tight,
        ),
        ("risk_off", "Elevated composite stress increased defensive quality and risk controls.", risk_off),
        (
            "market_damage",
            "Broad market damage increased valuation discipline and contrarian penalties.",
            market_damage,
        ),
        (
            "momentum_health",
            "Healthy momentum increased momentum/growth emphasis and short-squeeze awareness.",
            momentum_health,
        ),
        ("risk_on", "Risk-on conditions increased growth and price-momentum emphasis.", risk_on),
    ]
    drivers: list[_ScenarioDriver] = [
        _ScenarioDriver(key=key, detail=detail, value=round(float(value), 4))
        for key, detail, value in sorted(driver_candidates, key=lambda item: item[2], reverse=True)
        if value > 0
    ][:3]
    if drivers:
        rationale = " ".join(driver["detail"] for driver in drivers)
    else:
        rationale = "Signal factors did not cross deterministic tilt thresholds, so the preset stays close to balanced."

    return {
        "preset": AI_RECOMMENDED_PRESET,
        "scenario": scenario,
        "metric_scores": metric_scores,
        "brakes": brakes,
        "factor_inputs": _factor_inputs(signal_payload),
        "transforms": transforms,
        "drivers": drivers,
        "rationale": rationale,
    }


def _nonnegative_weight_group(
    values: Mapping[str, Any] | None,
    defaults: Mapping[str, float],
    *,
    group_name: str,
) -> dict[str, float]:
    raw = dict(values or {})
    out = {key: max(0.0, _safe_float(raw.get(key, default))) for key, default in defaults.items()}
    total = sum(out.values())
    if total <= 0:
        raise ValueError(f"{group_name} must include at least one positive weight.")
    return {key: value / total for key, value in out.items()}


def _clamped_brakes(values: Mapping[str, Any] | None) -> dict[str, float]:
    """Normalize UI brake scores into bounded relative intensities.

    Values in [0, 1] are already normalized. Values above 1 are interpreted as
    0-100 UI scores, so 80 means 0.80 brake intensity, not 80 score points.
    """
    raw = dict(values or {})
    out: dict[str, float] = {}
    for key, default in SCENARIO_BRAKE_DEFAULTS.items():
        value = _safe_float(raw.get(key, default))
        if not isfinite(value):
            value = default
        value = max(0.0, value)
        if value > 1.0:
            value = value / 100.0
        out[key] = float(min(1.0, value))
    return out


def _weights_close(
    values: Mapping[str, Any] | None, expected: Mapping[str, float], *, tolerance: float = 0.015
) -> bool:
    if not isinstance(values, Mapping):
        return False
    try:
        normalized = _nonnegative_weight_group(values, expected, group_name="balanced_default_check")
    except ValueError:
        return False
    return all(abs(normalized[key] - expected[key]) <= tolerance for key in expected)


def _brakes_are_default(values: Mapping[str, Any] | None) -> bool:
    return all(value == 0.0 for value in _clamped_brakes(values).values())


def _is_previous_balanced_default(raw: Mapping[str, Any]) -> bool:
    preset = str(raw.get("preset") or "balanced")
    if preset != "balanced":
        return False
    if raw.get("metric_scores") is not None:
        return False
    if not _weights_close(raw.get("factor_weights"), LEGACY_BALANCED_FACTOR_WEIGHTS):
        return False
    return _brakes_are_default(raw.get("brakes"))


def _weights_from_metric_scores(values: Mapping[str, Any] | None) -> dict[str, dict[str, float]]:
    raw = dict(values or {})
    scores = {
        key: max(0.0, _safe_float(raw.get(key, default))) for key, default in SCENARIO_METRIC_SCORE_DEFAULTS.items()
    }
    total = sum(scores.values())
    if total <= 0:
        raise ValueError("metric_scores must include at least one positive score.")

    fundamental_total = scores["revenue"] + scores["eps"]
    valuation_total = sum(scores[key] for key in VALUATION_COLUMNS)
    qualitative_total = sum(scores[key] for key in QUALITATIVE_COLUMNS)

    factor_weights = _nonnegative_weight_group(
        {
            "quality": scores["quality"],
            "price_momentum": scores["price_momentum"],
            "fundamental_momentum": fundamental_total,
            "valuation": valuation_total,
            "qualitative": qualitative_total,
        },
        SCENARIO_FACTOR_DEFAULTS,
        group_name="factor_weights",
    )
    fundamental_momentum_weights = (
        _nonnegative_weight_group(
            {"revenue": scores["revenue"], "eps": scores["eps"]},
            SCENARIO_FUNDAMENTAL_DEFAULTS,
            group_name="fundamental_momentum_weights",
        )
        if fundamental_total > 0
        else _nonnegative_weight_group(
            None,
            SCENARIO_FUNDAMENTAL_DEFAULTS,
            group_name="fundamental_momentum_weights",
        )
    )
    valuation_weights = (
        _nonnegative_weight_group(
            {key: scores[key] for key in VALUATION_COLUMNS},
            SCENARIO_VALUATION_DEFAULTS,
            group_name="valuation_weights",
        )
        if valuation_total > 0
        else _nonnegative_weight_group(None, SCENARIO_VALUATION_DEFAULTS, group_name="valuation_weights")
    )
    qualitative_weights = (
        _nonnegative_weight_group(
            {key: scores[key] for key in QUALITATIVE_COLUMNS},
            SCENARIO_QUALITATIVE_DEFAULTS,
            group_name="qualitative_weights",
        )
        if qualitative_total > 0
        else _nonnegative_weight_group(None, SCENARIO_QUALITATIVE_DEFAULTS, group_name="qualitative_weights")
    )

    return {
        "factor_weights": factor_weights,
        "fundamental_momentum_weights": fundamental_momentum_weights,
        "valuation_weights": valuation_weights,
        "qualitative_weights": qualitative_weights,
    }


def normalize_analyzer_scenario(scenario: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(scenario or {})
    preset = str(raw.get("preset") or "balanced")
    if preset in REMOVED_SCENARIO_PRESETS:
        raise ValueError(f"Unsupported analyzer preset: {preset}")
    preset_config = SCENARIO_PRESETS.get(preset)
    previous_balanced_default = _is_previous_balanced_default(raw)
    has_explicit_weights = not previous_balanced_default and any(
        raw.get(key) is not None
        for key in ("factor_weights", "fundamental_momentum_weights", "valuation_weights", "qualitative_weights")
    )

    if raw.get("metric_scores") is not None:
        weights = _weights_from_metric_scores(raw.get("metric_scores"))
    elif not has_explicit_weights and preset_config is not None:
        weights = _weights_from_metric_scores(preset_config["metric_scores"])
    else:
        weights = {
            "factor_weights": _nonnegative_weight_group(
                raw.get("factor_weights"),
                EXPLICIT_LEGACY_FACTOR_DEFAULTS if raw.get("factor_weights") is not None else SCENARIO_FACTOR_DEFAULTS,
                group_name="factor_weights",
            ),
            "fundamental_momentum_weights": _nonnegative_weight_group(
                raw.get("fundamental_momentum_weights"),
                SCENARIO_FUNDAMENTAL_DEFAULTS,
                group_name="fundamental_momentum_weights",
            ),
            "valuation_weights": _nonnegative_weight_group(
                raw.get("valuation_weights"),
                SCENARIO_VALUATION_DEFAULTS,
                group_name="valuation_weights",
            ),
            "qualitative_weights": _nonnegative_weight_group(
                raw.get("qualitative_weights"),
                SCENARIO_QUALITATIVE_DEFAULTS,
                group_name="qualitative_weights",
            ),
        }

    brakes_source = raw.get("brakes")
    if brakes_source is None and not has_explicit_weights and raw.get("metric_scores") is None and preset_config:
        brakes_source = preset_config["brakes"]

    return {
        "preset": preset,
        **weights,
        "brakes": _clamped_brakes(brakes_source),
    }


def default_analyzer_scenario(preset: str = "balanced") -> dict[str, Any]:
    return deepcopy(normalize_analyzer_scenario({"preset": preset}))
