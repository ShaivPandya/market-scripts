from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from typing import Any

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
            "drawdown_sensitivity": 0,
            "contrarian_penalty": 0,
            "short_squeeze_brake": 0,
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
    "short_defense": {
        "preset": "short_defense",
        "metric_scores": {
            "quality": 20,
            "price_momentum": 40,
            "revenue": 8,
            "eps": 4,
            "price_sales": 0,
            "price_operating_income": 2,
            "price_fcf": 5,
            "price_earnings": 0,
            "price_book": 3,
            "business_quality_qualitative": 6,
            "industry_quality": 5,
            "management_quality": 7,
        },
        "brakes": {
            "drawdown_sensitivity": 35,
            "contrarian_penalty": 25,
            "short_squeeze_brake": 80,
        },
    },
}

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
        normalized = _nonnegative_weight_group(values, expected, group_name="legacy_check")
    except ValueError:
        return False
    return all(abs(normalized[key] - expected[key]) <= tolerance for key in expected)


def _brakes_are_default(values: Mapping[str, Any] | None) -> bool:
    return all(value == 0.0 for value in _clamped_brakes(values).values())


def _is_legacy_balanced_default(raw: Mapping[str, Any]) -> bool:
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
    preset_config = SCENARIO_PRESETS.get(preset)
    legacy_balanced_default = _is_legacy_balanced_default(raw)
    has_explicit_weights = not legacy_balanced_default and any(
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
