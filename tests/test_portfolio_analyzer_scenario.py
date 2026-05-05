import math

import pandas as pd
import pytest
from pydantic import ValidationError

from api.routers.analyzer import AnalyzerRequest, _cache_key
from portfolio.portfolio_optimizer import portfolio_analyzer as analyzer_module
from portfolio.portfolio_optimizer.portfolio_analyzer import (
    INTERACTIVE_SIGNAL_ANCHOR_MIN_UNIQUE,
    INTERACTIVE_SIGNAL_ANCHOR_TOP_N,
    build_course_of_action,
    compute_valuation_signal,
    normalize_analyzer_scenario,
    overlay_anchor_long_equity_signals,
)


def test_normalize_scenario_rejects_all_zero_weight_group():
    with pytest.raises(ValueError):
        normalize_analyzer_scenario(
            {
                "factor_weights": {
                    "quality": 0,
                    "price_momentum": 0,
                    "fundamental_momentum": 0,
                    "valuation": 0,
                }
            }
        )


def test_analyzer_request_rejects_all_zero_nested_weight_group():
    with pytest.raises(ValidationError):
        AnalyzerRequest(
            scenario={
                "valuation_weights": {
                    "price_sales": 0,
                    "price_operating_income": 0,
                    "price_fcf": 0,
                    "price_earnings": 0,
                }
            }
        )


def test_analyzer_cache_key_changes_with_scenario_weights():
    quality_req = AnalyzerRequest(
        scenario={
            "factor_weights": {
                "quality": 1,
                "price_momentum": 0,
                "fundamental_momentum": 0,
                "valuation": 0,
            }
        }
    )
    value_req = AnalyzerRequest(
        scenario={
            "factor_weights": {
                "quality": 0,
                "price_momentum": 0,
                "fundamental_momentum": 0,
                "valuation": 1,
            }
        }
    )

    assert _cache_key(quality_req) != _cache_key(value_req)


def test_metric_scores_normalize_across_all_alpha_metrics():
    scenario = normalize_analyzer_scenario(
        {
            "metric_scores": {
                "quality": 10,
                "price_momentum": 0,
                "revenue": 20,
                "eps": 20,
                "price_sales": 0,
                "price_operating_income": 0,
                "price_fcf": 0,
                "price_earnings": 0,
            }
        }
    )

    assert math.isclose(scenario["factor_weights"]["quality"], 0.20)
    assert math.isclose(scenario["factor_weights"]["fundamental_momentum"], 0.80)
    assert math.isclose(scenario["fundamental_momentum_weights"]["revenue"], 0.50)
    assert math.isclose(scenario["fundamental_momentum_weights"]["eps"], 0.50)


def test_metric_score_cache_key_is_ratio_based_and_brakes_accept_scores():
    raw_score_req = AnalyzerRequest(
        scenario={
            "metric_scores": {
                "quality": 10,
                "price_momentum": 0,
                "revenue": 20,
                "eps": 20,
                "price_sales": 0,
                "price_operating_income": 0,
                "price_fcf": 0,
                "price_earnings": 0,
            },
            "brakes": {
                "drawdown_sensitivity": 60,
                "contrarian_penalty": 20,
                "short_squeeze_brake": 0,
            },
        }
    )
    normalized_req = AnalyzerRequest(
        scenario={
            "metric_scores": {
                "quality": 20,
                "price_momentum": 0,
                "revenue": 40,
                "eps": 40,
                "price_sales": 0,
                "price_operating_income": 0,
                "price_fcf": 0,
                "price_earnings": 0,
            },
            "brakes": {
                "drawdown_sensitivity": 0.6,
                "contrarian_penalty": 0.2,
                "short_squeeze_brake": 0,
            },
        }
    )

    assert _cache_key(raw_score_req) == _cache_key(normalized_req)


def test_valuation_signal_ranks_lower_positive_multiples_higher():
    raw = pd.DataFrame(
        {
            "price_sales": [2.0, 5.0, 9.0],
            "price_operating_income": [8.0, 12.0, 20.0],
            "price_fcf": [10.0, 18.0, 25.0],
            "price_earnings": [14.0, 22.0, 35.0],
        },
        index=["CHEAP", "MID", "EXPENSIVE"],
    )

    signal = compute_valuation_signal(
        raw,
        {
            "price_sales": 0.25,
            "price_operating_income": 0.25,
            "price_fcf": 0.25,
            "price_earnings": 0.25,
        },
    )

    assert signal["CHEAP"] > signal["MID"] > signal["EXPENSIVE"]


def test_valuation_signal_excludes_invalid_or_missing_multiples():
    raw = pd.DataFrame(
        {
            "price_sales": [2.0, 5.0, None],
            "price_operating_income": [8.0, 12.0, None],
            "price_fcf": [10.0, -2.0, None],
            "price_earnings": [14.0, 22.0, None],
        },
        index=["A", "B", "NON_EQUITY"],
    )

    signal = compute_valuation_signal(
        raw,
        {
            "price_sales": 0.25,
            "price_operating_income": 0.25,
            "price_fcf": 0.25,
            "price_earnings": 0.25,
        },
    )

    assert signal["A"] > signal["B"]
    assert math.isnan(signal["NON_EQUITY"])


def test_interactive_anchor_overlay_uses_reduced_scoring_universe(monkeypatch):
    captured: dict[str, int] = {}

    def fake_anchor_signals(**kwargs):
        captured["anchor_top_n"] = kwargs["anchor_top_n"]
        captured["anchor_min_unique"] = kwargs["anchor_min_unique"]
        return (
            pd.DataFrame(
                {
                    "composite_signal": [1.0],
                    "quality_signal": [0.5],
                    "eps_mom_signal": [0.25],
                    "rev_mom_signal": [0.75],
                    "price_mom_signal": [0.1],
                },
                index=["AAA"],
            ),
            {
                "signal_anchor_mode": "spdr_sector_top3_anchor",
                "signal_anchor_universe_size": 24,
                "signal_anchor_scoring_universe_size": 25,
                "signal_anchor_fallback_used": False,
            },
        )

    monkeypatch.setattr(analyzer_module, "generate_anchor_normalized_long_equity_signals", fake_anchor_signals)
    meta = pd.DataFrame({"direction": ["long"], "asset": ["equity"]}, index=["AAA"])
    signal = pd.Series([0.0], index=["AAA"])
    subcomponents = {
        "quality_signal": pd.Series([0.0], index=["AAA"]),
        "eps_mom_signal": pd.Series([0.0], index=["AAA"]),
        "rev_mom_signal": pd.Series([0.0], index=["AAA"]),
        "price_mom_signal": pd.Series([0.0], index=["AAA"]),
    }

    overlay_anchor_long_equity_signals(
        ["AAA"],
        meta,
        signal,
        subcomponents,
        anchor_top_n=INTERACTIVE_SIGNAL_ANCHOR_TOP_N,
        anchor_min_unique=INTERACTIVE_SIGNAL_ANCHOR_MIN_UNIQUE,
    )

    assert captured == {
        "anchor_top_n": INTERACTIVE_SIGNAL_ANCHOR_TOP_N,
        "anchor_min_unique": INTERACTIVE_SIGNAL_ANCHOR_MIN_UNIQUE,
    }


def _course_rows(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "asset": "equity",
        "direction": "long",
        "scenario_score": 0.0,
        "baseline_score": 0.0,
        "score_delta": 0.0,
        "scenario_penalty": 0.0,
        "quality_signal": 0.0,
        "price_mom_signal": 0.0,
        "fundamental_momentum_signal": 0.0,
        "rev_mom_signal": 0.0,
        "eps_mom_signal": 0.0,
        "valuation_signal": 0.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _balanced_course(rows: list[dict]) -> dict:
    return build_course_of_action(
        _course_rows(rows),
        normalize_analyzer_scenario(
            {
                "preset": "balanced",
                "factor_weights": {
                    "quality": 0.30,
                    "price_momentum": 0.40,
                    "fundamental_momentum": 0.30,
                    "valuation": 0.0,
                },
            }
        ),
    )


def _first_action(course: dict, ticker: str) -> dict:
    return next(item for item in course["action_queue"] if item["ticker"] == ticker)


def test_course_of_action_uses_absolute_score_not_positive_delta_for_longs():
    course = _balanced_course(
        [
            {
                "ticker": "WEAK",
                "direction": "long",
                "scenario_score": -1.10,
                "baseline_score": -1.30,
                "score_delta": 0.20,
                "quality_signal": -0.8,
                "price_mom_signal": -1.2,
                "fundamental_momentum_signal": -0.7,
            }
        ]
    )

    action = _first_action(course, "WEAK")
    assert action["action"] in {"Trim Long", "Review"}
    assert "upgrade" not in action["action"].lower()


def test_course_of_action_ugl_style_negative_price_momentum_trims_or_reviews_long():
    course = _balanced_course(
        [
            {
                "ticker": "UGL",
                "asset": "commodity",
                "direction": "long",
                "scenario_score": -1.55,
                "baseline_score": -1.41,
                "score_delta": -0.13,
                "quality_signal": math.nan,
                "price_mom_signal": -1.33,
                "fundamental_momentum_signal": math.nan,
                "rev_mom_signal": math.nan,
                "eps_mom_signal": math.nan,
                "valuation_signal": math.nan,
            }
        ]
    )

    action = _first_action(course, "UGL")
    assert action["action"] in {"Trim Long", "Review"}
    assert action["sizing_implication"]["implication"] in {"trim exposure", "review before sizing"}


def test_course_of_action_short_negative_score_is_press_short():
    course = _balanced_course(
        [
            {
                "ticker": "SHORT",
                "direction": "short",
                "scenario_score": -1.15,
                "score_delta": -0.40,
                "quality_signal": -0.9,
                "price_mom_signal": -1.1,
                "fundamental_momentum_signal": -0.7,
            }
        ]
    )

    assert _first_action(course, "SHORT")["action"] == "Press Short"


def test_course_of_action_short_positive_score_is_cover_short():
    course = _balanced_course(
        [
            {
                "ticker": "COVER",
                "direction": "short",
                "scenario_score": 1.20,
                "score_delta": 0.50,
                "quality_signal": 1.0,
                "price_mom_signal": 1.1,
                "fundamental_momentum_signal": 0.8,
            }
        ]
    )

    assert _first_action(course, "COVER")["action"] == "Cover Short"


def test_course_of_action_missing_equity_data_gates_strong_action():
    course = _balanced_course(
        [
            {
                "ticker": "MISSING",
                "direction": "long",
                "scenario_score": 1.30,
                "score_delta": 0.40,
                "quality_signal": math.nan,
                "price_mom_signal": 1.2,
                "fundamental_momentum_signal": math.nan,
                "rev_mom_signal": math.nan,
                "eps_mom_signal": math.nan,
            }
        ]
    )

    action = _first_action(course, "MISSING")
    assert action["action"] == "Review"
    assert action["gate_status"] == "review"
    assert "Insufficient applicable data coverage" in action["gate_reasons"]


def test_course_of_action_non_equity_missing_equity_metrics_are_not_missing():
    course = _balanced_course(
        [
            {
                "ticker": "CMDTY",
                "asset": "commodity",
                "direction": "long",
                "scenario_score": 1.10,
                "score_delta": 0.25,
                "quality_signal": math.nan,
                "price_mom_signal": 1.2,
                "fundamental_momentum_signal": math.nan,
                "rev_mom_signal": math.nan,
                "eps_mom_signal": math.nan,
                "valuation_signal": math.nan,
            }
        ]
    )

    action = _first_action(course, "CMDTY")
    assert action["gate_status"] == "pass"
    assert action["data_coverage"]["ratio"] == 1.0
    assert not any("Missing quality" in warning for warning in action["warnings"])


def test_course_of_action_factor_conflict_downgrades_to_review():
    course = _balanced_course(
        [
            {
                "ticker": "CONFLICT",
                "direction": "long",
                "scenario_score": 1.25,
                "score_delta": 0.35,
                "quality_signal": 1.3,
                "price_mom_signal": -1.2,
                "fundamental_momentum_signal": 1.0,
            }
        ]
    )

    action = _first_action(course, "CONFLICT")
    assert action["action"] == "Review"
    assert action["factor_conflict"] is True
    assert "Conflicting factor evidence" in action["gate_reasons"]
