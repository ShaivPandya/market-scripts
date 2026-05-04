import math

import pandas as pd
import pytest
from pydantic import ValidationError

from api.routers.analyzer import AnalyzerRequest, _cache_key
from portfolio.portfolio_optimizer import portfolio_analyzer as analyzer_module
from portfolio.portfolio_optimizer.portfolio_analyzer import (
    INTERACTIVE_SIGNAL_ANCHOR_MIN_UNIQUE,
    INTERACTIVE_SIGNAL_ANCHOR_TOP_N,
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
