import math

import pandas as pd
import pytest
from pydantic import ValidationError

from api.routers.analyzer import AnalyzerRequest, _cache_key
from portfolio.portfolio_optimizer.portfolio_analyzer import (
    compute_valuation_signal,
    normalize_analyzer_scenario,
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
