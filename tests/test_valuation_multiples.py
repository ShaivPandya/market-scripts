import math

import pandas as pd

from equities.valuation import multiples


def _stmt(rows: dict[str, list[float]]) -> pd.DataFrame:
    cols = pd.to_datetime(["2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30"])
    return pd.DataFrame(rows, index=cols).T


def test_compute_current_multiples_uses_ttm_and_latest_book_value():
    income = _stmt(
        {
            "Total Revenue": [100, 90, 80, 70],
            "EBIT": [25, 25, 25, 25],
            "Operating Income": [20, 20, 20, 20],
            "Net Income": [10, 10, 10, 10],
        }
    )
    cashflow = _stmt(
        {
            "Operating Cash Flow": [15, 15, 15, 15],
            "Capital Expenditure": [-5, -5, -5, -5],
        }
    )
    balance = _stmt(
        {
            "Stockholders Equity": [500, 480, 460, 440],
            "Total Debt": [200, 190, 180, 170],
            "Cash And Cash Equivalents": [50, 45, 40, 35],
        }
    )

    result = multiples.compute_current_multiples_from_statements(
        {"marketCap": 1000},
        quarterly_income=income,
        quarterly_cashflow=cashflow,
        quarterly_balance=balance,
    )

    metrics = result["metrics"]
    assert result["enterprise_value"] == 1150
    assert math.isclose(metrics["price_sales"]["value"], 1150 / 340)
    assert math.isclose(metrics["price_operating_income"]["value"], 1150 / 100)
    assert math.isclose(metrics["price_fcf"]["value"], 1150 / 40)
    assert math.isclose(metrics["price_earnings"]["value"], 1000 / 40)
    assert math.isclose(metrics["price_book"]["value"], 2.0)
    assert metrics["price_sales"]["label"] == "EV/S"
    assert metrics["price_operating_income"]["label"] == "EV/EBIT"
    assert metrics["price_operating_income"]["denominator_label"] == "TTM EBIT"
    assert metrics["price_fcf"]["label"] == "EV/FCF"
    assert metrics["price_book"]["period"] == "MRQ/latest"


def test_compute_current_multiples_marks_non_positive_denominators_not_meaningful():
    income = _stmt({"Total Revenue": [100, 100, 100, 100], "Net Income": [-5, -5, -5, -5]})

    result = multiples.compute_current_multiples_from_statements({"marketCap": 1000}, quarterly_income=income)

    assert result["metrics"]["price_sales"]["status"] == "degraded"
    assert result["metrics"]["price_sales"]["reason"] == "using_market_cap_enterprise_value_proxy"
    assert result["metrics"]["price_earnings"]["status"] == "not_meaningful"
    assert result["metrics"]["price_earnings"]["reason"] == "non_positive_denominator"
    assert result["metrics"]["price_earnings"]["value"] is None


def test_compute_current_multiples_uses_provider_fallbacks_for_pe_and_pb():
    result = multiples.compute_current_multiples_from_statements(
        {"marketCap": 1000, "trailingPE": 18.5, "priceToBook": 2.4}
    )

    assert result["metrics"]["price_earnings"]["value"] == 18.5
    assert result["metrics"]["price_earnings"]["status"] == "degraded"
    assert result["metrics"]["price_book"]["value"] == 2.4
    assert result["metrics"]["price_book"]["status"] == "degraded"


def test_resolve_profile_handles_banks_software_cyclicals_and_override():
    assert (
        multiples.resolve_profile({"sector": "Financial Services", "industry": "Banks - Regional"})["id"]
        == "bank_financial"
    )
    assert (
        multiples.resolve_profile(
            {"sector": "Technology", "industry": "Software - Application", "revenueGrowth": 0.25}
        )["id"]
        == "high_growth_software_saas"
    )
    assert (
        multiples.resolve_profile({"sector": "Energy", "industry": "Oil & Gas E&P"})["id"]
        == "capital_intensive_cyclical"
    )
    assert multiples.resolve_profile({"sector": "Energy"}, "mature_software")["selection_mode"] == "override"


def test_peer_context_requires_sample_and_orients_lower_multiples_as_cheaper(monkeypatch):
    monkeypatch.setattr(
        multiples,
        "resolve_peer_universe",
        lambda *args, **kwargs: (["A", "B", "C", "D", "E"], "mock_peer_set"),
    )
    monkeypatch.setattr(
        multiples,
        "fetch_valuation_metrics_batch",
        lambda peers, max_workers=None: pd.DataFrame(
            {
                "price_sales": [2.0, 4.0, 6.0, 8.0, 10.0],
                "price_operating_income": [8.0, 10.0, 12.0, 14.0, 16.0],
                "price_fcf": [10.0, 12.0, 14.0, 16.0, 18.0],
                "price_earnings": [12.0, 14.0, 16.0, 18.0, 20.0],
                "price_book": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            index=peers,
        ),
    )
    metrics = {
        key: {"value": value, "status": "ok"}
        for key, value in {
            "price_sales": 3.0,
            "price_operating_income": 9.0,
            "price_fcf": 11.0,
            "price_earnings": 13.0,
            "price_book": 0.8,
        }.items()
    }

    context = multiples.peer_context("TEST", {"sector": "Technology"}, metrics)

    assert context["source"] == "mock_peer_set"
    assert context["metric_stats"]["price_sales"]["status"] == "ok"
    assert context["metric_stats"]["price_sales"]["percentile"] == 75.0
    assert context["metric_stats"]["price_book"]["percentile"] == 100.0


def test_effective_profile_weights_drop_unusable_metrics():
    weights = multiples.effective_profile_weights(
        {"price_sales": 0.5, "price_book": 0.5},
        {
            "price_sales": {"value": 4.0, "status": "ok"},
            "price_book": {"value": None, "status": "missing"},
        },
    )

    assert weights["price_sales"] == 1.0
    assert weights["price_book"] == 0.0


def test_value_range_scenario_uses_ev_to_equity_for_enterprise_multiples():
    row = multiples.compute_value_range_scenario(
        "price_sales",
        {"multiple": 4.0, "denominator": 300.0},
        current_price=10.0,
        shares=100.0,
        net_debt=200.0,
    )

    assert row["status"] == "ok"
    assert row["equity_value"] == 1000.0
    assert row["expected_price"] == 10.0
    assert row["percent_change"] == 0.0


def test_value_range_scenario_uses_equity_value_for_pe_and_pb():
    row = multiples.compute_value_range_scenario(
        "price_earnings",
        {"multiple": 15.0, "denominator": 100.0},
        current_price=10.0,
        shares=100.0,
        net_debt=None,
    )

    assert row["status"] == "ok"
    assert row["equity_value"] == 1500.0
    assert row["expected_price"] == 15.0
    assert row["percent_change"] == 50.0


def test_value_range_assumption_persists_and_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(multiples, "VALUE_RANGE_LOCAL_PATH", tmp_path / "value_ranges.json")
    monkeypatch.setattr(multiples, "VALUE_RANGE_GCS_KEY", "tests/value_ranges.json")

    first = {
        "metric": "price_sales",
        "scenarios": {
            "bear": {"multiple": 4.0, "denominator": 100.0},
            "base": {"multiple": 5.0, "denominator": 110.0},
            "bull": {"multiple": 6.0, "denominator": 120.0},
        },
    }
    second = {
        "metric": "price_earnings",
        "scenarios": {
            "bear": {"multiple": 12.0, "denominator": 50.0},
            "base": {"multiple": 15.0, "denominator": 55.0},
            "bull": {"multiple": 18.0, "denominator": 60.0},
        },
    }

    multiples.write_value_range_assumption("zzrange", first)
    assert multiples.read_value_range_assumption("ZZRANGE") == first

    multiples.write_value_range_assumption("ZZRANGE", second)
    assert multiples.read_value_range_assumption("zzrange") == second
