import json
import math
import uuid

import pandas as pd

from equities.valuation import multiples


def _stmt(rows: dict[str, list[float]]) -> pd.DataFrame:
    cols = pd.to_datetime(["2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30"])
    return pd.DataFrame(rows, index=cols).T


def test_compute_current_multiples_uses_ttm_and_latest_book_value():
    income = _stmt(
        {
            "Total Revenue": [100, 90, 80, 70],
            "EBITDA": [30, 30, 30, 30],
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
    assert math.isclose(metrics["price_ebitda"]["value"], 1150 / 120)
    assert math.isclose(metrics["price_operating_income"]["value"], 1150 / 100)
    assert math.isclose(metrics["price_fcf"]["value"], 1150 / 40)
    assert math.isclose(metrics["price_earnings"]["value"], 1000 / 40)
    assert math.isclose(metrics["price_book"]["value"], 2.0)
    assert metrics["price_sales"]["label"] == "EV/S"
    assert metrics["price_ebitda"]["label"] == "EV/EBITDA"
    assert metrics["price_ebitda"]["denominator_label"] == "TTM EBITDA"
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


def test_mixed_currency_multiples_convert_financial_denominators(monkeypatch):
    monkeypatch.setattr(multiples, "fx_rate_to_base", lambda currency, base: {"rate": 0.03125, "as_of": "2026-05-08"})
    income = _stmt({"Total Revenue": [8000, 8000, 8000, 8000], "Net Income": [800, 800, 800, 800]})
    cashflow = _stmt({"Operating Cash Flow": [1000, 1000, 1000, 1000], "Capital Expenditure": [-200, -200, -200, -200]})
    balance = _stmt({"Stockholders Equity": [16000, 15000, 14000, 13000]})

    result = multiples.compute_current_multiples_from_statements(
        {
            "currency": "USD",
            "financialCurrency": "TWD",
            "marketCap": 1000,
            "enterpriseValue": 1200,
            "totalDebt": 3000,
            "totalCash": 1000,
        },
        quarterly_income=income,
        quarterly_cashflow=cashflow,
        quarterly_balance=balance,
    )

    assert result["currency_context"]["price_currency"] == "USD"
    assert result["currency_context"]["financial_currency"] == "TWD"
    assert result["enterprise_value"] == 1200
    assert result["net_debt"] == 62.5
    assert result["financial_data"]["net_debt"] == 2000
    assert math.isclose(result["metrics"]["price_sales"]["denominator"], 32000)
    assert math.isclose(result["metrics"]["price_sales"]["denominator_converted"], 1000)
    assert math.isclose(result["metrics"]["price_sales"]["value"], 1.2)
    assert result["metrics"]["price_sales"]["denominator_currency"] == "TWD"
    assert result["metrics"]["price_sales"]["denominator_converted_currency"] == "USD"


def test_minor_unit_quote_currency_scales_financial_denominators():
    income = _stmt({"Total Revenue": [10, 10, 10, 10]})

    for price_currency in ("GBp", "GBX"):
        result = multiples.compute_current_multiples_from_statements(
            {
                "currency": price_currency,
                "financialCurrency": "GBP",
                "marketCap": 10_000,
                "enterpriseValue": 12_000,
            },
            quarterly_income=income,
        )

        assert result["currency_context"]["price_currency"] == price_currency
        assert result["currency_context"]["financial_currency"] == "GBP"
        assert result["currency_context"]["financial_to_price_fx_rate"] == 100.0
        assert result["currency_context"]["conversion_status"] == "ok"
        assert math.isclose(result["metrics"]["price_sales"]["denominator"], 40.0)
        assert math.isclose(result["metrics"]["price_sales"]["denominator_converted"], 4000.0)
        assert math.isclose(result["metrics"]["price_sales"]["value"], 3.0)
        assert result["metrics"]["price_sales"]["denominator_currency"] == "GBP"
        assert result["metrics"]["price_sales"]["denominator_converted_currency"] == price_currency


def test_recomputed_enterprise_value_uses_converted_debt_and_cash(monkeypatch):
    monkeypatch.setattr(multiples, "fx_rate_to_base", lambda currency, base: {"rate": 0.03125, "as_of": "2026-05-08"})
    income = _stmt({"Total Revenue": [8000, 8000, 8000, 8000]})

    result = multiples.compute_current_multiples_from_statements(
        {
            "currency": "USD",
            "financialCurrency": "TWD",
            "marketCap": 1000,
            "totalDebt": 6400,
            "totalCash": 3200,
        },
        quarterly_income=income,
    )

    assert result["enterprise_value"] == 1100
    assert result["net_debt"] == 100
    assert math.isclose(result["metrics"]["price_sales"]["value"], 1100 / 1000)


def test_missing_fx_blocks_mixed_currency_statement_multiples(monkeypatch):
    monkeypatch.setattr(multiples, "fx_rate_to_base", lambda currency, base: None)
    income = _stmt({"Total Revenue": [8000, 8000, 8000, 8000]})

    result = multiples.compute_current_multiples_from_statements(
        {"currency": "USD", "financialCurrency": "TWD", "marketCap": 1000},
        quarterly_income=income,
    )

    assert result["currency_context"]["conversion_status"] == "missing_fx_rate"
    assert result["metrics"]["price_sales"]["status"] == "missing"
    assert result["metrics"]["price_sales"]["reason"] == "missing_fx_rate"
    quality = multiples.valuation_data_quality(result["metrics"], multiples._empty_peer_context())
    assert "FX conversion is unavailable for mixed-currency valuation inputs." in quality["warnings"]


def test_fetch_current_valuation_uses_daily_cache(monkeypatch):
    ticker = f"ZZCUR{uuid.uuid4().hex[:8]}".upper()
    calls = 0

    def _uncached(symbol, *, info=None):
        nonlocal calls
        calls += 1
        return {
            "market_cap": 1000.0,
            "enterprise_value": 1200.0,
            "net_debt": 200.0,
            "metrics": {
                "price_sales": {
                    "key": "price_sales",
                    "label": "EV/S",
                    "value": 3.0,
                    "status": "ok",
                }
            },
        }

    monkeypatch.setattr(multiples, "_fetch_current_valuation_uncached", _uncached)

    first = multiples.fetch_current_valuation(ticker, info={"marketCap": 1000})
    second = multiples.fetch_current_valuation(ticker, info={"marketCap": 2000})

    assert calls == 1
    assert first == second
    assert "_meta" not in first


def test_batch_row_uses_daily_cache(monkeypatch):
    ticker = f"ZZPEER{uuid.uuid4().hex[:8]}".upper()
    calls = 0
    override = None
    metrics = {
        key: {"key": key, "label": multiples.VALUATION_LABELS[key], "value": value, "status": "ok"}
        for key, value in {
            "price_sales": 3.0,
            "price_ebitda": 8.0,
            "price_operating_income": 10.0,
            "price_fcf": 12.0,
            "price_earnings": 15.0,
            "price_book": 2.0,
        }.items()
    }

    def _uncached(symbol):
        nonlocal calls
        calls += 1
        return {
            "info": {"sector": "Consumer Defensive", "industry": "Beverages"},
            "metrics": metrics,
            "sector": "Consumer Defensive",
            "industry": "Beverages",
        }

    monkeypatch.setattr(multiples, "_batch_row_uncached", _uncached)
    monkeypatch.setattr(multiples, "read_profile_override", lambda symbol: override)

    first = multiples._batch_row(ticker)
    override = "bank_financial"
    second = multiples._batch_row(ticker)

    assert calls == 1
    assert first["price_sales"] == second["price_sales"] == 3.0
    assert first["valuation_profile_id"] == "general_equity"
    assert second["valuation_profile_id"] == "bank_financial"
    assert first["price_book_profile_weight"] == 0.1
    assert second["price_book_profile_weight"] == 0.5
    assert "_meta" not in first


def test_profile_recomputes_with_cached_current_valuation(monkeypatch):
    ticker = f"ZZPROF{uuid.uuid4().hex[:8]}".upper()
    calls = 0
    override = None

    metrics = {
        key: {"key": key, "label": multiples.VALUATION_LABELS[key], "value": 10.0, "status": "ok"}
        for key in multiples.VALUATION_COLUMNS
    }

    def _uncached(symbol, *, info=None):
        nonlocal calls
        calls += 1
        return {"market_cap": 1000.0, "enterprise_value": 1100.0, "net_debt": 100.0, "metrics": metrics}

    monkeypatch.setattr(multiples, "_fetch_current_valuation_uncached", _uncached)
    monkeypatch.setattr(multiples, "_fetch_info", lambda symbol: {"marketCap": 1000.0, "sector": "Technology"})
    monkeypatch.setattr(multiples, "read_profile_override", lambda symbol: override)
    monkeypatch.setattr(multiples, "read_value_range_assumption", lambda symbol: None)

    first = multiples.get_position_valuation(ticker, include_peers=False)
    override = "bank_financial"
    second = multiples.get_position_valuation(ticker, include_peers=False)

    assert calls == 1
    assert first["profile"]["id"] == "general_equity"
    assert second["profile"]["id"] == "bank_financial"
    assert first["profile"]["effective_weights"]["price_book"] == 0.1
    assert second["profile"]["effective_weights"]["price_book"] == 0.5


def test_get_position_valuation_includes_52_week_market_data(monkeypatch):
    monkeypatch.setattr(
        multiples,
        "_fetch_info",
        lambda symbol: {
            "marketCap": 1000.0,
            "currentPrice": 10.0,
            "sharesOutstanding": 100.0,
            "fiftyTwoWeekHigh": 15.0,
            "fiftyTwoWeekLow": 6.0,
            "currency": "USD",
            "financialCurrency": "USD",
        },
    )
    monkeypatch.setattr(multiples, "read_profile_override", lambda symbol: None)
    monkeypatch.setattr(multiples, "read_value_range_assumption", lambda symbol: None)
    monkeypatch.setattr(
        multiples,
        "fetch_current_valuation",
        lambda symbol, *, info=None: {
            "market_cap": 1000.0,
            "enterprise_value": 1100.0,
            "net_debt": 100.0,
            "currency_context": multiples.currency_context_from_info(info or {}),
            "metrics": {
                "price_sales": {
                    "key": "price_sales",
                    "label": "EV/S",
                    "value": 1.1,
                    "denominator": 1000.0,
                    "status": "ok",
                }
            },
        },
    )

    result = multiples.get_position_valuation("zz52w", include_peers=False)

    assert result["market_data"]["fifty_two_week_high"] == 15.0
    assert result["market_data"]["fifty_two_week_low"] == 6.0


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


def test_value_range_assumption_persists_per_metric_and_updates_selected(tmp_path, monkeypatch):
    monkeypatch.setattr(multiples, "VALUE_RANGE_LOCAL_PATH", tmp_path / "value_ranges.json")
    monkeypatch.setattr(multiples, "VALUE_RANGE_GCS_KEY", "tests/value_ranges.json")

    first = {
        "metric": "price_sales",
        "denominator_currency": "USD",
        "scenarios": {
            "bear": {"multiple": 4.0, "denominator": 100.0},
            "base": {"multiple": 5.0, "denominator": 110.0},
            "bull": {"multiple": 6.0, "denominator": 120.0},
        },
    }
    second = {
        "metric": "price_earnings",
        "denominator_currency": "TWD",
        "scenarios": {
            "bear": {"multiple": 12.0, "denominator": 50.0},
            "base": {"multiple": 15.0, "denominator": 55.0},
            "bull": {"multiple": 18.0, "denominator": 60.0},
        },
    }

    multiples.write_value_range_assumption("zzrange", first)
    saved = multiples.read_value_range_assumption("ZZRANGE")
    assert saved["selected_metric"] == "price_sales"
    assert saved["metric_assumptions"]["price_sales"]["scenarios"] == first["scenarios"]
    assert saved["metric_assumptions"]["price_sales"]["denominator_currency"] == "USD"
    assert saved["metric_assumptions"]["price_sales"]["legacy_denominator_currency"] is False

    multiples.write_value_range_assumption("ZZRANGE", second)
    saved = multiples.read_value_range_assumption("zzrange")
    assert saved["selected_metric"] == "price_earnings"
    assert saved["metric_assumptions"]["price_sales"]["scenarios"] == first["scenarios"]
    assert saved["metric_assumptions"]["price_earnings"]["scenarios"] == second["scenarios"]
    assert saved["metric_assumptions"]["price_earnings"]["denominator_currency"] == "TWD"


def test_value_range_assumption_update_preserves_other_metric_currency_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(multiples, "VALUE_RANGE_LOCAL_PATH", tmp_path / "value_ranges.json")
    monkeypatch.setattr(multiples, "VALUE_RANGE_GCS_KEY", "tests/value_ranges.json")

    preserved = {
        "scenarios": {
            "bear": {"multiple": 8.0, "denominator": 50.0},
            "base": {"multiple": 10.0, "denominator": 55.0},
            "bull": {"multiple": 12.0, "denominator": 60.0},
        },
        "denominator_currency": "USD",
        "legacy_denominator_currency": True,
    }
    multiples._write_value_ranges(
        {
            "ZZMETA": {
                "selected_metric": "price_earnings",
                "metric_assumptions": {"price_earnings": preserved},
            }
        }
    )

    multiples.write_value_range_assumption(
        "ZZMETA",
        {
            "metric": "price_sales",
            "denominator_currency": "TWD",
            "scenarios": {
                "bear": {"multiple": 4.0, "denominator": 100.0},
                "base": {"multiple": 5.0, "denominator": 110.0},
                "bull": {"multiple": 6.0, "denominator": 120.0},
            },
        },
    )

    saved = multiples.read_value_range_assumption("ZZMETA")
    assert saved["selected_metric"] == "price_sales"
    assert saved["metric_assumptions"]["price_earnings"] == preserved


def test_value_range_assumption_preserves_denominator_currency(tmp_path, monkeypatch):
    monkeypatch.setattr(multiples, "VALUE_RANGE_LOCAL_PATH", tmp_path / "value_ranges.json")
    monkeypatch.setattr(multiples, "VALUE_RANGE_GCS_KEY", "tests/value_ranges.json")

    legacy = {
        "metric": "price_sales",
        "scenarios": {
            "bear": {"multiple": 4.0, "denominator": 100.0},
            "base": {"multiple": 5.0, "denominator": 110.0},
            "bull": {"multiple": 6.0, "denominator": 120.0},
        },
    }
    stamped = {**legacy, "denominator_currency": "TWD"}

    multiples.write_value_range_assumption("zzlegacy", legacy)
    saved = multiples.read_value_range_assumption("zzlegacy")
    assert "denominator_currency" not in saved["metric_assumptions"]["price_sales"]
    assert saved["metric_assumptions"]["price_sales"]["legacy_denominator_currency"] is False

    multiples.write_value_range_assumption("zzlegacy", stamped)
    saved = multiples.read_value_range_assumption("zzlegacy")
    assert saved["metric_assumptions"]["price_sales"]["denominator_currency"] == "TWD"
    assert saved["metric_assumptions"]["price_sales"]["legacy_denominator_currency"] is False


def test_legacy_flat_value_range_record_loads_as_single_legacy_metric(tmp_path, monkeypatch):
    path = tmp_path / "value_ranges.json"
    monkeypatch.setattr(multiples, "VALUE_RANGE_LOCAL_PATH", path)
    monkeypatch.setattr(multiples, "VALUE_RANGE_GCS_KEY", "tests/value_ranges.json")
    legacy = {
        "metric": "price_sales",
        "scenarios": {
            "bear": {"multiple": 4.0, "denominator": 100.0},
            "base": {"multiple": 5.0, "denominator": 110.0},
            "bull": {"multiple": 6.0, "denominator": 120.0},
        },
    }
    path.write_text(json.dumps({"ZZLEGACY": legacy}), encoding="utf-8")

    saved = multiples.read_value_range_assumption("ZZLEGACY")
    assert saved["selected_metric"] == "price_sales"
    assert list(saved["metric_assumptions"]) == ["price_sales"]
    assert saved["metric_assumptions"]["price_sales"]["legacy_denominator_currency"] is True


def test_delete_value_range_assumption_is_idempotent_and_metric_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(multiples, "VALUE_RANGE_LOCAL_PATH", tmp_path / "value_ranges.json")
    monkeypatch.setattr(multiples, "VALUE_RANGE_GCS_KEY", "tests/value_ranges.json")

    sales = {
        "metric": "price_sales",
        "denominator_currency": "USD",
        "scenarios": {
            "bear": {"multiple": 4.0, "denominator": 100.0},
            "base": {"multiple": 5.0, "denominator": 110.0},
            "bull": {"multiple": 6.0, "denominator": 120.0},
        },
    }
    earnings = {
        "metric": "price_earnings",
        "denominator_currency": "USD",
        "scenarios": {
            "bear": {"multiple": 12.0, "denominator": 50.0},
            "base": {"multiple": 15.0, "denominator": 55.0},
            "bull": {"multiple": 18.0, "denominator": 60.0},
        },
    }
    multiples.write_value_range_assumption("zzdelete", sales)
    multiples.write_value_range_assumption("zzdelete", earnings)

    result = multiples.delete_value_range_assumption("zzdelete", "price_sales")
    assert result["value_range"]["selected_metric"] == "price_earnings"
    assert "price_sales" not in result["value_range"]["metric_assumptions"]
    assert "price_earnings" in result["value_range"]["metric_assumptions"]

    result = multiples.delete_value_range_assumption("zzdelete", "price_sales")
    assert result["value_range"]["selected_metric"] == "price_earnings"
    assert "price_earnings" in result["value_range"]["metric_assumptions"]


def test_delete_value_range_assumption_removes_legacy_flat_record(tmp_path, monkeypatch):
    path = tmp_path / "value_ranges.json"
    monkeypatch.setattr(multiples, "VALUE_RANGE_LOCAL_PATH", path)
    monkeypatch.setattr(multiples, "VALUE_RANGE_GCS_KEY", "tests/value_ranges.json")
    path.write_text(
        json.dumps(
            {
                "ZZFLAT": {
                    "metric": "price_sales",
                    "scenarios": {
                        "bear": {"multiple": 4.0, "denominator": 100.0},
                        "base": {"multiple": 5.0, "denominator": 110.0},
                        "bull": {"multiple": 6.0, "denominator": 120.0},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = multiples.delete_value_range_assumption("zzflat", "price_sales")
    assert result["value_range"]["metric_assumptions"] == {}
    assert multiples.read_value_range_assumption("zzflat") is None


def test_value_range_payload_without_saved_assumptions_uses_default_metric_with_blank_scenarios():
    payload = multiples.value_range_payload(
        saved_assumption=None,
        metrics={
            "price_sales": {"value": 4.0, "denominator": 100.0},
            "price_fcf": {"value": 20.0, "denominator": 25.0},
        },
        peers=multiples._empty_peer_context(),
        effective_weights={"price_sales": 0.1, "price_fcf": 0.9},
        currency_context={"price_currency": "USD", "financial_currency": "USD"},
        market_data={"current_price": 10.0, "shares": 100.0, "net_debt": 0.0, "currency": "USD"},
    )

    assert payload["saved"] is False
    assert payload["source"] == "blank"
    assert payload["selected_metric"] == "price_fcf"
    assert payload["metric_assumptions"] == {}
    assert payload["scenarios"]["base"]["status"] == "missing"
    assert payload["scenarios"]["base"]["multiple"] is None
    assert payload["scenarios"]["base"]["denominator"] is None


def test_value_range_payload_computes_scenarios_for_each_saved_metric_with_metric_fx():
    currency_context = {
        "price_currency": "USD",
        "financial_currency": "TWD",
        "financial_to_price_fx_rate": 0.03125,
        "fx_rate_as_of": "2026-05-08",
        "conversion_status": "ok",
    }

    payload = multiples.value_range_payload(
        saved_assumption={
            "selected_metric": "price_sales",
            "metric_assumptions": {
                "price_sales": {
                    "denominator_currency": "TWD",
                    "scenarios": {
                        "bear": {"multiple": 4.0, "denominator": 32000.0},
                        "base": {"multiple": 5.0, "denominator": 32000.0},
                        "bull": {"multiple": 6.0, "denominator": 32000.0},
                    },
                },
                "price_earnings": {
                    "denominator_currency": "USD",
                    "scenarios": {
                        "bear": {"multiple": 8.0, "denominator": 200.0},
                        "base": {"multiple": 10.0, "denominator": 200.0},
                        "bull": {"multiple": 12.0, "denominator": 200.0},
                    },
                },
            },
        },
        metrics={},
        peers=multiples._empty_peer_context(),
        effective_weights={"price_sales": 1.0},
        currency_context=currency_context,
        market_data={"current_price": 10.0, "shares": 100.0, "net_debt": 100.0, "currency": "USD"},
    )

    assumptions = payload["metric_assumptions"]
    sales_base = assumptions["price_sales"]["computed_scenarios"]["base"]
    earnings_base = assumptions["price_earnings"]["computed_scenarios"]["base"]

    assert assumptions["price_sales"]["denominator_currency"] == "TWD"
    assert assumptions["price_earnings"]["denominator_currency"] == "USD"
    assert payload["scenarios"]["base"] == sales_base
    assert sales_base["denominator_currency"] == "TWD"
    assert sales_base["denominator_converted"] == 1000.0
    assert sales_base["expected_price"] == 49.0
    assert earnings_base["denominator_currency"] == "TWD"
    assert earnings_base["denominator"] == 6400.0
    assert earnings_base["denominator_converted"] == 200.0
    assert earnings_base["expected_price"] == 20.0


def test_legacy_value_range_without_currency_is_computed_as_price_currency(monkeypatch):
    monkeypatch.setattr(multiples, "fx_rate_to_base", lambda currency, base: {"rate": 0.03125, "as_of": "2026-05-08"})
    metrics = {
        "price_sales": {
            "value": 4.0,
            "denominator": 32000.0,
            "denominator_currency": "TWD",
            "denominator_converted": 1000.0,
            "status": "ok",
        }
    }
    currency_context = {
        "price_currency": "USD",
        "financial_currency": "TWD",
        "financial_to_price_fx_rate": 0.03125,
        "fx_rate_as_of": "2026-05-08",
        "conversion_status": "ok",
    }

    payload = multiples.value_range_payload(
        saved_assumption={
            "metric": "price_sales",
            "scenarios": {
                "bear": {"multiple": 4.0, "denominator": 1000.0},
                "base": {"multiple": 5.0, "denominator": 1000.0},
                "bull": {"multiple": 6.0, "denominator": 1000.0},
            },
        },
        metrics=metrics,
        peers=multiples._empty_peer_context(),
        effective_weights={"price_sales": 1.0},
        currency_context=currency_context,
        market_data={"current_price": 10.0, "shares": 100.0, "net_debt": 0.0, "currency": "USD"},
    )

    assert payload["legacy_denominator_currency"] is True
    assert payload["stored_denominator_currency"] == "USD"
    assert payload["denominator_currency"] == "TWD"
    assert payload["scenarios"]["base"]["denominator"] == 32000.0
    assert payload["scenarios"]["base"]["denominator_converted"] == 1000.0
    assert payload["scenarios"]["base"]["expected_price"] == 50.0


def test_value_range_payload_converts_major_denominator_to_minor_quote_currency():
    currency_context = multiples.currency_context_from_info({"currency": "GBp", "financialCurrency": "GBP"})

    payload = multiples.value_range_payload(
        saved_assumption={
            "metric": "price_earnings",
            "denominator_currency": "GBP",
            "scenarios": {
                "bear": {"multiple": 8.0, "denominator": 50.0},
                "base": {"multiple": 10.0, "denominator": 50.0},
                "bull": {"multiple": 12.0, "denominator": 50.0},
            },
        },
        metrics={},
        peers=multiples._empty_peer_context(),
        effective_weights={"price_earnings": 1.0},
        currency_context=currency_context,
        market_data={"current_price": 1000.0, "shares": 100.0, "net_debt": 0.0, "currency": "GBp"},
    )

    assert payload["denominator_currency"] == "GBP"
    assert payload["currency"] == "GBp"
    assert payload["denominator_to_price_fx_rate"] == 100.0
    assert payload["scenarios"]["base"]["denominator"] == 50.0
    assert payload["scenarios"]["base"]["denominator_converted"] == 5000.0
    assert payload["scenarios"]["base"]["expected_price"] == 500.0
    assert payload["scenarios"]["base"]["percent_change"] == -50.0


def test_stale_route_cache_without_currency_context_is_recomputed(monkeypatch):
    from api.cache import delete_cached, long_cache, set_cached
    from api.routers import valuation as valuation_router

    ticker = f"ZZCACHE{uuid.uuid4().hex[:8]}".upper()
    key = valuation_router.valuation_cache_key(ticker, None)
    delete_cached(long_cache, key)
    set_cached(long_cache, key, {"ticker": ticker, "metrics": {}})
    calls = 0

    def _fresh(symbol):
        nonlocal calls
        calls += 1
        return {
            "ticker": symbol,
            "currency_context": {"price_currency": "USD", "financial_currency": "USD"},
            "metrics": {},
        }

    monkeypatch.setattr(multiples, "read_profile_override", lambda symbol: None)
    monkeypatch.setattr(multiples, "get_position_valuation", _fresh)

    try:
        result = valuation_router.get_position_valuation_endpoint(ticker)
        assert calls == 1
        assert result["currency_context"]["price_currency"] == "USD"
    finally:
        delete_cached(long_cache, key)
