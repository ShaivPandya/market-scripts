import math
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from api.routers import analyzer as analyzer_router
from api.routers.analyzer import AnalyzerRequest, _cache_key, _compute_analyzer_result_cached
from portfolio.portfolio_optimizer import portfolio_analyzer as analyzer_module
from portfolio.portfolio_optimizer.portfolio_analyzer import (
    INTERACTIVE_SIGNAL_ANCHOR_MIN_UNIQUE,
    INTERACTIVE_SIGNAL_ANCHOR_TOP_N,
    build_course_of_action,
    compute_qualitative_signals,
    compute_valuation_signal,
    fetch_qualitative_metrics,
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
                    "price_book": 0,
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


def test_analyzer_cache_key_ignores_freshness_bucket():
    req = AnalyzerRequest()

    assert _cache_key(req, freshness_bucket=123) == _cache_key(req, freshness_bucket=123)
    assert _cache_key(req, freshness_bucket=123) == _cache_key(req, freshness_bucket=124)


def test_analyzer_cache_key_changes_with_source_token(monkeypatch):
    req = AnalyzerRequest()

    monkeypatch.setattr(analyzer_module, "analyzer_source_cache_token", lambda: {"portfolio_metadata_hash": "one"})
    first = _cache_key(req)
    monkeypatch.setattr(analyzer_module, "analyzer_source_cache_token", lambda: {"portfolio_metadata_hash": "two"})

    assert first != _cache_key(req)


def test_analyzer_short_cache_reuses_result_within_ttl(monkeypatch):
    calls: list[dict] = []
    source_token = {"test_token": uuid4().hex}

    def fake_get_data(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "error": None,
            "course_of_action": {
                "summary": {"as_of": "2026-05-04T14:15:00+00:00"},
                "action_queue": [],
            },
        }

    analyzer_router.short_cache.clear()
    monkeypatch.setattr(analyzer_module, "analyzer_source_cache_token", lambda: source_token)
    monkeypatch.setattr(analyzer_module, "get_data", fake_get_data)

    req = AnalyzerRequest()
    first = _compute_analyzer_result_cached(req)
    second = _compute_analyzer_result_cached(req)

    assert first["course_of_action"] == second["course_of_action"]
    assert len(calls) == 1


def _sample_analyzer_inputs() -> dict:
    tickers = ["AAA"]
    meta = pd.DataFrame(
        {
            "asset": pd.Series(["equity"], index=tickers, dtype="object"),
            "instrument_type": pd.Series(["security"], index=tickers, dtype="object"),
            "price_symbol": pd.Series(["AAA"], index=tickers, dtype="object"),
            "quantity": pd.Series([10.0], index=tickers, dtype="float64"),
            "contract_multiplier": pd.Series([1.0], index=tickers, dtype="float64"),
            "direction": pd.Series(["long"], index=tickers, dtype="object"),
            "direction_intended": pd.Series(["long"], index=tickers, dtype="object"),
            "contrarian": pd.Series([False], index=tickers, dtype="bool"),
            "contrarian_eligible": pd.Series([False], index=tickers, dtype="bool"),
            "drawdown_52w": pd.Series([0.12], index=tickers, dtype="float64"),
            "stabilized_10d": pd.Series([True], index=tickers, dtype="bool"),
            "days_since_new_low": pd.Series([12], index=tickers, dtype="int64"),
            "no_new_high_20d": pd.Series([True], index=tickers, dtype="bool"),
            "days_since_high": pd.Series([40], index=tickers, dtype="int64"),
            "avg20_roc63": pd.Series([0.04], index=tickers, dtype="float64"),
            "avg10_rel_roc": pd.Series([0.03], index=tickers, dtype="float64"),
        }
    )
    valuation_df = pd.DataFrame(
        {
            "price_sales": [2.0],
            "price_operating_income": [8.0],
            "price_fcf": [12.0],
            "price_earnings": [18.0],
            "price_book": [3.0],
            "valuation_profile_id": ["default"],
        },
        index=tickers,
    )
    qualitative_df = pd.DataFrame(
        {
            "business_quality_qual_score": [80.0],
            "business_quality_qual_confidence": [0.8],
            "business_quality_qual_status": ["available"],
            "business_quality_qual_evidence": ["durable"],
            "industry_quality_score": [70.0],
            "industry_quality_confidence": [0.7],
            "industry_quality_status": ["available"],
            "industry_quality_evidence": ["attractive"],
            "management_quality_score": [85.0],
            "management_quality_confidence": [0.75],
            "management_quality_status": ["available"],
            "management_quality_evidence": ["strong"],
            "overview_source_hash": ["overview"],
            "management_quality_source_hash": ["management"],
        },
        index=tickers,
    )
    return {
        "meta": meta,
        "tickers": tickers,
        "active_tickers": tickers,
        "valuation_tickers": tickers,
        "signal_effective": pd.Series([0.2], index=tickers, dtype="float64"),
        "signal_subcomponents": {
            "quality_signal": pd.Series([0.7], index=tickers, dtype="float64"),
            "eps_mom_signal": pd.Series([0.3], index=tickers, dtype="float64"),
            "rev_mom_signal": pd.Series([0.4], index=tickers, dtype="float64"),
            "price_mom_signal": pd.Series([0.5], index=tickers, dtype="float64"),
        },
        "signal_anchor_meta": {
            "signal_anchor_cache_status": "hit",
            "numpy_scalar": np.float64(1.25),
            "timestamp": pd.Timestamp("2026-05-07T12:00:00Z"),
            "set_value": {"b", "a"},
        },
        "valuation_df": valuation_df,
        "qualitative_df": qualitative_df,
        "direction_display": pd.Series(["long"], index=tickers, dtype="object"),
    }


def _reset_analyzer_input_cache(monkeypatch, tmp_path, *, now: datetime | None = None):
    monkeypatch.setenv("PORTFOLIO_ANALYZER_INPUT_CACHE_DIR", str(tmp_path))
    analyzer_module._ANALYZER_INPUTS_CACHE.clear()
    analyzer_module._ANALYZER_INPUTS_FLIGHTS.clear()
    if now is not None:
        monkeypatch.setattr(analyzer_module, "_analyzer_input_cache_now", lambda: now)


def test_phase_a_snapshot_reused_across_missions(monkeypatch, tmp_path):
    _reset_analyzer_input_cache(monkeypatch, tmp_path)
    calls = {"compute": 0, "apply": 0}
    source_token = {"tickers": ["AAA"], "portfolio_metadata_hash": uuid4().hex, "qualitative_sources": []}
    original_apply_scenario = analyzer_module._apply_scenario

    def fake_compute():
        calls["compute"] += 1
        return _sample_analyzer_inputs()

    def counting_apply_scenario(inputs, scenario_config):
        calls["apply"] += 1
        return original_apply_scenario(inputs, scenario_config)

    monkeypatch.setattr(analyzer_module, "analyzer_source_cache_token", lambda: source_token)
    monkeypatch.setattr(analyzer_module, "_compute_analyzer_inputs", fake_compute)
    monkeypatch.setattr(analyzer_module, "_apply_scenario", counting_apply_scenario)

    balanced = analyzer_module.analyze_portfolio({"preset": "balanced"})
    analyzer_module._ANALYZER_INPUTS_CACHE.clear()
    capital_preservation = analyzer_module.analyze_portfolio({"preset": "capital_preservation"})

    assert calls["compute"] == 1
    assert calls["apply"] == 2
    assert balanced["scenario"]["preset"] == "balanced"
    assert capital_preservation["scenario"]["preset"] == "capital_preservation"
    assert balanced["scenario"] != capital_preservation["scenario"]


def test_phase_a_snapshot_uses_sliding_freshness_not_fixed_bucket(monkeypatch, tmp_path):
    base = datetime(2026, 5, 7, 12, 4, 50, tzinfo=UTC)
    _reset_analyzer_input_cache(monkeypatch, tmp_path, now=base)
    snapshot_key = analyzer_module._analyzer_input_snapshot_key(
        {"tickers": ["AAA"], "portfolio_metadata_hash": "freshness", "qualitative_sources": []}
    )
    analyzer_module._write_analyzer_input_snapshot(snapshot_key, _sample_analyzer_inputs())
    path = analyzer_module._analyzer_input_snapshot_path(snapshot_key)
    os.utime(path, (base.timestamp(), base.timestamp()))

    monkeypatch.setattr(analyzer_module, "_analyzer_input_cache_now", lambda: base + timedelta(seconds=20))
    assert analyzer_module._read_analyzer_input_snapshot(snapshot_key) is not None
    assert math.isclose(path.stat().st_mtime, (base + timedelta(seconds=20)).timestamp(), abs_tol=0.01)

    monkeypatch.setattr(analyzer_module, "_analyzer_input_cache_now", lambda: base + timedelta(seconds=301))
    assert analyzer_module._read_analyzer_input_snapshot(snapshot_key) is not None

    os.utime(path, (base.timestamp(), base.timestamp()))
    assert analyzer_module._read_analyzer_input_snapshot(snapshot_key) is None


def test_phase_a_snapshot_touch_failure_does_not_break_read(monkeypatch, tmp_path):
    base = datetime(2026, 5, 7, 12, 0, 0, tzinfo=UTC)
    _reset_analyzer_input_cache(monkeypatch, tmp_path, now=base)
    snapshot_key = analyzer_module._analyzer_input_snapshot_key(
        {"tickers": ["AAA"], "portfolio_metadata_hash": "touch-failure", "qualitative_sources": []}
    )
    analyzer_module._write_analyzer_input_snapshot(snapshot_key, _sample_analyzer_inputs())
    path = analyzer_module._analyzer_input_snapshot_path(snapshot_key)
    os.utime(path, (base.timestamp(), base.timestamp()))

    monkeypatch.setattr(analyzer_module, "_analyzer_input_cache_now", lambda: base + timedelta(seconds=20))
    monkeypatch.setattr(
        analyzer_module.os,
        "utime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("touch failed")),
    )

    assert analyzer_module._read_analyzer_input_snapshot(snapshot_key) is not None

    monkeypatch.setattr(analyzer_module, "_analyzer_input_cache_now", lambda: base + timedelta(seconds=301))
    assert analyzer_module._read_analyzer_input_snapshot(snapshot_key) is None


def test_phase_a_snapshot_key_invalidates_for_metadata_sources_and_version(monkeypatch):
    first = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "asset": "equity",
                "direction": "long",
                "price_symbol": "AAA",
                "instrument_type": "security",
            }
        ]
    )
    second = first.copy()
    second.loc[0, "direction"] = "short"

    assert analyzer_module._hash_json(
        analyzer_module._normalize_position_source_records(first)
    ) != analyzer_module._hash_json(analyzer_module._normalize_position_source_records(second))
    assert analyzer_module._analyzer_input_snapshot_key(
        {"tickers": ["AAA"], "portfolio_metadata_hash": "same", "qualitative_sources": [{"overview_hash": "a"}]}
    ) != analyzer_module._analyzer_input_snapshot_key(
        {"tickers": ["AAA"], "portfolio_metadata_hash": "same", "qualitative_sources": [{"overview_hash": "b"}]}
    )

    original = analyzer_module._analyzer_input_snapshot_key(
        {"tickers": ["AAA"], "portfolio_metadata_hash": "same", "qualitative_sources": []}
    )
    monkeypatch.setattr(analyzer_module, "_ANALYZER_INPUTS_VERSION", "v-next")
    bumped = analyzer_module._analyzer_input_snapshot_key(
        {"tickers": ["AAA"], "portfolio_metadata_hash": "same", "qualitative_sources": []}
    )
    assert original != bumped


def test_phase_a_snapshot_round_trip_preserves_dtypes_and_json_safe_metadata():
    decoded = analyzer_module._decode_analyzer_input_snapshot(
        analyzer_module._encode_analyzer_input_snapshot(_sample_analyzer_inputs())
    )

    assert decoded["tickers"] == ["AAA"]
    assert decoded["meta"].index.tolist() == ["AAA"]
    assert decoded["meta"]["contrarian_eligible"].dtype == bool
    assert decoded["meta"]["stabilized_10d"].dtype == bool
    assert decoded["signal_subcomponents"]["quality_signal"].index.tolist() == ["AAA"]
    assert decoded["signal_anchor_meta"]["numpy_scalar"] == 1.25
    assert decoded["signal_anchor_meta"]["timestamp"] == "2026-05-07T12:00:00+00:00"
    assert decoded["signal_anchor_meta"]["set_value"] == ["a", "b"]


def test_phase_a_snapshot_corrupt_or_tmp_only_cache_falls_through_to_compute(monkeypatch, tmp_path):
    _reset_analyzer_input_cache(monkeypatch, tmp_path)
    source_token = {"tickers": ["AAA"], "portfolio_metadata_hash": "corrupt", "qualitative_sources": []}
    snapshot_key = analyzer_module._analyzer_input_snapshot_key(source_token)
    path = analyzer_module._analyzer_input_snapshot_path(snapshot_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a zip")
    path.with_name(f"{path.name}.tmp").write_bytes(b"partial")

    calls = {"compute": 0}

    def fake_compute():
        calls["compute"] += 1
        return _sample_analyzer_inputs()

    monkeypatch.setattr(analyzer_module, "analyzer_source_cache_token", lambda: source_token)
    monkeypatch.setattr(analyzer_module, "_compute_analyzer_inputs", fake_compute)

    inputs = analyzer_module._cached_analyzer_inputs()

    assert calls["compute"] == 1
    assert inputs["tickers"] == ["AAA"]


def test_phase_a_snapshot_cleanup_removes_old_local_snapshots(monkeypatch, tmp_path):
    base = datetime(2026, 5, 7, 12, 0, 0, tzinfo=UTC)
    _reset_analyzer_input_cache(monkeypatch, tmp_path, now=base)
    old_key = analyzer_module._analyzer_input_snapshot_key(
        {"tickers": ["AAA"], "portfolio_metadata_hash": "old", "qualitative_sources": []}
    )
    fresh_key = analyzer_module._analyzer_input_snapshot_key(
        {"tickers": ["AAA"], "portfolio_metadata_hash": "fresh", "qualitative_sources": []}
    )
    analyzer_module._write_analyzer_input_snapshot(old_key, _sample_analyzer_inputs())
    analyzer_module._write_analyzer_input_snapshot(fresh_key, _sample_analyzer_inputs())
    old_path = analyzer_module._analyzer_input_snapshot_path(old_key)
    fresh_path = analyzer_module._analyzer_input_snapshot_path(fresh_key)
    stale_time = (base - timedelta(hours=25)).timestamp()
    os.utime(old_path, (stale_time, stale_time))

    assert analyzer_module.cleanup_analyzer_input_snapshots(max_age_seconds=24 * 60 * 60) == 1
    assert not old_path.exists()
    assert fresh_path.exists()


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
                "price_book": 0,
                "business_quality_qualitative": 20,
                "industry_quality": 10,
                "management_quality": 10,
            }
        }
    )

    assert math.isclose(scenario["factor_weights"]["quality"], 1 / 9)
    assert math.isclose(scenario["factor_weights"]["fundamental_momentum"], 4 / 9)
    assert math.isclose(scenario["factor_weights"]["qualitative"], 4 / 9)
    assert math.isclose(scenario["fundamental_momentum_weights"]["revenue"], 0.50)
    assert math.isclose(scenario["fundamental_momentum_weights"]["eps"], 0.50)
    assert math.isclose(scenario["qualitative_weights"]["business_quality_qualitative"], 0.50)
    assert math.isclose(scenario["qualitative_weights"]["industry_quality"], 0.25)
    assert math.isclose(scenario["qualitative_weights"]["management_quality"], 0.25)


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
                "price_book": 0,
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
                "price_book": 0,
            },
            "brakes": {
                "drawdown_sensitivity": 0.6,
                "contrarian_penalty": 0.2,
                "short_squeeze_brake": 0,
            },
        }
    )

    assert _cache_key(raw_score_req) == _cache_key(normalized_req)


def test_default_balanced_mission_includes_small_valuation_sleeve():
    scenario = normalize_analyzer_scenario()

    assert math.isclose(scenario["factor_weights"]["quality"], 0.20)
    assert math.isclose(scenario["factor_weights"]["price_momentum"], 0.30)
    assert math.isclose(scenario["factor_weights"]["fundamental_momentum"], 0.21)
    assert math.isclose(scenario["factor_weights"]["valuation"], 0.09)
    assert math.isclose(scenario["factor_weights"]["qualitative"], 0.20)
    assert math.isclose(scenario["fundamental_momentum_weights"]["revenue"], 13 / 21)
    assert math.isclose(scenario["fundamental_momentum_weights"]["eps"], 8 / 21)
    assert "price_book" in scenario["valuation_weights"]
    assert math.isclose(scenario["qualitative_weights"]["business_quality_qualitative"], 0.40)


def test_core_db_default_mission_uses_shared_balanced_scenario():
    import portfolio.core_db as core_db

    assert core_db._default_optimization_scenario() == normalize_analyzer_scenario({"preset": "balanced"})


def test_legacy_balanced_default_is_upgraded_to_current_default():
    scenario = normalize_analyzer_scenario(
        {
            "preset": "balanced",
            "factor_weights": {
                "quality": 0.30,
                "price_momentum": 0.40,
                "fundamental_momentum": 0.30,
                "valuation": 0.0,
            },
            "fundamental_momentum_weights": {"revenue": 0.67, "eps": 0.33},
            "valuation_weights": {
                "price_sales": 0.25,
                "price_operating_income": 0.25,
                "price_fcf": 0.25,
                "price_earnings": 0.25,
            },
            "brakes": {
                "drawdown_sensitivity": 0,
                "contrarian_penalty": 0,
                "short_squeeze_brake": 0,
            },
        }
    )

    assert math.isclose(scenario["factor_weights"]["price_momentum"], 0.30)
    assert math.isclose(scenario["factor_weights"]["valuation"], 0.09)
    assert math.isclose(scenario["factor_weights"]["qualitative"], 0.20)
    assert "price_book" in scenario["valuation_weights"]


def test_preset_only_request_uses_named_mission_weights():
    scenario = normalize_analyzer_scenario({"preset": "value_dislocation"})
    req = AnalyzerRequest(scenario={"preset": "value_dislocation"})

    assert math.isclose(scenario["factor_weights"]["valuation"], 0.50)
    assert math.isclose(scenario["factor_weights"]["price_momentum"], 0.08)
    assert _cache_key(req) == _cache_key(
        AnalyzerRequest(
            scenario={
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
            }
        )
    )


def test_valuation_signal_ranks_lower_positive_multiples_higher():
    raw = pd.DataFrame(
        {
            "price_sales": [2.0, 5.0, 9.0],
            "price_operating_income": [8.0, 12.0, 20.0],
            "price_fcf": [10.0, 18.0, 25.0],
            "price_earnings": [14.0, 22.0, 35.0],
            "price_book": [1.5, 3.0, 6.0],
        },
        index=["CHEAP", "MID", "EXPENSIVE"],
    )

    signal = compute_valuation_signal(
        raw,
        {
            "price_sales": 0.25,
            "price_operating_income": 0.25,
            "price_fcf": 0.20,
            "price_earnings": 0.20,
            "price_book": 0.10,
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
            "price_book": [1.5, 3.0, None],
        },
        index=["A", "B", "NON_EQUITY"],
    )

    signal = compute_valuation_signal(
        raw,
        {
            "price_sales": 0.25,
            "price_operating_income": 0.25,
            "price_fcf": 0.20,
            "price_earnings": 0.20,
            "price_book": 0.10,
        },
    )

    assert signal["A"] > signal["B"]
    assert math.isnan(signal["NON_EQUITY"])


def test_valuation_signal_uses_profile_weights_when_available():
    raw = pd.DataFrame(
        {
            "price_sales": [1.0, 5.0, 10.0],
            "price_book": [10.0, 5.0, 1.0],
            "price_sales_profile_weight": [1.0, 1.0, 1.0],
            "price_book_profile_weight": [0.0, 0.0, 0.0],
        },
        index=["CHEAP_SALES", "MID", "EXPENSIVE_SALES"],
    )

    signal = compute_valuation_signal(raw, {"price_sales": 0.5, "price_book": 0.5})

    assert signal["CHEAP_SALES"] > signal["MID"] > signal["EXPENSIVE_SALES"]


def test_compute_qualitative_signals_maps_rubric_scores_to_signal_scale():
    raw = pd.DataFrame(
        {
            "business_quality_qual_score": [80, 35],
            "industry_quality_score": [70, 45],
            "management_quality_score": [90, 30],
        },
        index=["STRONG", "WEAK"],
    )

    signal, sub_signals = compute_qualitative_signals(
        raw,
        {
            "business_quality_qualitative": 0.4,
            "industry_quality": 0.3,
            "management_quality": 0.3,
        },
        ["STRONG", "WEAK"],
    )

    assert signal["STRONG"] > 0
    assert signal["WEAK"] < 0
    assert sub_signals["management_quality"]["STRONG"] > sub_signals["management_quality"]["WEAK"]


def test_fetch_qualitative_metrics_reuses_cache_for_same_source_hash(monkeypatch):
    import uuid

    import llm_utils

    calls = {"count": 0}
    overview_content = f"overview evidence {uuid.uuid4()}"
    management_content = f"management evidence {uuid.uuid4()}"
    monkeypatch.setattr(analyzer_module, "_read_overview_markdown", lambda ticker: overview_content)
    monkeypatch.setattr(analyzer_module, "_read_management_quality_markdown", lambda ticker: management_content)
    monkeypatch.setattr(llm_utils, "has_llm_api_key", lambda provider=None: True)

    def fake_score(**kwargs):
        calls["count"] += 1
        return {
            "ticker": kwargs["ticker"],
            "business_quality_qual_score": 82,
            "business_quality_qual_confidence": 0.8,
            "business_quality_qual_status": "available",
            "business_quality_qual_evidence": "durable business",
            "industry_quality_score": 74,
            "industry_quality_confidence": 0.7,
            "industry_quality_status": "available",
            "industry_quality_evidence": "attractive industry",
            "management_quality_score": 68,
            "management_quality_confidence": 0.6,
            "management_quality_status": "available",
            "management_quality_evidence": "solid management",
            "overview_source_hash": kwargs["overview_hash"],
            "management_quality_source_hash": kwargs["management_hash"],
        }

    monkeypatch.setattr(analyzer_module, "_score_qualitative_with_llm", fake_score)

    first = fetch_qualitative_metrics("QUALCACHE")
    second = fetch_qualitative_metrics("QUALCACHE")

    assert calls["count"] == 1
    assert first["business_quality_qual_score"] == second["business_quality_qual_score"] == 82


def test_fetch_qualitative_metrics_missing_documents_do_not_call_llm(monkeypatch):
    import llm_utils

    monkeypatch.setattr(analyzer_module, "_read_overview_markdown", lambda ticker: None)
    monkeypatch.setattr(analyzer_module, "_read_management_quality_markdown", lambda ticker: None)
    monkeypatch.setattr(llm_utils, "has_llm_api_key", lambda provider=None: True)

    def fail_score(**kwargs):
        raise AssertionError("LLM scoring should not run without source documents")

    monkeypatch.setattr(analyzer_module, "_score_qualitative_with_llm", fail_score)

    result = fetch_qualitative_metrics("NODOCS")

    assert result["business_quality_qual_status"] == "missing_overview"
    assert result["industry_quality_status"] == "missing_overview"
    assert result["management_quality_status"] == "missing_management_quality"
    assert math.isnan(result["business_quality_qual_score"])


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
        "qualitative_signal": 0.0,
        "business_quality_qual_signal": 0.0,
        "industry_quality_signal": 0.0,
        "management_quality_signal": 0.0,
        "business_quality_qual_status": "available",
        "industry_quality_status": "available",
        "management_quality_status": "available",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _balanced_course(rows: list[dict]) -> dict:
    return build_course_of_action(
        _course_rows(rows),
        normalize_analyzer_scenario({"preset": "balanced"}),
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


def test_course_of_action_missing_qualitative_evidence_warns_when_weighted():
    course = build_course_of_action(
        _course_rows(
            [
                {
                    "ticker": "QUALMISS",
                    "direction": "long",
                    "scenario_score": 1.10,
                    "score_delta": 0.40,
                    "quality_signal": 1.0,
                    "price_mom_signal": 1.0,
                    "fundamental_momentum_signal": 1.0,
                    "qualitative_signal": math.nan,
                    "business_quality_qual_signal": math.nan,
                    "industry_quality_signal": math.nan,
                    "management_quality_signal": math.nan,
                    "business_quality_qual_status": "missing_overview",
                    "industry_quality_status": "missing_overview",
                    "management_quality_status": "missing_management_quality",
                }
            ]
        ),
        normalize_analyzer_scenario(
            {
                "metric_scores": {
                    "quality": 20,
                    "price_momentum": 20,
                    "revenue": 20,
                    "eps": 0,
                    "price_sales": 0,
                    "price_operating_income": 0,
                    "price_fcf": 0,
                    "price_earnings": 0,
                    "price_book": 0,
                    "business_quality_qualitative": 20,
                    "industry_quality": 10,
                    "management_quality": 10,
                }
            }
        ),
    )

    action = _first_action(course, "QUALMISS")
    assert any("Missing qualitative evidence" in warning for warning in action["warnings"])


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


def test_portfolio_plus_ideas_universe_injects_only_enabled_non_duplicates(monkeypatch):
    monkeypatch.setattr(
        analyzer_module,
        "_get_positions_df",
        lambda: pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "asset": "equity",
                    "direction": "long",
                    "instrument_type": "security",
                    "quantity": 10,
                    "contract_multiplier": 1,
                }
            ]
        ),
    )

    class FakeReadService:
        def list_objects(self, object_type, limit=500):
            assert object_type == "InvestmentIdea"
            assert limit == 500
            return [
                {"id": "1", "ticker": "MSFT", "status": "watching", "metadata": {"analyzer_direction": "long"}},
                {"id": "2", "ticker": "TSLA", "status": "researching", "metadata": {"analyzer_direction": "short"}},
                {"id": "3", "ticker": "AAPL", "status": "watching", "metadata": {"analyzer_direction": "short"}},
                {"id": "4", "ticker": "META", "status": "accepted", "metadata": {"analyzer_direction": "long"}},
                {"id": "5", "ticker": "NFLX", "status": "watching", "metadata": {}},
            ]

    monkeypatch.setattr("ontology.runtime_read_service.OntologyRuntimeReadService", FakeReadService)

    universe = analyzer_module._positions_df_for_universe("portfolio_plus_ideas")

    assert universe["ticker"].tolist() == ["AAPL", "MSFT", "TSLA"]
    idea_rows = universe[universe["source_type"].eq("idea")]
    assert set(idea_rows["ticker"]) == {"MSFT", "TSLA"}
    assert idea_rows.set_index("ticker").loc["MSFT", "quantity"] == 0
    assert idea_rows.set_index("ticker").loc["TSLA", "direction"] == "short"


def test_idea_only_course_actions_use_initiate_and_pass_labels():
    course = _balanced_course(
        [
            {
                "ticker": "IDEALONG",
                "source_type": "idea",
                "direction": "long",
                "scenario_score": 0.90,
                "score_delta": 0.20,
            },
            {
                "ticker": "IDEASHORT",
                "source_type": "idea",
                "direction": "short",
                "scenario_score": -0.90,
                "score_delta": -0.20,
            },
            {
                "ticker": "BADLONG",
                "source_type": "idea",
                "direction": "long",
                "scenario_score": -0.90,
                "score_delta": -0.20,
            },
            {
                "ticker": "OWNED",
                "source_type": "portfolio",
                "direction": "long",
                "scenario_score": 0.90,
                "score_delta": 0.20,
            },
        ]
    )

    assert _first_action(course, "IDEALONG")["action"] == "Initiate Long"
    assert _first_action(course, "IDEASHORT")["action"] == "Initiate Short"
    assert _first_action(course, "BADLONG")["action"] == "Pass"
    assert _first_action(course, "OWNED")["action"] == "Increase Long"
