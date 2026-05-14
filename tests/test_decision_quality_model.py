from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_report.recommendations import validate_recommendations_payload
from decision_quality import DecisionQuality, apply_decision_quality_gates

CASES_DIR = Path("docs/decision_quality_evals/cases")


def _case_gold(case_name: str) -> dict:
    return json.loads((CASES_DIR / case_name).read_text(encoding="utf-8"))["gold_output"]


def _valid_dq() -> dict:
    return _case_gold("mu_ai_memory_cycle_2025.json")


@pytest.mark.parametrize(
    "case_name",
    [
        "mu_ai_memory_cycle_2025.json",
        "oklo_pre_revenue_smr_short_2025.json",
        "gbp_erm_break_short_1992.json",
        "nvda_ai_platform_long_2026.json",
    ],
)
def test_eval_gold_outputs_parse_as_decision_quality(case_name: str):
    dq = DecisionQuality.model_validate(_case_gold(case_name))

    assert dq.conviction.max_level == 5
    assert dq.actionability.status == "actionable"


def test_decision_quality_rejects_invalid_conviction_level():
    raw = _valid_dq()
    raw["conviction"]["level"] = 6

    with pytest.raises(ValidationError):
        DecisionQuality.model_validate(raw)


def test_decision_quality_rejects_wrong_max_conviction():
    raw = _valid_dq()
    raw["conviction"]["max_level"] = 10

    with pytest.raises(ValidationError):
        DecisionQuality.model_validate(raw)


def test_decision_quality_rejects_missing_invalidation_threshold():
    raw = _valid_dq()
    raw["invalidation"].pop("threshold")

    with pytest.raises(ValidationError):
        DecisionQuality.model_validate(raw)


def test_missing_catalyst_downgrades_actionable_decision():
    raw = _valid_dq()
    raw["catalyst_or_reason_now"] = {
        "event_or_condition": "",
        "expected_timeframe": "",
        "why_now": "",
        "source_evidence": [],
    }
    dq = DecisionQuality.model_validate(raw)

    gate = apply_decision_quality_gates(dq, current_action="buy", recommendation_status="clear")

    assert gate.final_action == "watch"
    assert gate.final_recommendation_status == "review_required"
    assert any(reason.code == "MISSING_CATALYST" for reason in gate.reasons)


def test_missing_evidence_against_caps_confidence_and_marks_review():
    raw = _valid_dq()
    raw["evidence_against"] = []
    dq = DecisionQuality.model_validate(raw)

    gate = apply_decision_quality_gates(dq, current_action="buy", recommendation_status="clear")

    assert gate.final_action == "buy"
    assert gate.final_recommendation_status == "review_required"
    assert gate.confidence_cap == 0.6
    assert any(reason.code == "MISSING_EVIDENCE_AGAINST" for reason in gate.reasons)


def test_add_without_sizing_delta_marks_review():
    raw = _valid_dq()
    raw["recommended_action"] = "add"
    raw["sizing_context"].pop("sizing_delta", None)
    dq = DecisionQuality.model_validate(raw)

    gate = apply_decision_quality_gates(dq, current_action="add", recommendation_status="clear")

    assert gate.final_action == "add"
    assert gate.final_recommendation_status == "review_required"
    assert any(reason.code == "MISSING_SIZING_DELTA" for reason in gate.reasons)


def test_stale_data_quality_blocks_actionable_decision():
    dq = DecisionQuality.model_validate(_valid_dq())

    gate = apply_decision_quality_gates(
        dq,
        current_action="buy",
        recommendation_status="clear",
        data_quality={"critical_data_quality": "stale"},
    )

    assert gate.final_action == "watch"
    assert gate.final_recommendation_status == "review_required"
    assert any(reason.code == "CRITICAL_DATA_QUALITY" for reason in gate.reasons)


def test_recommendation_validation_applies_decision_quality_gate():
    payload = {
        "report_type": "daily",
        "as_of": "2026-05-14",
        "stance": "Neutral / Watchful",
        "recommendation_status": "clear",
        "critical_data_quality": "ok",
        "blocked_reasons": [],
        "do_nothing_rationale": "",
        "what_changed": ["Test"],
        "recommended_actions": [
            {
                "action": "buy",
                "ticker": "MU",
                "instrument": "MU",
                "horizon": "1 trading day",
                "target_change": "initiate",
                "rationale": "Test",
                "evidence": ["Test"],
                "disconfirming_evidence": ["Risk"],
                "catalyst": "Earnings",
                "invalidation": "Breaks trend",
                "expected_onset_window": "1 week",
                "confidence": 0.8,
                "source_quality": "ok",
                "approval_required": False,
                "decision_quality": _valid_dq(),
            }
        ],
        "alternatives": [],
        "opportunity_cost": [],
    }

    normalized = validate_recommendations_payload(
        copy.deepcopy(payload),
        report_type="daily",
        as_of="2026-05-14",
        stance="Neutral / Watchful",
        data_quality={"critical_data_quality": "ok"},
    )

    action = normalized["recommended_actions"][0]
    assert action["action"] == "buy"
    assert action["approval_required"] is True
    assert action["decision_quality_gate"]["status"] == "pass"


def test_recommendation_validation_downgrades_missing_decision_quality():
    payload = {
        "report_type": "daily",
        "as_of": "2026-05-14",
        "stance": "Neutral / Watchful",
        "recommendation_status": "clear",
        "critical_data_quality": "ok",
        "blocked_reasons": [],
        "do_nothing_rationale": "",
        "what_changed": ["Test"],
        "recommended_actions": [
            {
                "action": "buy",
                "ticker": "MU",
                "instrument": "MU",
                "horizon": "1 trading day",
                "target_change": "initiate",
                "rationale": "Test",
                "evidence": ["Test"],
                "disconfirming_evidence": ["Risk"],
                "catalyst": "Earnings",
                "invalidation": "Breaks trend",
                "expected_onset_window": "1 week",
                "confidence": 0.8,
                "source_quality": "ok",
                "approval_required": False,
            }
        ],
        "alternatives": [],
        "opportunity_cost": [],
    }

    normalized = validate_recommendations_payload(
        payload,
        report_type="daily",
        as_of="2026-05-14",
        stance="Neutral / Watchful",
        data_quality={"critical_data_quality": "ok"},
    )

    action = normalized["recommended_actions"][0]
    assert normalized["recommendation_status"] == "review_required"
    assert action["action"] == "watch"
    assert action["approval_required"] is False
    assert action["decision_quality_gate"]["reasons"][0]["code"] == "MISSING_DECISION_QUALITY"
