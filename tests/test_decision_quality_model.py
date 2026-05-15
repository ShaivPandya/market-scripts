from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_report.recommendations import validate_recommendations_payload
from decision_quality import ACTIONABLE_ACTIONS, DecisionQuality, apply_decision_quality_gates, parse_decision_quality
from decision_quality.eval_runner import EvalCase, validate_case_input_refs

CASES_DIR = Path("docs/decision_quality_evals/cases")


def _case_gold(case_name: str) -> dict:
    return json.loads((CASES_DIR / case_name).read_text(encoding="utf-8"))["gold_output"]


def _case_paths() -> list[Path]:
    return sorted(CASES_DIR.glob("*.json"))


def _load_case(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_dq() -> dict:
    return _case_gold("mu_ai_memory_cycle_2025.json")


@pytest.mark.parametrize("case_path", _case_paths(), ids=lambda path: path.name)
def test_eval_gold_outputs_parse_as_decision_quality(case_path: Path):
    dq = DecisionQuality.model_validate(_load_case(case_path)["gold_output"])

    assert dq.conviction.max_level == 5


@pytest.mark.parametrize("case_path", _case_paths(), ids=lambda path: path.name)
def test_eval_gold_outputs_pass_decision_quality_gates(case_path: Path):
    dq = DecisionQuality.model_validate(_load_case(case_path)["gold_output"])

    gate = apply_decision_quality_gates(dq, current_action=dq.recommended_action, recommendation_status="clear")

    assert gate.status == "pass"
    assert gate.final_action == dq.recommended_action
    assert not any(reason.severity == "blocker" for reason in gate.reasons)


@pytest.mark.parametrize("case_path", _case_paths(), ids=lambda path: path.name)
def test_eval_gold_actionability_matches_action_type(case_path: Path):
    dq = DecisionQuality.model_validate(_load_case(case_path)["gold_output"])

    if dq.recommended_action in ACTIONABLE_ACTIONS:
        assert dq.actionability.status == "actionable"
    else:
        assert dq.actionability.status != "actionable"


@pytest.mark.parametrize("case_path", _case_paths(), ids=lambda path: path.name)
def test_eval_case_input_refs_exist_and_hash(case_path: Path):
    case_data = _load_case(case_path)
    errors = validate_case_input_refs(EvalCase(path=case_path, data=case_data), root=Path("."))

    assert errors == []


def test_eval_case_input_ref_hash_test_skips_null_path_and_hash():
    case_path = CASES_DIR / "scenario_simulator_uncertainty_disclosure_2026.json"
    case_data = _load_case(case_path)

    assert validate_case_input_refs(EvalCase(path=case_path, data=case_data), root=Path(".")) == []


def test_eval_case_input_ref_hash_test_detects_mismatch():
    case_path = CASES_DIR / "mu_ai_memory_cycle_2025.json"
    case_data = copy.deepcopy(_load_case(case_path))
    case_data["input_refs"][0]["sha256"] = hashlib.sha256(b"wrong").hexdigest()

    errors = validate_case_input_refs(EvalCase(path=case_path, data=case_data), root=Path("."))

    assert errors


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


def test_parse_decision_quality_normalizes_common_llm_aliases():
    raw = _valid_dq()
    raw["catalyst"] = {
        "primary": raw["catalyst_or_reason_now"]["event_or_condition"],
        "timeframe": raw["catalyst_or_reason_now"]["expected_timeframe"],
        "why_now": raw["catalyst_or_reason_now"]["why_now"],
        "evidence": raw["catalyst_or_reason_now"]["source_evidence"],
    }
    raw.pop("catalyst_or_reason_now")
    raw["invalidation"] = {
        "metric": "Revenue growth and gross margin",
        "threshold": "Revenue misses guidance or gross margin falls below thesis threshold",
        "timeframe": "Next 2 quarters",
        "implication": "The AI memory-cycle thesis is not confirming.",
    }
    raw["evidence_for"] = [{"summary": "HBM demand supports the thesis.", "source": "dossier"}]
    raw["evidence_against"] = [{"summary": "Memory cycles can reverse quickly.", "source_ref": "dossier"}]
    raw["price_action_read"] = {
        "what_price_did": "Closed above key moving averages.",
        "what_it_implies": "Price confirms the thesis.",
        "confirms_thesis": "Yes - price action supports the thesis.",
        "missing_data": "Volume confirmation.",
    }
    raw["conviction"]["raw_target_weight"] = "not specified"
    raw["sizing_context"]["sizing_delta"] = {
        "direction": "add",
        "amount": 200,
        "unit": "basis_points",
        "basis": "target",
        "condition": "Earnings confirm.",
    }

    dq, errors = parse_decision_quality(raw)

    assert errors == []
    assert dq is not None
    assert dq.invalidation.metric_or_event == "Revenue growth and gross margin"
    assert dq.evidence_against[0].source_refs == ["dossier"]
    assert dq.price_action_read.confirms_thesis is True
    assert dq.price_action_read.data_needed == ["Volume confirmation."]
    assert dq.conviction.raw_target_weight is None
    assert dq.sizing_context.sizing_delta.direction == "increase"
    assert dq.sizing_context.sizing_delta.unit == "bps"


def test_parse_decision_quality_uses_first_invalidation_when_model_returns_list():
    raw = _valid_dq()
    raw["invalidation"] = [
        {
            "observable": "Observable metric",
            "metric_or_event": "Metric event",
            "threshold": "Threshold",
            "timeframe": "Timeframe",
            "implication": "Implication",
        }
    ]

    dq, errors = parse_decision_quality(raw)

    assert errors == []
    assert dq is not None
    assert dq.invalidation.observable == "Observable metric"


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
