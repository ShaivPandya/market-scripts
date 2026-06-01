from __future__ import annotations

import copy
import json
from pathlib import Path

from decision_quality.eval_runner import (
    EvalCase,
    build_report,
    build_solver_payload,
    deterministic_score,
    load_cases,
    run_case,
    validate_case_input_refs,
)
from decision_quality.gates import apply_decision_quality_gates
from decision_quality.models import DecisionQuality

CASES_DIR = Path("docs/decision_quality_evals/cases")


def _case(name: str) -> EvalCase:
    path = CASES_DIR / name
    return EvalCase(path=path, data=json.loads(path.read_text(encoding="utf-8")))


def _judge_payload(total: int = 18) -> dict:
    dimensions = [
        "thesis_clarity",
        "mispricing",
        "catalyst_reason_now",
        "evidence_quality",
        "disconfirming_evidence",
        "invalidation",
        "price_action_market_behavior",
        "actionability_discipline",
        "confidence_calibration",
        "sizing_risk_context",
    ]
    return {
        "scores": {dimension: 2 for dimension in dimensions},
        "total": total,
        "leakage_detected": False,
        "fatal_issues": [],
        "notes": "Mock judge pass.",
    }


def test_load_cases_defaults_to_review_and_approved():
    cases = load_cases()

    assert cases
    assert all(case.status in {"review", "approved"} for case in cases)
    assert "scenario_simulator_uncertainty_disclosure_2026" in {case.case_id for case in cases}


def test_load_cases_filters_by_corpus_tag():
    cases = load_cases(statuses={"approved"}, corpus_tags={"structured_dq"})

    assert cases
    assert all("structured_dq" in case.data.get("corpus_tags", []) for case in cases)


def test_dry_run_payload_omits_gold_and_future_outcome_context(monkeypatch):
    def fail_call(*_args, **_kwargs):
        raise AssertionError("dry-run should not call the LLM")

    monkeypatch.setattr("decision_quality.eval_runner.call_llm_text", fail_call)
    result = run_case(_case("ackman_covid_credit_hedge_2020.json"), dry_run=True, judge=True)
    serialized = json.dumps(result, sort_keys=True)

    assert result["dry_run"] is True
    assert "gold_output" not in serialized
    assert "future_outcome_context" not in serialized
    assert "rubric_scores" not in serialized
    assert "human_notes" not in serialized
    assert "March 9" not in serialized
    assert "March 12" not in serialized
    assert "March 23" not in serialized


def test_null_path_and_hash_input_ref_is_valid():
    case = _case("scenario_simulator_uncertainty_disclosure_2026.json")

    assert validate_case_input_refs(case, root=Path(".")) == []


def test_valid_mocked_solver_output_passes_without_judge(monkeypatch):
    case = _case("mu_ai_memory_cycle_2025.json")

    def fake_call(*_args, **_kwargs):
        return json.dumps(case.gold_output), [], object()

    monkeypatch.setattr("decision_quality.eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=False)

    assert result["deterministic"]["passed"] is True
    assert result["decision_quality_gate"]["status"] == "pass"
    assert result["candidate"]["recommended_action"] == "buy"
    assert "judge" not in result


def test_wrong_action_fails_deterministic_scoring():
    case = _case("mu_ai_memory_cycle_2025.json")
    raw = copy.deepcopy(case.gold_output)
    raw["recommended_action"] = "watch"
    candidate = DecisionQuality.model_validate(raw)
    gate = apply_decision_quality_gates(
        candidate, current_action=candidate.recommended_action, recommendation_status="clear"
    )

    score = deterministic_score(case=case, candidate=candidate, gate=gate)

    assert score["passed"] is False
    assert any(check["name"] == "recommended_action" and not check["passed"] for check in score["checks"])


def test_missing_inputs_alignment_fails_when_gold_has_missing_inputs_but_candidate_omits_them():
    case = _case("mu_ai_memory_cycle_2025.json")
    raw = copy.deepcopy(case.gold_output)
    raw["actionability"]["missing_inputs"] = []
    candidate = DecisionQuality.model_validate(raw)
    gate = apply_decision_quality_gates(
        candidate, current_action=candidate.recommended_action, recommendation_status="clear"
    )

    score = deterministic_score(case=case, candidate=candidate, gate=gate)

    assert score["passed"] is False
    assert any(check["name"] == "missing_inputs_alignment" and not check["passed"] for check in score["checks"])


def test_missing_inputs_alignment_passes_when_candidate_surfaces_missing_inputs():
    case = _case("mu_ai_memory_cycle_2025.json")
    candidate = DecisionQuality.model_validate(case.gold_output)
    gate = apply_decision_quality_gates(
        candidate, current_action=candidate.recommended_action, recommendation_status="clear"
    )

    score = deterministic_score(case=case, candidate=candidate, gate=gate)

    assert any(check["name"] == "missing_inputs_alignment" and check["passed"] for check in score["checks"])


def test_missing_inputs_status_requires_missing_inputs_list():
    case = _case("mu_ai_memory_cycle_2025.json")
    raw = copy.deepcopy(case.gold_output)
    raw["recommended_action"] = "research"
    raw["actionability"]["status"] = "missing_inputs"
    raw["actionability"]["missing_inputs"] = []
    candidate = DecisionQuality.model_validate(raw)
    gate = apply_decision_quality_gates(
        candidate, current_action=candidate.recommended_action, recommendation_status="clear"
    )

    score = deterministic_score(case=case, candidate=candidate, gate=gate)

    assert score["passed"] is False
    assert any(check["name"] == "missing_inputs_alignment" and not check["passed"] for check in score["checks"])


def test_build_report_counts_failed_check_even_when_score_exceeds_threshold():
    report = build_report(
        [
            {
                "case_id": "wrong_action_high_score",
                "dry_run": False,
                "deterministic": {
                    "score": 91.67,
                    "passed": False,
                    "checks": [{"name": "recommended_action", "passed": False, "message": "wrong action"}],
                },
            }
        ],
        fail_under_deterministic=80.0,
        fail_under_judge=14.0,
    )

    assert report["summary"]["deterministic_failures"] == ["wrong_action_high_score"]


def test_missing_catalyst_fails_through_gate(monkeypatch):
    case = _case("mu_ai_memory_cycle_2025.json")
    raw = copy.deepcopy(case.gold_output)
    raw["catalyst_or_reason_now"] = {
        "event_or_condition": "",
        "expected_timeframe": "",
        "why_now": "",
        "source_evidence": [],
    }

    def fake_call(*_args, **_kwargs):
        return json.dumps(raw), [], object()

    monkeypatch.setattr("decision_quality.eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=False)

    assert result["deterministic"]["passed"] is False
    assert result["decision_quality_gate"]["status"] == "downgraded"
    assert any(reason["code"] == "MISSING_CATALYST" for reason in result["decision_quality_gate"]["reasons"])


def test_invalid_json_fails_cleanly(monkeypatch):
    case = _case("mu_ai_memory_cycle_2025.json")

    def fake_call(*_args, **_kwargs):
        return "not json", [], object()

    monkeypatch.setattr("decision_quality.eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=False)

    assert result["candidate"] is None
    assert result["deterministic"]["passed"] is False
    assert result["decision_quality_gate"]["reasons"][0]["code"] == "MISSING_DECISION_QUALITY"


def test_solver_call_error_fails_cleanly(monkeypatch):
    case = _case("mu_ai_memory_cycle_2025.json")

    def fake_call(*_args, **_kwargs):
        raise TypeError("missing provider credentials")

    monkeypatch.setattr("decision_quality.eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=False)

    assert result["candidate"] is None
    assert result["solver_error"]["type"] == "TypeError"
    assert result["deterministic"]["passed"] is False
    assert result["decision_quality_gate"]["status"] == "invalid"


def test_missing_invalidation_fails_through_gate(monkeypatch):
    case = _case("mu_ai_memory_cycle_2025.json")
    raw = copy.deepcopy(case.gold_output)
    raw["invalidation"] = {
        "observable": "",
        "metric_or_event": "",
        "threshold": "",
        "timeframe": "",
        "implication": "",
    }

    def fake_call(*_args, **_kwargs):
        return json.dumps(raw), [], object()

    monkeypatch.setattr("decision_quality.eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=False)

    assert result["deterministic"]["passed"] is False
    assert result["decision_quality_gate"]["status"] == "downgraded"
    assert any(reason["code"] == "MISSING_INVALIDATION" for reason in result["decision_quality_gate"]["reasons"])


def test_mocked_judge_result_is_included(monkeypatch):
    case = _case("mu_ai_memory_cycle_2025.json")
    responses = [json.dumps(case.gold_output), json.dumps(_judge_payload())]

    def fake_call(*_args, **_kwargs):
        return responses.pop(0), [], object()

    monkeypatch.setattr("decision_quality.eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=True)

    assert result["deterministic"]["passed"] is True
    assert result["judge"]["passed"] is True
    assert result["judge"]["total"] == 18
    assert responses == []


def test_judge_call_error_is_reported(monkeypatch):
    case = _case("mu_ai_memory_cycle_2025.json")
    responses = [json.dumps(case.gold_output)]

    def fake_call(*_args, **_kwargs):
        if responses:
            return responses.pop(0), [], object()
        raise TypeError("missing judge credentials")

    monkeypatch.setattr("decision_quality.eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=True)

    assert result["deterministic"]["passed"] is True
    assert result["judge"]["passed"] is False
    assert result["judge"]["error"]["type"] == "TypeError"


def test_build_solver_payload_keeps_null_path_ref_without_content():
    case = _case("scenario_simulator_uncertainty_disclosure_2026.json")
    payload = build_solver_payload(case)

    assert payload["input_refs"][0]["path"] is None
    assert "content" not in payload["input_refs"][0]
