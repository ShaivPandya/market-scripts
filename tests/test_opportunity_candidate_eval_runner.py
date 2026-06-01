from __future__ import annotations

import copy
import json
from pathlib import Path

from decision_quality.candidate_gates import apply_opportunity_candidate_gates
from decision_quality.eval_corpus import compare_reports, validate_approved_case_metadata
from decision_quality.opportunity_candidate import OpportunityCandidate
from decision_quality.opportunity_candidate_eval_runner import (
    EvalCase,
    build_report,
    build_solver_payload,
    deterministic_score,
    load_cases,
    run_case,
    validate_case_input_refs,
)

CASES_DIR = Path("docs/opportunity_candidate_evals/cases")
BASELINE_PATH = Path("docs/opportunity_candidate_evals/baselines/approved_corpus_baseline.json")


def _case(name: str) -> EvalCase:
    path = CASES_DIR / name
    return EvalCase(path=path, data=json.loads(path.read_text(encoding="utf-8")))


def _judge_payload(total: int = 18) -> dict:
    dimensions = ["trigger_clarity", "why_now", "missing_inputs", "triage_discipline"]
    return {
        "scores": {dimension: 4 for dimension in dimensions},
        "total": total,
        "leakage_detected": False,
        "fatal_issues": [],
        "notes": "Mock judge pass.",
    }


def test_load_cases_defaults_to_review_and_approved():
    cases = load_cases()

    assert cases
    assert all(case.status in {"review", "approved"} for case in cases)
    assert "opportunity_candidate_graduate_nvda_2026" in {case.case_id for case in cases}


def test_load_cases_filters_by_corpus_tag():
    cases = load_cases(statuses={"approved"}, corpus_tags={"opportunity_identification"})

    assert cases
    assert all("opportunity_identification" in case.data.get("corpus_tags", []) for case in cases)


def test_all_approved_cases_have_valid_metadata():
    cases = load_cases(statuses={"approved"})
    assert len(cases) == 5
    for case in cases:
        errors = validate_approved_case_metadata(case.data)
        assert errors == [], f"{case.case_id}: {errors}"


def test_committed_baseline_matches_approved_inventory():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    approved_ids = {case.case_id for case in load_cases(statuses={"approved"})}

    assert set(baseline["cases"]) == approved_ids
    assert all(entry["deterministic_passed"] for entry in baseline["cases"].values())


def test_dry_run_payload_omits_gold_and_rubric(monkeypatch):
    def fail_call(*_args, **_kwargs):
        raise AssertionError("dry-run should not call the LLM")

    monkeypatch.setattr("decision_quality.opportunity_candidate_eval_runner.call_llm_text", fail_call)
    result = run_case(_case("opportunity_candidate_graduate_nvda_2026.json"), dry_run=True, judge=True)
    serialized = json.dumps(result, sort_keys=True)

    assert result["dry_run"] is True
    assert result["gold_deterministic"]["passed"] is True
    assert "gold_output" not in serialized
    assert "rubric_scores" not in serialized
    assert "expected_graduation" not in serialized


def test_null_path_and_hash_input_ref_is_valid():
    case = _case("opportunity_candidate_graduate_nvda_2026.json")

    assert validate_case_input_refs(case, root=Path(".")) == []


def test_valid_mocked_solver_output_passes_without_judge(monkeypatch):
    case = _case("opportunity_candidate_graduate_nvda_2026.json")

    def fake_call(*_args, **_kwargs):
        return json.dumps(case.gold_output), [], object()

    monkeypatch.setattr("decision_quality.opportunity_candidate_eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=False)

    assert result["deterministic"]["passed"] is True
    assert result["opportunity_candidate_gate"]["status"] == "pass"
    assert result["candidate"]["next_action"] == "graduate_to_decision_quality"
    assert "judge" not in result


def test_wrong_next_action_fails_deterministic_scoring():
    case = _case("opportunity_candidate_graduate_nvda_2026.json")
    raw = copy.deepcopy(case.gold_output)
    raw["next_action"] = "research"
    candidate = OpportunityCandidate.model_validate(raw)
    gate = apply_opportunity_candidate_gates(candidate)

    score = deterministic_score(case=case, candidate=candidate, gate=gate)

    assert score["passed"] is False
    assert any(check["name"] == "next_action" and not check["passed"] for check in score["checks"])


def test_graduation_expectation_failure_is_detected():
    case = _case("opportunity_candidate_graduate_nvda_2026.json")
    raw = copy.deepcopy(case.gold_output)
    raw["next_action"] = "research"
    candidate = OpportunityCandidate.model_validate(raw)
    gate = apply_opportunity_candidate_gates(candidate)

    score = deterministic_score(case=case, candidate=candidate, gate=gate)

    assert score["passed"] is False
    assert any(check["name"] == "should_graduate" and not check["passed"] for check in score["checks"])


def test_scout_skeptic_expectations_pass_for_gold():
    case = _case("opportunity_candidate_scout_pass_skeptic_block_nvda_2026.json")
    candidate = OpportunityCandidate.model_validate(case.gold_output)
    gate = apply_opportunity_candidate_gates(candidate)

    score = deterministic_score(case=case, candidate=candidate, gate=gate, has_decision_quality=False)

    assert score["passed"] is True
    assert any(check["name"] == "expected_scout_status" and check["passed"] for check in score["checks"])
    assert any(check["name"] == "expected_skeptic_status" and check["passed"] for check in score["checks"])


def test_build_report_counts_failed_check_even_when_score_exceeds_threshold():
    report = build_report(
        [
            {
                "case_id": "wrong_action_high_score",
                "dry_run": False,
                "deterministic": {
                    "score": 91.67,
                    "passed": False,
                    "checks": [{"name": "next_action", "passed": False, "message": "wrong action"}],
                },
            }
        ],
        fail_under_deterministic=80.0,
        fail_under_judge=14.0,
    )

    assert report["summary"]["deterministic_failures"] == ["wrong_action_high_score"]


def test_invalid_json_fails_cleanly(monkeypatch):
    case = _case("opportunity_candidate_graduate_nvda_2026.json")

    def fake_call(*_args, **_kwargs):
        return "not json", [], object()

    monkeypatch.setattr("decision_quality.opportunity_candidate_eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=False)

    assert result["candidate"] is None
    assert result["deterministic"]["passed"] is False
    assert result["opportunity_candidate_gate"]["reasons"][0]["code"] == "MISSING_OPPORTUNITY_CANDIDATE"


def test_solver_call_error_fails_cleanly(monkeypatch):
    case = _case("opportunity_candidate_graduate_nvda_2026.json")

    def fake_call(*_args, **_kwargs):
        raise TypeError("missing provider credentials")

    monkeypatch.setattr("decision_quality.opportunity_candidate_eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=False)

    assert result["candidate"] is None
    assert result["solver_error"]["type"] == "TypeError"
    assert result["deterministic"]["passed"] is False
    assert result["opportunity_candidate_gate"]["status"] == "invalid"


def test_mocked_judge_result_is_included(monkeypatch):
    case = _case("opportunity_candidate_graduate_nvda_2026.json")
    responses = [json.dumps(case.gold_output), json.dumps(_judge_payload())]

    def fake_call(*_args, **_kwargs):
        return responses.pop(0), [], object()

    monkeypatch.setattr("decision_quality.opportunity_candidate_eval_runner.call_llm_text", fake_call)
    result = run_case(case, judge=True)

    assert result["deterministic"]["passed"] is True
    assert result["judge"]["passed"] is True
    assert result["judge"]["total"] == 18
    assert responses == []


def test_compare_reports_detects_regression():
    baseline = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "cases": {
            "opportunity_candidate_graduate_nvda_2026": {
                "case_id": "opportunity_candidate_graduate_nvda_2026",
                "deterministic_passed": True,
            }
        },
    }
    current = {
        "generated_at": "2026-01-02T00:00:00+00:00",
        "cases": [
            {
                "case_id": "opportunity_candidate_graduate_nvda_2026",
                "deterministic": {
                    "passed": False,
                    "checks": [{"name": "next_action", "passed": False}],
                },
            }
        ],
    }

    comparison = compare_reports(baseline, current)

    assert comparison["summary"]["regression_detected"] is True
    assert comparison["summary"]["new_deterministic_failures"] == ["opportunity_candidate_graduate_nvda_2026"]


def test_build_solver_payload_keeps_null_path_ref_without_content():
    case = _case("opportunity_candidate_graduate_nvda_2026.json")
    payload = build_solver_payload(case)

    assert payload["input_refs"][0]["path"] is None
    assert "content" not in payload["input_refs"][0]


def test_dry_run_report_flags_metadata_or_gold_failures():
    report = build_report(
        [
            {
                "case_id": "broken",
                "dry_run": True,
                "metadata_errors": ["missing corpus_tags"],
                "gold_deterministic": {"passed": True},
            }
        ],
        fail_under_deterministic=80.0,
        fail_under_judge=14.0,
    )

    assert report["summary"]["dry_run_failures"] == ["broken"]
