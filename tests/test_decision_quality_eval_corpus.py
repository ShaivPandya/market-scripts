from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_quality.chat_eval_runner import load_cases as load_chat_cases
from decision_quality.eval_corpus import (
    compare_reports,
    filter_cases,
    infer_corpus_tags_from_failure_tags,
    normalize_failure_tags,
    summarize_calibration,
    summarize_case_result,
    validate_approved_case_metadata,
    validate_review_case_metadata,
)
from decision_quality.eval_runner import load_cases as load_structured_cases


def test_compare_reports_detects_new_and_fixed_failures():
    baseline = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "cases": {
            "pass_to_fail": {"case_id": "pass_to_fail", "deterministic_passed": True},
            "fail_to_pass": {"case_id": "fail_to_pass", "deterministic_passed": False},
            "stable_pass": {"case_id": "stable_pass", "deterministic_passed": True},
        },
    }
    current = {
        "generated_at": "2026-01-02T00:00:00+00:00",
        "cases": [
            {
                "case_id": "pass_to_fail",
                "deterministic": {"passed": False, "checks": [{"name": "recommended_action", "passed": False}]},
            },
            {"case_id": "fail_to_pass", "deterministic": {"passed": True, "checks": []}},
            {"case_id": "stable_pass", "deterministic": {"passed": True, "checks": []}},
            {"case_id": "new_case", "deterministic": {"passed": True, "checks": []}},
        ],
    }

    comparison = compare_reports(baseline, current)

    assert comparison["summary"]["new_deterministic_failures"] == ["pass_to_fail"]
    assert comparison["summary"]["fixed_deterministic_failures"] == ["fail_to_pass"]
    assert comparison["summary"]["new_in_current"] == ["new_case"]
    assert comparison["summary"]["regression_detected"] is True
    assert comparison["summary"]["check_regressions"][0]["newly_failed_checks"] == ["recommended_action"]


def test_filter_cases_by_corpus_tag_and_tool_pack(tmp_path: Path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "a.json").write_text(
        json.dumps(
            {
                "id": "a",
                "status": "approved",
                "corpus_tags": ["routing_tool_use"],
                "tool_pack": "catalyst_status",
            }
        ),
        encoding="utf-8",
    )
    (cases_dir / "b.json").write_text(
        json.dumps(
            {
                "id": "b",
                "status": "approved",
                "corpus_tags": ["chat_behavior"],
                "tool_pack": "thesis_review",
            }
        ),
        encoding="utf-8",
    )

    cases = load_chat_cases(statuses={"approved"}, cases_dir=cases_dir)
    filtered = filter_cases(cases, corpus_tags={"routing_tool_use"}, tool_pack="catalyst_status")

    assert [case.case_id for case in filtered] == ["a"]


def test_normalize_failure_tags_and_infer_corpus_tags():
    tags, failure_type, errors = normalize_failure_tags(["stale_data", "wrong_routing"])
    assert errors == []
    assert tags == ["source_freshness", "wrong_routing"]
    assert failure_type == "source_freshness"
    assert infer_corpus_tags_from_failure_tags(tags) == ["routing_tool_use", "chat_behavior"]


def test_validate_review_case_metadata_requires_routing_labels():
    errors = validate_review_case_metadata(
        {
            "status": "review",
            "user_message": "route this",
            "failure_tags": ["wrong_routing"],
            "corpus_tags": ["routing_tool_use"],
        }
    )
    assert any("routing_expectations" in error for error in errors)

    clean = validate_review_case_metadata(
        {
            "status": "review",
            "user_message": "route this",
            "failure_tags": ["wrong_routing"],
            "failure_type": "wrong_routing",
            "corpus_tags": ["routing_tool_use"],
            "routing_expectations": {"intent_class": "catalyst_status"},
        }
    )
    assert clean == []


def test_validate_approved_case_metadata_requires_tags_and_dimensions():
    errors = validate_approved_case_metadata({"status": "approved"})
    assert any("corpus_tags" in error for error in errors)
    assert any("required_dq_dimensions" in error for error in errors)

    routing_only = validate_approved_case_metadata(
        {
            "status": "approved",
            "corpus_tags": ["routing_tool_use"],
            "failure_type": "wrong_routing",
            "tool_pack": "catalyst_status",
            "required_dq_dimensions": [],
        }
    )
    assert routing_only == []


@pytest.mark.parametrize(
    "cases_dir,loader",
    [
        (Path("docs/decision_quality_evals/cases"), load_structured_cases),
        (Path("docs/decision_quality_chat_evals/cases"), load_chat_cases),
    ],
)
def test_all_approved_cases_have_valid_metadata(cases_dir: Path, loader):
    cases = loader(statuses={"approved"}, cases_dir=cases_dir)
    assert cases
    for case in cases:
        errors = validate_approved_case_metadata(case.data)
        assert errors == [], f"{case.case_id}: {errors}"


def test_structured_approved_corpus_is_tagged_and_promoted():
    cases = load_structured_cases(statuses={"approved"})
    assert len(cases) == 14
    assert all("structured_dq" in case.data.get("corpus_tags", []) for case in cases)


def test_chat_workflow_cases_remain_unapproved():
    cases = load_chat_cases(statuses={"draft", "review"})
    by_id = {case.case_id: case for case in cases}
    assert by_id["builder_monitor_safe_mode_chat"].status == "draft"
    assert by_id["ontology_temporal_query_2026"].status == "draft"
    assert "workflow_boundary" in by_id["builder_monitor_safe_mode_chat"].data.get("corpus_tags", [])


def test_summarize_case_result_prefers_explicit_fields():
    summary = summarize_case_result(
        {
            "case_id": "unit",
            "status": "approved",
            "corpus_tags": ["structured_dq"],
            "deterministic": {
                "passed": False,
                "score": 50.0,
                "checks": [{"name": "recommended_action", "passed": False}],
            },
            "judge": {"passed": True, "total": 18},
        }
    )

    assert summary["deterministic_passed"] is False
    assert summary["judge_total"] == 18
    assert summary["deterministic_checks"] == [{"name": "recommended_action", "passed": False}]


def test_summarize_calibration_groups_outcome_cases():
    summary = summarize_calibration(
        [
            {
                "case_id": "outcome_a",
                "corpus_tags": ["structured_dq", "outcome_calibration"],
                "calibration_dimensions": {
                    "opportunity_type": "quality_compounder",
                    "confidence_bin": "high",
                    "actionability_stance": "actionable",
                    "data_quality_tier": "adequate",
                    "process_label": "good_process_bad_outcome",
                },
                "deterministic": {"passed": True},
            },
            {
                "case_id": "outcome_b",
                "corpus_tags": ["structured_dq", "outcome_calibration"],
                "calibration_dimensions": {
                    "opportunity_type": "quality_compounder",
                    "confidence_bin": "medium",
                    "actionability_stance": "missing_inputs",
                    "data_quality_tier": "degraded",
                    "process_label": "bad_process_good_outcome",
                },
                "deterministic": {"passed": False},
            },
            {
                "case_id": "plain_structured",
                "corpus_tags": ["structured_dq"],
                "deterministic": {"passed": True},
            },
        ]
    )

    assert summary["outcome_calibration_case_count"] == 2
    assert summary["by_opportunity_type"]["quality_compounder"]["case_count"] == 2
    assert summary["by_confidence_bin"]["high"]["deterministic_passed"] == 1
    assert summary["by_confidence_bin"]["medium"]["deterministic_failed"] == 1
    assert summary["by_process_label"]["good_process_bad_outcome"]["case_count"] == 1


def test_build_report_includes_calibration_summary():
    from decision_quality.eval_runner import build_report

    report = build_report(
        [
            {
                "case_id": "outcome_a",
                "corpus_tags": ["outcome_calibration"],
                "calibration_dimensions": {
                    "opportunity_type": "cyclical_upturn",
                    "confidence_bin": "high",
                    "actionability_stance": "actionable",
                    "data_quality_tier": "adequate",
                    "process_label": "good_process_good_outcome",
                },
                "deterministic": {"passed": True, "score": 100.0},
            }
        ],
        fail_under_deterministic=80.0,
        fail_under_judge=14.0,
    )

    calibration = report["summary"]["calibration_summary"]
    assert calibration["outcome_calibration_case_count"] == 1
    assert calibration["by_opportunity_type"]["cyclical_upturn"]["deterministic_passed"] == 1


def test_committed_structured_baseline_matches_approved_inventory():
    baseline_path = Path("docs/decision_quality_evals/baselines/approved_corpus_baseline.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    approved_ids = {case.case_id for case in load_structured_cases(statuses={"approved"})}

    assert set(baseline["cases"]) == approved_ids
    assert all(entry["deterministic_passed"] for entry in baseline["cases"].values())


def test_committed_chat_baseline_matches_approved_inventory():
    baseline_path = Path("docs/decision_quality_chat_evals/baselines/approved_corpus_baseline.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    approved_ids = {case.case_id for case in load_chat_cases(statuses={"approved"})}

    assert set(baseline["cases"]) == approved_ids
    assert all(entry["deterministic_passed"] for entry in baseline["cases"].values())
