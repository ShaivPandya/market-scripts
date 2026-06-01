from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_quality.chat_eval_runner import load_cases as load_chat_cases
from decision_quality.eval_corpus import (
    compare_reports,
    filter_cases,
    summarize_case_result,
    validate_approved_case_metadata,
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
