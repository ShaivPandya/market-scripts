"""Shared helpers for decision-quality eval corpus metadata, filtering, and baselines."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

CORPUS_TAGS = frozenset(
    {
        "structured_dq",
        "chat_behavior",
        "routing_tool_use",
        "opportunity_identification",
        "workflow_boundary",
    }
)

FAILURE_TYPES = frozenset(
    {
        "process_regression",
        "missing_invalidation",
        "missing_mispricing",
        "missing_catalyst",
        "generic_answer",
        "wrong_routing",
        "wrong_tools",
        "workflow_boundary_violation",
        "sizing_discipline",
        "source_freshness",
        "price_confirmation",
        "other",
    }
)

STANDARD_CHAT_DQ_DIMENSIONS = (
    "simple_thesis",
    "mispricing",
    "catalyst_or_reason_now",
    "evidence_for",
    "evidence_against",
    "price_action",
    "invalidation",
    "missing_inputs",
    "confidence_sizing",
    "trade_after_trade",
)

STRUCTURED_DQ_DIMENSIONS = (
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
)


class _CaseLike(Protocol):
    @property
    def case_id(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def data(self) -> dict[str, Any]: ...


def parse_csv_values(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def case_corpus_tags(case_data: dict[str, Any]) -> list[str]:
    raw = case_data.get("corpus_tags")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str) and item]


def case_failure_type(case_data: dict[str, Any]) -> str | None:
    value = case_data.get("failure_type")
    if value is None:
        return None
    return str(value)


def case_tool_pack(case_data: dict[str, Any]) -> str | None:
    value = case_data.get("tool_pack")
    if value is not None:
        return str(value) if value else None
    routing = case_data.get("routing_expectations")
    if isinstance(routing, dict):
        intent = routing.get("intent_class")
        if isinstance(intent, str) and intent:
            return intent
    return None


def case_required_dq_dimensions(case_data: dict[str, Any]) -> list[str]:
    raw = case_data.get("required_dq_dimensions")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw if isinstance(item, str) and item]
    chat_dims = case_data.get("required_decision_quality_dimensions")
    if isinstance(chat_dims, list) and chat_dims:
        return [str(item) for item in chat_dims if isinstance(item, str) and item]
    if "gold_output" in case_data:
        return list(STRUCTURED_DQ_DIMENSIONS)
    return []


def filter_cases(
    cases: list[_CaseLike],
    *,
    corpus_tags: set[str] | None = None,
    failure_type: str | None = None,
    tool_pack: str | None = None,
) -> list[_CaseLike]:
    filtered = cases
    if corpus_tags:
        filtered = [case for case in filtered if corpus_tags.intersection(case_corpus_tags(case.data))]
    if failure_type:
        filtered = [case for case in filtered if case_failure_type(case.data) == failure_type]
    if tool_pack:
        filtered = [case for case in filtered if case_tool_pack(case.data) == tool_pack]
    return filtered


def case_result_metadata(case_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "corpus_tags": case_corpus_tags(case_data),
        "failure_type": case_failure_type(case_data),
        "tool_pack": case_tool_pack(case_data),
        "required_dq_dimensions": case_required_dq_dimensions(case_data),
    }


def validate_approved_case_metadata(case_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    status = str(case_data.get("status") or "draft")
    if status != "approved":
        return errors

    tags = case_corpus_tags(case_data)
    if not tags:
        errors.append("approved cases must include at least one corpus_tags entry")
    unknown_tags = sorted(set(tags) - CORPUS_TAGS)
    if unknown_tags:
        errors.append(f"unknown corpus_tags: {', '.join(unknown_tags)}")

    failure = case_failure_type(case_data)
    if failure is not None and failure not in FAILURE_TYPES:
        errors.append(f"unknown failure_type: {failure}")

    dims = case_required_dq_dimensions(case_data)
    routing_only = tags == ["routing_tool_use"] or (
        "routing_tool_use" in tags and "chat_behavior" not in tags and "opportunity_identification" not in tags
    )
    if not dims and not routing_only:
        errors.append("approved cases must declare required_dq_dimensions or chat dimension expectations")

    refs = case_data.get("input_refs")
    if isinstance(refs, list):
        for idx, ref in enumerate(refs):
            if not isinstance(ref, dict):
                continue
            path_value = ref.get("path")
            sha = ref.get("sha256")
            if path_value and sha is None:
                errors.append(f"input_refs[{idx}] must include sha256 for approved cases with local paths")

    if "workflow_expectations" in case_data and "workflow_boundary" in tags:
        workflow = case_data.get("workflow_expectations")
        if not isinstance(workflow, dict):
            errors.append("workflow_boundary cases must include workflow_expectations")

    return errors


def summarize_case_result(result: dict[str, Any]) -> dict[str, Any]:
    deterministic = result.get("deterministic") or {}
    judge = result.get("judge") or {}
    checks = deterministic.get("checks") or []
    return {
        "case_id": result.get("case_id"),
        "status": result.get("status"),
        "corpus_tags": result.get("corpus_tags") or [],
        "failure_type": result.get("failure_type"),
        "tool_pack": result.get("tool_pack"),
        "deterministic_passed": bool(deterministic.get("passed")),
        "deterministic_score": deterministic.get("score"),
        "deterministic_checks": [
            {"name": check.get("name"), "passed": bool(check.get("passed"))}
            for check in checks
            if isinstance(check, dict)
        ],
        "judge_total": judge.get("total"),
        "judge_passed": judge.get("passed") if judge else None,
    }


def build_baseline_report(
    results: list[dict[str, Any]],
    *,
    corpus_tags: set[str] | None = None,
    status_filter: set[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    summaries = [summarize_case_result(result) for result in results]
    return {
        "baseline_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_tags_filter": sorted(corpus_tags) if corpus_tags else [],
        "status_filter": sorted(status_filter) if status_filter else [],
        "notes": notes,
        "case_count": len(summaries),
        "cases": {summary["case_id"]: summary for summary in summaries if summary.get("case_id")},
    }


def load_baseline(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_baseline(path: Path, baseline: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _case_map_from_report(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = report.get("cases")
    if isinstance(cases, dict):
        return {str(case_id): case for case_id, case in cases.items() if isinstance(case, dict)}
    if isinstance(cases, list):
        mapped: dict[str, dict[str, Any]] = {}
        for item in cases:
            if isinstance(item, dict) and item.get("case_id"):
                mapped[str(item["case_id"])] = item
        return mapped
    return {}


def _deterministic_passed(case_entry: dict[str, Any]) -> bool | None:
    if "deterministic_passed" in case_entry:
        return bool(case_entry["deterministic_passed"])
    deterministic = case_entry.get("deterministic")
    if isinstance(deterministic, dict) and "passed" in deterministic:
        return bool(deterministic["passed"])
    return None


def _failed_check_names(case_entry: dict[str, Any]) -> set[str]:
    checks = case_entry.get("deterministic_checks")
    if checks is None and isinstance(case_entry.get("deterministic"), dict):
        checks = case_entry["deterministic"].get("checks")
    if not isinstance(checks, list):
        return set()
    return {
        str(check["name"])
        for check in checks
        if isinstance(check, dict) and check.get("name") and not check.get("passed")
    }


def compare_reports(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    baseline_cases = _case_map_from_report(baseline)
    current_cases = _case_map_from_report(current)

    baseline_ids = set(baseline_cases)
    current_ids = set(current_cases)

    new_failures: list[str] = []
    fixed_failures: list[str] = []
    check_regressions: list[dict[str, Any]] = []
    judge_deltas: list[dict[str, Any]] = []

    for case_id in sorted(baseline_ids & current_ids):
        before = baseline_cases[case_id]
        after = current_cases[case_id]
        before_pass = _deterministic_passed(before)
        after_pass = _deterministic_passed(after)
        if before_pass is True and after_pass is False:
            new_failures.append(case_id)
        elif before_pass is False and after_pass is True:
            fixed_failures.append(case_id)

        before_failed = _failed_check_names(before)
        after_failed = _failed_check_names(after)
        newly_failed_checks = sorted(after_failed - before_failed)
        newly_passed_checks = sorted(before_failed - after_failed)
        if newly_failed_checks or newly_passed_checks:
            check_regressions.append(
                {
                    "case_id": case_id,
                    "newly_failed_checks": newly_failed_checks,
                    "newly_passed_checks": newly_passed_checks,
                }
            )

        before_judge = before.get("judge_total")
        if before_judge is None and isinstance(before.get("judge"), dict):
            before_judge = before["judge"].get("total")
        after_judge = after.get("judge_total")
        if after_judge is None and isinstance(after.get("judge"), dict):
            after_judge = after["judge"].get("total")
        if before_judge is not None and after_judge is not None and before_judge != after_judge:
            judge_deltas.append(
                {
                    "case_id": case_id,
                    "baseline_total": before_judge,
                    "current_total": after_judge,
                    "delta": after_judge - before_judge,
                }
            )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_generated_at": baseline.get("generated_at"),
        "current_generated_at": current.get("generated_at"),
        "summary": {
            "baseline_case_count": len(baseline_ids),
            "current_case_count": len(current_ids),
            "missing_from_current": sorted(baseline_ids - current_ids),
            "new_in_current": sorted(current_ids - baseline_ids),
            "new_deterministic_failures": new_failures,
            "fixed_deterministic_failures": fixed_failures,
            "check_regressions": check_regressions,
            "judge_deltas": judge_deltas,
            "regression_detected": bool(new_failures),
        },
    }
