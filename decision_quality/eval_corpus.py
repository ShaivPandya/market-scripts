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

# Richer active-learning tags grouped by failure area. Each tag maps to a canonical failure_type.
FAILURE_TAG_CATEGORIES: dict[str, frozenset[str]] = {
    "routing": frozenset({"wrong_routing", "wrong_tools"}),
    "hidden_dq": frozenset(
        {
            "missed_hidden_dq",
            "missing_invalidation",
            "missing_mispricing",
            "missing_catalyst",
            "generic_answer",
        }
    ),
    "source_quality": frozenset({"source_freshness", "price_confirmation", "stale_data"}),
    "opportunity_identification": frozenset({"missing_mispricing", "weak_opportunity_id"}),
    "synthesis_quality": frozenset({"generic_answer", "bad_synthesis", "process_regression"}),
    "policy_action_gating": frozenset(
        {
            "sizing_discipline",
            "workflow_boundary_violation",
            "overconfident_actionability",
        }
    ),
}

FAILURE_TAG_ALIASES: dict[str, str] = {
    "generic": "generic_answer",
    "stale_data": "source_freshness",
    "missed_hidden_dq": "missed_hidden_dq",
    "overconfident_actionability": "overconfident_actionability",
    "weak_opportunity_id": "weak_opportunity_id",
    "bad_synthesis": "bad_synthesis",
}

FAILURE_TAG_TO_TYPE: dict[str, str] = {
    "missed_hidden_dq": "generic_answer",
    "stale_data": "source_freshness",
    "overconfident_actionability": "sizing_discipline",
    "weak_opportunity_id": "missing_mispricing",
    "bad_synthesis": "generic_answer",
}

FAILURE_TAGS = (
    frozenset(tag for tags in FAILURE_TAG_CATEGORIES.values() for tag in tags)
    | FAILURE_TYPES
    | frozenset(FAILURE_TAG_ALIASES)
)

TRAINING_EXPORT_STATUSES = frozenset({"review", "approved"})

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


def case_failure_tags(case_data: dict[str, Any]) -> list[str]:
    raw = case_data.get("failure_tags")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str) and item]


def failure_tag_category(tag: str) -> str | None:
    canonical = FAILURE_TAG_ALIASES.get(tag, tag)
    for category, tags in FAILURE_TAG_CATEGORIES.items():
        if canonical in tags:
            return category
    if canonical in FAILURE_TYPES:
        return "routing" if canonical in {"wrong_routing", "wrong_tools"} else None
    return None


def failure_type_for_tag(tag: str) -> str:
    canonical = FAILURE_TAG_ALIASES.get(tag, tag)
    if canonical in FAILURE_TYPES:
        return canonical
    return FAILURE_TAG_TO_TYPE.get(canonical, "other")


def normalize_failure_tags(tags: list[str]) -> tuple[list[str], str, list[str]]:
    """Normalize CLI/user tags to canonical failure_tags and a primary failure_type."""
    errors: list[str] = []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = raw.strip()
        if not tag:
            continue
        canonical = FAILURE_TAG_ALIASES.get(tag, tag)
        if canonical not in FAILURE_TAGS and tag not in FAILURE_TAGS:
            errors.append(f"unknown failure tag: {tag}")
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    failure_type = failure_type_for_tag(normalized[0]) if normalized else "other"
    return normalized, failure_type, errors


def infer_corpus_tags_from_failure_tags(tags: list[str]) -> list[str]:
    categories = {failure_tag_category(tag) for tag in tags if failure_tag_category(tag)}
    inferred: list[str] = []
    if "routing" in categories:
        inferred.append("routing_tool_use")
    if categories.intersection({"hidden_dq", "synthesis_quality", "policy_action_gating", "source_quality"}):
        if "chat_behavior" not in inferred:
            inferred.append("chat_behavior")
    if "opportunity_identification" in categories:
        inferred.append("opportunity_identification")
    return inferred or ["chat_behavior"]


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


def _validate_failure_tags(case_data: dict[str, Any], errors: list[str]) -> None:
    failure_tags = case_failure_tags(case_data)
    if not failure_tags:
        return
    unknown = sorted({tag for tag in failure_tags if tag not in FAILURE_TAGS})
    if unknown:
        errors.append(f"unknown failure_tags: {', '.join(unknown)}")
    failure = case_failure_type(case_data)
    if failure is not None and failure not in FAILURE_TYPES:
        errors.append(f"unknown failure_type: {failure}")
    if failure_tags and failure and failure != failure_type_for_tag(failure_tags[0]):
        errors.append("failure_type must match the primary failure_tags entry")


def validate_review_case_metadata(case_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    status = str(case_data.get("status") or "draft")
    if status != "review":
        return errors

    if not str(case_data.get("user_message") or "").strip():
        errors.append("review cases must include user_message")

    failure_tags = case_failure_tags(case_data)
    failure = case_failure_type(case_data)
    if not failure_tags and failure is None:
        errors.append("review cases must include failure_tags or failure_type")
    _validate_failure_tags(case_data, errors)

    tags = case_corpus_tags(case_data)
    if not tags:
        errors.append("review cases must include at least one corpus_tags entry")
    unknown_tags = sorted(set(tags) - CORPUS_TAGS)
    if unknown_tags:
        errors.append(f"unknown corpus_tags: {', '.join(unknown_tags)}")

    categories = {failure_tag_category(tag) for tag in failure_tags if failure_tag_category(tag)}
    routing = case_data.get("routing_expectations")
    needs_routing = "routing" in categories or "routing_tool_use" in tags
    if needs_routing:
        if not isinstance(routing, dict) or not routing.get("intent_class"):
            errors.append("routing failures in review must include routing_expectations.intent_class")

    return errors


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

    _validate_failure_tags(case_data, errors)

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
