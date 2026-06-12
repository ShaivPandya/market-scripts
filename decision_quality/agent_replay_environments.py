"""Replayable agent environments and process-level reward functions (TL-94).

Provides a versioned, non-production replay harness for deterministic agent
evaluation. The first backend wraps approved chat eval fixtures and mock tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_quality.chat_eval_runner import (
    AgentChatRun,
    ChatEvalCase,
    deterministic_score,
    run_agent_chat_in_process,
    validate_case_input_refs,
)
from decision_quality.chat_eval_runner import (
    load_cases as load_chat_eval_cases,
)
from decision_quality.supervised_labels import assign_split, split_group_for_case

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_CASES_DIR = ROOT / "docs" / "agent_replay_environments" / "cases"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "agent_replay_environments"

ENVIRONMENT_SCHEMA_VERSION = 1
REWARD_SCHEMA_VERSION = 1

VALID_BACKENDS = frozenset({"chat_eval"})
VALID_PROBE_TYPES = frozenset(
    {
        "shortcut",
        "fabricated_source",
        "excessive_tool",
        "policy_boundary",
        "premature_stop",
    }
)

REWARD_COMPONENT_CATEGORIES = (
    "tool_selection",
    "argument_validity",
    "source_quality",
    "structured_output",
    "gate_compliance",
    "missing_input_recognition",
    "efficiency",
    "stopping_defer",
    "general",
)

_CHECK_PREFIX_TO_CATEGORY: tuple[tuple[str, str], ...] = (
    ("expected_tool_coverage", "tool_selection"),
    ("routing_required_tool_names", "tool_selection"),
    ("context_pack_required_tools", "tool_selection"),
    ("workflow_tool_metadata", "argument_validity"),
    ("tool_quality_", "source_quality"),
    ("context_pack_", "source_quality"),
    ("dimension_", "structured_output"),
    ("no_raw_json", "structured_output"),
    ("required_point_", "structured_output"),
    ("gate_", "gate_compliance"),
    ("scout_skeptic_sizer_", "gate_compliance"),
    ("forbid_actionable", "gate_compliance"),
    ("routing_", "gate_compliance"),
    ("missing_input", "missing_input_recognition"),
    ("tool_quality_missing", "missing_input_recognition"),
    ("elapsed_ms", "efficiency"),
    ("stance_", "stopping_defer"),
    ("expected_stance", "stopping_defer"),
    ("forbidden_", "stopping_defer"),
    ("nonempty_answer", "stopping_defer"),
    ("workflow_run_id", "stopping_defer"),
)


@dataclass(frozen=True)
class ReplayEnvironmentCase:
    """Versioned replay environment case."""

    path: Path
    data: dict[str, Any]
    backend: str = "chat_eval"

    @property
    def case_id(self) -> str:
        return str(self.data.get("id") or self.path.stem)

    @property
    def status(self) -> str:
        return str(self.data.get("status") or "draft")

    @property
    def probe_type(self) -> str | None:
        value = self.data.get("probe_type")
        return str(value) if value is not None else None

    @property
    def environment_schema_version(self) -> int:
        try:
            return int(self.data.get("environment_schema_version", ENVIRONMENT_SCHEMA_VERSION))
        except (TypeError, ValueError):
            return ENVIRONMENT_SCHEMA_VERSION

    def as_chat_eval_case(self) -> ChatEvalCase:
        return ChatEvalCase(path=self.path, data=self.data)


@dataclass(frozen=True)
class ReplayObservation:
    """Initial observation returned by ``reset``."""

    case_id: str
    user_message: str
    screen_context: dict[str, Any] | None
    mock_tools: dict[str, Any]
    backend: str
    environment_schema_version: int = ENVIRONMENT_SCHEMA_VERSION
    probe_type: str | None = None


@dataclass(frozen=True)
class RewardComponent:
    """One auditable process-reward component."""

    component_id: str
    category: str
    check_name: str
    passed: bool
    weight: float
    message: str


@dataclass
class ProcessRewardReport:
    """Decomposable process reward report."""

    reward_schema_version: int
    case_id: str
    total_score: float
    passed: bool
    components: list[RewardComponent] = field(default_factory=list)
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class EnvironmentRunResult:
    """Complete replay episode result."""

    case_id: str
    backend: str
    observation: ReplayObservation
    run: AgentChatRun
    deterministic: dict[str, Any]
    rewards: ProcessRewardReport
    trajectory: dict[str, Any]
    terminated: bool
    termination_reason: str
    probe_type: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def categorize_check(check_name: str) -> str:
    """Map a deterministic check name to a reward component category."""
    for prefix, category in _CHECK_PREFIX_TO_CATEGORY:
        if check_name.startswith(prefix) or check_name == prefix.rstrip("_"):
            return category
    return "general"


def validate_environment_case(case: ReplayEnvironmentCase) -> list[str]:
    """Validate environment contract fields."""
    errors: list[str] = []
    if case.environment_schema_version != ENVIRONMENT_SCHEMA_VERSION:
        errors.append(
            f"unsupported environment_schema_version={case.environment_schema_version}; "
            f"expected {ENVIRONMENT_SCHEMA_VERSION}"
        )
    if case.backend not in VALID_BACKENDS:
        errors.append(f"unsupported backend={case.backend!r}; expected one of {sorted(VALID_BACKENDS)}")
    if case.probe_type is not None and case.probe_type not in VALID_PROBE_TYPES:
        errors.append(f"unsupported probe_type={case.probe_type!r}")
    max_tool_calls = case.data.get("max_tool_calls")
    if max_tool_calls is not None:
        try:
            if int(max_tool_calls) < 0:
                errors.append("max_tool_calls must be non-negative")
        except (TypeError, ValueError):
            errors.append("max_tool_calls must be an integer")
    errors.extend(validate_case_input_refs(case.as_chat_eval_case()))
    return errors


def load_environment_cases(
    *,
    case_selectors: list[str] | None = None,
    statuses: set[str] | None = None,
    probe_types: set[str] | None = None,
    include_chat_eval_cases: bool = False,
    cases_dir: Path = DEFAULT_PROBE_CASES_DIR,
    chat_cases_dir: Path | None = None,
) -> list[ReplayEnvironmentCase]:
    """Load replay environment cases from probe fixtures and optional chat eval cases."""
    cases: list[ReplayEnvironmentCase] = []
    if cases_dir.exists():
        for path in sorted(cases_dir.glob("*.json")):
            data = _read_json(path)
            backend = str(data.get("backend") or "chat_eval")
            cases.append(ReplayEnvironmentCase(path=path, data=data, backend=backend))

    if include_chat_eval_cases:
        chat_dir = chat_cases_dir or (ROOT / "docs" / "decision_quality_chat_evals" / "cases")
        for chat_case in load_chat_eval_cases(cases_dir=chat_dir, statuses=statuses or {"approved", "review"}):
            data = dict(chat_case.data)
            data.setdefault("environment_schema_version", ENVIRONMENT_SCHEMA_VERSION)
            data.setdefault("backend", "chat_eval")
            cases.append(
                ReplayEnvironmentCase(path=chat_case.path, data=data, backend="chat_eval"),
            )

    if case_selectors:
        by_id = {case.case_id: case for case in cases}
        by_stem = {case.path.stem: case for case in cases}
        selected: list[ReplayEnvironmentCase] = []
        for selector in case_selectors:
            path = Path(selector)
            if path.exists():
                data = _read_json(path)
                selected.append(
                    ReplayEnvironmentCase(
                        path=path,
                        data=data,
                        backend=str(data.get("backend") or "chat_eval"),
                    )
                )
                continue
            match = by_id.get(selector) or by_stem.get(selector)
            if match is None:
                raise ValueError(f"Unknown replay environment case: {selector}")
            selected.append(match)
        cases = selected

    if statuses is not None:
        cases = [case for case in cases if case.status in statuses]

    if probe_types is not None:
        cases = [case for case in cases if case.probe_type in probe_types]

    return cases


def reset(case: ReplayEnvironmentCase) -> ReplayObservation:
    """Reset the environment and return the initial observation."""
    mock_tools = case.data.get("mock_tools")
    screen_context = case.data.get("screen_context")
    return ReplayObservation(
        case_id=case.case_id,
        user_message=str(case.data.get("user_message") or ""),
        screen_context=screen_context if isinstance(screen_context, dict) else None,
        mock_tools=mock_tools if isinstance(mock_tools, dict) else {},
        backend=case.backend,
        environment_schema_version=case.environment_schema_version,
        probe_type=case.probe_type,
    )


def _termination_reason(run: AgentChatRun, *, passed: bool) -> str:
    if run.error:
        return "error"
    if run.status_code != 200:
        return "http_failure"
    if not (run.final_text or "").strip():
        return "empty_response"
    if passed:
        return "completed"
    return "reward_failure"


def _apply_environment_checks(
    case: ReplayEnvironmentCase,
    run: AgentChatRun,
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append environment-specific checks such as max_tool_calls."""
    extended = list(checks)
    max_tool_calls = case.data.get("max_tool_calls")
    if max_tool_calls is not None:
        try:
            limit = int(max_tool_calls)
            actual = len(run.tool_names)
            extended.append(
                {
                    "name": "max_tool_calls",
                    "passed": actual <= limit,
                    "message": f"tool_calls={actual}, max={limit}",
                }
            )
        except (TypeError, ValueError):
            extended.append(
                {
                    "name": "max_tool_calls",
                    "passed": False,
                    "message": f"invalid max_tool_calls={max_tool_calls!r}",
                }
            )
    return extended


def score_process_rewards(
    case: ReplayEnvironmentCase,
    run: AgentChatRun,
    *,
    deterministic: dict[str, Any] | None = None,
) -> ProcessRewardReport:
    """Convert deterministic checks into decomposable process reward components."""
    det = deterministic if deterministic is not None else deterministic_score(case.as_chat_eval_case(), run)
    checks = _apply_environment_checks(case, run, list(det.get("checks") or []))
    components: list[RewardComponent] = []
    by_category: dict[str, dict[str, Any]] = {
        category: {"passed": 0, "failed": 0, "total": 0, "score": 0.0} for category in REWARD_COMPONENT_CATEGORIES
    }

    for idx, check in enumerate(checks):
        if not isinstance(check, dict):
            continue
        check_name = str(check.get("name") or f"check_{idx + 1}")
        passed = bool(check.get("passed"))
        category = categorize_check(check_name)
        if check_name == "max_tool_calls":
            category = "efficiency"
        component = RewardComponent(
            component_id=f"{case.case_id}:{check_name}",
            category=category,
            check_name=check_name,
            passed=passed,
            weight=1.0,
            message=str(check.get("message") or ""),
        )
        components.append(component)
        bucket = by_category.setdefault(category, {"passed": 0, "failed": 0, "total": 0, "score": 0.0})
        bucket["total"] += 1
        if passed:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    for bucket in by_category.values():
        total = bucket["total"]
        bucket["score"] = round((bucket["passed"] / total) * 100, 2) if total else 0.0

    passed_count = sum(1 for component in components if component.passed)
    total_score = round((passed_count / len(components)) * 100, 2) if components else 0.0
    all_passed = bool(components) and all(component.passed for component in components)

    return ProcessRewardReport(
        reward_schema_version=REWARD_SCHEMA_VERSION,
        case_id=case.case_id,
        total_score=total_score,
        passed=all_passed,
        components=components,
        by_category={key: value for key, value in by_category.items() if value["total"] > 0},
    )


def build_trajectory_steps(run: AgentChatRun, *, case_id: str) -> list[dict[str, Any]]:
    """Build ordered trajectory steps compatible with the TL-87 step vocabulary."""
    steps: list[dict[str, Any]] = []
    index = 0

    router_meta = run.done_payload.get("intent_router") if isinstance(run.done_payload, dict) else None
    applied = router_meta.get("applied") if isinstance(router_meta, dict) else None
    if isinstance(applied, dict):
        steps.append(
            {
                "step_id": f"{case_id}:route:0",
                "index": index,
                "kind": "route",
                "name": str(applied.get("intent_class") or "route"),
                "status": "ok",
                "payload": applied,
            }
        )
        index += 1

    for event_idx, (event_name, payload) in enumerate(run.events):
        if event_name not in {"tool_call", "tool_result"}:
            continue
        tool_name = payload.get("name") if isinstance(payload, dict) else None
        steps.append(
            {
                "step_id": f"{case_id}:tool:{event_idx}",
                "index": index,
                "kind": "tool_call" if event_name == "tool_call" else "tool_result",
                "name": str(tool_name) if tool_name is not None else None,
                "status": "ok",
                "payload": payload if isinstance(payload, dict) else {},
            }
        )
        index += 1

    dq_meta = run.done_payload.get("decision_quality_chat") if isinstance(run.done_payload, dict) else None
    if isinstance(dq_meta, dict):
        steps.append(
            {
                "step_id": f"{case_id}:gate:0",
                "index": index,
                "kind": "gate",
                "name": "decision_quality_chat",
                "status": str(dq_meta.get("gate_status") or "unknown"),
                "payload": dq_meta,
            }
        )
        index += 1

    steps.append(
        {
            "step_id": f"{case_id}:final:0",
            "index": index,
            "kind": "final",
            "name": "assistant_response",
            "status": "ok" if (run.final_text or "").strip() else "empty",
            "elapsed_ms": run.elapsed_ms,
            "payload": {
                "text_length": len(run.final_text or ""),
                "tool_names": list(run.tool_names),
            },
        }
    )
    return steps


def export_environment_trajectory(
    case: ReplayEnvironmentCase,
    run: AgentChatRun,
    *,
    rewards: ProcessRewardReport | None = None,
    trajectory_id: str | None = None,
) -> dict[str, Any]:
    """Export a replay trajectory consumable by dataset and eval pipelines."""
    captured_at = datetime.now(UTC).isoformat()
    split_group = split_group_for_case(case_id=case.case_id, case_data=case.data)
    message_hash = hashlib.sha256((run.final_text or case.case_id).encode("utf-8")).hexdigest()
    resolved_id = trajectory_id or str(uuid.uuid5(uuid.NAMESPACE_URL, f"replay:{case.case_id}:{message_hash}"))

    reward_report = rewards or score_process_rewards(case, run)
    final_disposition = "succeeded" if reward_report.passed and run.status_code == 200 else "failed"
    if run.error:
        final_disposition = "failed"
    elif run.status_code != 200:
        final_disposition = "failed"

    return {
        "trajectory_id": resolved_id,
        "schema_version": 1,
        "environment_schema_version": case.environment_schema_version,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "case_id": case.case_id,
        "backend": case.backend,
        "probe_type": case.probe_type,
        "captured_at": captured_at,
        "completed_at": captured_at,
        "final_disposition": final_disposition,
        "provider": "replay_environment",
        "model": "offline_replay",
        "dataset_split_group": split_group,
        "dataset_split": assign_split(split_group),
        "training_eligible": False,
        "exclusion_reasons": ["replay_environment_not_production_capture"],
        "steps": build_trajectory_steps(run, case_id=case.case_id),
        "reward_report": {
            "total_score": reward_report.total_score,
            "passed": reward_report.passed,
            "by_category": reward_report.by_category,
            "components": [
                {
                    "component_id": component.component_id,
                    "category": component.category,
                    "check_name": component.check_name,
                    "passed": component.passed,
                    "weight": component.weight,
                    "message": component.message,
                }
                for component in reward_report.components
            ],
        },
        "source_provenance": {
            "source": "agent_replay_environment",
            "case_path": str(case.path.relative_to(ROOT) if case.path.is_relative_to(ROOT) else case.path),
            "probe_type": case.probe_type,
        },
    }


def run_episode(
    case: ReplayEnvironmentCase,
    *,
    agent_runner: Callable[[ChatEvalCase], AgentChatRun] | None = None,
    dry_run: bool = False,
    run: AgentChatRun | None = None,
) -> EnvironmentRunResult:
    """Run one replay episode: reset, execute, score, and export trajectory."""
    validation_errors = validate_environment_case(case)
    observation = reset(case)

    if dry_run:
        empty_run = AgentChatRun(final_text="", events=[], tool_names=[], done_payload={})
        rewards = ProcessRewardReport(
            reward_schema_version=REWARD_SCHEMA_VERSION,
            case_id=case.case_id,
            total_score=0.0,
            passed=False,
            components=[
                RewardComponent(
                    component_id=f"{case.case_id}:dry_run",
                    category="general",
                    check_name="dry_run",
                    passed=True,
                    weight=0.0,
                    message="dry run only",
                )
            ],
        )
        trajectory = export_environment_trajectory(case, empty_run, rewards=rewards)
        trajectory["dry_run"] = True
        return EnvironmentRunResult(
            case_id=case.case_id,
            backend=case.backend,
            observation=observation,
            run=empty_run,
            deterministic={"score": 0.0, "passed": False, "checks": []},
            rewards=rewards,
            trajectory=trajectory,
            terminated=False,
            termination_reason="dry_run",
            probe_type=case.probe_type,
        )

    if validation_errors:
        failed_run = AgentChatRun(
            final_text="",
            events=[],
            tool_names=[],
            done_payload={},
            status_code=400,
            error="; ".join(validation_errors),
        )
        deterministic = {
            "score": 0.0,
            "passed": False,
            "checks": [{"name": "environment_validation", "passed": False, "message": "; ".join(validation_errors)}],
        }
        rewards = score_process_rewards(case, failed_run, deterministic=deterministic)
        return EnvironmentRunResult(
            case_id=case.case_id,
            backend=case.backend,
            observation=observation,
            run=failed_run,
            deterministic=deterministic,
            rewards=rewards,
            trajectory=export_environment_trajectory(case, failed_run, rewards=rewards),
            terminated=True,
            termination_reason="validation_failure",
            probe_type=case.probe_type,
        )

    chat_case = case.as_chat_eval_case()
    if run is None:
        runner = agent_runner or run_agent_chat_in_process
        run = runner(chat_case)

    deterministic = deterministic_score(chat_case, run)
    deterministic_checks = _apply_environment_checks(case, run, list(deterministic.get("checks") or []))
    deterministic = {
        **deterministic,
        "checks": deterministic_checks,
        "passed": all(bool(check.get("passed")) for check in deterministic_checks),
        "score": round(
            (sum(1 for check in deterministic_checks if check.get("passed")) / len(deterministic_checks)) * 100,
            2,
        )
        if deterministic_checks
        else 0.0,
    }
    rewards = score_process_rewards(case, run, deterministic=deterministic)
    trajectory = export_environment_trajectory(case, run, rewards=rewards)
    termination_reason = _termination_reason(run, passed=rewards.passed)

    return EnvironmentRunResult(
        case_id=case.case_id,
        backend=case.backend,
        observation=observation,
        run=run,
        deterministic=deterministic,
        rewards=rewards,
        trajectory=trajectory,
        terminated=True,
        termination_reason=termination_reason,
        probe_type=case.probe_type,
    )


def run_parallel_smoke(
    cases: list[ReplayEnvironmentCase],
    *,
    agent_runner: Callable[[ChatEvalCase], AgentChatRun] | None = None,
    max_workers: int = 4,
    dry_run: bool = False,
) -> list[EnvironmentRunResult]:
    """Run multiple replay cases concurrently for isolation/reproducibility checks."""
    if dry_run or max_workers <= 1 or len(cases) <= 1:
        return [run_episode(case, agent_runner=agent_runner, dry_run=dry_run) for case in cases]

    results: list[EnvironmentRunResult | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_episode, case, agent_runner=agent_runner, dry_run=dry_run): idx
            for idx, case in enumerate(cases)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
    return [result for result in results if result is not None]


def build_report(
    results: list[EnvironmentRunResult],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a summary report for one replay batch."""
    timestamp = generated_at or datetime.now(UTC).isoformat()
    return {
        "generated_at": timestamp,
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "case_count": len(results),
        "passed_count": sum(1 for result in results if result.rewards.passed),
        "results": [
            {
                "case_id": result.case_id,
                "probe_type": result.probe_type,
                "termination_reason": result.termination_reason,
                "reward_score": result.rewards.total_score,
                "reward_passed": result.rewards.passed,
                "deterministic_passed": result.deterministic.get("passed"),
                "trajectory_id": result.trajectory.get("trajectory_id"),
            }
            for result in results
        ],
    }


def _configure_offline_env() -> None:
    os.environ.setdefault("AGENT_GOVERNANCE_AUDIT_ENABLED", "false")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run replayable agent environments (TL-94).")
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_PROBE_CASES_DIR)
    parser.add_argument("--case", action="append", dest="case_selectors", default=[])
    parser.add_argument("--probe-type", action="append", dest="probe_types", default=[])
    parser.add_argument("--include-chat-eval-cases", action="store_true")
    parser.add_argument("--approved-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    _configure_offline_env()
    statuses = {"approved"} if args.approved_only else {"approved", "review"}
    probe_types = set(args.probe_types) if args.probe_types else None
    cases = load_environment_cases(
        case_selectors=args.case_selectors or None,
        statuses=statuses,
        probe_types=probe_types,
        include_chat_eval_cases=args.include_chat_eval_cases,
        cases_dir=args.cases_dir,
    )
    if not cases:
        raise SystemExit("No replay environment cases matched the selection.")

    if args.parallel:
        results = run_parallel_smoke(cases, max_workers=args.max_workers, dry_run=args.dry_run)
    else:
        results = [run_episode(case, dry_run=args.dry_run) for case in cases]

    report = build_report(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"replay_report_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
