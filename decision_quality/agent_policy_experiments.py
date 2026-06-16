"""Offline contextual-bandit and ranking experiments for agent-process choices.

This module is intentionally offline-only. It evaluates logged agent choices and
shadow candidates without changing production routing, rollout, or tool policy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from decision_quality.eval_corpus import compare_reports

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "agent_policy_experiments"

EXPERIMENT_SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
REPORT_VERSION = 1
RUNNER_VERSION = "agent_policy_experiments_v1"

ProblemType = Literal["contextual_bandit", "ranking"]
PolicyName = Literal["logged", "regex_baseline", "llm_candidate", "supervised_candidate", "highest_confidence"]
RewardSource = Literal["process_reward", "eval_score", "human_review", "outcome_label", "synthetic"]

PROCESS_REWARD_CATEGORIES = frozenset(
    {
        "tool_selection",
        "argument_validity",
        "source_quality",
        "structured_output",
        "gate_compliance",
        "missing_input_recognition",
        "efficiency",
        "stopping_defer",
        "routing",
        "general",
    }
)
DISALLOWED_REWARD_KEYS = frozenset(
    {
        "pnl",
        "realized_pnl",
        "realized_pnl_usd",
        "forward_return_pct",
        "benchmark_return_pct",
        "benchmark_relative_return_pct",
        "end_price",
        "future_price",
        "future_return",
    }
)
DISALLOWED_REWARD_CATEGORIES = frozenset({"pnl", "realized_pnl", "forward_return", "trading_return"})


class AgentPolicyExperimentError(ValueError):
    """Raised when an offline policy experiment cannot be constructed safely."""


class ActionCandidate(BaseModel):
    """One action available to a logged agent-process policy."""

    action_id: str
    source: str
    intent_class: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    gate_overrides: list[str] = Field(default_factory=list)

    @field_validator("tool_names")
    @classmethod
    def _dedupe_tools(cls, value: list[str]) -> list[str]:
        return sorted({str(item) for item in value if str(item or "").strip()})


class PropensityMetadata(BaseModel):
    """Logging-policy probabilities required for counterfactual comparisons."""

    logging_policy: str
    logged_action_id: str
    action_probability: float | None = Field(default=None, gt=0.0, le=1.0)
    candidate_probabilities: dict[str, float] = Field(default_factory=dict)

    @field_validator("candidate_probabilities")
    @classmethod
    def _valid_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for action_id, probability in value.items():
            prob = float(probability)
            if prob <= 0.0 or prob > 1.0:
                raise ValueError(f"Invalid propensity for {action_id}: {probability}")
            normalized[str(action_id)] = prob
        return normalized

    def probability_for(self, action_id: str) -> float | None:
        if action_id in self.candidate_probabilities:
            return float(self.candidate_probabilities[action_id])
        if action_id == self.logged_action_id:
            return self.action_probability
        return None


class RewardComponent(BaseModel):
    """Auditable reward component used for offline policy reports."""

    component_id: str
    category: str
    source: RewardSource
    value: float = Field(ge=-1.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_leaky_or_pnl_rewards(self) -> RewardComponent:
        category = self.category.strip().lower()
        if category in DISALLOWED_REWARD_CATEGORIES:
            raise ValueError(f"Reward category {self.category!r} is not allowed for TL-68")
        metadata_keys = {str(key).strip().lower() for key in self.metadata}
        leaked = sorted(metadata_keys & DISALLOWED_REWARD_KEYS)
        if leaked:
            raise ValueError(f"Reward component contains future/leaky fields: {', '.join(leaked)}")
        if self.source == "outcome_label" and category not in PROCESS_REWARD_CATEGORIES:
            raise ValueError("Outcome labels must map to a process-reward category, not direct returns")
        return self


class LoggedDecisionExample(BaseModel):
    """One logged context/action/reward row for offline policy evaluation."""

    schema_version: int = EXPERIMENT_SCHEMA_VERSION
    example_id: str
    source_type: str = "intent_router"
    context: dict[str, Any] = Field(default_factory=dict)
    action_candidates: list[ActionCandidate]
    logged_action_id: str
    propensity: PropensityMetadata | None = None
    reward_components: list[RewardComponent] = Field(default_factory=list)
    split_group: str
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: int) -> int:
        if value != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported experiment schema version: {value}")
        return value

    @model_validator(mode="after")
    def _valid_logged_action_and_reward(self) -> LoggedDecisionExample:
        action_ids = {candidate.action_id for candidate in self.action_candidates}
        if self.logged_action_id not in action_ids:
            raise ValueError("logged_action_id must reference an action candidate")
        if self.propensity and self.propensity.logged_action_id != self.logged_action_id:
            raise ValueError("propensity.logged_action_id must match logged_action_id")
        if not self.reward_components:
            raise ValueError("At least one reward component is required")
        return self


class ExperimentManifest(BaseModel):
    """Reproducible offline experiment configuration."""

    manifest_version: int = MANIFEST_VERSION
    experiment_id: str
    problem_type: ProblemType = "contextual_bandit"
    baseline_policy: PolicyName = "logged"
    candidate_policy: PolicyName = "highest_confidence"
    reward_components: list[str] = Field(default_factory=list)
    input_sources: list[str] = Field(default_factory=list)
    require_propensity: bool = True
    notes: str = ""

    @field_validator("manifest_version")
    @classmethod
    def _supported_manifest(cls, value: int) -> int:
        if value != MANIFEST_VERSION:
            raise ValueError(f"Unsupported manifest version: {value}")
        return value


class PolicyEvaluationRow(BaseModel):
    """Per-example evaluation result for one baseline/candidate pair."""

    example_id: str
    logged_action_id: str
    baseline_action_id: str | None
    candidate_action_id: str | None
    logged_reward: float
    candidate_propensity: float | None = None
    ips_reward: float | None = None
    deterministic_passed: bool
    deterministic_checks: list[dict[str, Any]]
    exclusion_reason: str | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AgentPolicyExperimentError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _stable_action_id(route: dict[str, Any], *, source: str) -> str:
    intent = str(route.get("intent_class") or "unknown")
    hidden_dq = "dq" if route.get("run_hidden_dq") else "no_dq"
    preflight = "preflight" if route.get("run_opportunity_preflight") else "no_preflight"
    workflow = str(route.get("workflow_name") or "no_workflow")
    tools = ",".join(sorted(str(item) for item in route.get("tool_names") or [] if str(item or "").strip()))
    return f"{source}:{intent}:{hidden_dq}:{preflight}:{workflow}:{tools}"


def action_candidate_from_route(route: dict[str, Any] | None, *, source: str | None = None) -> ActionCandidate | None:
    if not isinstance(route, dict) or not route:
        return None
    resolved_source = str(source or route.get("source") or "unknown").strip() or "unknown"
    metadata = dict(route)
    gate_overrides: list[str] = []
    for key in ("gate_override", "overrides_gate", "policy_override", "approval_override"):
        if metadata.get(key):
            gate_overrides.append(key)
    return ActionCandidate(
        action_id=str(route.get("action_id") or _stable_action_id(route, source=resolved_source)),
        source=resolved_source,
        intent_class=str(route.get("intent_class")) if route.get("intent_class") else None,
        tool_names=list(route.get("tool_names") or []),
        confidence=float(route["confidence"]) if route.get("confidence") is not None else None,
        metadata=metadata,
        gate_overrides=gate_overrides,
    )


def _dedupe_candidates(candidates: list[ActionCandidate | None]) -> list[ActionCandidate]:
    deduped: dict[str, ActionCandidate] = {}
    for candidate in candidates:
        if candidate is None:
            continue
        deduped.setdefault(candidate.action_id, candidate)
    return list(deduped.values())


def _reward_components_from_router_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = row.get("reward_components")
    if isinstance(explicit, list) and explicit:
        return explicit

    label_intent = row.get("label_intent_class")
    if label_intent:
        applied = row.get("applied_route") if isinstance(row.get("applied_route"), dict) else {}
        value = 1.0 if applied.get("intent_class") == label_intent else 0.0
        return [
            {
                "component_id": "human_router_label",
                "category": "routing",
                "source": "human_review",
                "value": value,
                "weight": 1.0,
                "metadata": {"label_intent_class": label_intent},
            }
        ]

    comparison = row.get("shadow_comparison") if isinstance(row.get("shadow_comparison"), dict) else {}
    if comparison:
        matched = bool(
            comparison.get("intent_match", False)
            and comparison.get("hidden_dq_match", True)
            and comparison.get("opportunity_preflight_match", True)
            and comparison.get("workflow_match", True)
            and not comparison.get("tool_only_in_candidate")
            and not comparison.get("tool_only_in_applied")
        )
        return [
            {
                "component_id": "router_shadow_agreement",
                "category": "routing",
                "source": "process_reward",
                "value": 1.0 if matched else 0.0,
                "weight": 1.0,
                "metadata": {"shadow_comparison": comparison},
            }
        ]

    return [
        {
            "component_id": "router_logged_event",
            "category": "routing",
            "source": "process_reward",
            "value": 0.5,
            "weight": 1.0,
            "metadata": {"default_reward": True},
        }
    ]


def logged_example_from_intent_router_row(row: dict[str, Any]) -> LoggedDecisionExample:
    """Convert a durable intent-router row into a TL-68 offline policy example."""

    regex = action_candidate_from_route(row.get("regex_baseline"), source="regex")
    llm = action_candidate_from_route(row.get("llm_candidate"), source="llm")
    supervised = action_candidate_from_route(row.get("supervised_candidate"), source="supervised")
    applied = action_candidate_from_route(row.get("applied_route"))

    candidates = _dedupe_candidates([regex, llm, supervised, applied])
    if not candidates:
        raise AgentPolicyExperimentError("Intent-router row does not contain any action candidates")

    logged_action_id = applied.action_id if applied is not None else candidates[0].action_id
    propensity_payload = row.get("propensity")
    propensity: PropensityMetadata | None = None
    if isinstance(propensity_payload, dict):
        propensity = PropensityMetadata.model_validate({**propensity_payload, "logged_action_id": logged_action_id})

    example_id = str(
        row.get("row_id")
        or f"intent-router:{row.get('session_id') or 'unknown'}:{row.get('client_turn_id') or 'unknown'}"
    )
    split_group = str(row.get("dataset_split_group") or row.get("session_id") or example_id)
    return LoggedDecisionExample(
        example_id=example_id,
        source_type="intent_router",
        context={
            "user_text": row.get("user_text"),
            "screen_context": row.get("screen_context"),
            "recent_session_features": row.get("recent_session_features") or [],
            "opportunity_candidate_metadata": row.get("opportunity_candidate_metadata"),
        },
        action_candidates=candidates,
        logged_action_id=logged_action_id,
        propensity=propensity,
        reward_components=[RewardComponent.model_validate(item) for item in _reward_components_from_router_row(row)],
        split_group=split_group,
        provenance={
            "row_id": row.get("row_id"),
            "session_id": row.get("session_id"),
            "client_turn_id": row.get("client_turn_id"),
            "captured_at": row.get("captured_at"),
            "capture_policy": row.get("capture_policy"),
            "sampling_reason": row.get("sampling_reason"),
        },
    )


def total_reward(example: LoggedDecisionExample) -> float:
    weighted = sum(component.value * component.weight for component in example.reward_components)
    total_weight = sum(component.weight for component in example.reward_components)
    if total_weight <= 0:
        raise AgentPolicyExperimentError(f"Example {example.example_id} has no positive reward weight")
    return max(-1.0, min(1.0, weighted / total_weight))


def validate_reward_policy(example: LoggedDecisionExample) -> list[str]:
    """Return reward-policy errors for leakage, direct P&L, or missing process evidence."""

    errors: list[str] = []
    sources = {component.source for component in example.reward_components}
    if sources == {"outcome_label"}:
        errors.append("outcome_label_cannot_be_sole_reward_source")
    if not sources.intersection({"process_reward", "eval_score", "human_review", "synthetic"}):
        errors.append("missing_process_or_review_reward_source")
    return errors


def select_action(example: LoggedDecisionExample, policy: PolicyName) -> ActionCandidate | None:
    by_id = {candidate.action_id: candidate for candidate in example.action_candidates}
    if policy == "logged":
        return by_id.get(example.logged_action_id)
    if policy == "highest_confidence":
        return max(
            example.action_candidates,
            key=lambda candidate: candidate.confidence if candidate.confidence is not None else -1.0,
            default=None,
        )
    source = {"regex_baseline": "regex", "llm_candidate": "llm", "supervised_candidate": "supervised"}[policy]
    for candidate in example.action_candidates:
        if candidate.source == source:
            return candidate
    return None


def _mean_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"count": 1, "mean": mean, "ci95_low": mean, "ci95_high": mean}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
    return {"count": len(values), "mean": mean, "ci95_low": mean - half_width, "ci95_high": mean + half_width}


def _case_entry(row: PolicyEvaluationRow, *, action_id: str | None, candidate: bool) -> dict[str, Any]:
    if candidate:
        checks = list(row.deterministic_checks)
        passed = row.deterministic_passed and row.exclusion_reason is None
    else:
        checks = [{"name": "baseline_action_available", "passed": row.baseline_action_id is not None}]
        passed = row.baseline_action_id is not None and row.exclusion_reason != "reward_policy"
    if candidate and row.exclusion_reason:
        checks.append({"name": row.exclusion_reason, "passed": False})
    return {
        "case_id": row.example_id,
        "deterministic_passed": passed,
        "deterministic_checks": checks,
        "judge_total": row.ips_reward,
        "action_id": action_id,
    }


def evaluate_examples(
    examples: list[LoggedDecisionExample],
    *,
    baseline_policy: PolicyName,
    candidate_policy: PolicyName,
    require_propensity: bool = True,
) -> tuple[list[PolicyEvaluationRow], list[dict[str, Any]]]:
    rows: list[PolicyEvaluationRow] = []
    exclusions: list[dict[str, Any]] = []

    for example in examples:
        reward_errors = validate_reward_policy(example)
        if reward_errors:
            reason = ",".join(reward_errors)
            exclusions.append({"example_id": example.example_id, "reason": reason})
            rows.append(
                PolicyEvaluationRow(
                    example_id=example.example_id,
                    logged_action_id=example.logged_action_id,
                    baseline_action_id=None,
                    candidate_action_id=None,
                    logged_reward=total_reward(example),
                    deterministic_passed=False,
                    deterministic_checks=[{"name": "reward_policy", "passed": False, "message": reason}],
                    exclusion_reason="reward_policy",
                )
            )
            continue

        baseline_action = select_action(example, baseline_policy)
        candidate_action = select_action(example, candidate_policy)
        checks: list[dict[str, Any]] = [
            {"name": "baseline_action_available", "passed": baseline_action is not None},
            {"name": "candidate_action_available", "passed": candidate_action is not None},
        ]
        exclusion_reason: str | None = None

        gate_violations = list((candidate_action.gate_overrides if candidate_action else []) or [])
        checks.append({"name": "gate_boundary", "passed": not gate_violations, "message": ",".join(gate_violations)})
        if gate_violations:
            exclusion_reason = "gate_boundary_violation"

        propensity_value: float | None = None
        ips_reward: float | None = None
        if candidate_action is None:
            exclusion_reason = exclusion_reason or "candidate_action_missing"
        elif candidate_action.action_id != example.logged_action_id:
            exclusion_reason = exclusion_reason or "counterfactual_action_unobserved"
        else:
            propensity_value = (
                example.propensity.probability_for(candidate_action.action_id) if example.propensity else None
            )
            if require_propensity and propensity_value is None:
                exclusion_reason = exclusion_reason or "missing_propensity"
            elif propensity_value is not None:
                ips_reward = total_reward(example) / propensity_value
            else:
                ips_reward = total_reward(example)

        if exclusion_reason:
            exclusions.append({"example_id": example.example_id, "reason": exclusion_reason})
        checks.append({"name": "observed_action_support", "passed": exclusion_reason is None})

        rows.append(
            PolicyEvaluationRow(
                example_id=example.example_id,
                logged_action_id=example.logged_action_id,
                baseline_action_id=baseline_action.action_id if baseline_action else None,
                candidate_action_id=candidate_action.action_id if candidate_action else None,
                logged_reward=total_reward(example),
                candidate_propensity=propensity_value,
                ips_reward=ips_reward,
                deterministic_passed=all(bool(check.get("passed")) for check in checks),
                deterministic_checks=checks,
                exclusion_reason=exclusion_reason,
            )
        )
    return rows, exclusions


def build_experiment_report(
    examples: list[LoggedDecisionExample],
    *,
    manifest: ExperimentManifest,
    generated_at: str | None = None,
) -> dict[str, Any]:
    rows, exclusions = evaluate_examples(
        examples,
        baseline_policy=manifest.baseline_policy,
        candidate_policy=manifest.candidate_policy,
        require_propensity=manifest.require_propensity,
    )
    evaluated_rows = [row for row in rows if row.exclusion_reason is None]
    ips_values = [float(row.ips_reward) for row in evaluated_rows if row.ips_reward is not None]
    logged_rewards = [row.logged_reward for row in rows]

    baseline_report = {
        "generated_at": generated_at or _now_iso(),
        "cases": {
            row.example_id: _case_entry(row, action_id=row.baseline_action_id, candidate=False)
            for row in rows
            if row.baseline_action_id is not None
        },
    }
    candidate_report = {
        "generated_at": generated_at or _now_iso(),
        "cases": {
            row.example_id: _case_entry(row, action_id=row.candidate_action_id, candidate=True)
            for row in rows
            if row.candidate_action_id is not None
        },
    }

    reward_sources = Counter(component.source for example in examples for component in example.reward_components)
    comparison = compare_reports(baseline_report, candidate_report)
    if generated_at is not None:
        comparison["generated_at"] = generated_at

    report = {
        "report_version": REPORT_VERSION,
        "runner_version": RUNNER_VERSION,
        "generated_at": generated_at or _now_iso(),
        "manifest": manifest.model_dump(mode="json"),
        "row_count": len(examples),
        "evaluated_count": len(evaluated_rows),
        "exclusion_counts": dict(Counter(item["reason"] for item in exclusions)),
        "exclusions": exclusions,
        "reward_source_counts": dict(reward_sources),
        "logged_reward": _mean_ci(logged_rewards),
        "candidate_ips_reward": _mean_ci(ips_values),
        "comparison": comparison,
        "rows": [row.model_dump(mode="json") for row in rows],
        "known_biases": [
            "Offline policy estimates only cover logged actions with valid propensities.",
            "Intent-router rows are the first policy-choice domain; full agent-loop choices remain out of scope.",
            "Outcome labels are bounded process evidence and cannot be direct P&L rewards.",
        ],
        "gate_boundary": {
            "non_overridable": True,
            "violations": [item for item in exclusions if item["reason"] == "gate_boundary_violation"],
        },
    }
    return report


def _load_examples_from_jsonl(path: Path) -> list[LoggedDecisionExample]:
    examples: list[LoggedDecisionExample] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise AgentPolicyExperimentError(f"{path}:{line_no} must contain JSON objects")
        try:
            if raw.get("action_candidates"):
                examples.append(LoggedDecisionExample.model_validate(raw))
            else:
                examples.append(logged_example_from_intent_router_row(raw))
        except (ValidationError, AgentPolicyExperimentError) as exc:
            raise AgentPolicyExperimentError(f"{path}:{line_no}: {exc}") from exc
    return examples


def load_examples(*, input_jsonl: Path | None, limit: int) -> list[LoggedDecisionExample]:
    if input_jsonl is not None:
        return _load_examples_from_jsonl(input_jsonl)[:limit]

    from api.intent_router_training_store import list_training_rows

    rows = list_training_rows(limit=limit)
    examples: list[LoggedDecisionExample] = []
    for row in rows:
        try:
            examples.append(logged_example_from_intent_router_row(row))
        except (ValidationError, AgentPolicyExperimentError):
            continue
    return examples


def _default_manifest() -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id=f"offline_policy_{_now_tag()}",
        baseline_policy="logged",
        candidate_policy="highest_confidence",
        reward_components=["routing", "gate_compliance", "tool_selection"],
        input_sources=["intent_router_training_rows"],
        notes="Default TL-68 offline contextual-bandit report.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline agent-policy experiments (TL-68).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser("report", help="Build an offline policy experiment report")
    report.add_argument("--manifest", type=Path)
    report.add_argument("--input-jsonl", type=Path)
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    report.add_argument("--limit", type=int, default=1000)
    report.add_argument("--baseline-policy", choices=list(PolicyName.__args__))  # type: ignore[attr-defined]
    report.add_argument("--candidate-policy", choices=list(PolicyName.__args__))  # type: ignore[attr-defined]
    report.add_argument("--allow-missing-propensity", action="store_true")
    report.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command != "report":
        raise AgentPolicyExperimentError(f"Unsupported command: {args.command}")

    manifest = ExperimentManifest.model_validate(_read_json(args.manifest)) if args.manifest else _default_manifest()
    if args.baseline_policy:
        manifest.baseline_policy = args.baseline_policy
    if args.candidate_policy:
        manifest.candidate_policy = args.candidate_policy
    if args.allow_missing_propensity:
        manifest.require_propensity = False

    examples = load_examples(input_jsonl=args.input_jsonl, limit=max(0, int(args.limit)))
    report = build_experiment_report(examples, manifest=manifest)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, default=str))
        return 0

    output_dir = Path(args.output_dir) / str(manifest.experiment_id)
    output_path = output_dir / "experiment_report.json"
    _write_json(output_path, report)
    print(f"Wrote offline policy experiment report: {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
