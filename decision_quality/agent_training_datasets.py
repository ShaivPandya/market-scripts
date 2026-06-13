"""Governed SFT and preference dataset curation for Talisman agent training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from api.agent_response_feedback import HUMAN_REVIEWED_SIGNAL, response_version_for_trajectory
from api.agent_trajectories import TRAJECTORY_REDACTION_POLICY, TrajectoryExportError
from decision_quality.eval_corpus import TRAINING_EXPORT_STATUSES
from decision_quality.supervised_labels import assign_split, check_split_leakage, split_group_for_case
from decision_quality.talisman_bench import build_inventory

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "agent_training_datasets"
DEFAULT_BENCH_MANIFEST = ROOT / "docs" / "talisman_bench" / "manifest.json"
DEFAULT_SEEDS_DIR = ROOT / "docs" / "agent_training_datasets" / "seeds"
DEFAULT_PREFERENCE_SEEDS_DIR = ROOT / "docs" / "agent_training_datasets" / "seeds" / "preference"

SFT_SCHEMA_VERSION = 1
PREFERENCE_SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
TRANSFORMATION_VERSION = "agent_training_datasets_v1"

SourceType = Literal["trajectory", "eval_fixture", "synthetic", "teacher"]
ReviewStatus = Literal["released", "pending", "rejected"]
SignalSource = Literal["human_reviewed", "eval_fixture", "synthetic", "teacher", "judge_assisted"]

EXCLUSION_REASONS = frozenset(
    {
        "release_gate_case",
        "release_gate_split_group",
        "duplicate",
        "unreviewed",
        "missing_trajectory",
        "response_version_mismatch",
        "ineligible_trajectory",
        "incomplete_trajectory",
        "unknown_schema",
        "teacher_not_human_approved",
        "missing_review_status",
        "missing_target_output",
        "missing_preference_pair",
        "invalid_seed_row",
        "conflicting_preference_labels",
        "low_confidence_preference",
    }
)


class AgentTrainingDatasetError(ValueError):
    """Raised when a governed dataset export cannot complete safely."""


class SftExample(BaseModel):
    schema_version: int = SFT_SCHEMA_VERSION
    example_id: str
    source_type: SourceType
    source_id: str
    task_class: str = "agent_turn"
    messages: list[dict[str, Any]]
    steps: list[dict[str, Any]] = Field(default_factory=list)
    target_output: str
    split_group: str
    split: str
    review_status: ReviewStatus = "released"
    eligibility: str = "eligible"
    provenance: dict[str, Any] = Field(default_factory=dict)
    redaction_manifest: dict[str, Any] = Field(default_factory=dict)
    sensitivity: str = "operational_private"
    signal_source: SignalSource
    transformation_version: str = TRANSFORMATION_VERSION
    content_hash: str

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: int) -> int:
        if value != SFT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported SFT schema version: {value}")
        return value


class PreferenceExample(BaseModel):
    schema_version: int = PREFERENCE_SCHEMA_VERSION
    example_id: str
    source_type: SourceType = "trajectory"
    source_id: str
    trajectory_id: str
    feedback_id: str
    response_version: str
    decision: Literal["reject", "correct"]
    messages: list[dict[str, Any]]
    steps: list[dict[str, Any]] = Field(default_factory=list)
    chosen: str | None = None
    rejected: str
    split_group: str
    split: str
    review_status: ReviewStatus = "released"
    eligibility: str = "eligible"
    provenance: dict[str, Any] = Field(default_factory=dict)
    redaction_manifest: dict[str, Any] = Field(default_factory=dict)
    sensitivity: str = "operational_private"
    signal_source: SignalSource = "human_reviewed"
    failure_tags: list[str] = Field(default_factory=list)
    transformation_version: str = TRANSFORMATION_VERSION
    content_hash: str

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: int) -> int:
        if value != PREFERENCE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported preference schema version: {value}")
        return value

    @field_validator("chosen")
    @classmethod
    def _correct_requires_chosen(cls, value: str | None, info) -> str | None:
        if info.data.get("decision") == "correct" and not value:
            raise ValueError("chosen is required when decision is correct")
        return value


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stable_hash(value: Any, *, length: int = 32) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _assistant_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _task_class_from_trajectory(payload: dict[str, Any]) -> str:
    route = payload.get("route")
    if isinstance(route, dict):
        intent = route.get("intent_class") or route.get("intent")
        if intent:
            return str(intent)
    return "agent_turn"


def _load_release_gate_inventory(*, manifest_path: Path = DEFAULT_BENCH_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return build_inventory(manifest, approved_only=True, root=ROOT)


def release_gate_case_ids(inventory: dict[str, Any] | None = None) -> set[str]:
    rows = (inventory or _load_release_gate_inventory()).get("rows") or []
    return {str(row.get("case_id") or "") for row in rows if row.get("case_id")}


def release_gate_split_groups(inventory: dict[str, Any] | None = None) -> set[str]:
    rows = (inventory or _load_release_gate_inventory()).get("rows") or []
    return {str(row.get("split_group") or "") for row in rows if row.get("split_group")}


def _record_exclusion(exclusions: list[dict[str, Any]], *, reason: str, source_id: str, detail: str = "") -> None:
    if reason not in EXCLUSION_REASONS:
        reason = "unknown_schema"
    exclusions.append({"reason": reason, "source_id": source_id, "detail": detail})


def _dedupe_key(row: dict[str, Any]) -> str:
    return str(row.get("content_hash") or row.get("example_id") or "")


def _validate_and_sort_rows(rows: list[dict[str, Any]], model: type[BaseModel]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for row in rows:
        validated.append(model.model_validate(row).model_dump(mode="json"))
    return sorted(validated, key=lambda item: str(item.get("example_id") or ""))


def trajectory_to_sft_row(
    *,
    trajectory: dict[str, Any],
    feedback: dict[str, Any],
    split_group: str,
    split: str,
    redaction_manifest: dict[str, Any],
) -> dict[str, Any]:
    messages = list(trajectory.get("messages") or [])
    target_output = _assistant_content(messages)
    if not target_output:
        raise AgentTrainingDatasetError("Trajectory is missing assistant target output")
    payload = {
        "example_id": f"sft:trajectory:{trajectory.get('trajectory_id')}:{feedback.get('response_version')}",
        "source_type": "trajectory",
        "source_id": str(trajectory.get("trajectory_id") or ""),
        "task_class": _task_class_from_trajectory(trajectory),
        "messages": messages,
        "steps": list(trajectory.get("steps") or []),
        "target_output": target_output,
        "split_group": split_group,
        "split": split,
        "review_status": "released",
        "provenance": {
            "trajectory_id": trajectory.get("trajectory_id"),
            "feedback_id": feedback.get("feedback_id"),
            "reviewer_actor_id": feedback.get("reviewer_actor_id"),
            "reviewed_at": feedback.get("reviewed_at"),
            "response_version": feedback.get("response_version"),
            "provider": trajectory.get("provider"),
            "model": trajectory.get("model"),
            "prompt_version": trajectory.get("prompt_version"),
            "code_version": trajectory.get("code_version"),
        },
        "redaction_manifest": redaction_manifest,
        "sensitivity": str(trajectory.get("sensitivity") or "operational_private"),
        "signal_source": "human_reviewed",
    }
    payload["content_hash"] = _stable_hash(
        {
            "messages": payload["messages"],
            "target_output": payload["target_output"],
            "source_id": payload["source_id"],
        }
    )
    return payload


def feedback_to_preference_row(
    *,
    trajectory: dict[str, Any],
    feedback: dict[str, Any],
    split_group: str,
    split: str,
    redaction_manifest: dict[str, Any],
) -> dict[str, Any]:
    messages = list(trajectory.get("messages") or [])
    rejected = _assistant_content(messages)
    if not rejected:
        raise AgentTrainingDatasetError("Trajectory is missing assistant content for preference export")
    decision = str(feedback.get("decision") or "")
    if decision not in {"reject", "correct"}:
        raise AgentTrainingDatasetError(f"Unsupported preference decision: {decision}")
    chosen = feedback.get("corrected_response") if decision == "correct" else None
    payload = {
        "example_id": f"pref:trajectory:{trajectory.get('trajectory_id')}:{feedback.get('response_version')}",
        "source_type": "trajectory",
        "source_id": str(trajectory.get("trajectory_id") or ""),
        "trajectory_id": str(trajectory.get("trajectory_id") or ""),
        "feedback_id": str(feedback.get("feedback_id") or ""),
        "response_version": str(feedback.get("response_version") or ""),
        "decision": decision,
        "messages": messages,
        "steps": list(trajectory.get("steps") or []),
        "chosen": chosen,
        "rejected": rejected,
        "split_group": split_group,
        "split": split,
        "review_status": "released",
        "provenance": {
            "trajectory_id": trajectory.get("trajectory_id"),
            "feedback_id": feedback.get("feedback_id"),
            "reviewer_actor_id": feedback.get("reviewer_actor_id"),
            "reviewed_at": feedback.get("reviewed_at"),
            "response_version": feedback.get("response_version"),
        },
        "redaction_manifest": redaction_manifest,
        "sensitivity": str(trajectory.get("sensitivity") or "operational_private"),
        "signal_source": "human_reviewed",
        "failure_tags": list(feedback.get("failure_tags") or []),
    }
    payload["content_hash"] = _stable_hash(
        {
            "messages": payload["messages"],
            "chosen": payload["chosen"],
            "rejected": payload["rejected"],
            "source_id": payload["source_id"],
        }
    )
    return payload


def eval_fixture_to_sft_row(
    *, case_id: str, case_data: dict[str, Any], source_path: str, corpus_id: str
) -> dict[str, Any]:
    user_text = case_data.get("user_question") or case_data.get("user_message") or ""
    gold = case_data.get("gold_output")
    if not user_text or not isinstance(gold, dict):
        raise AgentTrainingDatasetError(f"Eval fixture {case_id} is missing user text or gold output")
    target_output = json.dumps(gold, ensure_ascii=True, sort_keys=True, default=str)
    split_group = split_group_for_case(case_id=case_id, case_data=case_data)
    split = assign_split(split_group)
    messages = [
        {"role": "user", "content": str(user_text)},
        {"role": "assistant", "content": target_output},
    ]
    payload = {
        "example_id": f"sft:eval:{corpus_id}:{case_id}",
        "source_type": "eval_fixture",
        "source_id": case_id,
        "task_class": str(case_data.get("failure_type") or corpus_id),
        "messages": messages,
        "steps": [],
        "target_output": target_output,
        "split_group": split_group,
        "split": split,
        "review_status": "released",
        "provenance": {
            "case_id": case_id,
            "corpus_id": corpus_id,
            "source_path": source_path,
            "eval_status": str(case_data.get("status") or "draft"),
            "label_reviewer": "eval_fixture",
        },
        "redaction_manifest": {"policy": "eval_fixture_export_v1", "source": "approved_eval_exclusion"},
        "sensitivity": "benchmark_fixture",
        "signal_source": "eval_fixture",
    }
    payload["content_hash"] = _stable_hash(
        {"messages": payload["messages"], "target_output": payload["target_output"], "source_id": payload["source_id"]}
    )
    return payload


def seed_row_to_preference_row(row: dict[str, Any]) -> dict[str, Any]:
    source_type = str(row.get("source_type") or "")
    signal_source = str(row.get("signal_source") or source_type)
    if signal_source not in {"synthetic", "judge_assisted", "teacher"}:
        raise AgentTrainingDatasetError(f"Unsupported preference seed signal_source: {signal_source}")
    messages = row.get("messages")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(messages, list) or not chosen or not rejected:
        raise AgentTrainingDatasetError("Preference seed row requires messages, chosen, and rejected")
    source_id = str(row.get("source_id") or row.get("example_id") or _stable_hash(row, length=16))
    split_group = str(row.get("split_group") or f"preference:{source_id}")
    split = str(row.get("split") or assign_split(split_group))
    review_status = str(row.get("review_status") or "released")
    if review_status != "released":
        raise AgentTrainingDatasetError("Preference seed rows must be released for export")
    payload = {
        "example_id": str(row.get("example_id") or f"pref:{signal_source}:{source_id}"),
        "source_type": source_type if source_type in {"synthetic", "teacher"} else "synthetic",
        "source_id": source_id,
        "trajectory_id": str(row.get("trajectory_id") or ""),
        "feedback_id": str(row.get("feedback_id") or ""),
        "response_version": str(row.get("response_version") or ""),
        "decision": "correct",
        "messages": messages,
        "steps": list(row.get("steps") or []),
        "chosen": str(chosen),
        "rejected": str(rejected),
        "split_group": split_group,
        "split": split,
        "review_status": review_status,
        "provenance": dict(row.get("provenance") or {}),
        "redaction_manifest": dict(row.get("redaction_manifest") or {"policy": f"{signal_source}_preference_seed_v1"}),
        "sensitivity": str(row.get("sensitivity") or "synthetic"),
        "signal_source": signal_source,
        "failure_tags": list(row.get("failure_tags") or []),
    }
    payload["content_hash"] = _stable_hash(
        {
            "messages": payload["messages"],
            "chosen": payload["chosen"],
            "rejected": payload["rejected"],
            "source_id": payload["source_id"],
        }
    )
    return payload


def is_dpo_trainable_preference_row(row: dict[str, Any]) -> bool:
    """Return True when a preference row has a complete chosen/rejected pair for DPO."""

    return bool(str(row.get("chosen") or "").strip()) and bool(str(row.get("rejected") or "").strip())


def filter_dpo_trainable_preference_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split preference rows into DPO-trainable pairs and incomplete reject-only rows."""

    trainable: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for row in rows:
        if is_dpo_trainable_preference_row(row):
            trainable.append(row)
        else:
            incomplete.append(row)
    return trainable, incomplete


def preference_reward_source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("signal_source") or "unknown") for row in rows))


def seed_row_to_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    source_type = str(row.get("source_type") or "")
    if source_type not in {"synthetic", "teacher"}:
        raise AgentTrainingDatasetError(f"Unsupported seed source_type: {source_type}")
    messages = row.get("messages")
    target_output = row.get("target_output")
    if not isinstance(messages, list) or not target_output:
        raise AgentTrainingDatasetError("Seed row requires messages and target_output")
    source_id = str(row.get("source_id") or row.get("example_id") or _stable_hash(row, length=16))
    split_group = str(row.get("split_group") or f"{source_type}:{source_id}")
    split = str(row.get("split") or assign_split(split_group))
    payload = {
        "example_id": str(row.get("example_id") or f"sft:{source_type}:{source_id}"),
        "source_type": source_type,
        "source_id": source_id,
        "task_class": str(row.get("task_class") or source_type),
        "messages": messages,
        "steps": list(row.get("steps") or []),
        "target_output": str(target_output),
        "split_group": split_group,
        "split": split,
        "review_status": str(row.get("review_status") or "released"),
        "provenance": dict(row.get("provenance") or {}),
        "redaction_manifest": dict(row.get("redaction_manifest") or {"policy": f"{source_type}_seed_v1"}),
        "sensitivity": str(row.get("sensitivity") or "synthetic"),
        "signal_source": source_type,
    }
    payload["content_hash"] = _stable_hash(
        {"messages": payload["messages"], "target_output": payload["target_output"], "source_id": payload["source_id"]}
    )
    return payload


def _load_eval_fixture_rows(
    *,
    statuses: frozenset[str],
    release_gate_ids: set[str],
    exclusions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from decision_quality.chat_eval_runner import load_cases as load_chat_cases
    from decision_quality.eval_runner import load_cases as load_structured_cases
    from decision_quality.opportunity_candidate_eval_runner import load_cases as load_opportunity_cases
    from decision_quality.talisman_bench import corpus_configs

    manifest = json.loads(DEFAULT_BENCH_MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    status_set = set(statuses)
    for config in corpus_configs(manifest, root=ROOT):
        cases: list[Any]
        if config.runner == "structured":
            cases = load_structured_cases(statuses=status_set, cases_dir=config.cases_dir)
        elif config.runner == "chat":
            cases = load_chat_cases(statuses=status_set, cases_dir=config.cases_dir)
        else:
            cases = load_opportunity_cases(statuses=status_set, cases_dir=config.cases_dir)
        for case in cases:
            case_id = case.case_id
            if case_id in release_gate_ids:
                _record_exclusion(
                    exclusions,
                    reason="release_gate_case",
                    source_id=case_id,
                    detail="Approved TalismanBench case excluded from training export",
                )
                continue
            try:
                source_path = str(case.path.relative_to(ROOT) if case.path.is_relative_to(ROOT) else case.path)
                rows.append(
                    eval_fixture_to_sft_row(
                        case_id=case_id,
                        case_data=case.data,
                        source_path=source_path,
                        corpus_id=config.id,
                    )
                )
            except AgentTrainingDatasetError as exc:
                _record_exclusion(exclusions, reason="invalid_seed_row", source_id=case_id, detail=str(exc))
    return rows


def _load_preference_seed_rows(*, seeds_dir: Path, exclusions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not seeds_dir.exists():
        return rows
    for path in sorted(seeds_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows.append(seed_row_to_preference_row(row))
            except (json.JSONDecodeError, AgentTrainingDatasetError, ValidationError) as exc:
                _record_exclusion(
                    exclusions,
                    reason="invalid_seed_row",
                    source_id=f"{path.name}:{line_no}",
                    detail=str(exc),
                )
    return rows


def _detect_preference_conflicts(feedback_rows: list[dict[str, Any]]) -> set[str]:
    """Return trajectory ids with conflicting human-reviewed preference decisions."""

    grouped: dict[tuple[str, str], set[str]] = {}
    for label in feedback_rows:
        if label.get("signal_source") != HUMAN_REVIEWED_SIGNAL:
            continue
        if not label.get("training_eligible"):
            continue
        decision = str(label.get("decision") or "")
        if decision not in {"reject", "correct"}:
            continue
        trajectory_id = str(label.get("trajectory_id") or "")
        response_version = str(label.get("response_version") or "")
        key = (trajectory_id, response_version)
        grouped.setdefault(key, set()).add(decision)
    return {trajectory_id for (trajectory_id, _version), decisions in grouped.items() if len(decisions) > 1}


def _load_seed_rows(*, seeds_dir: Path, exclusions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not seeds_dir.exists():
        return rows
    for path in sorted(seeds_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows.append(seed_row_to_sft_row(row))
            except (json.JSONDecodeError, AgentTrainingDatasetError, ValidationError) as exc:
                _record_exclusion(
                    exclusions,
                    reason="invalid_seed_row",
                    source_id=f"{path.name}:{line_no}",
                    detail=str(exc),
                )
    return rows


def _trajectory_payload_index(
    row: dict[str, Any],
    *,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        **payload,
        "dataset_split_group": row.get("dataset_split_group") or payload.get("dataset_split_group"),
        "redaction_manifest": row.get("redaction_manifest") or {},
        "sensitivity": row.get("sensitivity") or payload.get("sensitivity"),
    }


def _trajectory_lookup_rows(
    feedback_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    from api.agent_trajectories import (
        _exportable_payload,
        _exportable_preference_payload,
        get_trajectory,
        list_trajectories,
    )

    metadata_by_id: dict[str, dict[str, Any]] = {}
    sft_index: dict[str, dict[str, Any]] = {}
    preference_index: dict[str, dict[str, Any]] = {}

    trajectory_ids = {str(label.get("trajectory_id") or "") for label in feedback_rows if label.get("trajectory_id")}
    for row in list_trajectories(limit=limit, training_eligible_only=True):
        trajectory_id = str(row.get("trajectory_id") or "")
        metadata_by_id[trajectory_id] = row
        trajectory_ids.add(trajectory_id)

    for trajectory_id in sorted(trajectory_ids):
        trajectory_row = metadata_by_id.get(trajectory_id)
        if trajectory_row is None:
            trajectory_row = get_trajectory(trajectory_id)
        if not trajectory_row or trajectory_row.get("tombstoned_at"):
            continue
        metadata_by_id[trajectory_id] = trajectory_row
        try:
            sft_index[trajectory_id] = _trajectory_payload_index(
                trajectory_row, payload=_exportable_payload(trajectory_row)
            )
        except TrajectoryExportError:
            pass
        try:
            preference_index[trajectory_id] = _trajectory_payload_index(
                trajectory_row,
                payload=_exportable_preference_payload(trajectory_row),
            )
        except TrajectoryExportError:
            pass

    return metadata_by_id, sft_index, preference_index


def build_training_dataset(
    *,
    limit: int = 5000,
    include_eval_fixtures: bool = True,
    include_seeds: bool = True,
    include_preference_seeds: bool = True,
    seeds_dir: Path = DEFAULT_SEEDS_DIR,
    preference_seeds_dir: Path = DEFAULT_PREFERENCE_SEEDS_DIR,
    manifest_path: Path = DEFAULT_BENCH_MANIFEST,
    export_version: str | None = None,
) -> dict[str, Any]:
    """Build governed SFT and preference datasets from safe sources."""

    exclusions: list[dict[str, Any]] = []
    inventory = _load_release_gate_inventory(manifest_path=manifest_path)
    blocked_case_ids = release_gate_case_ids(inventory)
    blocked_split_groups = release_gate_split_groups(inventory)

    from api.agent_response_feedback import export_human_reviewed_feedback

    feedback_rows = export_human_reviewed_feedback(limit=limit)
    conflicting_trajectory_ids = _detect_preference_conflicts(feedback_rows)
    metadata_by_id, sft_index, preference_index = _trajectory_lookup_rows(feedback_rows, limit=limit)

    sft_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []

    for label in feedback_rows:
        if label.get("signal_source") != HUMAN_REVIEWED_SIGNAL:
            _record_exclusion(exclusions, reason="unreviewed", source_id=str(label.get("feedback_id") or ""))
            continue
        trajectory_id = str(label.get("trajectory_id") or "")
        decision = str(label.get("decision") or "")
        if decision in {"reject", "correct"} and trajectory_id in conflicting_trajectory_ids:
            _record_exclusion(
                exclusions,
                reason="conflicting_preference_labels",
                source_id=trajectory_id,
                detail=str(label.get("feedback_id") or ""),
            )
            continue
        trajectory_lookup = sft_index if decision == "approve" else preference_index
        trajectory = trajectory_lookup.get(trajectory_id)
        if trajectory is None:
            _record_exclusion(exclusions, reason="missing_trajectory", source_id=trajectory_id)
            continue
        metadata = metadata_by_id.get(trajectory_id, {"sanitized_payload": trajectory})
        expected_version = response_version_for_trajectory(metadata)
        if str(label.get("response_version") or "") != expected_version:
            _record_exclusion(
                exclusions,
                reason="response_version_mismatch",
                source_id=trajectory_id,
                detail=str(label.get("feedback_id") or ""),
            )
            continue

        split_group = str(trajectory.get("dataset_split_group") or trajectory.get("split_group") or trajectory_id)
        if split_group in blocked_split_groups:
            _record_exclusion(
                exclusions,
                reason="release_gate_split_group",
                source_id=trajectory_id,
                detail=split_group,
            )
            continue
        split = assign_split(split_group)
        redaction_manifest = dict(trajectory.get("redaction_manifest") or {})
        if redaction_manifest.get("policy") != TRAJECTORY_REDACTION_POLICY:
            _record_exclusion(exclusions, reason="ineligible_trajectory", source_id=trajectory_id)
            continue

        try:
            if decision == "approve":
                sft_rows.append(
                    trajectory_to_sft_row(
                        trajectory=trajectory,
                        feedback=label,
                        split_group=split_group,
                        split=split,
                        redaction_manifest=redaction_manifest,
                    )
                )
            elif decision in {"reject", "correct"}:
                preference_rows.append(
                    feedback_to_preference_row(
                        trajectory=trajectory,
                        feedback=label,
                        split_group=split_group,
                        split=split,
                        redaction_manifest=redaction_manifest,
                    )
                )
        except AgentTrainingDatasetError as exc:
            _record_exclusion(exclusions, reason="incomplete_trajectory", source_id=trajectory_id, detail=str(exc))

    if include_eval_fixtures:
        sft_rows.extend(
            _load_eval_fixture_rows(
                statuses=TRAINING_EXPORT_STATUSES,
                release_gate_ids=blocked_case_ids,
                exclusions=exclusions,
            )
        )

    if include_seeds:
        sft_rows.extend(_load_seed_rows(seeds_dir=seeds_dir, exclusions=exclusions))
    if include_preference_seeds:
        preference_rows.extend(_load_preference_seed_rows(seeds_dir=preference_seeds_dir, exclusions=exclusions))

    sft_rows = _dedupe_rows(sft_rows, exclusions)
    preference_rows = _dedupe_rows(preference_rows, exclusions)
    dpo_trainable_rows, dpo_incomplete_rows = filter_dpo_trainable_preference_rows(preference_rows)

    combined_rows = sft_rows + preference_rows
    leakage_violations = check_split_leakage(combined_rows)
    contamination = _release_gate_contamination(combined_rows, blocked_case_ids, blocked_split_groups)
    if contamination:
        raise AgentTrainingDatasetError("Release-gate contamination detected: " + "; ".join(sorted(contamination)))
    if leakage_violations:
        raise AgentTrainingDatasetError("Split leakage detected: " + "; ".join(leakage_violations))

    sft_rows = _validate_and_sort_rows(sft_rows, SftExample)
    preference_rows = _validate_and_sort_rows(preference_rows, PreferenceExample)

    version = export_version or _now_tag()
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "version": version,
        "exported_at": _now_iso(),
        "transformation_version": TRANSFORMATION_VERSION,
        "sft_count": len(sft_rows),
        "preference_count": len(preference_rows),
        "dpo_trainable_count": len(dpo_trainable_rows),
        "dpo_incomplete_count": len(dpo_incomplete_rows),
        "preference_reward_source_counts": preference_reward_source_counts(preference_rows),
        "dpo_trainable_reward_source_counts": preference_reward_source_counts(dpo_trainable_rows),
        "exclusion_count": len(exclusions),
        "exclusion_counts": dict(Counter(item["reason"] for item in exclusions)),
        "exclusions": exclusions,
        "split_counts": dict(Counter(str(row.get("split") or "unknown") for row in sft_rows + preference_rows)),
        "source_counts": {
            "sft": dict(Counter(str(row.get("source_type") or "unknown") for row in sft_rows)),
            "preference": dict(Counter(str(row.get("source_type") or "unknown") for row in preference_rows)),
        },
        "signal_source_counts": {
            "sft": dict(Counter(str(row.get("signal_source") or "unknown") for row in sft_rows)),
            "preference": dict(Counter(str(row.get("signal_source") or "unknown") for row in preference_rows)),
        },
        "release_gate_excluded_case_count": len(blocked_case_ids),
        "release_gate_excluded_split_group_count": len(blocked_split_groups),
        "leakage_violations": leakage_violations,
        "leakage_check_passed": not leakage_violations,
        "bench_manifest_path": str(
            manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT) else manifest_path
        ),
    }
    manifest["content_hashes"] = {
        "sft.jsonl": _stable_hash(sft_rows),
        "preference.jsonl": _stable_hash(preference_rows),
        "manifest.json": _stable_hash(
            {key: value for key, value in manifest.items() if key not in {"content_hashes", "exported_at"}}
        ),
    }
    return {
        "version": version,
        "manifest": manifest,
        "sft_rows": sft_rows,
        "preference_rows": preference_rows,
    }


def _dedupe_rows(rows: list[dict[str, Any]], exclusions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        key = _dedupe_key(row)
        if not key or key in seen:
            _record_exclusion(
                exclusions,
                reason="duplicate",
                source_id=str(row.get("source_id") or row.get("example_id") or ""),
            )
            continue
        seen.add(key)
        kept.append(row)
    return kept


def _release_gate_contamination(
    rows: list[dict[str, Any]],
    blocked_case_ids: set[str],
    blocked_split_groups: set[str],
) -> list[str]:
    violations: list[str] = []
    for row in rows:
        raw_provenance = row.get("provenance")
        provenance: dict[str, Any] = raw_provenance if isinstance(raw_provenance, dict) else {}
        case_id = str(provenance.get("case_id") or row.get("source_id") or "")
        if case_id in blocked_case_ids:
            violations.append(f"case_id:{case_id}")
        split_group = str(row.get("split_group") or "")
        if split_group in blocked_split_groups and row.get("source_type") != "trajectory":
            violations.append(f"split_group:{split_group}")
    return violations


def write_training_dataset(
    bundle: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    version = str(bundle["version"])
    manifest = dict(bundle["manifest"])
    export_dir = output_dir / version
    if dry_run:
        manifest["dry_run"] = True
        manifest["export_dir"] = str(export_dir)
        return manifest

    export_dir.mkdir(parents=True, exist_ok=True)
    sft_path = export_dir / "sft.jsonl"
    preference_path = export_dir / "preference.jsonl"
    manifest_path = export_dir / "manifest.json"

    with sft_path.open("w", encoding="utf-8") as handle:
        for row in bundle["sft_rows"]:
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
    with preference_path.open("w", encoding="utf-8") as handle:
        for row in bundle["preference_rows"]:
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")

    manifest["sft_path"] = str(sft_path)
    manifest["preference_path"] = str(preference_path)
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    return manifest


def export_training_dataset(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int = 5000,
    include_eval_fixtures: bool = True,
    include_seeds: bool = True,
    include_preference_seeds: bool = True,
    seeds_dir: Path = DEFAULT_SEEDS_DIR,
    preference_seeds_dir: Path = DEFAULT_PREFERENCE_SEEDS_DIR,
    manifest_path: Path = DEFAULT_BENCH_MANIFEST,
    export_version: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    bundle = build_training_dataset(
        limit=limit,
        include_eval_fixtures=include_eval_fixtures,
        include_seeds=include_seeds,
        include_preference_seeds=include_preference_seeds,
        seeds_dir=seeds_dir,
        preference_seeds_dir=preference_seeds_dir,
        manifest_path=manifest_path,
        export_version=export_version,
    )
    manifest = write_training_dataset(bundle, output_dir=output_dir, dry_run=dry_run)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export governed Talisman agent training datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="Export versioned SFT and preference datasets")
    export_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    export_parser.add_argument("--limit", type=int, default=5000)
    export_parser.add_argument("--manifest-path", type=Path, default=DEFAULT_BENCH_MANIFEST)
    export_parser.add_argument("--seeds-dir", type=Path, default=DEFAULT_SEEDS_DIR)
    export_parser.add_argument("--export-version", type=str, default=None)
    export_parser.add_argument("--dry-run", action="store_true")
    export_parser.add_argument("--no-eval-fixtures", action="store_true")
    export_parser.add_argument("--no-seeds", action="store_true")
    export_parser.add_argument("--no-preference-seeds", action="store_true")
    export_parser.add_argument("--preference-seeds-dir", type=Path, default=DEFAULT_PREFERENCE_SEEDS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command != "export":
        raise SystemExit(f"Unsupported command: {args.command}")
    manifest = export_training_dataset(
        output_dir=args.output_dir,
        limit=args.limit,
        include_eval_fixtures=not args.no_eval_fixtures,
        include_seeds=not args.no_seeds,
        include_preference_seeds=not args.no_preference_seeds,
        seeds_dir=args.seeds_dir,
        preference_seeds_dir=args.preference_seeds_dir,
        manifest_path=args.manifest_path,
        export_version=args.export_version,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
