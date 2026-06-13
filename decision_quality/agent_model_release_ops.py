"""Gated model refresh, monitoring, and retirement operations (TL-96)."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from decision_quality.agent_inference_deployment import (
    InferenceDeploymentError,
    validate_deployment_eligibility,
)
from decision_quality.agent_model_training import (
    DEFAULT_REGISTRY_PATH,
    AgentModelTrainingError,
    _load_candidate_manifest,
    _read_json,
    _resolve_path,
    _write_json,
    disable_candidate,
    load_registry,
    validate_promotion_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "model_release_ops"
DEFAULT_RELEASE_RECORDS_DIR = DEFAULT_OUTPUT_DIR / "release_records"
DEFAULT_RETIREMENT_RECORDS_DIR = DEFAULT_OUTPUT_DIR / "retirement_records"

RELEASE_OPS_VERSION = 1
RELEASE_RECORD_VERSION = 1
RETIREMENT_RECORD_VERSION = 1

ReleaseDecisionType = Literal["promotion_approved", "rollout_approved", "rollback", "retirement", "refresh_review"]
AlertSeverity = Literal["info", "warning", "critical"]

DEFAULT_FALLBACK_RATE_THRESHOLD = 0.15
DEFAULT_GATE_FAILURE_THRESHOLD = 3
DEFAULT_REVIEWED_FEEDBACK_THRESHOLD = 5


class AgentModelReleaseOpsError(ValueError):
    """Raised when release operations cannot complete safely."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def fallback_rate_threshold() -> float:
    return _env_float("AGENT_MODEL_RELEASE_FALLBACK_RATE_THRESHOLD", DEFAULT_FALLBACK_RATE_THRESHOLD)


def gate_failure_threshold() -> int:
    return _env_int("AGENT_MODEL_RELEASE_GATE_FAILURE_THRESHOLD", DEFAULT_GATE_FAILURE_THRESHOLD)


def reviewed_feedback_threshold() -> int:
    return _env_int("AGENT_MODEL_RELEASE_REVIEWED_FEEDBACK_THRESHOLD", DEFAULT_REVIEWED_FEEDBACK_THRESHOLD)


class ReleaseLineage(BaseModel):
    candidate_id: str | None = None
    active_candidate_id: str | None = None
    rollback_candidate_id: str | None = None
    dataset_manifest_path: str | None = None
    bench_report_path: str | None = None
    model_card_path: str | None = None
    deployment_manifest_path: str | None = None
    artifact_digest: str | None = None
    feedback_ids: list[str] = Field(default_factory=list)
    trajectory_ids: list[str] = Field(default_factory=list)


class DriftAlert(BaseModel):
    alert_id: str
    severity: AlertSeverity
    alert_type: str
    candidate_id: str | None = None
    model: str | None = None
    task_class: str | None = None
    fallback_reason: str | None = None
    metric: str | None = None
    observed_value: float | None = None
    threshold: float | None = None
    message: str
    recommended_action: Literal["refresh_review", "rollback_review", "monitor"] = "monitor"


class RefreshTrigger(BaseModel):
    trigger_id: str
    triggered: bool
    reason: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ReleaseDryRunReport(BaseModel):
    schema_version: int = RELEASE_OPS_VERSION
    report_id: str
    generated_at: str
    dry_run: bool = True
    registry_path: str
    active_candidate_id: str | None = None
    candidate_summaries: list[dict[str, Any]] = Field(default_factory=list)
    refresh_triggers: list[RefreshTrigger] = Field(default_factory=list)
    promotion_evidence_errors: dict[str, list[str]] = Field(default_factory=dict)
    deployment_validation_errors: dict[str, list[str]] = Field(default_factory=dict)
    rollout_monitoring: dict[str, Any] = Field(default_factory=dict)
    drift_alerts: list[DriftAlert] = Field(default_factory=list)
    lineage: ReleaseLineage = Field(default_factory=ReleaseLineage)
    ready_for_promotion: bool = False
    ready_for_rollout: bool = False
    blocking_issues: list[str] = Field(default_factory=list)


class ReleaseDecisionRecord(BaseModel):
    schema_version: int = RELEASE_RECORD_VERSION
    record_id: str
    decision_type: ReleaseDecisionType
    candidate_id: str
    approver: str
    approval_note: str
    created_at: str
    rollback_candidate_id: str | None = None
    rollout_target: dict[str, Any] = Field(default_factory=dict)
    bench_report_path: str | None = None
    model_card_path: str | None = None
    deployment_manifest_path: str | None = None
    lineage: ReleaseLineage = Field(default_factory=ReleaseLineage)
    dry_run: bool = False


class RetirementRecord(BaseModel):
    schema_version: int = RETIREMENT_RECORD_VERSION
    record_id: str
    candidate_id: str
    approver: str
    retirement_note: str
    created_at: str
    lifecycle_state: str = "disabled"
    lineage_preserved: bool = True
    serving_cleanup_checklist: list[str] = Field(default_factory=list)
    registry_updated: bool = False
    dry_run: bool = False


def _extract_rollout_meta(trajectory: dict[str, Any]) -> dict[str, Any] | None:
    for container in (trajectory.get("raw_payload"), trajectory.get("sanitized_payload")):
        if not isinstance(container, dict):
            continue
        direct = container.get("owned_model_rollout")
        if isinstance(direct, dict):
            return direct
        nested = container.get("raw_payload")
        if isinstance(nested, dict):
            rollout_meta = nested.get("owned_model_rollout")
            if isinstance(rollout_meta, dict):
                return rollout_meta
    return None


def summarize_rollout_monitoring(
    *,
    trajectory_limit: int = 500,
    lookback_hours: int = 168,
) -> dict[str, Any]:
    from api.agent_trajectories import list_trajectories

    cutoff = datetime.now(UTC) - timedelta(hours=max(1, lookback_hours))
    trajectories = list_trajectories(limit=trajectory_limit)
    recent = []
    for trajectory in trajectories:
        captured_at = trajectory.get("captured_at")
        if not captured_at:
            continue
        try:
            captured = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
        except ValueError:
            continue
        if captured >= cutoff:
            recent.append(trajectory)

    by_task_class: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_candidate: Counter[str] = Counter()
    by_fallback_reason: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    gate_failures = 0
    rollout_observed = 0

    for trajectory in recent:
        meta = _extract_rollout_meta(trajectory)
        if not meta:
            continue
        rollout_observed += 1
        task_class = str(meta.get("task_class") or trajectory.get("task_class") or "unknown")
        by_task_class[task_class] += 1
        model = str(trajectory.get("model") or meta.get("candidate_model") or "unknown")
        by_model[model] += 1
        candidate_id = str(meta.get("candidate_id") or "")
        if candidate_id:
            by_candidate[candidate_id] += 1
        mode = str(meta.get("mode") or "unknown")
        by_mode[mode] += 1
        fallback_reason = meta.get("fallback_reason")
        if fallback_reason:
            by_fallback_reason[str(fallback_reason)] += 1
        for gate in trajectory.get("gate_outcomes") or []:
            if isinstance(gate, dict) and gate.get("passed") is False:
                gate_failures += 1

    total_with_fallback = sum(by_fallback_reason.values())
    fallback_rate = (total_with_fallback / rollout_observed) if rollout_observed else 0.0
    return {
        "lookback_hours": lookback_hours,
        "trajectory_sample_size": len(recent),
        "rollout_observed_count": rollout_observed,
        "fallback_rate": fallback_rate,
        "gate_failure_count": gate_failures,
        "by_task_class": dict(by_task_class),
        "by_model": dict(by_model),
        "by_candidate_id": dict(by_candidate),
        "by_fallback_reason": dict(by_fallback_reason),
        "by_mode": dict(by_mode),
    }


def _count_recent_feedback(*, lookback_hours: int = 168) -> dict[str, Any]:
    from api.agent_response_feedback import list_feedback

    cutoff = datetime.now(UTC) - timedelta(hours=max(1, lookback_hours))
    rows = list_feedback(limit=500)
    recent = []
    failure_tags: Counter[str] = Counter()
    for row in rows:
        reviewed_at = row.get("reviewed_at")
        if not reviewed_at:
            continue
        try:
            reviewed = datetime.fromisoformat(str(reviewed_at).replace("Z", "+00:00"))
        except ValueError:
            continue
        if reviewed >= cutoff:
            recent.append(row)
            for tag in row.get("failure_tags") or []:
                failure_tags[str(tag)] += 1
    training_eligible = sum(1 for row in recent if row.get("training_eligible"))
    return {
        "reviewed_count": len(recent),
        "training_eligible_count": training_eligible,
        "failure_tag_counts": dict(failure_tags),
        "feedback_ids": [str(row.get("feedback_id")) for row in recent if row.get("feedback_id")],
    }


def assess_refresh_triggers(
    *,
    registry: dict[str, Any],
    rollout_monitoring: dict[str, Any],
    feedback_summary: dict[str, Any],
    scheduled_review: bool = False,
) -> list[RefreshTrigger]:
    triggers: list[RefreshTrigger] = []
    active_candidate_id = registry.get("active_candidate_id")

    reviewed_count = int(feedback_summary.get("reviewed_count") or 0)
    triggers.append(
        RefreshTrigger(
            trigger_id="new_reviewed_data",
            triggered=reviewed_count >= reviewed_feedback_threshold(),
            reason="Reviewed human feedback crossed threshold for refresh review",
            detail={"reviewed_count": reviewed_count, "threshold": reviewed_feedback_threshold()},
        )
    )

    failure_tags = feedback_summary.get("failure_tag_counts") or {}
    failure_cluster = sum(int(count) for count in failure_tags.values())
    triggers.append(
        RefreshTrigger(
            trigger_id="failure_clusters",
            triggered=failure_cluster >= gate_failure_threshold(),
            reason="Failure-tag clusters crossed threshold for refresh review",
            detail={"failure_tag_total": failure_cluster, "threshold": gate_failure_threshold()},
        )
    )

    fallback_rate = float(rollout_monitoring.get("fallback_rate") or 0.0)
    triggers.append(
        RefreshTrigger(
            trigger_id="rollout_fallback_drift",
            triggered=fallback_rate >= fallback_rate_threshold()
            and rollout_monitoring.get("rollout_observed_count", 0) > 0,
            reason="Owned-model fallback rate crossed monitoring threshold",
            detail={
                "fallback_rate": fallback_rate,
                "threshold": fallback_rate_threshold(),
                "candidate_id": active_candidate_id,
            },
        )
    )

    gate_failures = int(rollout_monitoring.get("gate_failure_count") or 0)
    triggers.append(
        RefreshTrigger(
            trigger_id="gate_regression",
            triggered=gate_failures >= gate_failure_threshold(),
            reason="Deterministic gate failures crossed monitoring threshold",
            detail={"gate_failure_count": gate_failures, "threshold": gate_failure_threshold()},
        )
    )

    triggers.append(
        RefreshTrigger(
            trigger_id="scheduled_review",
            triggered=scheduled_review,
            reason="Scheduled release review requested",
            detail={},
        )
    )

    if not active_candidate_id:
        triggers.append(
            RefreshTrigger(
                trigger_id="missing_active_candidate",
                triggered=True,
                reason="Registry has no active approved candidate",
                detail={},
            )
        )
    return triggers


def build_drift_alerts(
    *,
    rollout_monitoring: dict[str, Any],
    candidate_id: str | None,
) -> list[DriftAlert]:
    alerts: list[DriftAlert] = []
    fallback_rate = float(rollout_monitoring.get("fallback_rate") or 0.0)
    if rollout_monitoring.get("rollout_observed_count", 0) > 0 and fallback_rate >= fallback_rate_threshold():
        alerts.append(
            DriftAlert(
                alert_id="fallback_rate_spike",
                severity="critical" if fallback_rate >= min(1.0, fallback_rate_threshold() * 2) else "warning",
                alert_type="rollout_fallback_rate",
                candidate_id=candidate_id,
                metric="fallback_rate",
                observed_value=fallback_rate,
                threshold=fallback_rate_threshold(),
                message="Owned-model fallback rate exceeded monitoring threshold",
                recommended_action="rollback_review",
            )
        )

    gate_failures = int(rollout_monitoring.get("gate_failure_count") or 0)
    if gate_failures >= gate_failure_threshold():
        alerts.append(
            DriftAlert(
                alert_id="gate_failure_cluster",
                severity="critical",
                alert_type="gate_regression",
                candidate_id=candidate_id,
                metric="gate_failure_count",
                observed_value=float(gate_failures),
                threshold=float(gate_failure_threshold()),
                message="Deterministic gate failures exceeded monitoring threshold",
                recommended_action="rollback_review",
            )
        )

    raw_by_task_class = rollout_monitoring.get("by_task_class")
    by_task_class: dict[str, Any] = raw_by_task_class if isinstance(raw_by_task_class, dict) else {}
    for task_class, count in by_task_class.items():
        raw_fallback_reasons = rollout_monitoring.get("by_fallback_reason")
        fallback_reasons: dict[str, Any] = raw_fallback_reasons if isinstance(raw_fallback_reasons, dict) else {}
        if count >= gate_failure_threshold() and fallback_reasons:
            dominant_reason = max(fallback_reasons, key=lambda reason: fallback_reasons[reason])
            alerts.append(
                DriftAlert(
                    alert_id=f"task_class_fallback:{task_class}",
                    severity="warning",
                    alert_type="task_class_fallback_cluster",
                    candidate_id=candidate_id,
                    task_class=str(task_class),
                    fallback_reason=str(dominant_reason),
                    metric="task_class_observations",
                    observed_value=float(count),
                    threshold=float(gate_failure_threshold()),
                    message=f"Task class {task_class} shows elevated owned-model fallback activity",
                    recommended_action="refresh_review",
                )
            )
    return alerts


def _candidate_summary(candidate_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    artifact_path = _resolve_path(str(entry.get("artifact_path") or ""))
    manifest_path = artifact_path / "candidate_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    model_card_path = manifest.get("model_card_path")
    return {
        "candidate_id": candidate_id,
        "lifecycle_state": entry.get("lifecycle_state"),
        "artifact_path": entry.get("artifact_path"),
        "artifact_digest": entry.get("artifact_digest") or manifest.get("artifact_digest"),
        "training_method": manifest.get("training_method") or entry.get("training_method"),
        "bench_report_path": entry.get("bench_report_path") or manifest.get("bench_report_path"),
        "model_card_path": model_card_path,
        "approved_at": entry.get("approved_at") or manifest.get("approved_at"),
    }


def _load_gateway_rollout_policy() -> dict[str, Any]:
    try:
        from api.llm_settings import get_gateway_policy_setting

        policy = get_gateway_policy_setting()
        rollout = policy.get("owned_model_rollout")
        return rollout if isinstance(rollout, dict) else {}
    except Exception:
        return {}


def run_release_dry_run(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    candidate_id: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    scheduled_review: bool = False,
    lookback_hours: int = 168,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    active_candidate_id = registry.get("active_candidate_id")
    target_candidate_id = candidate_id or active_candidate_id
    candidates = registry.get("candidates") or {}

    candidate_summaries = [
        _candidate_summary(cid, entry) for cid, entry in sorted(candidates.items()) if isinstance(entry, dict)
    ]

    promotion_errors: dict[str, list[str]] = {}
    deployment_errors: dict[str, list[str]] = {}
    blocking_issues: list[str] = []
    lineage = ReleaseLineage(
        active_candidate_id=active_candidate_id,
        rollback_candidate_id=active_candidate_id if target_candidate_id != active_candidate_id else None,
        candidate_id=target_candidate_id,
    )

    if target_candidate_id:
        entry = candidates.get(target_candidate_id)
        if not isinstance(entry, dict):
            blocking_issues.append(f"Unknown candidate_id: {target_candidate_id}")
        else:
            lifecycle = str(entry.get("lifecycle_state") or "").lower()
            if lifecycle in {"disabled", "deprecated"}:
                blocking_issues.append(f"Candidate {target_candidate_id} lifecycle_state={lifecycle} blocks rollout")
            artifact_dir = _resolve_path(str(entry.get("artifact_path") or ""))
            if artifact_dir.exists():
                manifest = _load_candidate_manifest(artifact_dir)
                lineage.dataset_manifest_path = str((manifest.get("dataset_manifest") or {}).get("version"))
                lineage.bench_report_path = str(
                    entry.get("bench_report_path") or manifest.get("bench_report_path") or ""
                )
                lineage.model_card_path = str(manifest.get("model_card_path") or "")
                lineage.artifact_digest = str(manifest.get("artifact_digest") or entry.get("artifact_digest") or "")
                promotion_errors[target_candidate_id] = validate_promotion_evidence(manifest)
                deployment_errors[target_candidate_id] = validate_deployment_eligibility(entry, manifest)
            else:
                blocking_issues.append(f"Artifact path missing for candidate {target_candidate_id}")

    feedback_summary = _count_recent_feedback(lookback_hours=lookback_hours)
    lineage.feedback_ids = list(feedback_summary.get("feedback_ids") or [])

    rollout_monitoring = summarize_rollout_monitoring(lookback_hours=lookback_hours)
    drift_alerts = build_drift_alerts(
        rollout_monitoring=rollout_monitoring,
        candidate_id=target_candidate_id,
    )
    refresh_triggers = assess_refresh_triggers(
        registry=registry,
        rollout_monitoring=rollout_monitoring,
        feedback_summary=feedback_summary,
        scheduled_review=scheduled_review,
    )

    gateway_rollout = _load_gateway_rollout_policy()
    ready_for_promotion = bool(
        target_candidate_id
        and not promotion_errors.get(target_candidate_id)
        and str((candidates.get(target_candidate_id) or {}).get("lifecycle_state") or "") != "approved"
    )
    ready_for_rollout = bool(
        target_candidate_id
        and not promotion_errors.get(target_candidate_id)
        and not deployment_errors.get(target_candidate_id)
        and str((candidates.get(target_candidate_id) or {}).get("lifecycle_state") or "") == "approved"
        and not blocking_issues
    )

    if (
        gateway_rollout.get("approved_candidate_id")
        and gateway_rollout.get("approved_candidate_id") != target_candidate_id
    ):
        blocking_issues.append("Gateway rollout policy references a different approved_candidate_id")

    report = ReleaseDryRunReport(
        report_id=f"release_dry_run_{_now_tag()}",
        generated_at=_now_iso(),
        registry_path=str(registry_path),
        active_candidate_id=active_candidate_id,
        candidate_summaries=candidate_summaries,
        refresh_triggers=refresh_triggers,
        promotion_evidence_errors=promotion_errors,
        deployment_validation_errors=deployment_errors,
        rollout_monitoring=rollout_monitoring,
        drift_alerts=drift_alerts,
        lineage=lineage,
        ready_for_promotion=ready_for_promotion,
        ready_for_rollout=ready_for_rollout,
        blocking_issues=blocking_issues,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{report.report_id}.json"
    payload = report.model_dump(mode="json")
    _write_json(report_path, payload)
    payload["report_path"] = str(report_path)
    return payload


def record_release_decision(
    *,
    candidate_id: str,
    decision_type: ReleaseDecisionType,
    approver: str,
    approval_note: str,
    rollback_candidate_id: str | None = None,
    bench_report_path: Path | None = None,
    deployment_manifest_path: Path | None = None,
    rollout_target: dict[str, Any] | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    output_dir: Path = DEFAULT_RELEASE_RECORDS_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    approver_norm = str(approver or "").strip()
    note_norm = str(approval_note or "").strip()
    if not approver_norm:
        raise AgentModelReleaseOpsError("approver is required")
    if not note_norm:
        raise AgentModelReleaseOpsError("approval_note is required")

    registry = load_registry(registry_path)
    entry = (registry.get("candidates") or {}).get(candidate_id)
    if not isinstance(entry, dict):
        raise AgentModelReleaseOpsError(f"Unknown candidate_id: {candidate_id}")

    artifact_dir = _resolve_path(str(entry.get("artifact_path") or ""))
    manifest = _load_candidate_manifest(artifact_dir)
    evidence_errors = validate_promotion_evidence(
        manifest,
        bench_report_path=bench_report_path,
    )
    if decision_type in {"promotion_approved", "rollout_approved"} and evidence_errors:
        raise AgentModelReleaseOpsError("; ".join(evidence_errors))

    lifecycle = str(entry.get("lifecycle_state") or "").lower()
    if decision_type == "rollout_approved" and lifecycle != "approved":
        raise AgentModelReleaseOpsError(f"Candidate {candidate_id} must be approved before rollout approval")
    if lifecycle == "disabled":
        raise AgentModelReleaseOpsError(f"Disabled candidate {candidate_id} cannot receive release approval")

    record = ReleaseDecisionRecord(
        record_id=f"release_decision_{candidate_id}_{_now_tag()}",
        decision_type=decision_type,
        candidate_id=candidate_id,
        approver=approver_norm,
        approval_note=note_norm,
        created_at=_now_iso(),
        rollback_candidate_id=rollback_candidate_id or registry.get("active_candidate_id"),
        rollout_target=dict(rollout_target or {}),
        bench_report_path=str(bench_report_path) if bench_report_path else str(entry.get("bench_report_path") or ""),
        model_card_path=str(manifest.get("model_card_path") or ""),
        deployment_manifest_path=str(deployment_manifest_path) if deployment_manifest_path else None,
        lineage=ReleaseLineage(
            candidate_id=candidate_id,
            active_candidate_id=registry.get("active_candidate_id"),
            rollback_candidate_id=rollback_candidate_id or registry.get("active_candidate_id"),
            dataset_manifest_path=str((manifest.get("dataset_manifest") or {}).get("version")),
            bench_report_path=str(bench_report_path or entry.get("bench_report_path") or ""),
            model_card_path=str(manifest.get("model_card_path") or ""),
            deployment_manifest_path=str(deployment_manifest_path) if deployment_manifest_path else None,
            artifact_digest=str(manifest.get("artifact_digest") or ""),
        ),
        dry_run=dry_run,
    )

    if dry_run:
        return record.model_dump(mode="json")

    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / f"{record.record_id}.json"
    payload = record.model_dump(mode="json")
    _write_json(record_path, payload)
    payload["record_path"] = str(record_path)
    return payload


def retirement_cleanup_checklist(*, candidate_id: str) -> list[str]:
    return [
        f"Disable candidate in registry: python -m decision_quality.agent_model_training disable --candidate-id {candidate_id}",
        "Set gateway owned_model_rollout.enabled=false with gateway_note audit entry",
        "Set AGENT_OWNED_MODEL_ROLLOUT_KILL_SWITCH=true or provider_lifecycle.talisman=disabled for emergency break-glass",
        "Scale inference service to zero or redeploy prior deployment manifest",
        "Rotate TALISMAN_API_KEY secret version if the retired candidate was the only served alias",
        "Verify inference readiness returns refused for the retired candidate",
        "Retain artifact directory, model card, bench report, and release records for audit lineage",
    ]


def retire_candidate(
    *,
    candidate_id: str,
    approver: str,
    retirement_note: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    output_dir: Path = DEFAULT_RETIREMENT_RECORDS_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    approver_norm = str(approver or "").strip()
    note_norm = str(retirement_note or "").strip()
    if not approver_norm:
        raise AgentModelReleaseOpsError("approver is required")
    if not note_norm:
        raise AgentModelReleaseOpsError("retirement_note is required")

    registry = load_registry(registry_path)
    entry = (registry.get("candidates") or {}).get(candidate_id)
    if not isinstance(entry, dict):
        raise AgentModelReleaseOpsError(f"Unknown candidate_id: {candidate_id}")

    record = RetirementRecord(
        record_id=f"retirement_{candidate_id}_{_now_tag()}",
        candidate_id=candidate_id,
        approver=approver_norm,
        retirement_note=note_norm,
        created_at=_now_iso(),
        serving_cleanup_checklist=retirement_cleanup_checklist(candidate_id=candidate_id),
        registry_updated=False,
        dry_run=dry_run,
    )

    if dry_run:
        return record.model_dump(mode="json")

    disable_candidate(candidate_id, registry_path=registry_path)
    record.registry_updated = True
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / f"{record.record_id}.json"
    payload = record.model_dump(mode="json")
    _write_json(record_path, payload)
    payload["record_path"] = str(record_path)
    return payload


def run_scheduled_refresh_dry_run(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    del payload
    report = run_release_dry_run(scheduled_review=True)
    return {
        "job": "agent_model_release_refresh",
        "dry_run": True,
        "report_id": report.get("report_id"),
        "report_path": report.get("report_path"),
        "triggered_refresh_reviews": [
            trigger["trigger_id"]
            for trigger in report.get("refresh_triggers") or []
            if isinstance(trigger, dict) and trigger.get("triggered")
        ],
        "drift_alert_count": len(report.get("drift_alerts") or []),
        "blocking_issue_count": len(report.get("blocking_issues") or []),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talisman owned-model release operations (TL-96)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run_parser = subparsers.add_parser("dry-run", help="Run release workflow dry-run without production mutation")
    dry_run_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    dry_run_parser.add_argument("--candidate-id", default=None)
    dry_run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    dry_run_parser.add_argument("--scheduled-review", action="store_true")
    dry_run_parser.add_argument("--lookback-hours", type=int, default=168)

    record_parser = subparsers.add_parser("record-decision", help="Record an immutable human release decision")
    record_parser.add_argument("--candidate-id", required=True)
    record_parser.add_argument(
        "--decision-type",
        choices=["promotion_approved", "rollout_approved", "rollback", "retirement", "refresh_review"],
        required=True,
    )
    record_parser.add_argument("--approver", required=True)
    record_parser.add_argument("--approval-note", required=True)
    record_parser.add_argument("--rollback-candidate-id", default=None)
    record_parser.add_argument("--bench-report", type=Path, default=None)
    record_parser.add_argument("--deployment-manifest", type=Path, default=None)
    record_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    record_parser.add_argument("--output-dir", type=Path, default=DEFAULT_RELEASE_RECORDS_DIR)
    record_parser.add_argument("--dry-run", action="store_true")

    retire_parser = subparsers.add_parser("retire", help="Disable a candidate and record retirement metadata")
    retire_parser.add_argument("--candidate-id", required=True)
    retire_parser.add_argument("--approver", required=True)
    retire_parser.add_argument("--retirement-note", required=True)
    retire_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    retire_parser.add_argument("--output-dir", type=Path, default=DEFAULT_RETIREMENT_RECORDS_DIR)
    retire_parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "dry-run":
            result = run_release_dry_run(
                registry_path=args.registry,
                candidate_id=args.candidate_id,
                output_dir=args.output_dir,
                scheduled_review=args.scheduled_review,
                lookback_hours=args.lookback_hours,
            )
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0

        if args.command == "record-decision":
            result = record_release_decision(
                candidate_id=args.candidate_id,
                decision_type=args.decision_type,
                approver=args.approver,
                approval_note=args.approval_note,
                rollback_candidate_id=args.rollback_candidate_id,
                bench_report_path=args.bench_report,
                deployment_manifest_path=args.deployment_manifest,
                registry_path=args.registry,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0

        if args.command == "retire":
            result = retire_candidate(
                candidate_id=args.candidate_id,
                approver=args.approver,
                retirement_note=args.retirement_note,
                registry_path=args.registry,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0

        raise AgentModelReleaseOpsError(f"Unsupported command: {args.command}")
    except (AgentModelReleaseOpsError, AgentModelTrainingError, InferenceDeploymentError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
