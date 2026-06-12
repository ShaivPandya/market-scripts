from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_quality.agent_model_release_ops import (
    AgentModelReleaseOpsError,
    assess_refresh_triggers,
    build_drift_alerts,
    record_release_decision,
    retire_candidate,
    run_release_dry_run,
    summarize_rollout_monitoring,
)


@pytest.fixture(autouse=True)
def _isolate_trajectory_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setenv("TALISMAN_ALLOW_SQLITE_STATE", "true")
    monkeypatch.setenv("DATABASE_URL", "")
    from api import agent_trajectories

    monkeypatch.setattr(agent_trajectories, "_SQLITE_PATH", tmp_path / "agent_trajectories.sqlite3")
    yield
    agent_trajectories.reset_agent_trajectory_store_for_tests()


def _write_dataset_bundle(tmp_path: Path, *, version: str = "test-dataset") -> Path:
    export_dir = tmp_path / "datasets" / version
    export_dir.mkdir(parents=True)
    sft_rows = [
        {
            "schema_version": 1,
            "example_id": "sft:synthetic:one",
            "source_type": "synthetic",
            "source_id": "seed-one",
            "task_class": "agent_turn",
            "messages": [
                {"role": "user", "content": "Review NVDA"},
                {"role": "assistant", "content": "NVDA looks constructive."},
            ],
            "steps": [],
            "target_output": "NVDA looks constructive.",
            "split_group": "seed-one",
            "split": "train",
            "review_status": "released",
            "signal_source": "synthetic",
            "transformation_version": "agent_training_datasets_v1",
            "content_hash": "abc123",
        }
    ]
    sft_path = export_dir / "sft.jsonl"
    with sft_path.open("w", encoding="utf-8") as handle:
        for row in sft_rows:
            handle.write(json.dumps(row) + "\n")

    manifest = {
        "manifest_version": 1,
        "version": version,
        "transformation_version": "agent_training_datasets_v1",
        "sft_count": 1,
        "preference_count": 0,
        "leakage_check_passed": True,
        "leakage_violations": [],
        "content_hashes": {"sft.jsonl": "hash-sft", "preference.jsonl": "hash-pref", "manifest.json": "hash-manifest"},
        "sft_path": str(sft_path),
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _passing_bench_report() -> dict:
    return {
        "benchmark_version": "test",
        "release_gate": {"passed": True, "hard_blocker_failures": [], "threshold_failures": []},
        "hard_blockers": [{"id": "deterministic_failure", "passed": True}],
    }


def _approved_candidate(tmp_path: Path) -> tuple[str, Path]:
    from decision_quality.agent_model_training import (
        build_default_trainer_config,
        load_trainer_config,
        promote_candidate,
        register_candidate,
        smoke_train,
    )

    dataset_manifest = _write_dataset_bundle(tmp_path)
    config = build_default_trainer_config(dataset_manifest_path=dataset_manifest)
    config_path = tmp_path / "trainer_config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    config = load_trainer_config(config_path)
    bench_report_path = tmp_path / "bench_pass.json"
    bench_report_path.write_text(json.dumps(_passing_bench_report()), encoding="utf-8")
    artifact_dir = Path(
        smoke_train(config, output_dir=tmp_path / "artifacts", bench_report_path=bench_report_path)["artifact_dir"]
    )
    registry_path = tmp_path / "registry.json"
    registered = register_candidate(
        artifact_dir=artifact_dir,
        config_path=config_path,
        bench_report_path=bench_report_path,
        registry_path=registry_path,
    )
    promote_candidate(
        registered["candidate_id"],
        registry_path=registry_path,
        bench_report_path=bench_report_path,
    )
    return registered["candidate_id"], registry_path


@pytest.fixture(autouse=True)
def _isolate_trajectory_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setenv("TALISMAN_ALLOW_SQLITE_STATE", "true")
    monkeypatch.setenv("DATABASE_URL", "")
    from api import agent_trajectories

    monkeypatch.setattr(agent_trajectories, "_SQLITE_PATH", tmp_path / "agent_trajectories.sqlite3")
    yield
    agent_trajectories.reset_agent_trajectory_store_for_tests()


def test_run_release_dry_run_writes_report_without_mutating_registry(tmp_path, monkeypatch):
    candidate_id, registry_path = _approved_candidate(tmp_path)
    monkeypatch.setattr(
        "decision_quality.agent_model_release_ops._count_recent_feedback",
        lambda **kwargs: {
            "reviewed_count": 0,
            "training_eligible_count": 0,
            "failure_tag_counts": {},
            "feedback_ids": [],
        },
    )
    monkeypatch.setattr(
        "decision_quality.agent_model_release_ops.summarize_rollout_monitoring",
        lambda **kwargs: {
            "lookback_hours": 168,
            "trajectory_sample_size": 0,
            "rollout_observed_count": 0,
            "fallback_rate": 0.0,
            "gate_failure_count": 0,
            "by_task_class": {},
            "by_model": {},
            "by_candidate_id": {},
            "by_fallback_reason": {},
            "by_mode": {},
        },
    )

    before = json.loads(registry_path.read_text(encoding="utf-8"))
    report = run_release_dry_run(
        registry_path=registry_path,
        candidate_id=candidate_id,
        output_dir=tmp_path / "release_ops",
    )
    after = json.loads(registry_path.read_text(encoding="utf-8"))

    assert before == after
    assert report["dry_run"] is True
    assert report["candidate_summaries"]
    assert Path(report["report_path"]).exists()
    assert report["ready_for_rollout"] is True
    assert candidate_id in report["promotion_evidence_errors"]
    assert report["promotion_evidence_errors"][candidate_id] == []


def test_record_release_decision_rejects_missing_evidence(tmp_path):
    from decision_quality.agent_model_training import (
        build_default_trainer_config,
        load_trainer_config,
        register_candidate,
        smoke_train,
    )

    dataset_manifest = _write_dataset_bundle(tmp_path)
    config = build_default_trainer_config(dataset_manifest_path=dataset_manifest)
    config_path = tmp_path / "trainer_config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    config = load_trainer_config(config_path)
    artifact_dir = Path(smoke_train(config, output_dir=tmp_path / "artifacts")["artifact_dir"])
    registry_path = tmp_path / "registry.json"
    registered = register_candidate(
        artifact_dir=artifact_dir,
        config_path=config_path,
        registry_path=registry_path,
    )

    with pytest.raises(AgentModelReleaseOpsError, match="release report"):
        record_release_decision(
            candidate_id=registered["candidate_id"],
            decision_type="promotion_approved",
            approver="operator@test",
            approval_note="Attempt promotion without bench evidence",
            registry_path=registry_path,
        )


def test_record_release_decision_dry_run_does_not_write_file(tmp_path):
    candidate_id, registry_path = _approved_candidate(tmp_path)
    bench_report_path = tmp_path / "bench_pass.json"
    result = record_release_decision(
        candidate_id=candidate_id,
        decision_type="rollout_approved",
        approver="operator@test",
        approval_note="Approved for shadow burn-in",
        bench_report_path=bench_report_path,
        registry_path=registry_path,
        output_dir=tmp_path / "release_records",
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert "record_path" not in result
    assert not list((tmp_path / "release_records").glob("*.json"))


def test_retire_candidate_disables_registry_entry(tmp_path):
    candidate_id, registry_path = _approved_candidate(tmp_path)
    result = retire_candidate(
        candidate_id=candidate_id,
        approver="operator@test",
        retirement_note="Retire after rollback drill",
        registry_path=registry_path,
        output_dir=tmp_path / "retirement_records",
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["active_candidate_id"] is None
    assert registry["candidates"][candidate_id]["lifecycle_state"] == "disabled"
    assert result["registry_updated"] is True
    assert Path(result["record_path"]).exists()
    assert result["lineage_preserved"] is True


def test_build_drift_alerts_flags_fallback_and_gate_failures(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_RELEASE_FALLBACK_RATE_THRESHOLD", "0.10")
    monkeypatch.setenv("AGENT_MODEL_RELEASE_GATE_FAILURE_THRESHOLD", "2")
    rollout_monitoring = {
        "rollout_observed_count": 10,
        "fallback_rate": 0.25,
        "gate_failure_count": 3,
        "by_task_class": {"synthesis": 4},
        "by_fallback_reason": {"endpoint_failure": 2},
    }
    alerts = build_drift_alerts(rollout_monitoring=rollout_monitoring, candidate_id="abc123")
    assert any(alert.alert_type == "rollout_fallback_rate" for alert in alerts)
    assert any(alert.alert_type == "gate_regression" for alert in alerts)


def test_assess_refresh_triggers_marks_missing_active_candidate():
    triggers = assess_refresh_triggers(
        registry={"active_candidate_id": None, "candidates": {}},
        rollout_monitoring={"fallback_rate": 0.0, "gate_failure_count": 0, "rollout_observed_count": 0},
        feedback_summary={"reviewed_count": 0, "failure_tag_counts": {}},
    )
    assert any(trigger.trigger_id == "missing_active_candidate" and trigger.triggered for trigger in triggers)


def test_summarize_rollout_monitoring_reads_trajectory_payload():
    from api.agent_trajectories import insert_trajectory

    trajectory_id = insert_trajectory(
        {
            "session_id": "sess-rollout",
            "client_turn_id": "turn-rollout",
            "final_disposition": "succeeded",
            "provider": "talisman",
            "model": "qwen-test",
            "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            "steps": [
                {
                    "step_id": "step-0",
                    "index": 0,
                    "kind": "final",
                    "name": "assistant",
                    "status": "ok",
                    "payload": {},
                }
            ],
            "raw_payload": {
                "owned_model_rollout": {
                    "mode": "shadow",
                    "task_class": "synthesis",
                    "candidate_id": "abc123",
                    "fallback_reason": "endpoint_failure",
                }
            },
        }
    )
    assert trajectory_id
    summary = summarize_rollout_monitoring(trajectory_limit=10, lookback_hours=168)
    assert summary["rollout_observed_count"] == 1
    assert summary["by_task_class"]["synthesis"] == 1
    assert summary["by_fallback_reason"]["endpoint_failure"] == 1
