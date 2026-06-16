from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_quality.agent_model_training import (
    AgentModelTrainingError,
    build_default_trainer_config,
    build_model_card,
    load_trainer_config,
    promote_candidate,
    register_candidate,
    smoke_train,
    validate_promotion_evidence,
    validate_trainer_config,
)
from decision_quality.eval_corpus import compare_reports


def _write_dataset_bundle(
    tmp_path: Path,
    *,
    version: str = "test-dataset",
    preference_rows: list[dict] | None = None,
) -> Path:
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
    preference_rows = preference_rows or [
        {
            "schema_version": 1,
            "example_id": "pref:trajectory:one",
            "source_type": "trajectory",
            "source_id": "traj-one",
            "trajectory_id": "traj-one",
            "feedback_id": "fb-one",
            "response_version": "rv-one",
            "decision": "correct",
            "messages": [{"role": "user", "content": "Review NVDA"}],
            "steps": [],
            "chosen": "Better answer.",
            "rejected": "Worse answer.",
            "split_group": "traj-one",
            "split": "train",
            "review_status": "released",
            "signal_source": "human_reviewed",
            "failure_tags": ["synthesis"],
            "transformation_version": "agent_training_datasets_v1",
            "content_hash": "pref123",
        }
    ]

    sft_path = export_dir / "sft.jsonl"
    preference_path = export_dir / "preference.jsonl"
    with sft_path.open("w", encoding="utf-8") as handle:
        for row in sft_rows:
            handle.write(json.dumps(row) + "\n")
    with preference_path.open("w", encoding="utf-8") as handle:
        for row in preference_rows:
            handle.write(json.dumps(row) + "\n")

    manifest = {
        "manifest_version": 1,
        "version": version,
        "transformation_version": "agent_training_datasets_v1",
        "sft_count": len(sft_rows),
        "preference_count": len(preference_rows),
        "dpo_trainable_count": len([row for row in preference_rows if row.get("chosen")]),
        "dpo_incomplete_count": len([row for row in preference_rows if not row.get("chosen")]),
        "preference_reward_source_counts": {"human_reviewed": len(preference_rows)},
        "dpo_trainable_reward_source_counts": {
            "human_reviewed": len([row for row in preference_rows if row.get("chosen")])
        },
        "leakage_check_passed": True,
        "leakage_violations": [],
        "content_hashes": {
            "sft.jsonl": "hash-sft",
            "preference.jsonl": "hash-pref",
            "manifest.json": "hash-manifest",
        },
        "sft_path": str(sft_path),
        "preference_path": str(preference_path),
        "manifest_path": str(export_dir / "manifest.json"),
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _write_registry_with_parent(tmp_path: Path, *, parent_id: str = "parent-sft") -> Path:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "registry_version": 1,
                "active_candidate_id": parent_id,
                "active_artifact_path": "outputs/agent_model_training/parent",
                "candidates": {
                    parent_id: {
                        "candidate_id": parent_id,
                        "artifact_path": "outputs/agent_model_training/parent",
                        "artifact_digest": "parent-digest",
                        "lifecycle_state": "approved",
                        "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "training_method": "sft",
                        "dataset_version": "parent-dataset",
                        "created_at": "2026-06-07T00:00:00+00:00",
                    }
                },
                "updated_at": "2026-06-07T00:00:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return registry_path


def _passing_bench_report() -> dict:
    return {
        "benchmark_version": "test",
        "release_gate": {"passed": True, "hard_blocker_failures": [], "threshold_failures": []},
        "hard_blockers": [{"id": "deterministic_failure", "passed": True}],
        "cases": {
            "case_a": {"deterministic_passed": True},
        },
    }


def _failing_parent_comparison_report() -> dict:
    return {
        "benchmark_version": "test",
        "release_gate": {"passed": True, "hard_blocker_failures": [], "threshold_failures": []},
        "cases": {
            "case_a": {"deterministic_passed": False},
        },
    }


def test_validate_preference_config_requires_parent_and_dpo_rows(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    config = build_default_trainer_config(
        dataset_manifest_path=dataset_manifest,
        training_method="preference",
    )
    errors = validate_trainer_config(config, registry_path=tmp_path / "missing.json")
    assert any("parent_candidate_id" in error for error in errors)

    registry_path = _write_registry_with_parent(tmp_path)
    config = build_default_trainer_config(
        dataset_manifest_path=dataset_manifest,
        training_method="preference",
        parent_candidate_id="parent-sft",
    )
    assert validate_trainer_config(config, registry_path=registry_path) == []

    incomplete_manifest = _write_dataset_bundle(
        tmp_path,
        version="incomplete",
        preference_rows=[
            {
                "schema_version": 1,
                "example_id": "pref:reject-only",
                "source_type": "trajectory",
                "source_id": "traj-reject",
                "trajectory_id": "traj-reject",
                "feedback_id": "fb-reject",
                "response_version": "rv-reject",
                "decision": "reject",
                "messages": [{"role": "user", "content": "Review NVDA"}],
                "steps": [],
                "chosen": None,
                "rejected": "Bad answer.",
                "split_group": "traj-reject",
                "split": "train",
                "review_status": "released",
                "signal_source": "human_reviewed",
                "failure_tags": [],
                "transformation_version": "agent_training_datasets_v1",
                "content_hash": "reject-only",
            }
        ],
    )
    incomplete_config = build_default_trainer_config(
        dataset_manifest_path=incomplete_manifest,
        training_method="preference",
        parent_candidate_id="parent-sft",
    )
    errors = validate_trainer_config(incomplete_config, registry_path=registry_path)
    assert any("DPO-trainable" in error for error in errors)


def test_smoke_preference_train_records_parent_lineage(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    registry_path = _write_registry_with_parent(tmp_path)
    config = build_default_trainer_config(
        dataset_manifest_path=dataset_manifest,
        training_method="preference",
        parent_candidate_id="parent-sft",
    )
    config_path = tmp_path / "preference_config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    loaded = load_trainer_config(config_path)

    first = smoke_train(
        loaded,
        output_dir=tmp_path / "artifacts",
        run_version="pref-smoke",
        registry_path=registry_path,
    )
    second = smoke_train(
        loaded,
        output_dir=tmp_path / "artifacts",
        run_version="pref-smoke",
        registry_path=registry_path,
    )
    assert first["candidate_id"] == second["candidate_id"]

    model_card = json.loads(Path(first["model_card_path"]).read_text(encoding="utf-8"))
    assert model_card["training_method"] == "preference"
    assert model_card["parent_candidate_id"] == "parent-sft"
    assert model_card["dataset_lineage"]["dpo_trainable_count"] == 1


def test_model_card_parent_comparison_detects_regression():
    parent_report = _passing_bench_report()
    child_report = _failing_parent_comparison_report()
    card = build_model_card(
        candidate_id="pref-child",
        config=build_default_trainer_config(
            dataset_manifest_path=Path("outputs/agent_training_datasets/test/manifest.json"),
            training_method="preference",
            parent_candidate_id="parent-sft",
        ),
        dataset_manifest={
            "version": "test",
            "transformation_version": "agent_training_datasets_v1",
            "content_hashes": {},
            "preference_count": 1,
            "dpo_trainable_count": 1,
            "leakage_check_passed": True,
        },
        metrics={"train_rows": 1},
        bench_report=child_report,
        parent_bench_report=parent_report,
    )
    assert card["parent_bench_comparison"]["summary"]["regression_detected"] is True


def test_promotion_refuses_preference_candidate_without_parent_report(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    registry_path = _write_registry_with_parent(tmp_path)
    config = build_default_trainer_config(
        dataset_manifest_path=dataset_manifest,
        training_method="preference",
        parent_candidate_id="parent-sft",
    )
    config_path = tmp_path / "preference_config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    bench_report_path = tmp_path / "bench_pass.json"
    bench_report_path.write_text(json.dumps(_passing_bench_report()), encoding="utf-8")

    artifact_dir = Path(
        smoke_train(
            load_trainer_config(config_path),
            output_dir=tmp_path / "artifacts",
            bench_report_path=bench_report_path,
            registry_path=registry_path,
        )["artifact_dir"]
    )
    registered = register_candidate(
        artifact_dir=artifact_dir,
        config_path=config_path,
        bench_report_path=bench_report_path,
        registry_path=registry_path,
    )

    with pytest.raises(AgentModelTrainingError, match="SFT parent TalismanBench report"):
        promote_candidate(
            registered["candidate_id"],
            registry_path=registry_path,
            bench_report_path=bench_report_path,
        )


def test_promotion_refuses_parent_regression(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    registry_path = _write_registry_with_parent(tmp_path)
    config = build_default_trainer_config(
        dataset_manifest_path=dataset_manifest,
        training_method="preference",
        parent_candidate_id="parent-sft",
    )
    config_path = tmp_path / "preference_config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    parent_bench = tmp_path / "parent_bench.json"
    child_bench = tmp_path / "child_bench.json"
    parent_bench.write_text(json.dumps(_passing_bench_report()), encoding="utf-8")
    child_bench.write_text(json.dumps(_failing_parent_comparison_report()), encoding="utf-8")

    artifact_dir = Path(
        smoke_train(
            load_trainer_config(config_path),
            output_dir=tmp_path / "artifacts",
            bench_report_path=child_bench,
            parent_bench_report_path=parent_bench,
            registry_path=registry_path,
        )["artifact_dir"]
    )
    register_candidate(
        artifact_dir=artifact_dir,
        config_path=config_path,
        bench_report_path=child_bench,
        parent_bench_report_path=parent_bench,
        registry_path=registry_path,
    )
    manifest_path = artifact_dir / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_promotion_evidence(
        manifest,
        bench_report_path=child_bench,
        parent_bench_report_path=parent_bench,
    )
    assert any("regressed vs SFT parent" in error for error in errors)

    comparison = compare_reports(_passing_bench_report(), _failing_parent_comparison_report())
    assert comparison["summary"]["new_deterministic_failures"] == ["case_a"]
