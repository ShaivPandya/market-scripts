from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_quality.agent_model_training import (
    AgentModelTrainingError,
    TrainerConfig,
    build_default_trainer_config,
    build_model_card,
    deprecate_candidate,
    disable_candidate,
    load_trainer_config,
    promote_candidate,
    register_candidate,
    smoke_train,
    trainer_config_hash,
    validate_model_card,
    validate_promotion_evidence,
    validate_trainer_config,
)


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


def _write_trainer_config(tmp_path: Path, dataset_manifest: Path) -> Path:
    config = build_default_trainer_config(dataset_manifest_path=dataset_manifest)
    config_path = tmp_path / "trainer_config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return config_path


def _passing_bench_report() -> dict:
    return {
        "benchmark_version": "test",
        "release_gate": {"passed": True, "hard_blocker_failures": [], "threshold_failures": []},
        "hard_blockers": [{"id": "deterministic_failure", "passed": True}],
    }


def _failing_bench_report() -> dict:
    return {
        "benchmark_version": "test",
        "release_gate": {
            "passed": False,
            "hard_blocker_failures": ["deterministic_failure"],
            "threshold_failures": [],
        },
        "hard_blockers": [{"id": "deterministic_failure", "passed": False}],
    }


def test_trainer_config_hash_is_stable(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    first = build_default_trainer_config(dataset_manifest_path=dataset_manifest)
    second = build_default_trainer_config(dataset_manifest_path=dataset_manifest)
    assert trainer_config_hash(first) == trainer_config_hash(second)


def test_validate_trainer_config_rejects_missing_dataset(tmp_path):
    config = TrainerConfig(
        base_model_id="Qwen/Qwen2.5-7B-Instruct",
        dataset_manifest_path=str(tmp_path / "missing.json"),
    )
    errors = validate_trainer_config(config)
    assert any("does not exist" in error for error in errors)


def test_validate_trainer_config_rejects_leakage_failure(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    manifest["leakage_check_passed"] = False
    dataset_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    config = build_default_trainer_config(dataset_manifest_path=dataset_manifest)
    errors = validate_trainer_config(config)
    assert any("leakage_check_passed" in error for error in errors)


def test_smoke_train_produces_reproducible_artifacts(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path, version="smoke-v1")
    config_path = _write_trainer_config(tmp_path, dataset_manifest)
    config = load_trainer_config(config_path)

    first = smoke_train(config, output_dir=tmp_path / "artifacts", run_version="pinned-smoke")
    second = smoke_train(config, output_dir=tmp_path / "artifacts", run_version="pinned-smoke")

    assert first["candidate_id"] == second["candidate_id"]
    assert first["artifact_digest"] == second["artifact_digest"]
    assert Path(first["model_card_path"]).exists()


def test_model_card_contains_required_fields(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    config = build_default_trainer_config(dataset_manifest_path=dataset_manifest)
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    card = build_model_card(
        candidate_id="candidate-test",
        config=config,
        dataset_manifest=dataset,
        metrics={"train_rows": 1},
    )
    assert validate_model_card(card) == []


def test_register_candidate_writes_registry_entry(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    config_path = _write_trainer_config(tmp_path, dataset_manifest)
    config = load_trainer_config(config_path)
    train_result = smoke_train(config, output_dir=tmp_path / "artifacts")
    artifact_dir = Path(train_result["artifact_dir"])
    registry_path = tmp_path / "registry.json"

    registered = register_candidate(
        artifact_dir=artifact_dir,
        config_path=config_path,
        registry_path=registry_path,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered["candidate_id"] in registry["candidates"]
    assert registry["candidates"][registered["candidate_id"]]["lifecycle_state"] == "candidate"


def test_promotion_refuses_missing_bench_report(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    config_path = _write_trainer_config(tmp_path, dataset_manifest)
    config = load_trainer_config(config_path)
    artifact_dir = Path(smoke_train(config, output_dir=tmp_path / "artifacts")["artifact_dir"])
    registry_path = tmp_path / "registry.json"
    registered = register_candidate(
        artifact_dir=artifact_dir,
        config_path=config_path,
        registry_path=registry_path,
    )

    with pytest.raises(AgentModelTrainingError, match="release report"):
        promote_candidate(registered["candidate_id"], registry_path=registry_path)


def test_promotion_refuses_failed_release_gate(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    config_path = _write_trainer_config(tmp_path, dataset_manifest)
    config = load_trainer_config(config_path)
    bench_report_path = tmp_path / "bench_fail.json"
    bench_report_path.write_text(json.dumps(_failing_bench_report()), encoding="utf-8")

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

    with pytest.raises(AgentModelTrainingError, match="release gate failed"):
        promote_candidate(
            registered["candidate_id"],
            registry_path=registry_path,
            bench_report_path=bench_report_path,
        )


def test_promotion_succeeds_with_passing_bench_report(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    config_path = _write_trainer_config(tmp_path, dataset_manifest)
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

    promoted = promote_candidate(
        registered["candidate_id"],
        registry_path=registry_path,
        bench_report_path=bench_report_path,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert promoted["lifecycle_state"] == "approved"
    assert registry["active_candidate_id"] == registered["candidate_id"]


def test_deprecate_and_disable_clear_active_alias(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    config_path = _write_trainer_config(tmp_path, dataset_manifest)
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

    deprecate_candidate(registered["candidate_id"], registry_path=registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["active_candidate_id"] is None
    assert registry["candidates"][registered["candidate_id"]]["lifecycle_state"] == "deprecated"

    disable_candidate(registered["candidate_id"], registry_path=registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["candidates"][registered["candidate_id"]]["lifecycle_state"] == "disabled"


def test_validate_promotion_evidence_checks_artifact_digest(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path)
    config_path = _write_trainer_config(tmp_path, dataset_manifest)
    config = load_trainer_config(config_path)
    artifact_dir = Path(
        smoke_train(config, output_dir=tmp_path / "artifacts", run_version="digest-check")["artifact_dir"]
    )
    register_candidate(artifact_dir=artifact_dir, config_path=config_path, registry_path=tmp_path / "registry.json")
    manifest = json.loads((artifact_dir / "candidate_manifest.json").read_text(encoding="utf-8"))

    (artifact_dir / "adapter_config.json").write_text('{"tampered": true}', encoding="utf-8")
    errors = validate_promotion_evidence(manifest, bench_report_path=tmp_path / "missing.json")
    assert any("digest mismatch" in error for error in errors)


def test_end_to_end_smoke_pipeline(tmp_path):
    dataset_manifest = _write_dataset_bundle(tmp_path, version="e2e")
    config_path = _write_trainer_config(tmp_path, dataset_manifest)
    config = load_trainer_config(config_path)
    bench_report_path = tmp_path / "bench_pass.json"
    bench_report_path.write_text(json.dumps(_passing_bench_report()), encoding="utf-8")

    train_result = smoke_train(
        config,
        output_dir=tmp_path / "pipeline",
        bench_report_path=bench_report_path,
    )
    registry_path = tmp_path / "registry.json"
    registered = register_candidate(
        artifact_dir=Path(train_result["artifact_dir"]),
        config_path=config_path,
        bench_report_path=bench_report_path,
        registry_path=registry_path,
    )
    promoted = promote_candidate(
        registered["candidate_id"],
        registry_path=registry_path,
        bench_report_path=bench_report_path,
    )

    manifest = json.loads((Path(train_result["artifact_dir"]) / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["lifecycle_state"] == "approved"
    assert promoted["active_candidate_id"] == registered["candidate_id"]
