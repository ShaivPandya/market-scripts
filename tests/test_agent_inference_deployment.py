from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_quality.agent_inference_deployment import (
    InferenceDeploymentError,
    build_deployment_manifest,
    resolve_deployment_candidate,
    validate_deployment_eligibility,
    validate_deployment_manifest,
    write_deployment_manifest,
)
from decision_quality.agent_model_training import (
    build_default_trainer_config,
    load_trainer_config,
    promote_candidate,
    register_candidate,
    smoke_train,
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


def _promoted_candidate(tmp_path: Path) -> tuple[str, Path, Path]:
    dataset_manifest = _write_dataset_bundle(tmp_path)
    config_path = _write_trainer_config(tmp_path, dataset_manifest)
    bench_report_path = tmp_path / "bench_pass.json"
    bench_report_path.write_text(json.dumps(_passing_bench_report()), encoding="utf-8")

    config = load_trainer_config(config_path)
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
    return registered["candidate_id"], registry_path, artifact_dir


def test_validate_deployment_eligibility_rejects_candidate_lifecycle(tmp_path):
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
    resolved_id, entry, manifest = resolve_deployment_candidate(
        candidate_id=registered["candidate_id"],
        registry_path=registry_path,
    )
    errors = validate_deployment_eligibility(entry, manifest, require_approved=True)
    assert any("approved" in error for error in errors)


def test_build_deployment_manifest_for_approved_candidate(tmp_path):
    candidate_id, registry_path, _artifact_dir = _promoted_candidate(tmp_path)
    manifest = build_deployment_manifest(
        candidate_id=candidate_id,
        registry_path=registry_path,
        combination_id="qwen-managed-gpu",
        environment="nonprod",
    )
    assert manifest["candidate_id"] == candidate_id
    assert manifest["combination_id"] == "qwen-managed-gpu"
    assert manifest["served_model_name"] == "qwen2.5-7b-instruct"
    assert manifest["model_tier_aliases"]["mid"] == "qwen2.5-7b-instruct"
    assert validate_deployment_manifest(manifest) == []


def test_build_deployment_manifest_rejects_disabled_candidate(tmp_path):
    candidate_id, registry_path, _artifact_dir = _promoted_candidate(tmp_path)
    from decision_quality.agent_model_training import disable_candidate

    disable_candidate(candidate_id, registry_path=registry_path)
    with pytest.raises(InferenceDeploymentError, match="approved"):
        build_deployment_manifest(candidate_id=candidate_id, registry_path=registry_path)


def test_build_deployment_manifest_rejects_digest_mismatch(tmp_path):
    candidate_id, registry_path, artifact_dir = _promoted_candidate(tmp_path)
    (artifact_dir / "adapter_config.json").write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(InferenceDeploymentError, match="digest mismatch"):
        build_deployment_manifest(candidate_id=candidate_id, registry_path=registry_path)


def test_write_deployment_manifest_creates_versioned_path(tmp_path):
    candidate_id, registry_path, _artifact_dir = _promoted_candidate(tmp_path)
    manifest = build_deployment_manifest(candidate_id=candidate_id, registry_path=registry_path)
    output_path = write_deployment_manifest(manifest, output_dir=tmp_path / "deployments")
    assert output_path.exists()
    assert output_path.name == f"{candidate_id}.json"
    assert "nonprod" in str(output_path)
