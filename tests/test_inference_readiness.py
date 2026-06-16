from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_quality.agent_inference_deployment import (
    InferenceDeploymentError,
    build_deployment_manifest,
)
from decision_quality.agent_model_training import (
    build_default_trainer_config,
    load_trainer_config,
    promote_candidate,
    register_candidate,
    smoke_train,
)
from decision_quality.inference_readiness import (
    READINESS_STATUS_NOT_READY,
    READINESS_STATUS_REFUSED,
    assess_readiness,
    build_vllm_serve_command,
    check_startup_eligibility,
    startup_check,
)


def _promoted_candidate(tmp_path: Path) -> tuple[str, Path, Path]:
    export_dir = tmp_path / "datasets" / "readiness"
    export_dir.mkdir(parents=True)
    sft_path = export_dir / "sft.jsonl"
    sft_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "example_id": "sft:synthetic:one",
                "source_type": "synthetic",
                "source_id": "seed-one",
                "task_class": "agent_turn",
                "messages": [{"role": "user", "content": "Review NVDA"}],
                "steps": [],
                "target_output": "ok",
                "split_group": "seed-one",
                "split": "train",
                "review_status": "released",
                "signal_source": "synthetic",
                "transformation_version": "agent_training_datasets_v1",
                "content_hash": "abc123",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "version": "readiness",
                "transformation_version": "agent_training_datasets_v1",
                "sft_count": 1,
                "preference_count": 0,
                "leakage_check_passed": True,
                "leakage_violations": [],
                "content_hashes": {"sft.jsonl": "hash-sft"},
                "sft_path": str(sft_path),
            }
        ),
        encoding="utf-8",
    )
    config = build_default_trainer_config(dataset_manifest_path=manifest_path)
    config_path = tmp_path / "trainer_config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    config = load_trainer_config(config_path)
    bench_report_path = tmp_path / "bench_pass.json"
    bench_report_path.write_text(
        json.dumps(
            {
                "benchmark_version": "test",
                "release_gate": {"passed": True, "hard_blocker_failures": [], "threshold_failures": []},
            }
        ),
        encoding="utf-8",
    )
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


def test_check_startup_eligibility_accepts_approved_candidate(tmp_path):
    candidate_id, registry_path, _artifact_dir = _promoted_candidate(tmp_path)
    manifest = build_deployment_manifest(candidate_id=candidate_id, registry_path=registry_path)
    result = check_startup_eligibility(
        deployment_manifest=manifest,
        registry_path=registry_path,
    )
    assert result["eligible"] is True
    assert result["candidate_id"] == candidate_id


def test_check_startup_eligibility_refuses_disabled_candidate(tmp_path):
    candidate_id, registry_path, _artifact_dir = _promoted_candidate(tmp_path)
    from decision_quality.agent_model_training import disable_candidate

    manifest = build_deployment_manifest(candidate_id=candidate_id, registry_path=registry_path)
    manifest_path = tmp_path / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    disable_candidate(candidate_id, registry_path=registry_path)
    with pytest.raises(InferenceDeploymentError):
        startup_check(deployment_manifest_path=manifest_path, registry_path=registry_path)


def test_assess_readiness_distinguishes_health_and_model_loaded(tmp_path):
    candidate_id, registry_path, _artifact_dir = _promoted_candidate(tmp_path)
    manifest = build_deployment_manifest(candidate_id=candidate_id, registry_path=registry_path)

    not_ready = assess_readiness(
        deployment_manifest=manifest,
        registry_path=registry_path,
        model_loaded=False,
    )
    assert not_ready["status"] == READINESS_STATUS_NOT_READY
    assert not_ready["ready"] is False

    ready = assess_readiness(
        deployment_manifest=manifest,
        registry_path=registry_path,
        model_loaded=True,
        served_model_aliases=[manifest["served_model_name"]],
    )
    assert ready["ready"] is True
    assert ready["identity"]["candidate_id"] == candidate_id


def test_assess_readiness_refuses_when_registry_invalid(tmp_path):
    candidate_id, registry_path, _artifact_dir = _promoted_candidate(tmp_path)
    manifest = build_deployment_manifest(candidate_id=candidate_id, registry_path=registry_path)
    from decision_quality.agent_model_training import disable_candidate

    disable_candidate(candidate_id, registry_path=registry_path)
    result = assess_readiness(
        deployment_manifest=manifest,
        registry_path=registry_path,
        model_loaded=True,
        served_model_aliases=[manifest["served_model_name"]],
    )
    assert result["status"] == READINESS_STATUS_REFUSED
    assert result["ready"] is False
    assert result["governance_errors"]


def test_build_vllm_serve_command_uses_manifest_serving_metadata(tmp_path):
    candidate_id, registry_path, _artifact_dir = _promoted_candidate(tmp_path)
    manifest = build_deployment_manifest(
        candidate_id=candidate_id,
        registry_path=registry_path,
        combination_id="qwen-managed-gpu",
    )
    command = build_vllm_serve_command(manifest)
    assert command[:3] == ["vllm", "serve", manifest["base_model_id"]]
    assert "--enable-auto-tool-choice" in command
    assert "--tool-call-parser" in command
    assert "hermes" in command
    assert "--max-model-len" in command
