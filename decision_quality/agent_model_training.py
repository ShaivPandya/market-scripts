"""Reproducible SFT/LoRA and preference-optimization training for Talisman agent models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "agent_model_training"
DEFAULT_MODEL_DIR = ROOT / "data" / "agent_model_candidates"
DEFAULT_REGISTRY_PATH = DEFAULT_MODEL_DIR / "registry.json"
DEFAULT_CANDIDATE_MATRIX = ROOT / "docs" / "talisman_bench" / "candidate_matrix.json"

CONFIG_SCHEMA_VERSION = 1
CANDIDATE_MANIFEST_VERSION = 1
MODEL_CARD_VERSION = 1
TRAINER_VERSION = "agent_model_training_v1"
PREFERENCE_TRAINER_VERSION = "agent_preference_training_v1"

LifecycleState = Literal["candidate", "approved", "deprecated", "disabled"]
TrainerBackend = Literal["smoke", "trl", "peft"]
TrainingMethod = Literal["sft", "preference"]
PreferenceAlgorithm = Literal["smoke", "dpo"]

REQUIRED_MODEL_CARD_KEYS = (
    "model_card_version",
    "candidate_id",
    "base_model_id",
    "dataset_lineage",
    "intended_task_classes",
    "limitations",
    "known_failures",
    "license",
    "metrics",
)

PROMOTION_REQUIRED_MANIFEST_KEYS = (
    "manifest_version",
    "candidate_id",
    "artifact_digest",
    "base_model_id",
    "dataset_manifest",
    "trainer_config_hash",
    "artifact_path",
    "model_card_path",
    "lifecycle_state",
)


class AgentModelTrainingError(ValueError):
    """Raised when training or registry operations cannot complete safely."""


class LoraConfig(BaseModel):
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    use_qlora: bool = False


class TrainingHyperparameters(BaseModel):
    epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    seed: int = 42
    warmup_ratio: float = 0.03


class ServeMetadata(BaseModel):
    served_model_name: str | None = None
    combination_id: str | None = None
    endpoint_protocol: str = "openai_compatible"


class TrainerConfig(BaseModel):
    schema_version: int = CONFIG_SCHEMA_VERSION
    trainer_version: str = TRAINER_VERSION
    training_method: TrainingMethod = "sft"
    base_model_id: str
    base_model_revision: str | None = None
    dataset_manifest_path: str
    chat_template: str = "qwen2.5"
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    trainer_backend: TrainerBackend = "smoke"
    preference_algorithm: PreferenceAlgorithm = "smoke"
    parent_candidate_id: str | None = None
    code_revision: str | None = None
    serve: ServeMetadata = Field(default_factory=ServeMetadata)

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: int) -> int:
        if value != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported trainer config schema version: {value}")
        return value


class CandidateManifest(BaseModel):
    manifest_version: int = CANDIDATE_MANIFEST_VERSION
    candidate_id: str
    artifact_digest: str
    base_model_id: str
    base_model_revision: str | None = None
    training_method: TrainingMethod = "sft"
    parent_candidate_id: str | None = None
    dataset_manifest: dict[str, Any]
    trainer_config_hash: str
    config_path: str
    artifact_path: str
    model_card_path: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    bench_report_path: str | None = None
    parent_bench_report_path: str | None = None
    lifecycle_state: LifecycleState = "candidate"
    created_at: str
    trainer_version: str = TRAINER_VERSION


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _stable_hash(value: Any, *, length: int = 32) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AgentModelTrainingError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("TALISMAN_CODE_REVISION")


IMMUTABLE_ARTIFACT_FILES = ("adapter_config.json", "metrics.json", "model_card.json")


def _artifact_digest(artifact_dir: Path, *, files: tuple[str, ...] = IMMUTABLE_ARTIFACT_FILES) -> str:
    if not artifact_dir.exists():
        raise AgentModelTrainingError(f"Artifact path does not exist: {artifact_dir}")
    digest = hashlib.sha256()
    for name in files:
        path = artifact_dir / name
        if not path.exists():
            raise AgentModelTrainingError(f"Missing immutable artifact file: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_trainer_config(path: Path) -> TrainerConfig:
    return TrainerConfig.model_validate(_read_json(path))


def _load_parent_candidate(
    parent_candidate_id: str,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    candidates = registry.get("candidates")
    entry = candidates.get(parent_candidate_id) if isinstance(candidates, dict) else None
    if not isinstance(entry, dict):
        raise AgentModelTrainingError(f"Unknown parent_candidate_id: {parent_candidate_id}")
    if entry.get("lifecycle_state") != "approved":
        raise AgentModelTrainingError(
            f"parent_candidate_id {parent_candidate_id} must be approved before preference training"
        )
    return entry


def validate_trainer_config(
    config: TrainerConfig,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> list[str]:
    errors: list[str] = []
    dataset_path = _resolve_path(config.dataset_manifest_path)
    if not dataset_path.exists():
        errors.append(f"dataset manifest does not exist: {dataset_path}")
        return errors

    dataset_manifest = _read_json(dataset_path)
    if not dataset_manifest.get("leakage_check_passed", False):
        errors.append("dataset manifest leakage_check_passed must be true")

    if config.training_method == "sft":
        if int(dataset_manifest.get("sft_count") or 0) < 1:
            errors.append("dataset manifest must include at least one SFT example")
        return errors

    if config.training_method != "preference":
        errors.append(f"Unsupported training_method: {config.training_method}")
        return errors

    if not config.parent_candidate_id:
        errors.append("parent_candidate_id is required for preference training")
    else:
        try:
            _load_parent_candidate(config.parent_candidate_id, registry_path=registry_path)
        except AgentModelTrainingError as exc:
            errors.append(str(exc))

    dpo_trainable_count = int(dataset_manifest.get("dpo_trainable_count") or 0)
    if dpo_trainable_count < 1:
        errors.append("dataset manifest must include at least one DPO-trainable preference example")

    return errors


def trainer_config_hash(config: TrainerConfig) -> str:
    payload = config.model_dump(mode="json")
    payload.pop("code_revision", None)
    return _stable_hash(payload)


def load_dataset_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if not manifest.get("leakage_check_passed", False):
        raise AgentModelTrainingError("Dataset manifest failed leakage checks")
    return manifest


def build_model_card(
    *,
    candidate_id: str,
    config: TrainerConfig,
    dataset_manifest: dict[str, Any],
    metrics: dict[str, Any],
    bench_report: dict[str, Any] | None = None,
    parent_bench_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    limitations = [
        "Smoke-trained candidates are for pipeline validation only.",
        "Production routing requires TalismanBench release evidence and approved registry promotion.",
    ]
    if config.training_method == "preference":
        limitations.append(
            "Preference-trained candidates must improve targeted dimensions without new hard-blocker regressions."
        )
    known_failures: list[str] = []
    if bench_report:
        blockers = [
            item.get("id")
            for item in bench_report.get("hard_blockers") or []
            if isinstance(item, dict) and not item.get("passed")
        ]
        if blockers:
            known_failures.extend(str(item) for item in blockers)

    parent_comparison = None
    if bench_report and parent_bench_report:
        from decision_quality.eval_corpus import compare_reports

        parent_comparison = compare_reports(parent_bench_report, bench_report)

    return {
        "model_card_version": MODEL_CARD_VERSION,
        "candidate_id": candidate_id,
        "base_model_id": config.base_model_id,
        "base_model_revision": config.base_model_revision,
        "trainer_version": config.trainer_version,
        "trainer_backend": config.trainer_backend,
        "training_method": config.training_method,
        "preference_algorithm": config.preference_algorithm if config.training_method == "preference" else None,
        "parent_candidate_id": config.parent_candidate_id,
        "dataset_lineage": {
            "version": dataset_manifest.get("version"),
            "transformation_version": dataset_manifest.get("transformation_version"),
            "content_hashes": dataset_manifest.get("content_hashes"),
            "sft_count": dataset_manifest.get("sft_count"),
            "preference_count": dataset_manifest.get("preference_count"),
            "dpo_trainable_count": dataset_manifest.get("dpo_trainable_count"),
            "preference_reward_source_counts": dataset_manifest.get("preference_reward_source_counts"),
            "dpo_trainable_reward_source_counts": dataset_manifest.get("dpo_trainable_reward_source_counts"),
            "leakage_check_passed": dataset_manifest.get("leakage_check_passed"),
        },
        "intended_task_classes": ["agent_turn", "routing", "tool_use", "structured_output"],
        "limitations": limitations,
        "known_failures": known_failures,
        "license": "Derived from base model license; see base_model_id.",
        "metrics": metrics,
        "bench_report_summary": (
            {
                "release_gate_passed": (bench_report.get("release_gate") or {}).get("passed"),
                "benchmark_version": bench_report.get("benchmark_version"),
            }
            if bench_report
            else None
        ),
        "parent_bench_comparison": parent_comparison,
        "serve": config.serve.model_dump(mode="json"),
        "created_at": _now_iso(),
    }


def validate_model_card(card: dict[str, Any]) -> list[str]:
    return [f"model card missing required key: {key}" for key in REQUIRED_MODEL_CARD_KEYS if key not in card]


def _dataset_path_from_manifest(dataset_manifest: dict[str, Any], *, key: str, fallback_name: str) -> Path:
    dataset_dir = _resolve_path(str(dataset_manifest.get("manifest_path") or "")).parent
    raw_path = dataset_manifest.get(key) or dataset_dir / fallback_name
    path = Path(str(raw_path))
    return path if path.is_absolute() else _resolve_path(str(path))


def _smoke_train_rows(dataset_manifest: dict[str, Any], *, sft_path: Path) -> list[dict[str, Any]]:
    if not sft_path.exists():
        raise AgentModelTrainingError(f"SFT dataset missing: {sft_path}")
    rows = [json.loads(line) for line in sft_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = [row for row in rows if str(row.get("split") or "") in {"train", "validation"}]
    return train_rows or rows


def _smoke_preference_rows(dataset_manifest: dict[str, Any], *, preference_path: Path) -> list[dict[str, Any]]:
    from decision_quality.agent_training_datasets import filter_dpo_trainable_preference_rows

    if not preference_path.exists():
        raise AgentModelTrainingError(f"Preference dataset missing: {preference_path}")
    rows = [json.loads(line) for line in preference_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    trainable_rows, _incomplete = filter_dpo_trainable_preference_rows(rows)
    train_rows = [row for row in trainable_rows if str(row.get("split") or "") in {"train", "validation"}]
    return train_rows or trainable_rows


def smoke_train(
    config: TrainerConfig,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    bench_report_path: Path | None = None,
    parent_bench_report_path: Path | None = None,
    run_version: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Produce a deterministic smoke artifact directory without GPU training."""
    errors = validate_trainer_config(config, registry_path=registry_path)
    if errors:
        raise AgentModelTrainingError("; ".join(errors))

    dataset_path = _resolve_path(config.dataset_manifest_path)
    dataset_manifest = load_dataset_manifest(dataset_path)
    if config.training_method == "preference":
        preference_path = _dataset_path_from_manifest(
            dataset_manifest,
            key="preference_path",
            fallback_name="preference.jsonl",
        )
        train_rows = _smoke_preference_rows(dataset_manifest, preference_path=preference_path)
        parent_entry = _load_parent_candidate(str(config.parent_candidate_id), registry_path=registry_path)
    else:
        sft_path = _dataset_path_from_manifest(dataset_manifest, key="sft_path", fallback_name="sft.jsonl")
        train_rows = _smoke_train_rows(dataset_manifest, sft_path=sft_path)
        parent_entry = None

    reward_source_counts = dict(Counter(str(row.get("signal_source") or "unknown") for row in train_rows))
    metrics = {
        "backend": "smoke",
        "training_method": config.training_method,
        "preference_algorithm": config.preference_algorithm if config.training_method == "preference" else None,
        "train_rows": len(train_rows),
        "reward_source_counts": reward_source_counts,
        "loss": 0.0,
        "eval_loss": 0.0,
        "seed": config.training.seed,
        "trainer_config_hash": trainer_config_hash(config),
        "parent_candidate_id": config.parent_candidate_id,
        "parent_artifact_path": parent_entry.get("artifact_path") if parent_entry else None,
    }

    version = run_version or _now_tag()
    created_at = f"{version}T00:00:00+00:00" if run_version else _now_iso()
    artifact_dir = output_dir / version
    if artifact_dir.exists() and run_version:
        for child in artifact_dir.iterdir():
            if child.is_file():
                child.unlink()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = {
        "base_model_id": config.base_model_id,
        "base_model_revision": config.base_model_revision,
        "lora": config.lora.model_dump(mode="json"),
        "training": config.training.model_dump(mode="json"),
        "chat_template": config.chat_template,
        "trainer_backend": config.trainer_backend,
        "trainer_version": config.trainer_version,
        "training_method": config.training_method,
        "preference_algorithm": config.preference_algorithm,
        "parent_candidate_id": config.parent_candidate_id,
        "dataset_version": dataset_manifest.get("version"),
        "dataset_content_hashes": dataset_manifest.get("content_hashes"),
        "code_revision": config.code_revision or _git_revision(),
    }
    _write_json(artifact_dir / "adapter_config.json", adapter_config)
    _write_json(artifact_dir / "metrics.json", metrics)

    bench_report = _read_json(bench_report_path) if bench_report_path and bench_report_path.exists() else None
    parent_bench_report = (
        _read_json(parent_bench_report_path) if parent_bench_report_path and parent_bench_report_path.exists() else None
    )
    candidate_id = _stable_hash(
        {
            "trainer_config_hash": metrics["trainer_config_hash"],
            "dataset_hashes": dataset_manifest.get("content_hashes"),
            "trainer_version": config.trainer_version,
            "trainer_backend": config.trainer_backend,
            "training_method": config.training_method,
            "parent_candidate_id": config.parent_candidate_id,
        },
        length=16,
    )
    model_card = build_model_card(
        candidate_id=candidate_id,
        config=config,
        dataset_manifest=dataset_manifest,
        metrics=metrics,
        bench_report=bench_report,
        parent_bench_report=parent_bench_report,
    )
    model_card["created_at"] = created_at
    _write_json(artifact_dir / "model_card.json", model_card)

    artifact_digest = _artifact_digest(artifact_dir)
    training_manifest = {
        "artifact_digest": artifact_digest,
        "candidate_id": candidate_id,
        "trainer_config_hash": metrics["trainer_config_hash"],
        "dataset_manifest_path": _repo_relative_path(dataset_path),
        "created_at": created_at,
        "trainer_version": TRAINER_VERSION,
        "run_version": version,
    }
    _write_json(artifact_dir / "training_manifest.json", training_manifest)

    return {
        "artifact_dir": str(artifact_dir),
        "artifact_digest": artifact_digest,
        "candidate_id": candidate_id,
        "metrics": metrics,
        "model_card_path": str(artifact_dir / "model_card.json"),
    }


def register_candidate(
    *,
    artifact_dir: Path,
    config_path: Path,
    bench_report_path: Path | None = None,
    parent_bench_report_path: Path | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Register an immutable candidate manifest from a trained artifact directory."""
    if not artifact_dir.exists():
        raise AgentModelTrainingError(f"Artifact directory does not exist: {artifact_dir}")

    config = load_trainer_config(config_path)
    dataset_path = _resolve_path(config.dataset_manifest_path)
    dataset_manifest = load_dataset_manifest(dataset_path)

    model_card_path = artifact_dir / "model_card.json"
    if not model_card_path.exists():
        raise AgentModelTrainingError(f"Missing model card: {model_card_path}")
    model_card = _read_json(model_card_path)
    card_errors = validate_model_card(model_card)
    if card_errors:
        raise AgentModelTrainingError("; ".join(card_errors))

    training_manifest_path = artifact_dir / "training_manifest.json"
    training_manifest = _read_json(training_manifest_path) if training_manifest_path.exists() else {}
    artifact_digest = str(training_manifest.get("artifact_digest") or _artifact_digest(artifact_dir))
    candidate_id = str(
        training_manifest.get("candidate_id")
        or model_card.get("candidate_id")
        or _stable_hash(artifact_digest, length=16)
    )

    metrics_path = artifact_dir / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.exists() else dict(model_card.get("metrics") or {})

    manifest = CandidateManifest(
        candidate_id=candidate_id,
        artifact_digest=artifact_digest,
        base_model_id=config.base_model_id,
        base_model_revision=config.base_model_revision,
        training_method=config.training_method,
        parent_candidate_id=config.parent_candidate_id,
        dataset_manifest={
            "path": _repo_relative_path(dataset_path),
            "version": dataset_manifest.get("version"),
            "content_hashes": dataset_manifest.get("content_hashes"),
            "leakage_check_passed": dataset_manifest.get("leakage_check_passed"),
            "transformation_version": dataset_manifest.get("transformation_version"),
            "preference_count": dataset_manifest.get("preference_count"),
            "dpo_trainable_count": dataset_manifest.get("dpo_trainable_count"),
            "preference_reward_source_counts": dataset_manifest.get("preference_reward_source_counts"),
        },
        trainer_config_hash=trainer_config_hash(config),
        config_path=_repo_relative_path(config_path),
        artifact_path=_repo_relative_path(artifact_dir),
        model_card_path=_repo_relative_path(model_card_path),
        metrics=metrics,
        bench_report_path=_repo_relative_path(bench_report_path) if bench_report_path else None,
        parent_bench_report_path=(_repo_relative_path(parent_bench_report_path) if parent_bench_report_path else None),
        lifecycle_state="candidate",
        created_at=_now_iso(),
        trainer_version=config.trainer_version,
    )

    manifest_path = artifact_dir / "candidate_manifest.json"
    manifest_payload = manifest.model_dump(mode="json")
    _write_json(manifest_path, manifest_payload)

    registry = load_registry(registry_path)
    candidates = dict(registry.get("candidates") or {})
    candidates[candidate_id] = {
        "candidate_id": candidate_id,
        "artifact_path": manifest.artifact_path,
        "artifact_digest": artifact_digest,
        "lifecycle_state": "candidate",
        "base_model_id": config.base_model_id,
        "training_method": config.training_method,
        "parent_candidate_id": config.parent_candidate_id,
        "dataset_version": dataset_manifest.get("version"),
        "created_at": manifest.created_at,
        "bench_report_path": manifest.bench_report_path,
        "parent_bench_report_path": manifest.parent_bench_report_path,
    }
    registry["candidates"] = candidates
    registry["updated_at"] = _now_iso()
    save_registry(registry, registry_path)

    return {
        "candidate_id": candidate_id,
        "manifest_path": str(manifest_path),
        "registry_path": str(registry_path),
        "lifecycle_state": "candidate",
    }


def load_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    if not registry_path.exists():
        return {
            "registry_version": 1,
            "active_candidate_id": None,
            "active_artifact_path": None,
            "candidates": {},
            "updated_at": _now_iso(),
        }
    return _read_json(registry_path)


def save_registry(registry: dict[str, Any], registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(registry_path, registry)


def _load_candidate_manifest(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "candidate_manifest.json"
    if not manifest_path.exists():
        raise AgentModelTrainingError(f"Missing candidate manifest: {manifest_path}")
    return _read_json(manifest_path)


def validate_promotion_evidence(
    manifest: dict[str, Any],
    *,
    bench_report_path: Path | None = None,
    parent_bench_report_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    for key in PROMOTION_REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            errors.append(f"candidate manifest missing required key: {key}")

    dataset_manifest = manifest.get("dataset_manifest")
    if not isinstance(dataset_manifest, dict):
        errors.append("dataset_manifest must be an object")
    elif not dataset_manifest.get("leakage_check_passed"):
        errors.append("dataset manifest leakage_check_passed must be true")

    artifact_path = _resolve_path(str(manifest.get("artifact_path") or ""))
    if not artifact_path.exists():
        errors.append(f"artifact path does not exist: {artifact_path}")
    else:
        expected_digest = str(manifest.get("artifact_digest") or "")
        if expected_digest and _artifact_digest(artifact_path) != expected_digest:
            errors.append("artifact digest mismatch; candidate artifacts are mutable or incomplete")

    model_card_path = _resolve_path(str(manifest.get("model_card_path") or ""))
    if not model_card_path.exists():
        errors.append(f"model card does not exist: {model_card_path}")
    else:
        errors.extend(validate_model_card(_read_json(model_card_path)))

    resolved_bench = bench_report_path
    if resolved_bench is None and manifest.get("bench_report_path"):
        resolved_bench = _resolve_path(str(manifest["bench_report_path"]))
    if resolved_bench is None or not resolved_bench.exists():
        errors.append("TalismanBench release report is required for promotion")
    else:
        bench_report = _read_json(resolved_bench)
        release_gate = bench_report.get("release_gate") or {}
        if not release_gate.get("passed"):
            blockers = release_gate.get("hard_blocker_failures") or []
            thresholds = release_gate.get("threshold_failures") or []
            errors.append(
                "TalismanBench release gate failed: " + ", ".join([*blockers, *thresholds])
                if blockers or thresholds
                else "release_gate.passed is false"
            )

    training_method = str(manifest.get("training_method") or "sft")
    if training_method == "preference":
        if not manifest.get("parent_candidate_id"):
            errors.append("preference candidate missing parent_candidate_id")
        resolved_parent_bench = parent_bench_report_path
        if resolved_parent_bench is None and manifest.get("parent_bench_report_path"):
            resolved_parent_bench = _resolve_path(str(manifest["parent_bench_report_path"]))
        if resolved_parent_bench is None or not resolved_parent_bench.exists():
            errors.append("SFT parent TalismanBench report is required for preference promotion")
        elif resolved_bench and resolved_bench.exists():
            from decision_quality.eval_corpus import compare_reports

            comparison = compare_reports(_read_json(resolved_parent_bench), _read_json(resolved_bench))
            if comparison.get("summary", {}).get("regression_detected"):
                new_failures = comparison["summary"].get("new_deterministic_failures") or []
                errors.append("Preference candidate regressed vs SFT parent: " + ", ".join(new_failures))
    return errors


def promote_candidate(
    candidate_id: str,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    bench_report_path: Path | None = None,
    parent_bench_report_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    candidates = registry.get("candidates") or {}
    entry = candidates.get(candidate_id)
    if not entry:
        raise AgentModelTrainingError(f"Unknown candidate_id: {candidate_id}")

    artifact_dir = _resolve_path(str(entry.get("artifact_path") or ""))
    manifest = _load_candidate_manifest(artifact_dir)
    if manifest.get("candidate_id") != candidate_id:
        raise AgentModelTrainingError("candidate manifest id does not match registry entry")

    errors = validate_promotion_evidence(
        manifest,
        bench_report_path=bench_report_path,
        parent_bench_report_path=parent_bench_report_path,
    )
    if errors and not force:
        raise AgentModelTrainingError("; ".join(errors))

    manifest["lifecycle_state"] = "approved"
    manifest["approved_at"] = _now_iso()
    _write_json(artifact_dir / "candidate_manifest.json", manifest)

    entry["lifecycle_state"] = "approved"
    entry["approved_at"] = manifest["approved_at"]
    candidates[candidate_id] = entry
    registry["candidates"] = candidates
    registry["active_candidate_id"] = candidate_id
    registry["active_artifact_path"] = entry.get("artifact_path")
    registry["updated_at"] = _now_iso()
    save_registry(registry, registry_path)

    return {
        "candidate_id": candidate_id,
        "lifecycle_state": "approved",
        "active_candidate_id": candidate_id,
        "promotion_errors": errors,
        "forced": force,
    }


def set_candidate_lifecycle(
    candidate_id: str,
    *,
    lifecycle_state: LifecycleState,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    if lifecycle_state not in {"deprecated", "disabled", "candidate"}:
        raise AgentModelTrainingError(
            f"Unsupported lifecycle transition via set_candidate_lifecycle: {lifecycle_state}"
        )

    registry = load_registry(registry_path)
    candidates = registry.get("candidates") or {}
    entry = candidates.get(candidate_id)
    if not entry:
        raise AgentModelTrainingError(f"Unknown candidate_id: {candidate_id}")

    artifact_dir = _resolve_path(str(entry.get("artifact_path") or ""))
    manifest = _load_candidate_manifest(artifact_dir)
    manifest["lifecycle_state"] = lifecycle_state
    manifest[f"{lifecycle_state}_at"] = _now_iso()
    _write_json(artifact_dir / "candidate_manifest.json", manifest)

    entry["lifecycle_state"] = lifecycle_state
    candidates[candidate_id] = entry

    if registry.get("active_candidate_id") == candidate_id and lifecycle_state in {"deprecated", "disabled"}:
        registry["active_candidate_id"] = None
        registry["active_artifact_path"] = None

    registry["candidates"] = candidates
    registry["updated_at"] = _now_iso()
    save_registry(registry, registry_path)

    return {"candidate_id": candidate_id, "lifecycle_state": lifecycle_state}


def deprecate_candidate(candidate_id: str, *, registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    return set_candidate_lifecycle(candidate_id, lifecycle_state="deprecated", registry_path=registry_path)


def disable_candidate(candidate_id: str, *, registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    return set_candidate_lifecycle(candidate_id, lifecycle_state="disabled", registry_path=registry_path)


def build_default_trainer_config(
    *,
    dataset_manifest_path: Path,
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    combination_id: str = "qwen-local-vllm",
    training_method: TrainingMethod = "sft",
    parent_candidate_id: str | None = None,
) -> TrainerConfig:
    combination: dict[str, Any] | None = None
    if DEFAULT_CANDIDATE_MATRIX.exists():
        matrix = _read_json(DEFAULT_CANDIDATE_MATRIX)
        for entry in matrix.get("combinations") or []:
            if isinstance(entry, dict) and entry.get("id") == combination_id:
                combination = entry
                break

    served_model_name = None
    if combination:
        served_model_name = str(combination.get("served_model_name") or "")

    trainer_version = PREFERENCE_TRAINER_VERSION if training_method == "preference" else TRAINER_VERSION
    return TrainerConfig(
        base_model_id=base_model_id,
        base_model_revision=str((combination or {}).get("model_revision") or "") or None,
        dataset_manifest_path=_repo_relative_path(dataset_manifest_path),
        chat_template="qwen2.5",
        trainer_backend="smoke",
        training_method=training_method,
        preference_algorithm="smoke" if training_method == "preference" else "smoke",
        parent_candidate_id=parent_candidate_id,
        trainer_version=trainer_version,
        code_revision=_git_revision(),
        serve=ServeMetadata(
            served_model_name=served_model_name,
            combination_id=combination_id,
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talisman agent model SFT/LoRA training and registry")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate a trainer config")
    validate_parser.add_argument("--config", type=Path, required=True)

    smoke_parser = subparsers.add_parser("smoke-train", help="Run deterministic smoke training")
    smoke_parser.add_argument("--config", type=Path, required=True)
    smoke_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    smoke_parser.add_argument("--bench-report", type=Path, default=None)
    smoke_parser.add_argument("--parent-bench-report", type=Path, default=None)
    smoke_parser.add_argument("--run-version", type=str, default=None)
    smoke_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)

    register_parser = subparsers.add_parser("register-candidate", help="Register a trained artifact")
    register_parser.add_argument("--artifact-dir", type=Path, required=True)
    register_parser.add_argument("--config", type=Path, required=True)
    register_parser.add_argument("--bench-report", type=Path, default=None)
    register_parser.add_argument("--parent-bench-report", type=Path, default=None)
    register_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)

    promote_parser = subparsers.add_parser("promote", help="Promote a candidate to approved")
    promote_parser.add_argument("--candidate-id", required=True)
    promote_parser.add_argument("--bench-report", type=Path, default=None)
    promote_parser.add_argument("--parent-bench-report", type=Path, default=None)
    promote_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    promote_parser.add_argument("--force", action="store_true")

    deprecate_parser = subparsers.add_parser("deprecate", help="Mark a candidate deprecated")
    deprecate_parser.add_argument("--candidate-id", required=True)
    deprecate_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)

    disable_parser = subparsers.add_parser("disable", help="Disable a candidate")
    disable_parser.add_argument("--candidate-id", required=True)
    disable_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)

    init_parser = subparsers.add_parser("init-config", help="Write a default trainer config")
    init_parser.add_argument("--dataset-manifest", type=Path, required=True)
    init_parser.add_argument("--output", type=Path, required=True)
    init_parser.add_argument("--base-model-id", default="Qwen/Qwen2.5-7B-Instruct")
    init_parser.add_argument("--combination-id", default="qwen-local-vllm")
    init_parser.add_argument("--training-method", choices=["sft", "preference"], default="sft")
    init_parser.add_argument("--parent-candidate-id", default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_trainer_config(args.config)
            errors = validate_trainer_config(config)
            payload = {"valid": not errors, "errors": errors, "trainer_config_hash": trainer_config_hash(config)}
            print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))
            return 0 if not errors else 1

        if args.command == "smoke-train":
            config = load_trainer_config(args.config)
            result = smoke_train(
                config,
                output_dir=args.output_dir,
                bench_report_path=args.bench_report,
                parent_bench_report_path=args.parent_bench_report,
                run_version=args.run_version,
                registry_path=args.registry,
            )
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0

        if args.command == "register-candidate":
            result = register_candidate(
                artifact_dir=args.artifact_dir,
                config_path=args.config,
                bench_report_path=args.bench_report,
                parent_bench_report_path=args.parent_bench_report,
                registry_path=args.registry,
            )
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0

        if args.command == "promote":
            result = promote_candidate(
                args.candidate_id,
                registry_path=args.registry,
                bench_report_path=args.bench_report,
                parent_bench_report_path=args.parent_bench_report,
                force=args.force,
            )
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0

        if args.command == "deprecate":
            result = deprecate_candidate(args.candidate_id, registry_path=args.registry)
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0

        if args.command == "disable":
            result = disable_candidate(args.candidate_id, registry_path=args.registry)
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0

        if args.command == "init-config":
            config = build_default_trainer_config(
                dataset_manifest_path=args.dataset_manifest,
                base_model_id=args.base_model_id,
                combination_id=args.combination_id,
                training_method=args.training_method,
                parent_candidate_id=args.parent_candidate_id,
            )
            _write_json(args.output, config.model_dump(mode="json"))
            print(json.dumps({"config_path": str(args.output)}, indent=2))
            return 0

        raise AgentModelTrainingError(f"Unsupported command: {args.command}")
    except (AgentModelTrainingError, ValidationError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
