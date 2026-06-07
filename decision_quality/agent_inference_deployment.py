"""Governed inference deployment validation and manifest generation (TL-95)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from decision_quality.agent_model_training import (
    DEFAULT_CANDIDATE_MATRIX,
    DEFAULT_REGISTRY_PATH,
    AgentModelTrainingError,
    _artifact_digest,
    _read_json,
    _resolve_path,
    _write_json,
    load_registry,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_OUTPUT_DIR = ROOT / "outputs" / "inference_deployments"

DEPLOYMENT_MANIFEST_VERSION = 1
INFERENCE_RUNTIME_VERSION = "vllm_openai_compatible_v1"
ALLOWED_DEPLOYMENT_ENVIRONMENTS = {"nonprod", "staging", "production"}
ALLOWED_DEPLOYMENT_LIFECYCLE = {"approved"}


class InferenceDeploymentError(ValueError):
    """Raised when inference deployment validation or manifest generation fails."""


class DeploymentManifest(BaseModel):
    manifest_version: int = DEPLOYMENT_MANIFEST_VERSION
    environment: str
    candidate_id: str
    artifact_digest: str
    artifact_path: str
    base_model_id: str
    served_model_name: str
    combination_id: str
    endpoint_protocol: str = "openai_compatible"
    runtime_version: str = INFERENCE_RUNTIME_VERSION
    serving: dict[str, Any] = Field(default_factory=dict)
    model_tier_aliases: dict[str, str] = Field(default_factory=dict)
    registry_updated_at: str | None = None
    created_at: str
    code_revision: str | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


def _load_candidate_matrix() -> dict[str, Any]:
    if not DEFAULT_CANDIDATE_MATRIX.exists():
        raise InferenceDeploymentError(f"Candidate matrix not found: {DEFAULT_CANDIDATE_MATRIX}")
    return _read_json(DEFAULT_CANDIDATE_MATRIX)


def _combination_by_id(combination_id: str) -> dict[str, Any]:
    matrix = _load_candidate_matrix()
    for entry in matrix.get("combinations") or []:
        if isinstance(entry, dict) and str(entry.get("id") or "") == combination_id:
            return entry
    raise InferenceDeploymentError(f"Unknown combination_id: {combination_id}")


def resolve_deployment_candidate(
    *,
    candidate_id: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Resolve registry entry and on-disk candidate manifest."""
    registry = load_registry(registry_path)
    resolved_id = (candidate_id or registry.get("active_candidate_id") or "").strip()
    if not resolved_id:
        raise InferenceDeploymentError("No candidate_id provided and registry has no active_candidate_id")

    candidates = registry.get("candidates") or {}
    entry = candidates.get(resolved_id)
    if not isinstance(entry, dict):
        raise InferenceDeploymentError(f"Unknown candidate_id in registry: {resolved_id}")

    artifact_path = _resolve_path(str(entry.get("artifact_path") or ""))
    manifest_path = artifact_path / "candidate_manifest.json"
    if not manifest_path.exists():
        raise InferenceDeploymentError(f"Missing candidate manifest: {manifest_path}")

    manifest = _read_json(manifest_path)
    if str(manifest.get("candidate_id") or "") != resolved_id:
        raise InferenceDeploymentError("candidate manifest id does not match registry entry")

    return resolved_id, entry, manifest


def validate_deployment_eligibility(
    entry: dict[str, Any],
    manifest: dict[str, Any],
    *,
    require_approved: bool = True,
) -> list[str]:
    """Validate registry lifecycle and immutable artifact digest before deploy."""
    errors: list[str] = []
    lifecycle = str(entry.get("lifecycle_state") or manifest.get("lifecycle_state") or "").lower()
    if require_approved and lifecycle != "approved":
        errors.append(f"candidate lifecycle must be approved, got: {lifecycle or 'unknown'}")
    if lifecycle == "disabled":
        errors.append("disabled candidates cannot be deployed")

    artifact_path = _resolve_path(str(manifest.get("artifact_path") or entry.get("artifact_path") or ""))
    if not artifact_path.exists():
        errors.append(f"artifact path does not exist: {artifact_path}")
        return errors

    expected_digest = str(manifest.get("artifact_digest") or entry.get("artifact_digest") or "")
    if not expected_digest:
        errors.append("artifact_digest is required for deployment")
    else:
        try:
            actual_digest = _artifact_digest(artifact_path)
        except AgentModelTrainingError as exc:
            errors.append(str(exc))
        else:
            if actual_digest != expected_digest:
                errors.append("artifact digest mismatch; candidate artifacts are mutable or incomplete")

    model_card_path = _resolve_path(str(manifest.get("model_card_path") or ""))
    if not model_card_path.exists():
        errors.append(f"model card does not exist: {model_card_path}")

    return errors


def _serve_metadata_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    model_card_path = _resolve_path(str(manifest.get("model_card_path") or ""))
    if not model_card_path.exists():
        return {}
    model_card = _read_json(model_card_path)
    serve = model_card.get("serve")
    return dict(serve) if isinstance(serve, dict) else {}


def default_model_tier_aliases(served_model_name: str) -> dict[str, str]:
    return {
        "low": served_model_name,
        "mid": served_model_name,
        "high": served_model_name,
    }


def build_deployment_manifest(
    *,
    candidate_id: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    combination_id: str | None = None,
    environment: str = "nonprod",
    runtime_version: str | None = None,
) -> dict[str, Any]:
    """Build a versioned deployment manifest for an approved registry candidate."""
    env = str(environment or "nonprod").strip().lower()
    if env not in ALLOWED_DEPLOYMENT_ENVIRONMENTS:
        raise InferenceDeploymentError(f"environment must be one of {sorted(ALLOWED_DEPLOYMENT_ENVIRONMENTS)}")

    resolved_id, entry, manifest = resolve_deployment_candidate(
        candidate_id=candidate_id,
        registry_path=registry_path,
    )
    errors = validate_deployment_eligibility(entry, manifest, require_approved=True)
    if errors:
        raise InferenceDeploymentError("; ".join(errors))

    serve = _serve_metadata_from_manifest(manifest)
    resolved_combination_id = (combination_id or str(serve.get("combination_id") or "") or "qwen-managed-gpu").strip()
    combination = _combination_by_id(resolved_combination_id)

    served_model_name = str(
        serve.get("served_model_name") or combination.get("served_model_name") or manifest.get("base_model_id") or ""
    ).strip()
    if not served_model_name:
        raise InferenceDeploymentError("served_model_name could not be resolved for deployment")

    serving = dict(combination.get("serving") or {})
    registry = load_registry(registry_path)
    deployment = DeploymentManifest(
        environment=env,
        candidate_id=resolved_id,
        artifact_digest=str(manifest.get("artifact_digest") or entry.get("artifact_digest") or ""),
        artifact_path=str(manifest.get("artifact_path") or entry.get("artifact_path") or ""),
        base_model_id=str(manifest.get("base_model_id") or entry.get("base_model_id") or ""),
        served_model_name=served_model_name,
        combination_id=resolved_combination_id,
        endpoint_protocol=str(serve.get("endpoint_protocol") or "openai_compatible"),
        runtime_version=runtime_version or INFERENCE_RUNTIME_VERSION,
        serving=serving,
        model_tier_aliases=default_model_tier_aliases(served_model_name),
        registry_updated_at=str(registry.get("updated_at") or "") or None,
        created_at=_now_iso(),
        code_revision=_git_revision(),
    )
    return deployment.model_dump(mode="json")


def validate_deployment_manifest(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        manifest = DeploymentManifest.model_validate(payload)
    except ValidationError as exc:
        return [str(exc)]

    if manifest.manifest_version != DEPLOYMENT_MANIFEST_VERSION:
        errors.append(f"unsupported manifest_version: {manifest.manifest_version}")
    if not manifest.candidate_id:
        errors.append("candidate_id is required")
    if not manifest.artifact_digest:
        errors.append("artifact_digest is required")
    if not manifest.served_model_name:
        errors.append("served_model_name is required")
    if not manifest.combination_id:
        errors.append("combination_id is required")
    if not manifest.serving:
        errors.append("serving configuration is required")

    try:
        _combination_by_id(manifest.combination_id)
    except InferenceDeploymentError as exc:
        errors.append(str(exc))

    return errors


def write_deployment_manifest(
    manifest: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_MANIFEST_OUTPUT_DIR,
) -> Path:
    candidate_id = str(manifest.get("candidate_id") or "unknown")
    environment = str(manifest.get("environment") or "nonprod")
    output_path = output_dir / environment / f"{candidate_id}.json"
    _write_json(output_path, manifest)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talisman governed inference deployment manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate deployment eligibility")
    validate_parser.add_argument("--candidate-id", default=None)
    validate_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)

    build_parser = subparsers.add_parser("build-manifest", help="Build a deployment manifest")
    build_parser.add_argument("--candidate-id", default=None)
    build_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    build_parser.add_argument("--combination-id", default=None)
    build_parser.add_argument("--environment", default="nonprod")
    build_parser.add_argument("--output", type=Path, default=None)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_MANIFEST_OUTPUT_DIR)

    manifest_validate_parser = subparsers.add_parser(
        "validate-manifest", help="Validate an existing deployment manifest file"
    )
    manifest_validate_parser.add_argument("--manifest", type=Path, required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate":
            resolved_id, entry, manifest = resolve_deployment_candidate(
                candidate_id=args.candidate_id,
                registry_path=args.registry,
            )
            errors = validate_deployment_eligibility(entry, manifest, require_approved=True)
            payload = {"candidate_id": resolved_id, "eligible": not errors, "errors": errors}
            print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))
            return 0 if not errors else 1

        if args.command == "build-manifest":
            manifest = build_deployment_manifest(
                candidate_id=args.candidate_id,
                registry_path=args.registry,
                combination_id=args.combination_id,
                environment=args.environment,
            )
            output_path = args.output or write_deployment_manifest(
                manifest,
                output_dir=args.output_dir,
            )
            result = {"manifest_path": str(output_path), "manifest": manifest}
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0

        if args.command == "validate-manifest":
            manifest = _read_json(args.manifest)
            errors = validate_deployment_manifest(manifest)
            payload = {"valid": not errors, "errors": errors}
            print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))
            return 0 if not errors else 1

        raise InferenceDeploymentError(f"Unsupported command: {args.command}")
    except (InferenceDeploymentError, AgentModelTrainingError, ValidationError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
