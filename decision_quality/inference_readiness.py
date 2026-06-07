"""Inference service health/readiness and governed vLLM startup helpers (TL-95)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from decision_quality.agent_inference_deployment import (
    InferenceDeploymentError,
    resolve_deployment_candidate,
    validate_deployment_eligibility,
    validate_deployment_manifest,
)
from decision_quality.agent_model_training import _read_json

ROOT = Path(__file__).resolve().parents[1]

READINESS_STATUS_HEALTHY = "healthy"
READINESS_STATUS_NOT_READY = "not_ready"
READINESS_STATUS_REFUSED = "refused"


def _falsey_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"", "0", "false", "no", "off", "disabled"}


def load_deployment_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise InferenceDeploymentError(f"Deployment manifest not found: {path}")
    manifest = _read_json(path)
    errors = validate_deployment_manifest(manifest)
    if errors:
        raise InferenceDeploymentError("; ".join(errors))
    return manifest


def check_startup_eligibility(
    *,
    deployment_manifest: dict[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    """Refuse startup when registry lifecycle or artifact digest is no longer valid."""
    candidate_id = str(deployment_manifest.get("candidate_id") or "")
    resolved_id, entry, manifest = resolve_deployment_candidate(
        candidate_id=candidate_id,
        registry_path=registry_path,
    )
    if resolved_id != candidate_id:
        raise InferenceDeploymentError("deployment manifest candidate_id does not match registry")

    errors = validate_deployment_eligibility(entry, manifest, require_approved=True)
    expected_digest = str(deployment_manifest.get("artifact_digest") or "")
    actual_digest = str(manifest.get("artifact_digest") or entry.get("artifact_digest") or "")
    if expected_digest and actual_digest and expected_digest != actual_digest:
        errors.append("deployment manifest artifact_digest does not match registry")

    if errors:
        raise InferenceDeploymentError("; ".join(errors))

    return {
        "eligible": True,
        "candidate_id": resolved_id,
        "artifact_digest": actual_digest,
        "served_model_name": deployment_manifest.get("served_model_name"),
        "combination_id": deployment_manifest.get("combination_id"),
        "runtime_version": deployment_manifest.get("runtime_version"),
    }


def assess_readiness(
    *,
    deployment_manifest: dict[str, Any],
    registry_path: Path,
    model_loaded: bool,
    served_model_aliases: list[str] | None = None,
) -> dict[str, Any]:
    """Distinguish process health from model-loaded readiness."""
    candidate_id = str(deployment_manifest.get("candidate_id") or "")
    served_model_name = str(deployment_manifest.get("served_model_name") or "")
    identity = {
        "candidate_id": candidate_id,
        "artifact_digest": deployment_manifest.get("artifact_digest"),
        "served_model_name": served_model_name,
        "combination_id": deployment_manifest.get("combination_id"),
        "runtime_version": deployment_manifest.get("runtime_version"),
        "endpoint_protocol": deployment_manifest.get("endpoint_protocol"),
    }

    try:
        check_startup_eligibility(
            deployment_manifest=deployment_manifest,
            registry_path=registry_path,
        )
        governance_errors: list[str] = []
        status = READINESS_STATUS_HEALTHY
    except InferenceDeploymentError as exc:
        governance_errors = [str(exc)]
        status = READINESS_STATUS_REFUSED

    alias_errors: list[str] = []
    if model_loaded and served_model_name:
        aliases = {str(alias) for alias in (served_model_aliases or [])}
        if served_model_name not in aliases:
            alias_errors.append(f"configured served_model_name '{served_model_name}' not exposed by endpoint")

    ready = status == READINESS_STATUS_HEALTHY and model_loaded and not alias_errors
    if status == READINESS_STATUS_HEALTHY and not model_loaded:
        status = READINESS_STATUS_NOT_READY

    return {
        "status": status,
        "ready": ready,
        "model_loaded": model_loaded,
        "identity": identity,
        "governance_errors": governance_errors,
        "alias_errors": alias_errors,
    }


def build_vllm_serve_command(manifest: dict[str, Any]) -> list[str]:
    """Build a vLLM serve command from deployment manifest serving metadata."""
    base_model_id = str(manifest.get("base_model_id") or "").strip()
    if not base_model_id:
        raise InferenceDeploymentError("base_model_id is required to build vLLM serve command")

    served_model_name = str(manifest.get("served_model_name") or "").strip()
    serving = dict(manifest.get("serving") or {})
    command = [
        "vllm",
        "serve",
        base_model_id,
        "--served-model-name",
        served_model_name or base_model_id,
        "--host",
        os.environ.get("INFERENCE_HOST", "0.0.0.0"),
        "--port",
        os.environ.get("INFERENCE_PORT", os.environ.get("PORT", "8080")),
    ]

    max_model_len = serving.get("max_model_len")
    if max_model_len is not None:
        command.extend(["--max-model-len", str(max_model_len)])

    gpu_memory_utilization = serving.get("gpu_memory_utilization")
    if gpu_memory_utilization is not None:
        command.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])

    if serving.get("enable_auto_tool_choice"):
        command.append("--enable-auto-tool-choice")

    tool_call_parser = serving.get("tool_call_parser")
    if tool_call_parser:
        command.extend(["--tool-call-parser", str(tool_call_parser)])

    extra_args = (os.environ.get("INFERENCE_VLLM_EXTRA_ARGS") or "").strip()
    if extra_args:
        command.extend(shlex.split(extra_args))

    return command


def startup_check(
    *,
    deployment_manifest_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    manifest = load_deployment_manifest(deployment_manifest_path)
    return check_startup_eligibility(
        deployment_manifest=manifest,
        registry_path=registry_path,
    )


def run_vllm_serve(
    *,
    deployment_manifest_path: Path,
    registry_path: Path,
) -> int:
    manifest = load_deployment_manifest(deployment_manifest_path)
    check_startup_eligibility(deployment_manifest=manifest, registry_path=registry_path)
    command = build_vllm_serve_command(manifest)
    print(
        json.dumps(
            {
                "event": "inference_startup",
                "candidate_id": manifest.get("candidate_id"),
                "artifact_digest": manifest.get("artifact_digest"),
                "served_model_name": manifest.get("served_model_name"),
                "command": command,
            },
            indent=2,
            ensure_ascii=True,
        ),
        file=sys.stderr,
    )
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talisman inference readiness and startup")
    subparsers = parser.add_subparsers(dest="command", required=True)

    startup_parser = subparsers.add_parser("startup-check", help="Validate deploy eligibility")
    startup_parser.add_argument("--deployment-manifest", type=Path, required=True)
    startup_parser.add_argument("--registry", type=Path, required=True)

    readiness_parser = subparsers.add_parser("readiness", help="Assess readiness state")
    readiness_parser.add_argument("--deployment-manifest", type=Path, required=True)
    readiness_parser.add_argument("--registry", type=Path, required=True)
    readiness_parser.add_argument("--model-loaded", action="store_true")
    readiness_parser.add_argument("--served-model-alias", action="append", default=[])

    serve_parser = subparsers.add_parser("serve", help="Run governed vLLM serve")
    serve_parser.add_argument("--deployment-manifest", type=Path, required=True)
    serve_parser.add_argument("--registry", type=Path, required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "startup-check":
            result = startup_check(
                deployment_manifest_path=args.deployment_manifest,
                registry_path=args.registry,
            )
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0

        if args.command == "readiness":
            manifest = load_deployment_manifest(args.deployment_manifest)
            result = assess_readiness(
                deployment_manifest=manifest,
                registry_path=args.registry,
                model_loaded=bool(args.model_loaded),
                served_model_aliases=list(args.served_model_alias or []),
            )
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
            return 0 if result["ready"] or result["status"] == READINESS_STATUS_NOT_READY else 1

        if args.command == "serve":
            if _falsey_env("INFERENCE_ALLOW_SERVE"):
                raise InferenceDeploymentError("Refusing to start vLLM without INFERENCE_ALLOW_SERVE=1")
            return run_vllm_serve(
                deployment_manifest_path=args.deployment_manifest,
                registry_path=args.registry,
            )

        raise InferenceDeploymentError(f"Unsupported command: {args.command}")
    except InferenceDeploymentError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
