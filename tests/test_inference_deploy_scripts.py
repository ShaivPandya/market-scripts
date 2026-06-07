from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_config_example_includes_talisman_and_inference_settings() -> None:
    config = (ROOT / "infra/gcp/config.example.sh").read_text()
    assert "TALISMAN_BASE_URL=TALISMAN_BASE_URL:latest" in config
    assert "TALISMAN_API_KEY=TALISMAN_API_KEY:latest" in config
    assert 'INFERENCE_SERVICE="talisman-inference-nonprod"' in config
    assert 'INFERENCE_COMBINATION_ID="qwen-managed-gpu"' in config
    assert "INFERENCE_SECRETS=(" in config


def test_setup_secrets_includes_talisman_bindings() -> None:
    script = (ROOT / "infra/gcp/setup-secrets.sh").read_text()
    assert "TALISMAN_API_KEY" in script
    assert "TALISMAN_BASE_URL" in script
    assert "TALISMAN_BASE_URL TALISMAN_API_KEY" in script


def test_deploy_inference_service_script_requires_manifest_gate() -> None:
    script = (ROOT / "infra/gcp/deploy-inference-service.sh").read_text()
    assert "decision_quality.agent_inference_deployment build-manifest" in script
    assert "--no-allow-unauthenticated" in script
    assert "--gpu-type=" in script
    assert "gsutil cp" in script


def test_inference_dockerfile_uses_governed_entrypoint() -> None:
    dockerfile = (ROOT / "infra/gcp/Dockerfile.inference").read_text()
    assert "decision_quality.inference_readiness" in dockerfile
    assert "INFERENCE_ALLOW_SERVE=1" in dockerfile
    assert "candidate_matrix.json" in dockerfile


def test_inference_monitoring_assets_exist() -> None:
    assert (ROOT / "infra/gcp/setup-inference-monitoring.sh").exists()
    alerts = (ROOT / "infra/gcp/monitoring-inference-alerts.json").read_text()
    assert "inference_startup_refused_count" in alerts
    assert "inference_generation_latency_seconds" in alerts
