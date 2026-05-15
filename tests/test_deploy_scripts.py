import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARGS_RE = re.compile(r"--args=([^\\\s]+)")


def test_firebase_hosting_spa_rewrite_does_not_capture_missing_assets() -> None:
    config = json.loads((ROOT / "firebase.json").read_text())
    hosting = config["hosting"]
    rewrites = hosting["rewrites"]

    assert rewrites[0]["source"] == "/api/**"

    index_rewrites = [rewrite for rewrite in rewrites if rewrite.get("destination") == "/index.html"]
    assert len(index_rewrites) == 1

    spa_rewrite = index_rewrites[0]
    assert rewrites[1] == spa_rewrite
    assert spa_rewrite.get("source") != "**"
    assert "regex" in spa_rewrite

    no_cache_route_headers = [
        header_rule
        for header_rule in hosting["headers"]
        if header_rule.get("regex")
        and any(
            header["key"].lower() == "cache-control" and "no-cache" in header["value"]
            for header in header_rule["headers"]
        )
    ]
    assert len(no_cache_route_headers) == 1
    assert spa_rewrite["regex"] == no_cache_route_headers[0]["regex"]

    spa_pattern = re.compile(spa_rewrite["regex"])
    for path in ["/", "/index.html", "/sizer", "/sizer/results", "/analyzer"]:
        assert spa_pattern.fullmatch(path)

    for path in [
        "/api/health",
        "/assets/DefinitelyMissing-test.js",
        "/assets/index-old.js",
        "/favicon.ico",
    ]:
        assert not spa_pattern.fullmatch(path)


def test_deploy_scripts_do_not_repeat_gcloud_args_values() -> None:
    for script_path in sorted((ROOT / "infra/gcp").glob("*.sh")):
        for line_number, line in enumerate(script_path.read_text().splitlines(), start=1):
            match = ARGS_RE.search(line)
            if not match:
                continue

            args = [arg for arg in match.group(1).strip("'\"").split(",") if arg]
            duplicates = sorted({arg for arg in args if args.count(arg) > 1})

            assert not duplicates, f"{script_path}:{line_number} repeats --args values: {duplicates}"


def test_sizer_worker_deploy_removed() -> None:
    assert not (ROOT / "infra/gcp/deploy-sizer-worker.sh").exists()

    config_example = (ROOT / "infra/gcp/config.example.sh").read_text()
    backend = (ROOT / "infra/gcp/deploy-backend.sh").read_text()

    assert "SIZER_WORKER_" not in config_example
    assert "deploy-sizer-worker.sh" not in backend
    assert "sizer worker pool deploy" not in backend


def test_ontology_worker_deploy_uses_env_for_duplicate_ontology_values() -> None:
    script = (ROOT / "infra/gcp/deploy-ontology-worker.sh").read_text()

    assert "--args=-m,api.job_worker_loop,run \\" in script
    assert "--job-type,ontology,--queue,ontology" not in script
    assert '"ASYNC_DISPATCH_BACKEND_ONTOLOGY=warm_worker"' in script
    assert '"ASYNC_QUEUE_ONTOLOGY=ontology"' in script
    assert '"JOB_WORKER_JOB_TYPE=ontology"' in script
    assert '"JOB_WORKER_QUEUE=ontology"' in script


def test_analyzer_worker_deploy_uses_env_for_duplicate_analyzer_values() -> None:
    script = (ROOT / "infra/gcp/deploy-analyzer-worker.sh").read_text()

    assert "--args=-m,api.job_worker_loop,run \\" in script
    assert "--job-type,analyzer,--queue,analyzer" not in script
    assert '"ASYNC_DISPATCH_BACKEND_ANALYZER=warm_worker"' in script
    assert '"ASYNC_QUEUE_ANALYZER=analyzer"' in script
    assert '"JOB_WORKER_JOB_TYPE=analyzer"' in script
    assert '"JOB_WORKER_QUEUE=analyzer"' in script
    assert '"ASYNC_ANALYZER_COMPLETED_TTL_SECONDS=300"' in script
    assert '--cpu="${ANALYZER_WORKER_CPU:-1}"' in script
    assert '--memory="${ANALYZER_WORKER_MEMORY:-1Gi}"' in script


def test_api_deploy_routes_only_heavier_noninteractive_jobs_to_cloud_run_jobs() -> None:
    script = (ROOT / "infra/gcp/deploy-api.sh").read_text()

    assert '"ASYNC_DISPATCH_BACKEND_ANALYZER=cloud_run_jobs"' in script
    assert '"ASYNC_DISPATCH_BACKEND_SIZER=inline"' in script
    assert '"ASYNC_DISPATCH_BACKEND_ONTOLOGY=cloud_run_jobs"' in script
    assert '"AGENT_CHAT_DISPATCH_BACKEND=warm_worker"' in script
    assert '"ASYNC_QUEUE_ANALYZER=analyzer"' in script
    assert '"ASYNC_ANALYZER_COMPLETED_TTL_SECONDS=300"' in script


def test_watch_trigger_monitor_scheduler_is_disabled_by_default() -> None:
    script = (ROOT / "infra/gcp/setup-scheduler.sh").read_text()

    assert "SCHEDULE_WATCH_TRIGGER_MONITOR:-0" in script
    assert "delete_scheduler_job_if_present watch-trigger-monitor" in script
    assert 'upsert_api_job watch-trigger-monitor "${WATCH_TRIGGER_MONITOR_SCHEDULE:-30 14-22 * * 1-5}"' in script


def test_common_gcp_env_requires_postgres_for_api_and_ontology_worker() -> None:
    lib = (ROOT / "infra/gcp/lib.sh").read_text()

    assert "STATE_DB_BACKEND=postgres" in lib
    assert "ASYNC_JOB_STALE_GRACE_SECONDS=${ASYNC_JOB_STALE_GRACE_SECONDS:-300}" in lib
    assert "ASYNC_TIMEOUT_SIZER_SECONDS=${ASYNC_TIMEOUT_SIZER_SECONDS:-180}" in lib

    for script_name in ("deploy-api.sh", "deploy-ontology-worker.sh"):
        script = (ROOT / "infra/gcp" / script_name).read_text()
        assert "mapfile -t COMMON_ENV < <(common_env_vars)" in script
        assert '"${COMMON_ENV[@]}"' in script


def test_backend_deploy_includes_analyzer_worker_paths() -> None:
    script = (ROOT / "infra/gcp/deploy-backend.sh").read_text()

    assert '"analyzer worker pool deploy"' in script
    assert "/infra/gcp/deploy-analyzer-worker.sh" in script
    assert 'log "Deploying analyzer worker pool"' in script


def test_generic_async_job_defaults_to_smaller_fallback_resources() -> None:
    script = (ROOT / "infra/gcp/deploy-async-job.sh").read_text()

    assert '--cpu="${ASYNC_JOB_CPU:-1}"' in script
    assert '--memory="${ASYNC_JOB_MEMORY:-1Gi}"' in script
    assert '"ASYNC_ANALYZER_COMPLETED_TTL_SECONDS=300"' in script


# ---------------------------------------------------------------------------
# SHA-33: release manifest + release env vars
# ---------------------------------------------------------------------------


def test_backend_deploy_invokes_manifest_generation() -> None:
    """deploy-backend.sh must call the release manifest generator."""
    script = (ROOT / "infra/gcp/deploy-backend.sh").read_text()

    assert "release_manifest" in script or "release-manifest" in script
    assert "PYTHON_BIN" in script
    assert "infra.gcp.release_manifest" in script
    assert "--image-uri" in script
    assert "--output" in script


def test_backend_deploy_preserves_dirty_tree_guard() -> None:
    """The dirty-tree deploy guard must remain intact after manifest integration."""
    script = (ROOT / "infra/gcp/deploy-backend.sh").read_text()

    assert "ALLOW_DIRTY" in script
    assert "Working tree is dirty" in script
    assert 'git -C "${_repo_root}" diff --quiet' in script


def test_backend_deploy_manifest_rollback_refs() -> None:
    """deploy-backend.sh should attempt to pass prior manifest for rollback refs."""
    script = (ROOT / "infra/gcp/deploy-backend.sh").read_text()

    assert "--prior-manifest" in script
    assert "release-manifest.json" in script


def test_api_deploy_includes_release_env_vars() -> None:
    """deploy-api.sh must set TALISMAN_RELEASE_* env vars for the health endpoints."""
    script = (ROOT / "infra/gcp/deploy-api.sh").read_text()

    assert "TALISMAN_RELEASE_GIT_SHA=" in script
    assert "TALISMAN_RELEASE_GIT_SHA_SHORT=" in script
    assert "TALISMAN_RELEASE_IMAGE_TAG=" in script
    assert "TALISMAN_RELEASE_ENVIRONMENT=" in script


def test_api_deploy_does_not_expose_secrets_as_release_vars() -> None:
    """Release env vars must not include any secret references."""
    script = (ROOT / "infra/gcp/deploy-api.sh").read_text()

    # Find lines with TALISMAN_RELEASE and check none reference secrets
    for line in script.splitlines():
        if "TALISMAN_RELEASE" in line:
            assert "SECRET" not in line.upper(), f"Release var references a secret: {line.strip()}"
            assert "PASSWORD" not in line.upper(), f"Release var references a password: {line.strip()}"
            assert "API_KEY" not in line.upper(), f"Release var references an API key: {line.strip()}"


# ---------------------------------------------------------------------------
# SHA-34: deploy smoke hook
# ---------------------------------------------------------------------------


def test_backend_deploy_invokes_smoke_after_api_deploy() -> None:
    """deploy-backend.sh must run smoke tests after deploying the API."""
    script = (ROOT / "infra/gcp/deploy-backend.sh").read_text()

    # Smoke must appear after deploy-api.sh
    api_pos = script.index("deploy-api.sh")
    smoke_pos = script.index("run-backend-smoke.sh")
    assert smoke_pos > api_pos, "smoke must run after API deploy"

    # Smoke must appear before "Backend deploy complete"
    complete_pos = script.index("Backend deploy complete")
    assert smoke_pos < complete_pos, "smoke must run before declaring deploy complete"


def test_backend_deploy_supports_smoke_escape_hatch() -> None:
    """deploy-backend.sh must support RUN_DEPLOY_SMOKE=0."""
    script = (ROOT / "infra/gcp/deploy-backend.sh").read_text()

    assert "RUN_DEPLOY_SMOKE" in script
    assert "RUN_DEPLOY_SMOKE:-1" in script or "RUN_DEPLOY_SMOKE:-0" in script or "RUN_DEPLOY_SMOKE=0" in script


def test_backend_deploy_passes_image_tag_to_smoke() -> None:
    """deploy-backend.sh should pass EXPECTED_IMAGE_TAG to smoke."""
    script = (ROOT / "infra/gcp/deploy-backend.sh").read_text()

    assert "EXPECTED_IMAGE_TAG" in script


def test_smoke_runner_does_not_pass_secrets_as_cli_args() -> None:
    """run-backend-smoke.sh must not pass secrets as positional/flag CLI args."""
    smoke_script = (ROOT / "infra/gcp/run-backend-smoke.sh").read_text()

    for line in smoke_script.splitlines():
        stripped = line.strip()
        # Skip comments and blank lines
        if not stripped or stripped.startswith("#"):
            continue
        # Check we never pass secret values as CLI arguments directly
        if "--smoke-password" in stripped or "--proxy-secret" in stripped:
            raise AssertionError(f"Secret passed as CLI arg: {stripped}")


def test_api_deploy_includes_migration_head_env_var() -> None:
    """deploy-api.sh must set TALISMAN_RELEASE_MIGRATION_HEAD."""
    script = (ROOT / "infra/gcp/deploy-api.sh").read_text()

    assert "TALISMAN_RELEASE_MIGRATION_HEAD" in script


def test_config_example_includes_smoke_hash_secret() -> None:
    """config.example.sh must bind AUTH_SMOKE_PASSWORD_HASH for the API."""
    config = (ROOT / "infra/gcp/config.example.sh").read_text()

    assert "AUTH_SMOKE_PASSWORD_HASH" in config
