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


def test_sizer_worker_deploy_uses_env_for_duplicate_sizer_values() -> None:
    script = (ROOT / "infra/gcp/deploy-sizer-worker.sh").read_text()
    config_example = (ROOT / "infra/gcp/config.example.sh").read_text()

    assert "--args=-m,api.job_worker_loop,run \\" in script
    assert "--job-type,sizer,--queue,sizer" not in script
    assert "SIZER_WORKER_INSTANCES=0 is incompatible" in script
    assert 'SIZER_WORKER_INSTANCES="1"' in script
    assert 'SIZER_WORKER_INSTANCES="1"' in config_example
    assert '"ASYNC_DISPATCH_BACKEND_SIZER=warm_worker"' in script
    assert '"JOB_WORKER_JOB_TYPE=sizer"' in script
    assert '"JOB_WORKER_QUEUE=sizer"' in script


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


def test_api_deploy_routes_noninteractive_jobs_to_cloud_run_jobs() -> None:
    script = (ROOT / "infra/gcp/deploy-api.sh").read_text()

    assert '"ASYNC_DISPATCH_BACKEND_ANALYZER=cloud_run_jobs"' in script
    assert '"ASYNC_DISPATCH_BACKEND_SIZER=warm_worker"' in script
    assert '"ASYNC_DISPATCH_BACKEND_ONTOLOGY=cloud_run_jobs"' in script
    assert '"AGENT_CHAT_DISPATCH_BACKEND=warm_worker"' in script
    assert '"ASYNC_QUEUE_ANALYZER=analyzer"' in script
    assert '"ASYNC_ANALYZER_COMPLETED_TTL_SECONDS=300"' in script


def test_common_gcp_env_enables_ontology_read_model_for_api_and_ontology_worker() -> None:
    lib = (ROOT / "infra/gcp/lib.sh").read_text()

    assert "ONTOLOGY_READ_MODEL=true" in lib

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
