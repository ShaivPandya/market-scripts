import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARGS_RE = re.compile(r"--args=([^\\\s]+)")


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

    assert "--args=-m,api.job_worker_loop,run \\" in script
    assert "--job-type,sizer,--queue,sizer" not in script
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


def test_api_deploy_routes_analyzer_to_warm_worker() -> None:
    script = (ROOT / "infra/gcp/deploy-api.sh").read_text()

    assert '"ASYNC_DISPATCH_BACKEND_ANALYZER=warm_worker"' in script
    assert '"ASYNC_QUEUE_ANALYZER=analyzer"' in script
    assert '"ASYNC_ANALYZER_COMPLETED_TTL_SECONDS=300"' in script


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
