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
