"""Compatibility wrapper for the warm agent chat worker."""

from __future__ import annotations

import os
import sys
import threading

from api.job_worker_loop import main as _generic_main
from api.job_worker_loop import preload_job_path
from api.job_worker_loop import run_loop as _generic_run_loop
from api.job_worker_loop import run_once as _generic_run_once

DEFAULT_JOB_TYPE = "agent_chat_turn"
DEFAULT_QUEUE_NAME = "agent"


def _env_float(name: str, default: float) -> float:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return max(0.01, float(value))
    except ValueError:
        return default


def _preload_agent_path() -> None:
    """Import the heavy agent modules once so turns do not pay cold import cost."""
    preload_job_path(DEFAULT_JOB_TYPE)


def run_once(*, job_type: str = DEFAULT_JOB_TYPE, queue_name: str = DEFAULT_QUEUE_NAME) -> bool:
    return _generic_run_once(job_type=job_type, queue_name=queue_name)


def run_loop(
    *,
    job_type: str = DEFAULT_JOB_TYPE,
    queue_name: str = DEFAULT_QUEUE_NAME,
    poll_interval_s: float | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    interval = (
        poll_interval_s if poll_interval_s is not None else _env_float("AGENT_WORKER_POLL_INTERVAL_SECONDS", 0.25)
    )
    _generic_run_loop(
        job_type=job_type,
        queue_name=queue_name,
        poll_interval_s=interval,
        stop_event=stop_event or threading.Event(),
    )


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    command = args[0] if args else "run"
    if command in {"run", "run-once"}:
        args = [command, "--job-type", DEFAULT_JOB_TYPE, "--queue", DEFAULT_QUEUE_NAME, *args[1:]]
    elif "--job-type" not in args:
        args = ["--job-type", DEFAULT_JOB_TYPE, "--queue", DEFAULT_QUEUE_NAME, *args]
    os.environ.setdefault("JOB_WORKER_POLL_INTERVAL_SECONDS", os.getenv("AGENT_WORKER_POLL_INTERVAL_SECONDS", "0.25"))
    return _generic_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
