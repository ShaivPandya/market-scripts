"""Generic warm polling worker for durable async jobs."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from typing import Any

from api.async_job_runner import perform_job
from api.job_queue import claim_queued_job
from api.logging_config import configure_logging
from api.observability import capture_exception, init_sentry

logger = logging.getLogger("api.job_worker_loop")

DEFAULT_JOB_TYPE = "agent_chat_turn"
DEFAULT_QUEUE_NAME = "agent"
DEFAULT_POLL_INTERVAL_S = 0.25

_PRELOAD_MODULES: dict[str, tuple[str, ...]] = {
    "agent_chat_turn": (
        "api.agent_chat_worker",
        "api.routers.agent",
    ),
    "analyzer": (
        "portfolio.portfolio_optimizer.portfolio_analyzer",
        "api.routers.analyzer",
    ),
    "sizer": (
        "portfolio.portfolio_optimizer.portfolio_sizer",
        "api.routers.sizer",
    ),
    "ontology": (
        "ontology.service",
        "api.routers.ontology",
    ),
}


def _env_float(name: str, default: float) -> float:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return max(0.01, float(value))
    except ValueError:
        return default


def _is_transient_claim_error(exc: BaseException) -> bool:
    exc_type = type(exc)
    if exc_type.__name__ == "PoolTimeout" and exc_type.__module__.startswith("psycopg_pool"):
        return True
    try:
        import psycopg
    except ImportError:
        return False
    return isinstance(exc, psycopg.OperationalError)


def _db_error_backoff_s(poll_interval_s: float) -> float:
    return _env_float("JOB_WORKER_DB_ERROR_BACKOFF_SECONDS", max(1.0, poll_interval_s))


def _worker_env_prefix(job_type: str) -> str:
    return job_type.upper().replace("-", "_")


def _poll_interval_for_job(job_type: str) -> float:
    specific = f"{_worker_env_prefix(job_type)}_WORKER_POLL_INTERVAL_SECONDS"
    if os.getenv(specific):
        return _env_float(specific, DEFAULT_POLL_INTERVAL_S)
    return _env_float("JOB_WORKER_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_S)


def preload_job_path(job_type: str) -> None:
    """Import expensive modules once so jobs do not pay per-run cold import cost."""
    import importlib

    for module_name in _PRELOAD_MODULES.get(job_type, ()):
        importlib.import_module(module_name)


def run_once(*, job_type: str = DEFAULT_JOB_TYPE, queue_name: str = DEFAULT_QUEUE_NAME) -> bool:
    row = claim_queued_job(job_type, queue_name=queue_name)
    if row is None:
        return False

    job_id = str(row.get("job_id") or "")
    if not job_id:
        logger.error("claimed async job without job_id job_type=%s queue=%s row=%s", job_type, queue_name, row)
        return True

    try:
        perform_job(job_id)
    except Exception as exc:
        logger.exception("warm worker failed job_type=%s queue=%s job_id=%s", job_type, queue_name, job_id)
        capture_exception(
            exc,
            tags={"job_type": job_type, "queue": queue_name, "phase": "warm_worker"},
            context={"job_id": job_id},
        )
    return True


def run_loop(
    *,
    job_type: str = DEFAULT_JOB_TYPE,
    queue_name: str = DEFAULT_QUEUE_NAME,
    poll_interval_s: float | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    preload_job_path(job_type)
    interval = poll_interval_s if poll_interval_s is not None else _poll_interval_for_job(job_type)
    stop = stop_event or threading.Event()
    logger.info(
        "warm worker loop started job_type=%s queue=%s poll_interval_s=%.3f",
        job_type,
        queue_name,
        interval,
    )

    while not stop.is_set():
        try:
            claimed = run_once(job_type=job_type, queue_name=queue_name)
        except Exception as exc:
            if not _is_transient_claim_error(exc):
                raise
            logger.warning(
                "warm worker claim skipped after transient postgres connection error job_type=%s queue=%s",
                job_type,
                queue_name,
                exc_info=True,
            )
            stop.wait(_db_error_backoff_s(interval))
            continue
        if not claimed:
            stop.wait(interval)

    logger.info("warm worker loop stopped job_type=%s queue=%s", job_type, queue_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a warm async job worker.")
    parser.add_argument("command", nargs="?", choices=("run", "run-once"), default="run")
    parser.add_argument("--job-type", default=os.getenv("JOB_WORKER_JOB_TYPE") or DEFAULT_JOB_TYPE)
    parser.add_argument("--queue", dest="queue_name", default=os.getenv("JOB_WORKER_QUEUE") or DEFAULT_QUEUE_NAME)
    parser.add_argument("--poll-interval", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging(json_format=(os.getenv("ENVIRONMENT") or "").strip().lower() == "production")
    init_sentry(component="job_worker")
    args = _parser().parse_args(list(argv if argv is not None else sys.argv[1:]))

    if args.command == "run-once":
        preload_job_path(args.job_type)
        return 0 if run_once(job_type=args.job_type, queue_name=args.queue_name) else 1

    stop = threading.Event()

    def _handle_signal(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    run_loop(
        job_type=args.job_type,
        queue_name=args.queue_name,
        poll_interval_s=args.poll_interval,
        stop_event=stop,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
