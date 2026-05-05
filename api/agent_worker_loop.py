"""Warm polling worker for durable agent chat turns."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from typing import Any

from api.async_job_runner import perform_job
from api.job_queue import claim_queued_job
from api.logging_config import configure_logging

logger = logging.getLogger("api.agent_worker_loop")

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
    import api.agent_chat_worker  # noqa: F401
    import api.routers.agent  # noqa: F401


def run_once(*, job_type: str = DEFAULT_JOB_TYPE, queue_name: str = DEFAULT_QUEUE_NAME) -> bool:
    row = claim_queued_job(job_type, queue_name=queue_name)
    if row is None:
        return False

    job_id = str(row.get("job_id") or "")
    if not job_id:
        logger.error("claimed agent job without job_id row=%s", row)
        return True

    try:
        perform_job(job_id)
    except Exception:
        logger.exception("agent worker failed job_id=%s", job_id)
    return True


def run_loop(
    *,
    job_type: str = DEFAULT_JOB_TYPE,
    queue_name: str = DEFAULT_QUEUE_NAME,
    poll_interval_s: float | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    _preload_agent_path()
    interval = (
        poll_interval_s if poll_interval_s is not None else _env_float("AGENT_WORKER_POLL_INTERVAL_SECONDS", 0.25)
    )
    stop = stop_event or threading.Event()
    logger.info("agent worker loop started job_type=%s queue=%s poll_interval_s=%.3f", job_type, queue_name, interval)

    while not stop.is_set():
        claimed = run_once(job_type=job_type, queue_name=queue_name)
        if not claimed:
            stop.wait(interval)

    logger.info("agent worker loop stopped")


def main(argv: list[str] | None = None) -> int:
    configure_logging(json_format=(os.getenv("ENVIRONMENT") or "").strip().lower() == "production")
    args = list(argv if argv is not None else sys.argv[1:])
    command = args.pop(0) if args else "run"
    if command not in {"run", "run-once"}:
        print("Usage: python -m api.agent_worker_loop run|run-once", file=sys.stderr)
        return 2

    if command == "run-once":
        _preload_agent_path()
        return 0 if run_once() else 1

    stop = threading.Event()

    def _handle_signal(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    run_loop(stop_event=stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
