"""RQ enqueueing and execution helpers for registered async jobs."""

from __future__ import annotations

import os
import threading
from typing import Any

from fastapi.responses import JSONResponse

from api.job_queue import (
    complete_job,
    create_or_reuse_job,
    fail_job,
    get_job,
    mark_job_running,
    postgres_jobs_enabled,
    set_rq_job_id,
    update_job_progress,
)
from api.job_registry import cache_key_for_payload, get_job_spec, import_string, parse_request


def _env_backend() -> str:
    explicit = (os.getenv("ASYNC_JOB_BACKEND") or "").strip().lower()
    if explicit:
        return explicit
    if postgres_jobs_enabled() and (os.getenv("REDIS_URL") or "").strip():
        return "rq"
    return "local"


def _redis_url() -> str:
    url = (os.getenv("REDIS_URL") or "").strip()
    if not url:
        raise RuntimeError("REDIS_URL is required when ASYNC_JOB_BACKEND=rq.")
    return url


def _rq_queue(queue_name: str, timeout_s: int):
    try:
        from redis import Redis
        from rq import Queue
    except ImportError as exc:
        raise RuntimeError("rq and redis are required when ASYNC_JOB_BACKEND=rq.") from exc

    connection = Redis.from_url(_redis_url())
    return Queue(queue_name, connection=connection, default_timeout=timeout_s)


def _enqueue_rq_job(job_type: str, job_id: str) -> None:
    spec = get_job_spec(job_type)
    queue = _rq_queue(spec.queue_name, spec.timeout_s)
    rq_job = queue.enqueue(
        perform_job,
        job_id,
        job_id=job_id,
        timeout=spec.timeout_s,
        result_ttl=spec.completed_ttl_s,
        failure_ttl=spec.failed_ttl_s,
        description=f"{job_type}:{job_id}",
    )
    set_rq_job_id(job_id, rq_job.id)


def _enqueue_local_job(job_id: str) -> None:
    def _run() -> None:
        try:
            perform_job(job_id)
        except Exception:
            return

    thread = threading.Thread(target=_run, name=f"async-job-{job_id}", daemon=True)
    thread.start()


def enqueue_registered_job(
    job_type: str,
    payload: dict[str, Any],
    *,
    cache_key: str | None = None,
    reuse_completed: bool = True,
) -> tuple[dict[str, Any], str]:
    spec = get_job_spec(job_type)
    key = cache_key if cache_key is not None else cache_key_for_payload(spec, payload)
    row, disposition = create_or_reuse_job(
        job_type,
        payload=payload,
        cache_key=key,
        queue_name=spec.queue_name,
        initial_progress=spec.initial_progress,
        reuse_completed=reuse_completed,
    )
    if disposition != "created":
        return row, disposition

    try:
        if _env_backend() == "rq":
            _enqueue_rq_job(job_type, str(row["job_id"]))
        else:
            _enqueue_local_job(str(row["job_id"]))
    except Exception as exc:
        fail_job(str(row["job_id"]), str(exc) or "Failed to enqueue async job", result_ttl_seconds=spec.failed_ttl_s)
        raise
    return row, disposition


def perform_job(job_id: str) -> dict[str, Any] | None:
    row = get_job(job_id)
    if not row:
        raise RuntimeError(f"Unknown async job: {job_id}")

    job_type = str(row.get("job_type") or "")
    spec = get_job_spec(job_type)
    payload = row.get("payload_json")
    if not isinstance(payload, dict):
        payload = {}

    try:
        req = parse_request(spec, payload)
        mark_job_running(job_id)

        if spec.supports_progress:
            update_job_progress(job_id, {"phase": "running", "done": 0, "total": 0})

            def progress_callback(phase: str, done: int, total: int) -> None:
                update_job_progress(job_id, {"phase": phase, "done": done, "total": total})

            result = import_string(spec.compute_func)(req, progress_callback=progress_callback)
        else:
            result = import_string(spec.compute_func)(req)

        if not isinstance(result, dict):
            result = {"result": result}

        if spec.supports_progress:
            final_count = result.get("final_count", 0)
            update_job_progress(job_id, {"phase": "done", "done": final_count, "total": final_count})
        complete_job(job_id, result, result_ttl_seconds=spec.completed_ttl_s)
        return result
    except Exception as exc:
        fail_job(job_id, str(exc) or spec.error_message, result_ttl_seconds=spec.failed_ttl_s)
        raise


def job_response(row: dict[str, Any]) -> dict[str, Any]:
    job_id = str(row.get("job_id") or "")
    status = str(row.get("status") or "queued")
    progress = row.get("progress_json")
    if progress == {}:
        progress = None

    if status == "completed":
        payload: dict[str, Any] = {"job_id": job_id, "status": "done", "result": row.get("result_json")}
    elif status == "failed":
        payload = {"job_id": job_id, "status": "error", "error": row.get("error") or "Job failed"}
    else:
        payload = {"job_id": job_id, "status": status}

    if isinstance(progress, dict) and progress:
        payload["progress"] = progress
    return payload


def poll_registered_job(job_id: str) -> dict[str, Any]:
    row = get_job(job_id)
    if not row:
        raise KeyError(job_id)
    return job_response(row)


def enqueue_response(row: dict[str, Any], poll_path: str) -> JSONResponse:
    payload = job_response(row)
    status_code = 200 if payload.get("status") == "done" else 202
    headers = {"Location": poll_path.format(job_id=payload.get("job_id"))}
    return JSONResponse(payload, status_code=status_code, headers=headers)
