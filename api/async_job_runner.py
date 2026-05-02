"""Dispatch and execution helpers for registered async jobs."""

from __future__ import annotations

import logging
import os
import sys
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.responses import JSONResponse

from api.exceptions import AsyncJobDispatchError
from api.job_queue import (
    ACTIVE_STATUSES,
    complete_job,
    create_or_reuse_job,
    fail_job,
    get_job,
    list_active_jobs,
    mark_job_running,
    update_job_progress,
)
from api.job_registry import cache_key_for_payload, get_job_spec, import_string, parse_request

logger = logging.getLogger("api.async_job_runner")


def _normalize_backend(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _env_backend() -> str:
    explicit = _normalize_backend(os.getenv("ASYNC_JOB_BACKEND") or "")
    if explicit:
        if explicit == "rq":
            return "cloud_run_jobs"
        return explicit
    cloud_run_enabled = (os.getenv("CLOUD_RUN_JOBS_ENABLED") or "").strip().lower()
    if cloud_run_enabled in {"1", "true", "yes"}:
        return "cloud_run_jobs"
    return "local"


def _stale_grace_seconds() -> int:
    value = (os.getenv("ASYNC_JOB_STALE_GRACE_SECONDS") or "").strip()
    if not value:
        return 300
    try:
        return max(0, int(value))
    except ValueError:
        return 300


def _as_aware_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _active_reference_time(row: dict[str, Any]) -> datetime | None:
    status = str(row.get("status") or "")
    if status == "running":
        return _as_aware_utc(row.get("started_at")) or _as_aware_utc(row.get("created_at"))
    if status == "queued":
        return _as_aware_utc(row.get("created_at"))
    return None


def _sync_stale_active_job(row: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    status = str(row.get("status") or "")
    if status not in ACTIVE_STATUSES:
        return row

    job_type = str(row.get("job_type") or "")
    spec = get_job_spec(job_type)
    reference_time = _active_reference_time(row)
    if reference_time is None:
        return row

    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=UTC)
    expires_at = reference_time + timedelta(seconds=spec.timeout_s + _stale_grace_seconds())
    if checked_at <= expires_at:
        return row

    job_id = str(row.get("job_id") or "")
    if not job_id:
        return row

    error = (
        f"Async job exceeded timeout before completion (timeout={spec.timeout_s}s, grace={_stale_grace_seconds()}s)."
    )
    fail_job(job_id, error, result_ttl_seconds=spec.failed_ttl_s)
    refreshed = get_job(job_id)
    return refreshed or row


def fail_stale_active_jobs(now: datetime | None = None) -> int:
    failed = 0
    for row in list_active_jobs():
        synced = _sync_stale_active_job(row, now=now)
        if str(row.get("status") or "") in ACTIVE_STATUSES and str(synced.get("status") or "") == "failed":
            failed += 1
    return failed


def _enqueue_cloud_run_job(job_type: str, job_id: str) -> None:
    from api.cloud_run_jobs import dispatch_cloud_run_job

    dispatch_cloud_run_job(job_type, job_id)


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
        row = _sync_stale_active_job(row)
        if str(row.get("status") or "") != "failed":
            return row, disposition

        row, disposition = create_or_reuse_job(
            job_type,
            payload=payload,
            cache_key=key,
            queue_name=spec.queue_name,
            initial_progress=spec.initial_progress,
            reuse_completed=reuse_completed,
        )
        if disposition != "created":
            return _sync_stale_active_job(row), disposition

    try:
        if _env_backend() in {"cloud_run_jobs", "cloudrunjobs"}:
            _enqueue_cloud_run_job(job_type, str(row["job_id"]))
        else:
            _enqueue_local_job(str(row["job_id"]))
    except Exception as exc:
        detail = str(exc) or "Failed to enqueue async job"
        fail_job(str(row["job_id"]), detail, result_ttl_seconds=spec.failed_ttl_s)
        raise AsyncJobDispatchError(detail) from exc
    return row, disposition


def perform_job(job_id: str) -> dict[str, Any] | None:
    row = get_job(job_id)
    if not row:
        raise RuntimeError(f"Unknown async job: {job_id}")
    status = str(row.get("status") or "")
    if status == "completed":
        result = row.get("result_json")
        return result if isinstance(result, dict) else None
    if status == "failed":
        logger.info("skip terminal async job job_id=%s status=%s", job_id, status)
        return None

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
    row = _sync_stale_active_job(row)
    return job_response(row)


def enqueue_response(row: dict[str, Any], poll_path: str) -> JSONResponse:
    payload = job_response(row)
    status_code = 200 if payload.get("status") == "done" else 202
    headers = {"Location": poll_path.format(job_id=payload.get("job_id"))}
    return JSONResponse(payload, status_code=status_code, headers=headers)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    command = args.pop(0) if args else "run"
    if command != "run":
        print("Usage: python -m api.async_job_runner run [job_id]", file=sys.stderr)
        return 2

    job_id = args[0] if args else (os.getenv("ASYNC_JOB_ID") or "").strip()
    if not job_id:
        print("ASYNC_JOB_ID is required when no job_id argument is provided.", file=sys.stderr)
        return 2

    try:
        perform_job(job_id)
    except Exception:
        logger.exception("async job execution failed job_id=%s", job_id)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
