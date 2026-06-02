"""Dispatch and execution helpers for registered async jobs."""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import sys
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.responses import JSONResponse

from api.audit import emit_audit_event
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
from api.job_registry import (
    cache_key_for_payload,
    completed_ttl_for_request,
    get_job_spec,
    import_string,
    parse_request,
)
from api.observability import capture_exception, init_sentry

logger = logging.getLogger("api.async_job_runner")


def _hash_cache_key(cache_key: str | None) -> str | None:
    if not cache_key:
        return None
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]


def _job_actor(row_or_payload: dict[str, Any] | None) -> Any:
    if not isinstance(row_or_payload, dict):
        return None
    payload = row_or_payload.get("payload_json") if "payload_json" in row_or_payload else row_or_payload
    return payload.get("actor") if isinstance(payload, dict) else None


def _emit_job_audit(
    action_name: str,
    *,
    row: dict[str, Any] | None = None,
    job_id: str | None = None,
    job_type: str | None = None,
    status: str,
    metadata: dict[str, Any] | None = None,
    after_summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    resolved_job_id = job_id or str((row or {}).get("job_id") or "")
    resolved_job_type = job_type or str((row or {}).get("job_type") or "")
    emit_audit_event(
        action_name,
        "async_job",
        status,
        actor=_job_actor(row),
        object_refs=[
            {"type": "async_job", "id": resolved_job_id},
            {"type": "async_job_type", "id": resolved_job_type},
        ],
        metadata={
            "job_id": resolved_job_id,
            "job_type": resolved_job_type,
            "backend": _env_backend(),
            "cache_key_hash": _hash_cache_key(str((row or {}).get("cache_key") or "")),
            **(metadata or {}),
        },
        after_summary=after_summary,
        error=error,
    )


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


def _env_bool(name: str, *, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return True
    if raw in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _success_read_audit_enabled() -> bool:
    return _env_bool("ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED", default=False)


def _agent_chat_dispatch_backend() -> str | None:
    value = _normalize_backend(os.getenv("AGENT_CHAT_DISPATCH_BACKEND") or "")
    if not value:
        return None
    return _normalize_dispatch_backend(value)


def _normalize_dispatch_backend(value: str) -> str:
    if value in {"warm_worker", "warm_workers", "postgres_poll", "postgres_poller"}:
        return "warm_worker"
    if value in {"cloud_run_jobs", "cloudrunjobs", "rq"}:
        return "cloud_run_jobs"
    if value in {"inline", "sync", "synchronous"}:
        return "inline"
    if value in {"local", "memory", "in_memory", "in_process"}:
        return "local"
    return value


def _job_dispatch_backend(job_type: str) -> str | None:
    env_key = f"ASYNC_DISPATCH_BACKEND_{job_type.upper().replace('-', '_')}"
    value = _normalize_backend(os.getenv(env_key) or "")
    if not value:
        return None
    return _normalize_dispatch_backend(value)


def _dispatch_backend_for_job(job_type: str) -> str:
    if override := _job_dispatch_backend(job_type):
        return override
    if job_type == "agent_chat_turn":
        return _agent_chat_dispatch_backend() or _env_backend()
    return _env_backend()


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
    stale_grace_s = spec.stale_grace_s if spec.stale_grace_s is not None else _stale_grace_seconds()
    expires_at = reference_time + timedelta(seconds=spec.timeout_s + stale_grace_s)
    if checked_at <= expires_at:
        return row

    job_id = str(row.get("job_id") or "")
    if not job_id:
        return row

    error = f"Async job exceeded timeout before completion (timeout={spec.timeout_s}s, grace={stale_grace_s}s)."
    fail_job(job_id, error, result_ttl_seconds=spec.failed_ttl_s)
    refreshed = get_job(job_id)
    _emit_job_audit(
        "async_job.stale_failed",
        row=refreshed or row,
        status="failed",
        after_summary={"status": "failed", "reason": "stale_timeout"},
        error=error,
    )
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
            _emit_job_audit(
                "async_job.reused",
                row=row,
                status=str(row.get("status") or disposition),
                metadata={"disposition": disposition, "reuse_completed": reuse_completed},
                after_summary={"status": row.get("status"), "disposition": disposition},
            )
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
            _emit_job_audit(
                "async_job.reused",
                row=row,
                status=str(row.get("status") or disposition),
                metadata={"disposition": disposition, "reuse_completed": reuse_completed},
                after_summary={"status": row.get("status"), "disposition": disposition},
            )
            return _sync_stale_active_job(row), disposition

    try:
        dispatch_backend = _dispatch_backend_for_job(job_type)
        if dispatch_backend == "warm_worker":
            logger.info("async job queued for warm worker job_type=%s job_id=%s", job_type, row["job_id"])
        elif dispatch_backend in {"cloud_run_jobs", "cloudrunjobs"}:
            _enqueue_cloud_run_job(job_type, str(row["job_id"]))
        elif dispatch_backend == "inline":
            logger.info("async job running inline job_type=%s job_id=%s", job_type, row["job_id"])
            try:
                perform_job(str(row["job_id"]))
            except Exception:
                logger.exception("inline async job failed job_type=%s job_id=%s", job_type, row["job_id"])
            row = get_job(str(row["job_id"])) or row
        else:
            _enqueue_local_job(str(row["job_id"]))
    except Exception as exc:
        detail = str(exc) or "Failed to enqueue async job"
        fail_job(str(row["job_id"]), detail, result_ttl_seconds=spec.failed_ttl_s)
        capture_exception(
            exc,
            tags={"job_type": job_type, "backend": _dispatch_backend_for_job(job_type)},
            context={"job_id": str(row.get("job_id") or ""), "phase": "dispatch"},
        )
        failed = get_job(str(row["job_id"])) or row
        _emit_job_audit(
            "async_job.dispatch_failed",
            row=failed,
            status="failed",
            metadata={"disposition": disposition},
            after_summary={"status": "failed", "disposition": disposition},
            error=detail,
        )
        raise AsyncJobDispatchError(detail) from exc
    _emit_job_audit(
        "async_job.enqueued",
        row=row,
        status=str(row.get("status") or "queued"),
        metadata={"disposition": disposition, "reuse_completed": reuse_completed},
        after_summary={"status": row.get("status"), "disposition": disposition},
    )
    return row, disposition


def perform_job(job_id: str) -> dict[str, Any] | None:
    row = get_job(job_id)
    if not row:
        raise RuntimeError(f"Unknown async job: {job_id}")
    status = str(row.get("status") or "")
    if status == "completed":
        result = row.get("result_json")
        return result if isinstance(result, dict) else None
    if status in {"failed", "cancelled"}:
        logger.info("skip terminal async job job_id=%s status=%s", job_id, status)
        return None

    job_type = str(row.get("job_type") or "")
    spec = get_job_spec(job_type)
    payload = row.get("payload_json")
    if not isinstance(payload, dict):
        payload = {}

    try:
        mark_job_running(job_id)
        running_row = get_job(job_id) or row
        if str(running_row.get("status") or "") == "cancelled":
            logger.info("skip cancelled async job job_id=%s", job_id)
            return None
        _emit_job_audit(
            "async_job.running",
            row=running_row,
            status="running",
            after_summary={"status": "running"},
        )
        req = parse_request(spec, payload)
        completed_ttl_s = completed_ttl_for_request(spec, req)

        if spec.supports_progress:
            update_job_progress(job_id, {"phase": "running", "done": 0, "total": 0})

            def progress_callback(phase: str, done: int, total: int) -> None:
                update_job_progress(job_id, {"phase": phase, "done": done, "total": total})

            compute = import_string(spec.compute_func)
            params = inspect.signature(compute).parameters
            kwargs: dict[str, Any] = {"progress_callback": progress_callback}
            if "job_id" in params:
                kwargs["job_id"] = job_id
            result = compute(req, **kwargs)
        else:
            compute = import_string(spec.compute_func)
            params = inspect.signature(compute).parameters
            if "job_id" in params:
                result = compute(req, job_id=job_id)
            else:
                result = compute(req)

        if not isinstance(result, dict):
            result = {"result": result}

        if spec.supports_progress:
            final_count = result.get("final_count", 0)
            update_job_progress(job_id, {"phase": "done", "done": final_count, "total": final_count})
        complete_job(job_id, result, result_ttl_seconds=completed_ttl_s)
        completed_row = get_job(job_id) or row
        if str(completed_row.get("status") or "") == "cancelled":
            _emit_job_audit(
                "async_job.cancelled",
                row=completed_row,
                status="cancelled",
                after_summary={"status": "cancelled"},
            )
            return None
        _emit_job_audit(
            "async_job.completed",
            row=completed_row,
            status="succeeded",
            after_summary={
                "status": "completed",
                "result_keys": sorted(result.keys()),
                "final_count": result.get("final_count"),
            },
        )
        return result
    except Exception as exc:
        cancelled_row = get_job(job_id) or row
        if str(cancelled_row.get("status") or "") == "cancelled":
            _emit_job_audit(
                "async_job.cancelled",
                row=cancelled_row,
                status="cancelled",
                after_summary={"status": "cancelled"},
            )
            return None
        error = str(exc) or spec.error_message
        fail_job(job_id, error, result_ttl_seconds=spec.failed_ttl_s)
        failed_row = get_job(job_id) or row
        _emit_job_audit(
            "async_job.failed",
            row=failed_row,
            status="failed",
            after_summary={"status": "failed"},
            error=error,
        )
        capture_exception(
            exc,
            tags={"job_type": job_type, "backend": _env_backend()},
            context={"job_id": job_id, "phase": "perform_job"},
        )
        raise


def job_response(row: dict[str, Any]) -> dict[str, Any]:
    job_id = str(row.get("job_id") or "")
    job_type = str(row.get("job_type") or "")
    status = str(row.get("status") or "queued")
    progress = row.get("progress_json")
    if progress == {}:
        progress = None

    if status == "completed":
        payload: dict[str, Any] = {"job_id": job_id, "status": "done", "result": row.get("result_json")}
    elif status == "failed":
        payload = {"job_id": job_id, "status": "error", "error": row.get("error") or "Job failed"}
    elif status == "cancelled":
        payload = {"job_id": job_id, "status": "cancelled", "error": row.get("error") or "Job cancelled"}
    else:
        payload = {"job_id": job_id, "status": status}

    if job_type:
        payload["timeout_s"] = get_job_spec(job_type).timeout_s

    if isinstance(progress, dict) and progress:
        payload["progress"] = progress
    return payload


def poll_registered_job(job_id: str, *, row: dict[str, Any] | None = None) -> dict[str, Any]:
    row = row or get_job(job_id)
    if not row:
        raise KeyError(job_id)
    row = _sync_stale_active_job(row)
    response = job_response(row)
    if _success_read_audit_enabled():
        _emit_job_audit(
            "async_job.read",
            row=row,
            status=str(row.get("status") or response.get("status") or "unknown"),
            after_summary={"status": response.get("status"), "has_progress": bool(response.get("progress"))},
        )
    return response


def enqueue_response(row: dict[str, Any], poll_path: str) -> JSONResponse:
    payload = job_response(row)
    status_code = 200 if payload.get("status") == "done" else 202
    headers = {"Location": poll_path.format(job_id=payload.get("job_id"))}
    return JSONResponse(payload, status_code=status_code, headers=headers)


def main(argv: list[str] | None = None) -> int:
    from api.logging_config import configure_logging

    configure_logging(json_format=(os.getenv("ENVIRONMENT") or "").strip().lower() == "production")
    init_sentry(component="async_job_runner")

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
    except Exception as exc:
        logger.exception("async job execution failed job_id=%s", job_id)
        capture_exception(exc, tags={"phase": "cloud_run_job"}, context={"job_id": job_id})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
