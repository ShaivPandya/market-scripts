"""Async job status storage.

Production uses Postgres as the durable source of truth. Local development and
tests use an in-process fallback unless async jobs are explicitly configured for
Postgres-backed execution, so a production ``DATABASE_URL`` in ``.env`` does not
leak into local dashboard workflows.
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from api.postgres import connect, database_url

ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("completed", "failed")

_memory_jobs: dict[str, dict[str, Any]] = {}
_memory_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(UTC)


def postgres_jobs_enabled() -> bool:
    backend = (os.getenv("ASYNC_JOB_BACKEND") or "").strip().lower()
    if backend in {"local", "memory", "in-memory", "in_process", "in-process"}:
        return False
    if backend in {"cloud_run_jobs", "cloud-run-jobs", "cloudrunjobs", "rq", "postgres"}:
        return bool(database_url())
    return os.getenv("ENVIRONMENT", "development").strip().lower() == "production" and bool(database_url())


def _expires_in(seconds: int | None) -> datetime | None:
    if seconds is None:
        return None
    return _now() + timedelta(seconds=max(0, int(seconds)))


def _memory_timestamp(row: dict[str, Any], *keys: str) -> datetime:
    for key in keys:
        value = row.get(key)
        if isinstance(value, datetime):
            return value
    return datetime.min.replace(tzinfo=UTC)


def _memory_result_expires_after(row: dict[str, Any], now: datetime) -> bool:
    expires_at = row.get("result_expires_at")
    return expires_at is None or (isinstance(expires_at, datetime) and expires_at > now)


def _memory_result_expired(row: dict[str, Any], cutoff: datetime) -> bool:
    expires_at = row.get("result_expires_at")
    return isinstance(expires_at, datetime) and expires_at < cutoff


def _memory_select_completed(job_type: str, cache_key: str | None, now: datetime) -> dict[str, Any] | None:
    if not cache_key:
        return None
    candidates = [
        job
        for job in _memory_jobs.values()
        if job.get("job_type") == job_type
        and job.get("cache_key") == cache_key
        and job.get("status") == "completed"
        and _memory_result_expires_after(job, now)
    ]
    if not candidates:
        return None
    return dict(sorted(candidates, key=lambda row: _memory_timestamp(row, "updated_at", "created_at"), reverse=True)[0])


def _memory_select_active(job_type: str, cache_key: str | None) -> dict[str, Any] | None:
    if not cache_key:
        return None
    candidates = [
        job
        for job in _memory_jobs.values()
        if job.get("job_type") == job_type
        and job.get("cache_key") == cache_key
        and job.get("status") in ACTIVE_STATUSES
    ]
    if not candidates:
        return None
    return dict(sorted(candidates, key=lambda row: _memory_timestamp(row, "created_at"))[0])


def _select_completed_postgres(job_type: str, cache_key: str | None) -> dict[str, Any] | None:
    if not cache_key:
        return None
    now = _now()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM async_jobs
            WHERE job_type = %s
              AND cache_key = %s
              AND status = 'completed'
              AND (result_expires_at IS NULL OR result_expires_at > %s)
            ORDER BY completed_at DESC NULLS LAST, updated_at DESC
            LIMIT 1
            """,
            (job_type, cache_key, now),
        ).fetchone()
        return dict(row) if row else None


def _select_active_postgres(job_type: str, cache_key: str | None) -> dict[str, Any] | None:
    if not cache_key:
        return None
    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM async_jobs
            WHERE job_type = %s
              AND cache_key = %s
              AND status = ANY(%s)
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (job_type, cache_key, list(ACTIVE_STATUSES)),
        ).fetchone()
        return dict(row) if row else None


def create_or_reuse_job(
    job_type: str,
    *,
    payload: dict[str, Any] | None = None,
    cache_key: str | None = None,
    queue_name: str | None = None,
    initial_progress: dict[str, Any] | None = None,
    job_id: str | None = None,
    reuse_completed: bool = True,
) -> tuple[dict[str, Any], str]:
    """Create a queued job or return an existing reusable job.

    Returns ``(row, disposition)`` where disposition is one of:
    ``created``, ``active``, or ``completed``.

    Postgres dedupe relies on the migration's partial unique index:
    ``UNIQUE (job_type, cache_key) WHERE status IN ('queued','running')
    AND cache_key IS NOT NULL``.  The insert uses the same conflict target so
    concurrent requests collapse to one active row.
    """
    now = _now()
    if not postgres_jobs_enabled():
        with _memory_lock:
            if reuse_completed:
                completed = _memory_select_completed(job_type, cache_key, now)
                if completed:
                    return completed, "completed"
            active = _memory_select_active(job_type, cache_key)
            if active:
                return active, "active"
            jid = job_id or str(uuid.uuid4())
            row = {
                "job_id": jid,
                "job_type": job_type,
                "status": "queued",
                "payload_json": payload or {},
                "result_json": None,
                "error": None,
                "cache_key": cache_key,
                "queue_name": queue_name,
                "rq_job_id": None,
                "progress_json": initial_progress,
                "cloud_run_job_name": None,
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "updated_at": now,
                "result_expires_at": None,
            }
            _memory_jobs[jid] = row
            return dict(row), "created"

    from psycopg.types.json import Jsonb

    if reuse_completed:
        completed = _select_completed_postgres(job_type, cache_key)
        if completed:
            return completed, "completed"

    jid = job_id or str(uuid.uuid4())
    for _attempt in range(2):
        with connect() as conn:
            row = conn.execute(
                """
                INSERT INTO async_jobs
                    (job_id, job_type, status, payload_json, cache_key, queue_name, progress_json, created_at, updated_at)
                VALUES (%s, %s, 'queued', %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_type, cache_key)
                    WHERE status IN ('queued', 'running') AND cache_key IS NOT NULL
                DO NOTHING
                RETURNING *
                """,
                (jid, job_type, Jsonb(payload or {}), cache_key, queue_name, Jsonb(initial_progress or {}), now, now),
            ).fetchone()
            conn.commit()
            if row:
                return dict(row), "created"

        active = _select_active_postgres(job_type, cache_key)
        if active:
            return active, "active"
        if reuse_completed:
            completed = _select_completed_postgres(job_type, cache_key)
            if completed:
                return completed, "completed"

    # If the conflicting active row completed or failed between the INSERT and
    # re-selects, make one final non-recursive attempt with a fresh id.
    jid = str(uuid.uuid4())
    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO async_jobs
                (job_id, job_type, status, payload_json, cache_key, queue_name, progress_json, created_at, updated_at)
            VALUES (%s, %s, 'queued', %s, %s, %s, %s, %s, %s)
            ON CONFLICT (job_type, cache_key)
                WHERE status IN ('queued', 'running') AND cache_key IS NOT NULL
            DO NOTHING
            RETURNING *
            """,
            (jid, job_type, Jsonb(payload or {}), cache_key, queue_name, Jsonb(initial_progress or {}), now, now),
        ).fetchone()
        conn.commit()
        if row:
            return dict(row), "created"

    active = _select_active_postgres(job_type, cache_key)
    if active:
        return active, "active"
    raise RuntimeError(f"Could not create or reuse async job for {job_type}.")


def create_job(
    job_type: str,
    *,
    payload: dict[str, Any] | None = None,
    job_id: str | None = None,
    cloud_run_job_name: str | None = None,
) -> dict[str, Any]:
    jid = job_id or str(uuid.uuid4())
    now = _now()
    if not postgres_jobs_enabled():
        with _memory_lock:
            row = {
                "job_id": jid,
                "job_type": job_type,
                "status": "queued",
                "payload_json": payload or {},
                "result_json": None,
                "error": None,
                "cache_key": None,
                "queue_name": None,
                "rq_job_id": None,
                "progress_json": None,
                "cloud_run_job_name": cloud_run_job_name,
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "updated_at": now,
                "result_expires_at": None,
            }
            _memory_jobs[jid] = row
            return dict(row)

    from psycopg.types.json import Jsonb

    payload_json = Jsonb(payload or {})
    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO async_jobs
                (job_id, job_type, status, payload_json, cloud_run_job_name, created_at, updated_at)
            VALUES (%s, %s, 'queued', %s, %s, %s, %s)
            ON CONFLICT (job_id) DO UPDATE SET
                payload_json = EXCLUDED.payload_json,
                cloud_run_job_name = EXCLUDED.cloud_run_job_name,
                updated_at = EXCLUDED.updated_at
            RETURNING *
            """,
            (jid, job_type, payload_json, cloud_run_job_name, now, now),
        ).fetchone()
        conn.commit()
        return dict(row)


def set_cloud_run_job_name(job_id: str, cloud_run_job_name: str) -> None:
    now = _now()
    if not postgres_jobs_enabled():
        with _memory_lock:
            if job_id in _memory_jobs:
                _memory_jobs[job_id]["cloud_run_job_name"] = cloud_run_job_name
                _memory_jobs[job_id]["updated_at"] = now
        return
    with connect() as conn:
        conn.execute(
            "UPDATE async_jobs SET cloud_run_job_name = %s, updated_at = %s WHERE job_id = %s",
            (cloud_run_job_name, now, job_id),
        )
        conn.commit()


def mark_job_running(job_id: str) -> None:
    now = _now()
    if not postgres_jobs_enabled():
        with _memory_lock:
            if job_id in _memory_jobs:
                _memory_jobs[job_id]["status"] = "running"
                _memory_jobs[job_id]["started_at"] = _memory_jobs[job_id].get("started_at") or now
                _memory_jobs[job_id]["updated_at"] = now
        return
    with connect() as conn:
        conn.execute(
            "UPDATE async_jobs SET status = 'running', started_at = COALESCE(started_at, %s), updated_at = %s WHERE job_id = %s",
            (now, now, job_id),
        )
        conn.commit()


def update_job_progress(job_id: str, progress: dict[str, Any] | None) -> None:
    now = _now()
    if not postgres_jobs_enabled():
        with _memory_lock:
            if job_id in _memory_jobs:
                _memory_jobs[job_id]["progress_json"] = progress
                _memory_jobs[job_id]["updated_at"] = now
        return
    from psycopg.types.json import Jsonb

    with connect() as conn:
        conn.execute(
            "UPDATE async_jobs SET progress_json = %s, updated_at = %s WHERE job_id = %s",
            (Jsonb(progress or {}), now, job_id),
        )
        conn.commit()


def complete_job(job_id: str, result: dict[str, Any] | None = None, *, result_ttl_seconds: int | None = None) -> None:
    now = _now()
    expires_at = _expires_in(result_ttl_seconds)
    if not postgres_jobs_enabled():
        with _memory_lock:
            if job_id in _memory_jobs:
                _memory_jobs[job_id]["status"] = "completed"
                _memory_jobs[job_id]["result_json"] = result or {}
                _memory_jobs[job_id]["completed_at"] = now
                _memory_jobs[job_id]["updated_at"] = now
                _memory_jobs[job_id]["result_expires_at"] = expires_at
                _memory_jobs[job_id]["error"] = None
        return
    from psycopg.types.json import Jsonb

    with connect() as conn:
        conn.execute(
            """
            UPDATE async_jobs
            SET status = 'completed',
                result_json = %s,
                completed_at = %s,
                updated_at = %s,
                result_expires_at = %s,
                error = NULL
            WHERE job_id = %s
            """,
            (Jsonb(result or {}), now, now, expires_at, job_id),
        )
        conn.commit()


def fail_job(job_id: str, error: str, *, result_ttl_seconds: int | None = None) -> None:
    now = _now()
    expires_at = _expires_in(result_ttl_seconds)
    if not postgres_jobs_enabled():
        with _memory_lock:
            if job_id in _memory_jobs:
                _memory_jobs[job_id]["status"] = "failed"
                _memory_jobs[job_id]["error"] = error
                _memory_jobs[job_id]["completed_at"] = now
                _memory_jobs[job_id]["updated_at"] = now
                _memory_jobs[job_id]["result_expires_at"] = expires_at
        return
    with connect() as conn:
        conn.execute(
            """
            UPDATE async_jobs
            SET status = 'failed',
                error = %s,
                completed_at = %s,
                updated_at = %s,
                result_expires_at = %s
            WHERE job_id = %s
            """,
            (error, now, now, expires_at, job_id),
        )
        conn.commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    if not postgres_jobs_enabled():
        with _memory_lock:
            row = _memory_jobs.get(job_id)
            return dict(row) if row else None
    with connect() as conn:
        row = conn.execute("SELECT * FROM async_jobs WHERE job_id = %s", (job_id,)).fetchone()
        return dict(row) if row else None


def count_active_jobs() -> int:
    if not postgres_jobs_enabled():
        with _memory_lock:
            return sum(1 for row in _memory_jobs.values() if row.get("status") in ACTIVE_STATUSES)
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM async_jobs WHERE status = ANY(%s)", (list(ACTIVE_STATUSES),)
        ).fetchone()
        return int(row["count"]) if row else 0


def list_active_jobs() -> list[dict[str, Any]]:
    if not postgres_jobs_enabled():
        with _memory_lock:
            return [dict(row) for row in _memory_jobs.values() if row.get("status") in ACTIVE_STATUSES]
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM async_jobs
            WHERE status = ANY(%s)
            ORDER BY created_at ASC
            """,
            (list(ACTIVE_STATUSES),),
        ).fetchall()
        return [dict(row) for row in rows]


def clear_memory_jobs() -> None:
    """Clear local fallback jobs. Used by tests and local cache invalidation."""
    with _memory_lock:
        _memory_jobs.clear()


def sweep_expired_jobs(now: datetime | None = None) -> int:
    cutoff = now or _now()
    if not postgres_jobs_enabled():
        with _memory_lock:
            to_delete = [
                job_id
                for job_id, row in _memory_jobs.items()
                if row.get("status") in TERMINAL_STATUSES and _memory_result_expired(row, cutoff)
            ]
            for job_id in to_delete:
                _memory_jobs.pop(job_id, None)
            return len(to_delete)
    with connect() as conn:
        rows = conn.execute(
            """
            DELETE FROM async_jobs
            WHERE status = ANY(%s)
              AND result_expires_at IS NOT NULL
              AND result_expires_at < %s
            RETURNING job_id
            """,
            (list(TERMINAL_STATUSES), cutoff),
        ).fetchall()
        conn.commit()
        return len(rows)


def cloud_run_job_name(job_type: str | None = None) -> str:
    if value := (os.getenv("ASYNC_CLOUD_RUN_JOB") or os.getenv("ASYNC_JOB_RUNNER_JOB") or "").strip():
        return value
    if job_type:
        env_key = f"CLOUD_RUN_JOB_{job_type.upper().replace('-', '_')}"
        if value := (os.getenv(env_key) or "").strip():
            return value
    return "talisman-async-job"
