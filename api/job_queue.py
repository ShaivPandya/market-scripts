"""Postgres-backed status rows for work executed outside API requests."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any

from api.postgres import connect, database_url

ACTIVE_STATUSES = ("queued", "running")


def _now() -> datetime:
    return datetime.now(UTC)


def postgres_jobs_enabled() -> bool:
    return bool(database_url())


def create_job(
    job_type: str,
    *,
    payload: dict[str, Any] | None = None,
    job_id: str | None = None,
    cloud_run_job_name: str | None = None,
) -> dict[str, Any]:
    from psycopg.types.json import Jsonb

    jid = job_id or str(uuid.uuid4())
    now = _now()
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


def mark_job_running(job_id: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE async_jobs SET status = 'running', started_at = COALESCE(started_at, %s), updated_at = %s WHERE job_id = %s",
            (_now(), _now(), job_id),
        )
        conn.commit()


def complete_job(job_id: str, result: dict[str, Any] | None = None) -> None:
    from psycopg.types.json import Jsonb

    now = _now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE async_jobs
            SET status = 'completed', result_json = %s, completed_at = %s, updated_at = %s
            WHERE job_id = %s
            """,
            (Jsonb(result or {}), now, now, job_id),
        )
        conn.commit()


def fail_job(job_id: str, error: str) -> None:
    now = _now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE async_jobs
            SET status = 'failed', error = %s, completed_at = %s, updated_at = %s
            WHERE job_id = %s
            """,
            (error, now, now, job_id),
        )
        conn.commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    if not postgres_jobs_enabled():
        return None
    with connect() as conn:
        row = conn.execute("SELECT * FROM async_jobs WHERE job_id = %s", (job_id,)).fetchone()
        return dict(row) if row else None


def count_active_jobs() -> int:
    if not postgres_jobs_enabled():
        return 0
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM async_jobs WHERE status = ANY(%s)", (list(ACTIVE_STATUSES),)
        ).fetchone()
        return int(row["count"]) if row else 0


def cloud_run_job_name(job_type: str) -> str:
    env_key = f"CLOUD_RUN_JOB_{job_type.upper().replace('-', '_')}"
    return os.getenv(env_key, f"market-scripts-{job_type}")
