"""Durable event storage for replayable async job output."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from api.job_queue import postgres_jobs_enabled
from api.postgres import connect

_memory_events: dict[str, list[dict[str, Any]]] = {}
_memory_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(UTC)


def _row_to_event(row: Any) -> dict[str, Any]:
    return {
        "seq": int(row["seq"]),
        "event_type": str(row["event_type"]),
        "payload": row["payload_json"] if row["payload_json"] is not None else {},
        "created_at": row["created_at"].isoformat()
        if hasattr(row["created_at"], "isoformat")
        else str(row["created_at"]),
    }


def append_job_event(job_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append one event to a job's replay log and return the stored event."""
    body = payload or {}
    now = _now()
    if not postgres_jobs_enabled():
        with _memory_lock:
            events = _memory_events.setdefault(job_id, [])
            event = {
                "seq": len(events) + 1,
                "event_type": event_type,
                "payload": body,
                "created_at": now.isoformat(),
            }
            events.append(event)
            return dict(event)

    from psycopg.types.json import Jsonb

    with connect() as conn:
        # Lock the parent job row so concurrent API cancellation and worker
        # writes cannot race on the next sequence number.
        conn.execute("SELECT job_id FROM async_jobs WHERE job_id = %s FOR UPDATE", (job_id,)).fetchone()
        row = conn.execute(
            """
            INSERT INTO async_job_events (job_id, seq, event_type, payload_json, created_at)
            SELECT %s,
                   COALESCE(MAX(seq), 0) + 1,
                   %s,
                   %s,
                   %s
            FROM async_job_events
            WHERE job_id = %s
            RETURNING seq, event_type, payload_json, created_at
            """,
            (job_id, event_type, Jsonb(body), now, job_id),
        ).fetchone()
        conn.commit()
        return _row_to_event(row)


def list_job_events(job_id: str, *, after_seq: int = 0, limit: int = 500) -> list[dict[str, Any]]:
    """Return events with sequence numbers greater than ``after_seq``."""
    after = max(0, int(after_seq))
    capped_limit = max(1, min(int(limit), 1000))
    if not postgres_jobs_enabled():
        with _memory_lock:
            events = _memory_events.get(job_id, [])
            return [dict(event) for event in events if int(event["seq"]) > after][:capped_limit]

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT seq, event_type, payload_json, created_at
            FROM async_job_events
            WHERE job_id = %s AND seq > %s
            ORDER BY seq ASC
            LIMIT %s
            """,
            (job_id, after, capped_limit),
        ).fetchall()
        return [_row_to_event(row) for row in rows]


def clear_memory_events() -> None:
    """Clear local fallback events. Used by tests and local cache invalidation."""
    with _memory_lock:
        _memory_events.clear()
