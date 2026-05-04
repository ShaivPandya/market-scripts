"""Durable JSON snapshot storage for expensive computed API payloads."""

from __future__ import annotations

import copy
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from api.postgres import use_postgres_state
from api.snapshot_keys import DEFAULT_SNAPSHOT_MAX_AGE_SECONDS
from ontology.temporal_repository import SnapshotVersionWrite, TemporalOntologyRepository, payload_hash

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SQLITE_PATH = _REPO_ROOT / "data_cache" / "computed_snapshots.sqlite3"


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_key: str
    payload: dict[str, Any] | None
    as_of_date: str | None
    fetched_at: str
    status: str
    error: str | None
    version: int
    artifact_uri: str | None


def snapshots_required() -> bool:
    """Whether request paths should fail instead of falling back to live compute."""
    return use_postgres_state()


def _now_iso() -> str:
    return datetime.now().isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _sqlite_connect() -> sqlite3.Connection:
    _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS computed_snapshots (
            snapshot_key TEXT PRIMARY KEY,
            payload_json TEXT,
            as_of_date TEXT,
            fetched_at TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            artifact_uri TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_computed_snapshots_status ON computed_snapshots(status)")
    conn.commit()
    return conn


def _row_to_record(row: Any) -> SnapshotRecord | None:
    if row is None:
        return None
    payload_raw = _row_get(row, "payload_json")
    if isinstance(payload_raw, str):
        payload = json.loads(payload_raw) if payload_raw else None
    elif isinstance(payload_raw, dict) or payload_raw is None:
        payload = payload_raw
    else:
        payload = dict(payload_raw)
    as_of_raw = _row_get(row, "as_of_date", _row_get(row, "as_of"))
    fetched_raw = _row_get(row, "fetched_at", _row_get(row, "load_time"))
    return SnapshotRecord(
        snapshot_key=str(_row_get(row, "snapshot_key")),
        payload=payload,
        as_of_date=_iso_or_none(as_of_raw),
        fetched_at=_iso_or_none(fetched_raw) or _now_iso(),
        status=str(_row_get(row, "status")),
        error=_row_get(row, "error"),
        version=int(_row_get(row, "version", 1) or 1),
        artifact_uri=_row_get(row, "artifact_uri"),
    )


def write_snapshot_success(
    snapshot_key: str,
    payload: dict[str, Any],
    *,
    as_of_date: str | None,
    version: int = 1,
    artifact_uri: str | None = None,
    fetched_at: str | None = None,
) -> SnapshotRecord:
    fetched = fetched_at or _now_iso()
    if use_postgres_state():
        row = TemporalOntologyRepository().write_computed_snapshot_version(
            SnapshotVersionWrite(
                snapshot_key=snapshot_key,
                payload_hash=payload_hash(payload),
                payload=payload,
                artifact_uri=artifact_uri,
                as_of=as_of_date,
                load_time=fetched,
                valid_from=as_of_date or fetched,
                status="ok",
                quality="ok",
            )
        )
        record = _row_to_record(row)
        assert record is not None
        return record

    with _sqlite_connect() as conn:
        conn.execute(
            """
            INSERT INTO computed_snapshots
                (snapshot_key, payload_json, as_of_date, fetched_at, status, error, version, artifact_uri)
            VALUES (?, ?, ?, ?, 'ok', NULL, ?, ?)
            ON CONFLICT(snapshot_key)
            DO UPDATE SET
                payload_json = excluded.payload_json,
                as_of_date = excluded.as_of_date,
                fetched_at = excluded.fetched_at,
                status = 'ok',
                error = NULL,
                version = excluded.version,
                artifact_uri = excluded.artifact_uri
            """,
            (snapshot_key, json.dumps(payload), as_of_date, fetched, int(version), artifact_uri),
        )
        conn.commit()
    record = read_snapshot(snapshot_key)
    assert record is not None
    return record


def write_snapshot_failure(snapshot_key: str, error: str, *, version: int = 1) -> SnapshotRecord | None:
    """Record a failed refresh without discarding the last successful payload."""
    if use_postgres_state():
        prior = read_snapshot(snapshot_key)
        payload = prior.payload if prior else None
        fetched = _now_iso()
        row = TemporalOntologyRepository().write_computed_snapshot_version(
            SnapshotVersionWrite(
                snapshot_key=snapshot_key,
                payload_hash=payload_hash({"error": error, "prior_payload": payload}),
                payload=payload,
                artifact_uri=prior.artifact_uri if prior else None,
                as_of=prior.as_of_date if prior else None,
                load_time=fetched,
                valid_from=fetched,
                status="error",
                quality="degraded",
                error=error,
            )
        )
        return _row_to_record(row)

    with _sqlite_connect() as conn:
        cur = conn.execute(
            "UPDATE computed_snapshots SET status = 'error', error = ?, version = ? WHERE snapshot_key = ?",
            (error, int(version), snapshot_key),
        )
        if cur.rowcount == 0:
            conn.execute(
                """
                INSERT INTO computed_snapshots
                    (snapshot_key, payload_json, as_of_date, fetched_at, status, error, version, artifact_uri)
                VALUES (?, NULL, NULL, ?, 'error', ?, ?, NULL)
                """,
                (snapshot_key, _now_iso(), error, int(version)),
            )
        conn.commit()
    return read_snapshot(snapshot_key)


def read_snapshot(snapshot_key: str) -> SnapshotRecord | None:
    if use_postgres_state():
        row = TemporalOntologyRepository().read_computed_snapshot_version(snapshot_key)
        return _row_to_record(row)

    with _sqlite_connect() as conn:
        row = conn.execute("SELECT * FROM computed_snapshots WHERE snapshot_key = ?", (snapshot_key,)).fetchone()
        return _row_to_record(row)


def read_snapshot_at(
    snapshot_key: str,
    *,
    as_of: str | datetime,
    tx_as_of: str | datetime | None = None,
) -> SnapshotRecord | None:
    if use_postgres_state():
        row = TemporalOntologyRepository().read_computed_snapshot_version(
            snapshot_key,
            as_of=as_of,
            tx_as_of=tx_as_of,
        )
        return _row_to_record(row)
    return read_snapshot(snapshot_key)


def attach_snapshot_meta(
    payload: dict[str, Any],
    record: SnapshotRecord,
    *,
    max_age_seconds: int = DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    raw_meta = out.get("_meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    try:
        age_seconds = max(0, round((datetime.now() - _parse_dt(record.fetched_at)).total_seconds()))
    except Exception:
        age_seconds = None
    snapshot_meta = {
        "key": record.snapshot_key,
        "as_of": record.as_of_date,
        "fetched_at": record.fetched_at,
        "data_age_seconds": age_seconds,
        "stale": bool(age_seconds is not None and age_seconds > max_age_seconds),
        "refresh_status": record.status,
        "error": record.error,
        "version": record.version,
    }
    if record.artifact_uri:
        snapshot_meta["artifact_uri"] = record.artifact_uri
    meta["snapshot"] = snapshot_meta
    out["_meta"] = meta
    return out


def get_snapshot_response(
    snapshot_key: str,
    *,
    max_age_seconds: int = DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
) -> dict[str, Any] | None:
    record = read_snapshot(snapshot_key)
    if record is None or record.payload is None:
        return None
    return attach_snapshot_meta(record.payload, record, max_age_seconds=max_age_seconds)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)
