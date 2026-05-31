"""Durable JSON snapshot storage for expensive computed API payloads."""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.postgres import use_postgres_state
from api.snapshot_keys import DEFAULT_SNAPSHOT_MAX_AGE_SECONDS
from ontology.sources.source_registry import source_registry_metadata_for_snapshot
from ontology.temporal_repository import SnapshotVersionWrite, TemporalOntologyRepository, payload_hash
from utils.market_freshness import evaluate_source_freshness

logger = logging.getLogger(__name__)

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
    quality: str = "ok"
    payload_hash: str | None = None


def snapshots_required() -> bool:
    """Whether request paths should fail instead of falling back to live compute."""
    return use_postgres_state()


def _now_iso() -> str:
    return datetime.now().isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


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
        quality=str(_row_get(row, "quality", "ok") or "ok"),
        payload_hash=_row_get(row, "payload_hash"),
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
        _materialize_typed_snapshot(row, payload=payload)
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
        if payload is not None:
            _materialize_typed_snapshot(row, payload=payload)
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


def list_snapshot_records() -> list[SnapshotRecord]:
    """Return current computed snapshot records across the active state backend."""
    if use_postgres_state():
        repo = TemporalOntologyRepository()
        with repo._connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM ontology_current_computed_snapshot_read_model
                    ORDER BY snapshot_key
                    """
                ).fetchall()
            except Exception:
                rollback = getattr(conn, "rollback", None)
                if callable(rollback):
                    rollback()
                rows = conn.execute(
                    """
                    SELECT *
                    FROM computed_snapshot_versions
                    WHERE tx_to IS NULL
                      AND valid_from <= clock_timestamp()
                      AND (valid_to IS NULL OR valid_to > clock_timestamp())
                    ORDER BY snapshot_key, load_time DESC
                    """
                ).fetchall()
        return [record for record in (_row_to_record(row) for row in rows) if record is not None]

    with _sqlite_connect() as conn:
        rows = conn.execute("SELECT * FROM computed_snapshots ORDER BY snapshot_key").fetchall()
    return [record for record in (_row_to_record(row) for row in rows) if record is not None]


def delete_snapshot(snapshot_key: str) -> None:
    """Remove the current snapshot so the next read recomputes it."""
    if use_postgres_state():
        tx_to = datetime.now(UTC)
        repo = TemporalOntologyRepository()
        with repo._connect() as conn:
            conn.execute(
                """
                UPDATE computed_snapshot_versions
                SET tx_to = %s
                WHERE snapshot_key = %s
                  AND tx_to IS NULL
                """,
                (tx_to, snapshot_key),
            )
            conn.commit()
        return

    with _sqlite_connect() as conn:
        conn.execute("DELETE FROM computed_snapshots WHERE snapshot_key = ?", (snapshot_key,))
        conn.commit()


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
    source_registry = source_registry_metadata_for_snapshot(record.snapshot_key)
    freshness = _snapshot_freshness(record, source_registry=source_registry, max_age_seconds=max_age_seconds)
    snapshot_meta = {
        "key": record.snapshot_key,
        "as_of": record.as_of_date,
        "fetched_at": record.fetched_at,
        "data_age_seconds": age_seconds,
        "stale": not freshness.get("fresh", False),
        "refresh_status": record.status,
        "error": record.error,
        "version": record.version,
        "freshness_policy": freshness.get("policy"),
        "expected_as_of_date": freshness.get("expected_as_of_date") or freshness.get("oldest_acceptable_date"),
        "observed_as_of_date": freshness.get("observed_as_of_date"),
        "calendar_id": freshness.get("calendar_id"),
        "freshness_reason": freshness.get("reason"),
    }
    if record.artifact_uri:
        snapshot_meta["artifact_uri"] = record.artifact_uri
    meta["snapshot"] = snapshot_meta
    if source_registry:
        meta["source_registry"] = source_registry
    out["_meta"] = meta
    return out


def _snapshot_freshness(
    record: SnapshotRecord,
    *,
    source_registry: dict[str, Any] | None,
    max_age_seconds: int,
) -> dict[str, Any]:
    registry = source_registry if isinstance(source_registry, dict) else {}
    policy = str(registry.get("freshness_policy") or "elapsed").strip().lower()
    value = record.fetched_at if policy in {"", "elapsed"} else record.as_of_date or record.fetched_at
    state = evaluate_source_freshness(
        value,
        policy=policy or "elapsed",
        max_age_seconds=max_age_seconds,
        max_age_days=_int_or_none(registry.get("freshness_max_age_days")),
        calendar_id=str(registry.get("freshness_calendar_id") or "") or None,
    )
    return state.to_dict()


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


def _materialize_typed_snapshot(row: Any, *, payload: dict[str, Any]) -> None:
    snapshot_key = str(_row_get(row, "snapshot_key", ""))
    if snapshot_key != "signal_aggregator:current:v1":
        return
    try:
        from ontology.market_regime_writeback import materialize_signal_aggregator_snapshot

        materialize_signal_aggregator_snapshot(
            snapshot_key=snapshot_key,
            snapshot_version_id=_iso_or_none(_row_get(row, "snapshot_id")),
            payload=payload,
            as_of_date=_iso_or_none(_row_get(row, "as_of")),
            fetched_at=_iso_or_none(_row_get(row, "load_time")),
            status=str(_row_get(row, "status", "ok") or "ok"),
            quality=str(_row_get(row, "quality", "ok") or "ok"),
            error=_row_get(row, "error"),
            provenance_id=_row_get(row, "provenance_event_id"),
        )
    except Exception:
        logger.warning("Failed to materialize typed ontology snapshot %s", snapshot_key, exc_info=True)


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)
