"""Durable storage for first-class risk snapshots."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.postgres import connect, use_postgres_state

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SQLITE_PATH = _REPO_ROOT / "data_cache" / "position_risk.sqlite3"


def write_position_risk_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Persist and return a complete position risk snapshot payload."""
    ticker = _ticker(snapshot.get("ticker"))
    result_id = str(snapshot.get("result_id") or snapshot.get("run_id") or "").strip()
    if not ticker:
        raise ValueError("position risk snapshot requires ticker")
    if not result_id:
        raise ValueError("position risk snapshot requires result_id")

    if use_postgres_state():
        _write_postgres(ticker, result_id, snapshot)
    else:
        _write_sqlite(ticker, result_id, snapshot)
    return snapshot


def read_latest_position_risk(ticker: str) -> dict[str, Any] | None:
    """Read the latest persisted risk snapshot for *ticker*."""
    ticker_norm = _ticker(ticker)
    if not ticker_norm:
        return None

    if use_postgres_state():
        return _read_latest_postgres(ticker_norm)
    return _read_latest_sqlite(ticker_norm)


def read_position_risk_snapshot(snapshot_id: str) -> dict[str, Any] | None:
    """Read one persisted position risk snapshot by id."""
    sid = str(snapshot_id or "").strip()
    if not sid:
        return None
    if use_postgres_state():
        return _read_position_snapshot_postgres(sid)
    return _read_position_snapshot_sqlite(sid)


def write_portfolio_risk_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Persist and return a complete portfolio risk snapshot payload."""
    result_id = str(snapshot.get("result_id") or snapshot.get("run_id") or "").strip()
    if not result_id:
        raise ValueError("portfolio risk snapshot requires result_id")
    if use_postgres_state():
        _write_portfolio_postgres(result_id, snapshot)
    else:
        _write_portfolio_sqlite(result_id, snapshot)
    return snapshot


def read_latest_portfolio_risk() -> dict[str, Any] | None:
    """Read the latest persisted portfolio risk snapshot."""
    if use_postgres_state():
        return _read_latest_portfolio_postgres()
    return _read_latest_portfolio_sqlite()


def _write_sqlite(ticker: str, result_id: str, snapshot: dict[str, Any]) -> None:
    with _sqlite_connect() as conn:
        conn.execute(
            """
            INSERT INTO position_risk_snapshots (
                id,
                ticker,
                as_of,
                computed_at,
                risk_score,
                risk_level,
                confidence,
                quality,
                source_status_json,
                evidence_json,
                degraded_modules_json,
                input_snapshots_json,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id)
            DO UPDATE SET
                ticker = excluded.ticker,
                as_of = excluded.as_of,
                computed_at = excluded.computed_at,
                risk_score = excluded.risk_score,
                risk_level = excluded.risk_level,
                confidence = excluded.confidence,
                quality = excluded.quality,
                source_status_json = excluded.source_status_json,
                evidence_json = excluded.evidence_json,
                degraded_modules_json = excluded.degraded_modules_json,
                input_snapshots_json = excluded.input_snapshots_json,
                payload_json = excluded.payload_json
            """,
            (
                result_id,
                ticker,
                _string_or_none(snapshot.get("as_of")),
                str(snapshot.get("computed_at") or ""),
                _float_or_zero(snapshot.get("risk_score")),
                str(snapshot.get("risk_level") or "unknown"),
                _float_or_zero(snapshot.get("confidence")),
                str(snapshot.get("quality") or "missing"),
                json.dumps(snapshot.get("source_status") or {}),
                json.dumps(snapshot.get("evidence") or []),
                json.dumps(snapshot.get("degraded_modules") or []),
                json.dumps(snapshot.get("input_snapshots") or {}),
                json.dumps(snapshot),
            ),
        )
        conn.commit()


def _read_latest_sqlite(ticker: str) -> dict[str, Any] | None:
    with _sqlite_connect() as conn:
        row = conn.execute(
            """
            SELECT payload_json
            FROM position_risk_snapshots
            WHERE ticker = ?
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
    if row is None:
        return None
    payload_raw = row["payload_json"] if isinstance(row, sqlite3.Row) else row[0]
    return json.loads(payload_raw) if payload_raw else None


def _read_position_snapshot_sqlite(snapshot_id: str) -> dict[str, Any] | None:
    with _sqlite_connect() as conn:
        row = conn.execute(
            "SELECT payload_json FROM position_risk_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
    if row is None:
        return None
    payload_raw = row["payload_json"] if isinstance(row, sqlite3.Row) else row[0]
    return json.loads(payload_raw) if payload_raw else None


def _write_portfolio_sqlite(result_id: str, snapshot: dict[str, Any]) -> None:
    with _sqlite_connect() as conn:
        conn.execute(
            """
            INSERT INTO portfolio_risk_snapshots (
                id,
                as_of,
                computed_at,
                average_risk_score,
                max_risk_score,
                confidence,
                quality,
                position_count,
                source_status_json,
                degraded_modules_json,
                input_snapshots_json,
                position_snapshot_ids_json,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id)
            DO UPDATE SET
                as_of = excluded.as_of,
                computed_at = excluded.computed_at,
                average_risk_score = excluded.average_risk_score,
                max_risk_score = excluded.max_risk_score,
                confidence = excluded.confidence,
                quality = excluded.quality,
                position_count = excluded.position_count,
                source_status_json = excluded.source_status_json,
                degraded_modules_json = excluded.degraded_modules_json,
                input_snapshots_json = excluded.input_snapshots_json,
                position_snapshot_ids_json = excluded.position_snapshot_ids_json,
                payload_json = excluded.payload_json
            """,
            (
                result_id,
                _string_or_none(snapshot.get("as_of")),
                str(snapshot.get("computed_at") or datetime.now(UTC).isoformat()),
                _float_or_zero(snapshot.get("average_risk_score")),
                _float_or_zero(snapshot.get("max_risk_score")),
                _float_or_zero(snapshot.get("confidence")),
                str(snapshot.get("quality") or "missing"),
                _int_or_zero(snapshot.get("position_count")),
                json.dumps(snapshot.get("source_status") or {}),
                json.dumps(snapshot.get("degraded_modules") or []),
                json.dumps(snapshot.get("input_snapshots") or {}),
                json.dumps(snapshot.get("position_snapshot_ids") or {}),
                json.dumps(snapshot),
            ),
        )
        conn.commit()


def _read_latest_portfolio_sqlite() -> dict[str, Any] | None:
    with _sqlite_connect() as conn:
        row = conn.execute(
            """
            SELECT payload_json
            FROM portfolio_risk_snapshots
            ORDER BY computed_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    payload_raw = row["payload_json"] if isinstance(row, sqlite3.Row) else row[0]
    return json.loads(payload_raw) if payload_raw else None


def _write_postgres(ticker: str, result_id: str, snapshot: dict[str, Any]) -> None:
    from psycopg.types.json import Jsonb

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO position_risk_snapshots (
                id,
                ticker,
                as_of,
                computed_at,
                risk_score,
                risk_level,
                confidence,
                quality,
                source_status_json,
                evidence_json,
                degraded_modules_json,
                input_snapshots_json,
                payload_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(id)
            DO UPDATE SET
                ticker = excluded.ticker,
                as_of = excluded.as_of,
                computed_at = excluded.computed_at,
                risk_score = excluded.risk_score,
                risk_level = excluded.risk_level,
                confidence = excluded.confidence,
                quality = excluded.quality,
                source_status_json = excluded.source_status_json,
                evidence_json = excluded.evidence_json,
                degraded_modules_json = excluded.degraded_modules_json,
                input_snapshots_json = excluded.input_snapshots_json,
                payload_json = excluded.payload_json
            """,
            (
                result_id,
                ticker,
                _string_or_none(snapshot.get("as_of")),
                str(snapshot.get("computed_at") or ""),
                _float_or_zero(snapshot.get("risk_score")),
                str(snapshot.get("risk_level") or "unknown"),
                _float_or_zero(snapshot.get("confidence")),
                str(snapshot.get("quality") or "missing"),
                Jsonb(snapshot.get("source_status") or {}),
                Jsonb(snapshot.get("evidence") or []),
                Jsonb(snapshot.get("degraded_modules") or []),
                Jsonb(snapshot.get("input_snapshots") or {}),
                Jsonb(snapshot),
            ),
        )
        conn.commit()


def _read_latest_postgres(ticker: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT payload_json
            FROM position_risk_snapshots
            WHERE upper(ticker) = upper(%s)
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
    if not row:
        return None
    payload = row.get("payload_json") if isinstance(row, dict) else row[0]
    return _payload_dict(payload)


def _read_position_snapshot_postgres(snapshot_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT payload_json FROM position_risk_snapshots WHERE id = %s",
            (snapshot_id,),
        ).fetchone()
    if not row:
        return None
    payload = row.get("payload_json") if isinstance(row, dict) else row[0]
    return _payload_dict(payload)


def _write_portfolio_postgres(result_id: str, snapshot: dict[str, Any]) -> None:
    from psycopg.types.json import Jsonb

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO portfolio_risk_snapshots (
                id,
                as_of,
                computed_at,
                average_risk_score,
                max_risk_score,
                confidence,
                quality,
                position_count,
                source_status_json,
                degraded_modules_json,
                input_snapshots_json,
                position_snapshot_ids_json,
                payload_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(id)
            DO UPDATE SET
                as_of = excluded.as_of,
                computed_at = excluded.computed_at,
                average_risk_score = excluded.average_risk_score,
                max_risk_score = excluded.max_risk_score,
                confidence = excluded.confidence,
                quality = excluded.quality,
                position_count = excluded.position_count,
                source_status_json = excluded.source_status_json,
                degraded_modules_json = excluded.degraded_modules_json,
                input_snapshots_json = excluded.input_snapshots_json,
                position_snapshot_ids_json = excluded.position_snapshot_ids_json,
                payload_json = excluded.payload_json
            """,
            (
                result_id,
                _string_or_none(snapshot.get("as_of")),
                str(snapshot.get("computed_at") or datetime.now(UTC).isoformat()),
                _float_or_zero(snapshot.get("average_risk_score")),
                _float_or_zero(snapshot.get("max_risk_score")),
                _float_or_zero(snapshot.get("confidence")),
                str(snapshot.get("quality") or "missing"),
                _int_or_zero(snapshot.get("position_count")),
                Jsonb(snapshot.get("source_status") or {}),
                Jsonb(snapshot.get("degraded_modules") or []),
                Jsonb(snapshot.get("input_snapshots") or {}),
                Jsonb(snapshot.get("position_snapshot_ids") or {}),
                Jsonb(snapshot),
            ),
        )
        conn.commit()


def _read_latest_portfolio_postgres() -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT payload_json
            FROM portfolio_risk_snapshots
            ORDER BY computed_at DESC
            LIMIT 1
            """
        ).fetchone()
    if not row:
        return None
    payload = row.get("payload_json") if isinstance(row, dict) else row[0]
    return _payload_dict(payload)


def _payload_dict(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    return None


def _sqlite_connect() -> sqlite3.Connection:
    _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS position_risk_snapshots (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            as_of TEXT,
            computed_at TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            confidence REAL NOT NULL,
            quality TEXT NOT NULL,
            source_status_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            degraded_modules_json TEXT NOT NULL,
            input_snapshots_json TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_position_risk_snapshots_ticker_time "
        "ON position_risk_snapshots(ticker, computed_at DESC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_risk_snapshots (
            id TEXT PRIMARY KEY,
            as_of TEXT,
            computed_at TEXT NOT NULL,
            average_risk_score REAL NOT NULL,
            max_risk_score REAL NOT NULL,
            confidence REAL NOT NULL,
            quality TEXT NOT NULL,
            position_count INTEGER NOT NULL,
            source_status_json TEXT NOT NULL,
            degraded_modules_json TEXT NOT NULL,
            input_snapshots_json TEXT NOT NULL,
            position_snapshot_ids_json TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_portfolio_risk_snapshots_time ON portfolio_risk_snapshots(computed_at DESC)"
    )
    conn.commit()
    return conn


def _ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
