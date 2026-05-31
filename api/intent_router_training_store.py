"""Durable storage for intent-router supervised training rows."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.postgres import connect, use_postgres_state

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SQLITE_PATH = _REPO_ROOT / "data_cache" / "intent_router_training.sqlite3"

TRAINING_ROW_SCHEMA_VERSION = 1
DEFAULT_REDACTION_POLICY = "router_training_v1"
DEFAULT_RETENTION_CLASS = "router_training_365d"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS intent_router_training_rows (
    row_id TEXT PRIMARY KEY,
    session_id TEXT,
    client_turn_id TEXT,
    captured_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    capture_policy TEXT NOT NULL DEFAULT 'shadow_all',
    redaction_policy TEXT NOT NULL DEFAULT 'router_training_v1',
    retention_class TEXT NOT NULL DEFAULT 'router_training_365d',
    sampling_reason TEXT,
    applied_source TEXT,
    confidence_threshold REAL,
    fallback_reason TEXT,
    user_text TEXT NOT NULL,
    screen_context_json TEXT,
    recent_session_features_json TEXT,
    regex_baseline_json TEXT,
    llm_candidate_json TEXT,
    shadow_comparison_json TEXT,
    applied_route_json TEXT,
    opportunity_candidate_metadata_json TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    label_intent_class TEXT,
    label_run_hidden_dq INTEGER,
    label_run_opportunity_preflight INTEGER,
    label_workflow_name TEXT,
    label_tool_names_json TEXT,
    label_reviewer TEXT,
    labeled_at TEXT
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_intent_router_training_rows_captured_at ON intent_router_training_rows(captured_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_intent_router_training_rows_session_turn ON intent_router_training_rows(session_id, client_turn_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_intent_router_training_rows_session_turn ON intent_router_training_rows(session_id, client_turn_id) WHERE session_id IS NOT NULL AND client_turn_id IS NOT NULL",
)


def _sqlite_connect() -> sqlite3.Connection:
    _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_SQLITE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    for statement in _CREATE_INDEXES:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True, default=str)


def _json_load(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    return None


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        data = dict(row)
    else:
        data = dict(row)

    parsed: dict[str, Any] = {}
    for key, value in data.items():
        if key.endswith("_json"):
            parsed[key[:-5]] = _json_load(value)
        elif key in {"label_run_hidden_dq", "label_run_opportunity_preflight"}:
            parsed[key] = bool(value) if value is not None else None
        else:
            parsed[key] = value

    payload = parsed.pop("payload", None)
    if isinstance(payload, dict):
        merged = dict(payload)
        for key, value in parsed.items():
            if value is not None:
                merged.setdefault(key, value)
        return merged
    return parsed


def insert_training_row(row: dict[str, Any]) -> str | None:
    """Persist one training row. Returns row_id or None on duplicate/skip."""
    row_id = str(row.get("row_id") or uuid.uuid4())
    captured_at = str(row.get("captured_at") or datetime.now(UTC).isoformat())
    payload = dict(row)
    payload.setdefault("row_id", row_id)
    payload.setdefault("captured_at", captured_at)
    payload.setdefault("schema_version", TRAINING_ROW_SCHEMA_VERSION)
    payload.setdefault("redaction_policy", DEFAULT_REDACTION_POLICY)
    payload.setdefault("retention_class", DEFAULT_RETENTION_CLASS)

    try:
        if use_postgres_state():
            return _insert_postgres(payload, row_id=row_id, captured_at=captured_at)
        return _insert_sqlite(payload, row_id=row_id, captured_at=captured_at)
    except Exception:
        logger.exception("intent_router_training_row_insert_failed row_id=%s", row_id)
        return None


def update_opportunity_candidate_metadata(
    *,
    session_id: str | None,
    client_turn_id: str | None,
    opportunity_candidate_metadata: dict[str, Any],
) -> bool:
    """Attach OC metadata to an existing row when available post-preflight."""
    if not session_id or not client_turn_id:
        return False
    metadata_json = _json_dump(opportunity_candidate_metadata)
    try:
        if use_postgres_state():
            with connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE intent_router_training_rows
                    SET opportunity_candidate_metadata_json = %s::jsonb,
                        payload_json = jsonb_set(
                            COALESCE(payload_json, '{}'::jsonb),
                            '{opportunity_candidate_metadata}',
                            %s::jsonb,
                            true
                        )
                    WHERE session_id = %s AND client_turn_id = %s
                    """,
                    (metadata_json, metadata_json, session_id, client_turn_id),
                )
                return cur.rowcount > 0
        with _sqlite_connect() as conn:
            cur = conn.execute(
                """
                UPDATE intent_router_training_rows
                SET opportunity_candidate_metadata_json = ?,
                    payload_json = ?
                WHERE session_id = ? AND client_turn_id = ?
                """,
                (
                    metadata_json,
                    _json_dump({**{"opportunity_candidate_metadata": opportunity_candidate_metadata}}),
                    session_id,
                    client_turn_id,
                ),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        logger.exception(
            "intent_router_training_row_oc_update_failed session_id=%s client_turn_id=%s",
            session_id,
            client_turn_id,
        )
        return False


def list_training_rows(
    *,
    limit: int = 1000,
    labeled_only: bool = False,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """List persisted training rows newest-first."""
    if use_postgres_state():
        return _list_postgres(limit=limit, labeled_only=labeled_only, since=since)
    return _list_sqlite(limit=limit, labeled_only=labeled_only, since=since)


def apply_human_label(
    *,
    row_id: str,
    label: dict[str, Any],
    reviewer: str,
) -> bool:
    """Apply reviewer labels to one training row."""
    labeled_at = datetime.now(UTC).isoformat()
    try:
        if use_postgres_state():
            with connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE intent_router_training_rows
                    SET label_intent_class = %s,
                        label_run_hidden_dq = %s,
                        label_run_opportunity_preflight = %s,
                        label_workflow_name = %s,
                        label_tool_names_json = %s::jsonb,
                        label_reviewer = %s,
                        labeled_at = %s
                    WHERE row_id = %s
                    """,
                    (
                        label.get("intent_class"),
                        label.get("run_hidden_dq"),
                        label.get("run_opportunity_preflight"),
                        label.get("workflow_name"),
                        _json_dump(label.get("tool_names")),
                        reviewer,
                        labeled_at,
                        row_id,
                    ),
                )
                conn.commit()
                return cur.rowcount > 0
        with _sqlite_connect() as conn:
            cur = conn.execute(
                """
                UPDATE intent_router_training_rows
                SET label_intent_class = ?,
                    label_run_hidden_dq = ?,
                    label_run_opportunity_preflight = ?,
                    label_workflow_name = ?,
                    label_tool_names_json = ?,
                    label_reviewer = ?,
                    labeled_at = ?
                WHERE row_id = ?
                """,
                (
                    label.get("intent_class"),
                    int(label["run_hidden_dq"]) if label.get("run_hidden_dq") is not None else None,
                    int(label["run_opportunity_preflight"])
                    if label.get("run_opportunity_preflight") is not None
                    else None,
                    label.get("workflow_name"),
                    _json_dump(label.get("tool_names")),
                    reviewer,
                    labeled_at,
                    row_id,
                ),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        logger.exception("intent_router_training_row_label_failed row_id=%s", row_id)
        return False


def _insert_sqlite(payload: dict[str, Any], *, row_id: str, captured_at: str) -> str | None:
    with _sqlite_connect() as conn:
        try:
            conn.execute(
                """
                INSERT INTO intent_router_training_rows (
                    row_id, session_id, client_turn_id, captured_at, schema_version,
                    capture_policy, redaction_policy, retention_class, sampling_reason,
                    applied_source, confidence_threshold, fallback_reason, user_text,
                    screen_context_json, recent_session_features_json, regex_baseline_json,
                    llm_candidate_json, shadow_comparison_json, applied_route_json,
                    opportunity_candidate_metadata_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    payload.get("session_id"),
                    payload.get("client_turn_id"),
                    captured_at,
                    payload.get("schema_version", TRAINING_ROW_SCHEMA_VERSION),
                    payload.get("capture_policy", "shadow_all"),
                    payload.get("redaction_policy", DEFAULT_REDACTION_POLICY),
                    payload.get("retention_class", DEFAULT_RETENTION_CLASS),
                    payload.get("sampling_reason"),
                    payload.get("applied_source"),
                    payload.get("confidence_threshold"),
                    payload.get("fallback_reason"),
                    payload.get("user_text"),
                    _json_dump(payload.get("screen_context")),
                    _json_dump(payload.get("recent_session_features")),
                    _json_dump(payload.get("regex_baseline")),
                    _json_dump(payload.get("llm_candidate")),
                    _json_dump(payload.get("shadow_comparison")),
                    _json_dump(payload.get("applied_route")),
                    _json_dump(payload.get("opportunity_candidate_metadata")),
                    _json_dump(payload),
                ),
            )
            conn.commit()
            return row_id
        except sqlite3.IntegrityError:
            return None


def _insert_postgres(payload: dict[str, Any], *, row_id: str, captured_at: str) -> str | None:
    session_id = payload.get("session_id")
    client_turn_id = payload.get("client_turn_id")
    if session_id and client_turn_id:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT row_id FROM intent_router_training_rows
                WHERE session_id = %s AND client_turn_id = %s
                LIMIT 1
                """,
                (session_id, client_turn_id),
            )
            if cur.fetchone():
                return None

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO intent_router_training_rows (
                row_id, session_id, client_turn_id, captured_at, schema_version,
                capture_policy, redaction_policy, retention_class, sampling_reason,
                applied_source, confidence_threshold, fallback_reason, user_text,
                screen_context_json, recent_session_features_json, regex_baseline_json,
                llm_candidate_json, shadow_comparison_json, applied_route_json,
                opportunity_candidate_metadata_json, payload_json
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s::jsonb, %s::jsonb, %s::jsonb,
                %s::jsonb, %s::jsonb, %s::jsonb,
                %s::jsonb, %s::jsonb
            )
            """,
            (
                row_id,
                payload.get("session_id"),
                payload.get("client_turn_id"),
                captured_at,
                payload.get("schema_version", TRAINING_ROW_SCHEMA_VERSION),
                payload.get("capture_policy", "shadow_all"),
                payload.get("redaction_policy", DEFAULT_REDACTION_POLICY),
                payload.get("retention_class", DEFAULT_RETENTION_CLASS),
                payload.get("sampling_reason"),
                payload.get("applied_source"),
                payload.get("confidence_threshold"),
                payload.get("fallback_reason"),
                payload.get("user_text"),
                _json_dump(payload.get("screen_context")),
                _json_dump(payload.get("recent_session_features")),
                _json_dump(payload.get("regex_baseline")),
                _json_dump(payload.get("llm_candidate")),
                _json_dump(payload.get("shadow_comparison")),
                _json_dump(payload.get("applied_route")),
                _json_dump(payload.get("opportunity_candidate_metadata")),
                _json_dump(payload),
            ),
        )
        conn.commit()
        return row_id


def _list_sqlite(*, limit: int, labeled_only: bool, since: str | None) -> list[dict[str, Any]]:
    clauses = ["1=1"]
    params: list[Any] = []
    if labeled_only:
        clauses.append("label_intent_class IS NOT NULL")
    if since:
        clauses.append("captured_at >= ?")
        params.append(since)
    params.append(limit)
    with _sqlite_connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM intent_router_training_rows
            WHERE {' AND '.join(clauses)}
            ORDER BY captured_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _list_postgres(*, limit: int, labeled_only: bool, since: str | None) -> list[dict[str, Any]]:
    clauses = ["1=1"]
    params: list[Any] = []
    if labeled_only:
        clauses.append("label_intent_class IS NOT NULL")
    if since:
        clauses.append("captured_at >= %s")
        params.append(since)
    params.append(limit)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT row_id, session_id, client_turn_id, captured_at, schema_version,
                   capture_policy, redaction_policy, retention_class, sampling_reason,
                   applied_source, confidence_threshold, fallback_reason, user_text,
                   screen_context_json, recent_session_features_json, regex_baseline_json,
                   llm_candidate_json, shadow_comparison_json, applied_route_json,
                   opportunity_candidate_metadata_json, payload_json,
                   label_intent_class, label_run_hidden_dq, label_run_opportunity_preflight,
                   label_workflow_name, label_tool_names_json, label_reviewer, labeled_at
            FROM intent_router_training_rows
            WHERE {' AND '.join(clauses)}
            ORDER BY captured_at DESC
            LIMIT %s
            """,
            tuple(params),
        )
        columns = [desc[0] for desc in cur.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
    return [_row_to_dict(row) for row in rows]
