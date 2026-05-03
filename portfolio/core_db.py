"""
core_db.py -- SQLite-backed canonical process model for the investing OS.

Houses catalysts, kill conditions, workflow runs, action items, watch triggers,
research notes, and pending approvals. Follows the same connection pattern as
thesis_db.py (WAL mode, thread-safe singleton, _get_conn/_init_db).

Public API:
  Catalysts:
    create_catalyst(ticker, description, ...)        -> dict
    get_catalysts(ticker)                            -> list[dict]
    update_catalyst_status(id, status, evidence)     -> dict

  Kill Conditions:
    create_kill_condition(ticker, condition, ...)     -> dict
    get_kill_conditions(ticker)                      -> list[dict]
    update_kill_condition_status(id, status)          -> dict

  Workflow Runs:
    create_workflow_run(run_id, workflow_name, ticker) -> dict
    complete_workflow_run(run_id, synthesis, artifacts, tool_sections) -> dict
    fail_workflow_run(run_id, error)                  -> dict
    get_workflow_runs(workflow_name, ticker, limit)    -> list[dict]
    get_workflow_run(run_id)                          -> dict | None

  Action Items:
    create_action_item(ticker, action_type, ...)      -> dict
    get_action_items(status, ticker)                  -> list[dict]
    complete_action_item(id, resolution_note)          -> dict
    dismiss_action_item(id)                           -> dict

  Watch Triggers:
    create_watch_trigger(ticker, trigger_type, ...)    -> dict
    get_watch_triggers(status, ticker)                -> list[dict]
    fire_watch_trigger(id)                            -> dict
    cancel_watch_trigger(id)                          -> dict

  Research Notes:
    create_research_note(ticker, title, content, ...) -> dict
    get_research_notes(ticker, limit)                 -> list[dict]

  Pending Approvals:
    create_pending_approval(entity_type, ...)          -> dict
    get_pending_approvals(status, ticker)              -> list[dict]
    get_pending_approval(id)                           -> dict | None
    resolve_approval(id, status, resolved_note)        -> dict

  Report Runs:
    upsert_report_run(record)                          -> dict
    get_report_runs(report_type, limit)                -> list[dict]

  Thesis Claims:
    create_thesis_claim(record)                        -> dict
    get_thesis_claims(ticker, status)                  -> list[dict]
    update_thesis_claim(id, updates)                   -> dict

  Recommendations:
    create_recommendation(record)                      -> dict
    upsert_recommendation(record)                      -> dict
    get_recommendations(report_type, status, ticker)   -> list[dict]
    get_recommendation(id)                             -> dict | None
    get_latest_recommendation(report_type)             -> dict | None
    update_recommendation_approval(id, approval_id, status) -> dict
    update_recommendation_outcome(id, status, outcome) -> dict
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from api.postgres import use_postgres_state
from api.postgres_compat import PostgresCompatConnection

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "core.db"
APPROVAL_APPLICATION_STATUSES = ("pending", "applying", "applied", "failed", "not_applicable")
APPROVAL_APPLICATION_LEASE = timedelta(minutes=15)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

WATCH_TRIGGER_TYPES = (
    "price_level",
    "technical",
    "fundamental",
    "fundamental_news",
    "event",
    "news_event",
    "macro",
    "custom",
)

_CREATE_CATALYSTS = """
CREATE TABLE IF NOT EXISTS catalysts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    description TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'fundamental'
                CHECK (category IN ('fundamental', 'technical', 'macro', 'event', 'regulatory')),
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'played_out', 'failed', 'superseded')),
    target_date TEXT,
    evidence    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    created_by  TEXT NOT NULL DEFAULT 'user'
                CHECK (created_by IN ('backfill', 'user', 'agent', 'workflow'))
)
"""

_CREATE_KILL_CONDITIONS = """
CREATE TABLE IF NOT EXISTS kill_conditions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    condition   TEXT NOT NULL,
    metric      TEXT,
    threshold   TEXT,
    status      TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'triggered', 'retired')),
    triggered_at TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    created_by  TEXT NOT NULL DEFAULT 'user'
                CHECK (created_by IN ('backfill', 'user', 'agent', 'workflow'))
)
"""

_CREATE_WORKFLOW_RUNS = """
CREATE TABLE IF NOT EXISTS workflow_runs (
    run_id        TEXT PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    ticker        TEXT,
    status        TEXT NOT NULL DEFAULT 'running'
                  CHECK (status IN ('running', 'completed', 'failed')),
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    tool_sections TEXT,
    synthesis     TEXT,
    artifacts     TEXT,
    error         TEXT
)
"""

_CREATE_REPORT_RUNS = """
CREATE TABLE IF NOT EXISTS report_runs (
    report_id           TEXT PRIMARY KEY,
    report_type         TEXT NOT NULL
                        CHECK (report_type IN ('daily', 'weekly')),
    as_of               TEXT NOT NULL,
    source              TEXT NOT NULL DEFAULT 'github_actions',
    source_run_id       TEXT,
    source_url          TEXT,
    status              TEXT NOT NULL DEFAULT 'completed'
                        CHECK (status IN ('completed', 'failed')),
    report_hash         TEXT,
    input_hash          TEXT,
    summary_json        TEXT,
    artifact_paths_json TEXT,
    issue_url           TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    synced_at           TEXT NOT NULL,
    error               TEXT
)
"""

_CREATE_ACTION_ITEMS = """
CREATE TABLE IF NOT EXISTS action_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT,
    action_type     TEXT NOT NULL
                    CHECK (action_type IN ('review', 'resize', 'research', 'exit', 'enter', 'hedge', 'other')),
    description     TEXT NOT NULL,
    urgency         TEXT NOT NULL DEFAULT 'normal'
                    CHECK (urgency IN ('low', 'normal', 'high', 'urgent')),
    status          TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open', 'completed', 'dismissed', 'superseded')),
    source_type     TEXT NOT NULL DEFAULT 'user'
                    CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id       TEXT,
    created_at      TEXT NOT NULL,
    completed_at    TEXT,
    resolution_note TEXT
)
"""

_CREATE_WATCH_TRIGGERS = """
CREATE TABLE IF NOT EXISTS watch_triggers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT,
    trigger_type TEXT NOT NULL
                 CHECK (trigger_type IN ('price_level', 'technical', 'fundamental', 'fundamental_news', 'event', 'news_event', 'macro', 'custom')),
    condition    TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active'
                 CHECK (status IN ('active', 'fired', 'expired', 'cancelled')),
    source_type  TEXT NOT NULL DEFAULT 'user'
                 CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id    TEXT,
    created_at   TEXT NOT NULL,
    fired_at     TEXT,
    expires_at   TEXT,
    definition_json TEXT,
    last_checked_at TEXT,
    last_result_json TEXT,
    last_evidence TEXT
)
"""

_CREATE_THESIS_CLAIMS = """
CREATE TABLE IF NOT EXISTS thesis_claims (
    id                           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                       TEXT NOT NULL,
    claim                        TEXT NOT NULL,
    expected_evidence            TEXT,
    disconfirming_evidence       TEXT,
    source_requirements_json     TEXT,
    cadence                      TEXT,
    confidence                   REAL,
    status                       TEXT NOT NULL DEFAULT 'active'
                                 CHECK (status IN ('active', 'supported', 'challenged', 'disconfirmed', 'retired')),
    linked_catalyst_ids_json     TEXT,
    linked_kill_condition_ids_json TEXT,
    source_type                  TEXT NOT NULL DEFAULT 'user'
                                 CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id                    TEXT,
    created_at                   TEXT NOT NULL,
    updated_at                   TEXT NOT NULL
)
"""

_CREATE_RESEARCH_NOTES = """
CREATE TABLE IF NOT EXISTS research_notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    note_type   TEXT NOT NULL DEFAULT 'general'
                CHECK (note_type IN ('general', 'earnings', 'catalyst_update', 'risk_assessment', 'workflow_output')),
    source_type TEXT NOT NULL DEFAULT 'user'
                CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id   TEXT,
    created_at  TEXT NOT NULL
)
"""

_CREATE_PENDING_APPROVALS = """
CREATE TABLE IF NOT EXISTS pending_approvals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type     TEXT NOT NULL,
    entity_id       INTEGER,
    ticker          TEXT,
    action_id       TEXT,
    action_schema_version INTEGER,
    action_input_hash TEXT,
    proposed_change TEXT NOT NULL,
    reason          TEXT,
    source_type     TEXT NOT NULL DEFAULT 'workflow'
                    CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id       TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    resolved_note   TEXT,
    application_status       TEXT NOT NULL DEFAULT 'pending'
                             CHECK (application_status IN ('pending', 'applying', 'applied', 'failed', 'not_applicable')),
    application_attempts     INTEGER NOT NULL DEFAULT 0,
    application_started_at   TEXT,
    application_completed_at TEXT,
    application_error        TEXT
)
"""

_CREATE_ACTION_RUNS = """
CREATE TABLE IF NOT EXISTS action_runs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id            TEXT NOT NULL,
    action_schema_version INTEGER NOT NULL DEFAULT 1,
    actor_type           TEXT NOT NULL,
    actor_id             TEXT,
    source_type          TEXT,
    source_id            TEXT,
    approval_id          INTEGER,
    parent_action_run_id INTEGER,
    input_hash           TEXT,
    input_json           TEXT,
    output_json          TEXT,
    status               TEXT NOT NULL DEFAULT 'running'
                         CHECK (status IN ('running', 'succeeded', 'failed', 'rolled_back')),
    error                TEXT,
    started_at           TEXT NOT NULL,
    completed_at         TEXT
)
"""

_CREATE_ACTION_EVENTS = """
CREATE TABLE IF NOT EXISTS action_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    action_run_id INTEGER NOT NULL,
    event_type    TEXT NOT NULL,
    message       TEXT,
    payload_json  TEXT,
    created_at    TEXT NOT NULL
)
"""

_CREATE_RECOMMENDATIONS = """
CREATE TABLE IF NOT EXISTS recommendations (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_type                 TEXT NOT NULL
                                CHECK (report_type IN ('daily', 'weekly')),
    as_of                       TEXT NOT NULL,
    created_at                  TEXT NOT NULL,
    source_report_path          TEXT,
    source_json_path            TEXT,
    stance                      TEXT NOT NULL,
    recommendation_status       TEXT NOT NULL
                                CHECK (recommendation_status IN ('clear', 'blocked', 'error')),
    critical_data_quality       TEXT NOT NULL
                                CHECK (critical_data_quality IN ('ok', 'degraded', 'stale', 'failed')),
    blocked_reasons_json        TEXT,
    what_changed_json           TEXT,
    do_nothing_rationale        TEXT,
    action                      TEXT NOT NULL
                                CHECK (action IN ('buy', 'sell', 'hold', 'watch', 'avoid', 'reduce', 'exit', 'rebalance', 'hedge', 'do_nothing')),
    ticker                      TEXT,
    instrument                  TEXT NOT NULL,
    horizon                     TEXT,
    target_change               TEXT,
    rationale                   TEXT NOT NULL,
    confidence                  REAL,
    source_quality              TEXT NOT NULL
                                CHECK (source_quality IN ('ok', 'degraded', 'stale', 'failed')),
    status                      TEXT NOT NULL DEFAULT 'open'
                                CHECK (status IN ('open', 'blocked', 'error', 'superseded', 'closed')),
    evidence_json               TEXT,
    disconfirming_evidence_json TEXT,
    catalyst                    TEXT,
    invalidation                TEXT,
    expected_onset_window       TEXT,
    alternatives_json           TEXT,
    opportunity_cost_json       TEXT,
    approval_id                 INTEGER,
    approval_status             TEXT NOT NULL DEFAULT 'none'
                                CHECK (approval_status IN ('none', 'pending', 'approved', 'rejected')),
    outcome_status              TEXT NOT NULL DEFAULT 'pending'
                                CHECK (outcome_status IN ('pending', 'evaluated', 'unavailable')),
    outcome_json                TEXT,
    model                       TEXT,
    prompt_hash                 TEXT,
    input_hash                  TEXT,
    validation_status           TEXT,
    source_quality_summary_json TEXT,
    report_id                   TEXT,
    idempotency_key             TEXT
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_catalysts_ticker ON catalysts(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_kill_conditions_ticker ON kill_conditions(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_name ON workflow_runs(workflow_name)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_ticker ON workflow_runs(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_started ON workflow_runs(started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_report_runs_type_asof ON report_runs(report_type, as_of DESC)",
    "CREATE INDEX IF NOT EXISTS idx_report_runs_source ON report_runs(source_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_action_items_status ON action_items(status)",
    "CREATE INDEX IF NOT EXISTS idx_action_items_ticker ON action_items(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_watch_triggers_status ON watch_triggers(status)",
    "CREATE INDEX IF NOT EXISTS idx_watch_triggers_ticker ON watch_triggers(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_thesis_claims_ticker ON thesis_claims(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_thesis_claims_status ON thesis_claims(status)",
    "CREATE INDEX IF NOT EXISTS idx_research_notes_ticker ON research_notes(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_status ON pending_approvals(status)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_ticker ON pending_approvals(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_application_status ON pending_approvals(application_status)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_action_id ON pending_approvals(action_id)",
    "CREATE INDEX IF NOT EXISTS idx_action_runs_action_id ON action_runs(action_id)",
    "CREATE INDEX IF NOT EXISTS idx_action_runs_status ON action_runs(status)",
    "CREATE INDEX IF NOT EXISTS idx_action_runs_approval ON action_runs(approval_id)",
    "CREATE INDEX IF NOT EXISTS idx_action_events_run ON action_events(action_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_report ON recommendations(report_type, as_of DESC)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_report_id ON recommendations(report_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_recommendations_idempotency ON recommendations(idempotency_key)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_ticker ON recommendations(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_approval ON recommendations(approval_status)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_outcome ON recommendations(outcome_status)",
]

# ---------------------------------------------------------------------------
# Connection management (same pattern as thesis_db.py)
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_conn: sqlite3.Connection | PostgresCompatConnection | None = None


def _get_conn() -> sqlite3.Connection | PostgresCompatConnection:
    global _conn
    if _conn is not None:
        try:
            _conn.execute("SELECT 1")
        except Exception:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
    if _conn is None:
        with _lock:
            if _conn is None:
                if use_postgres_state():
                    _conn = PostgresCompatConnection(
                        identity_tables={
                            "catalysts",
                            "kill_conditions",
                            "thesis_claims",
                            "action_items",
                            "watch_triggers",
                            "research_notes",
                            "pending_approvals",
                            "action_runs",
                            "action_events",
                            "recommendations",
                        }
                    )
                else:
                    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                    _conn.execute("PRAGMA journal_mode=WAL")
                    _conn.row_factory = sqlite3.Row
                    _conn.execute("PRAGMA foreign_keys = ON")
                    _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection) -> None:
    for stmt in [
        _CREATE_CATALYSTS,
        _CREATE_KILL_CONDITIONS,
        _CREATE_WORKFLOW_RUNS,
        _CREATE_REPORT_RUNS,
        _CREATE_ACTION_ITEMS,
        _CREATE_WATCH_TRIGGERS,
        _CREATE_THESIS_CLAIMS,
        _CREATE_RESEARCH_NOTES,
        _CREATE_PENDING_APPROVALS,
        _CREATE_ACTION_RUNS,
        _CREATE_ACTION_EVENTS,
        _CREATE_RECOMMENDATIONS,
    ]:
        conn.execute(stmt)
    _ensure_sqlite_columns(conn)
    for idx in _INDEXES:
        conn.execute(idx)
    conn.commit()


def _ensure_sqlite_columns(conn: sqlite3.Connection) -> None:
    """Apply small additive schema upgrades for local SQLite databases."""

    def _columns(table: str) -> set[str]:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _add_missing(table: str, columns: dict[str, str]) -> None:
        existing = _columns(table)
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    _add_missing(
        "watch_triggers",
        {
            "definition_json": "TEXT",
            "last_checked_at": "TEXT",
            "last_result_json": "TEXT",
            "last_evidence": "TEXT",
        },
    )
    _add_missing(
        "recommendations",
        {
            "report_id": "TEXT",
            "idempotency_key": "TEXT",
        },
    )
    _add_missing(
        "pending_approvals",
        {
            "application_status": (
                "TEXT NOT NULL DEFAULT 'pending' "
                "CHECK (application_status IN ('pending', 'applying', 'applied', 'failed', 'not_applicable'))"
            ),
            "application_attempts": "INTEGER NOT NULL DEFAULT 0",
            "application_started_at": "TEXT",
            "application_completed_at": "TEXT",
            "application_error": "TEXT",
            "action_id": "TEXT",
            "action_schema_version": "INTEGER",
            "action_input_hash": "TEXT",
        },
    )
    conn.execute(
        "UPDATE pending_approvals "
        "SET application_status = 'applied', "
        "application_completed_at = COALESCE(application_completed_at, resolved_at, created_at) "
        "WHERE status = 'approved' AND application_status = 'pending'"
    )
    conn.execute(
        "UPDATE pending_approvals "
        "SET application_status = 'not_applicable', "
        "application_completed_at = COALESCE(application_completed_at, resolved_at, created_at) "
        "WHERE status IN ('rejected', 'expired') AND application_status = 'pending'"
    )
    _ensure_sqlite_watch_trigger_types(conn)


def _ensure_sqlite_watch_trigger_types(conn: sqlite3.Connection) -> None:
    """Rebuild legacy SQLite watch_triggers tables whose CHECK enum is stale."""

    row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'watch_triggers'").fetchone()
    create_sql = str(row[0] if row else "")
    if not create_sql or all(trigger_type in create_sql for trigger_type in ("news_event", "fundamental_news")):
        return
    if "CHECK" not in create_sql or "trigger_type" not in create_sql:
        return

    legacy_table = "watch_triggers_legacy_trigger_type_upgrade"
    conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
    conn.execute(f"ALTER TABLE watch_triggers RENAME TO {legacy_table}")
    conn.execute(_CREATE_WATCH_TRIGGERS)

    legacy_cols = {str(col[1]) for col in conn.execute(f"PRAGMA table_info({legacy_table})").fetchall()}
    target_cols = {str(col[1]) for col in conn.execute("PRAGMA table_info(watch_triggers)").fetchall()}
    copy_cols = [
        col
        for col in (
            "id",
            "ticker",
            "trigger_type",
            "condition",
            "status",
            "source_type",
            "source_id",
            "created_at",
            "fired_at",
            "expires_at",
            "definition_json",
            "last_checked_at",
            "last_result_json",
            "last_evidence",
        )
        if col in legacy_cols and col in target_cols
    ]
    cols_sql = ", ".join(copy_cols)
    conn.execute(f"INSERT INTO watch_triggers ({cols_sql}) SELECT {cols_sql} FROM {legacy_table}")
    conn.execute(f"DROP TABLE {legacy_table}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_dict(row: Any | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return cast(dict[str, Any], dict(row))


def _require_row_dict(row: Any | None) -> dict[str, Any]:
    if row is None:
        raise RuntimeError("Expected database row.")
    return cast(dict[str, Any], dict(row))


def _rows_to_list(rows: Iterable[Any]) -> list[dict[str, Any]]:
    return [_require_row_dict(r) for r in rows]


def _parse_json_field(d: dict, field: str) -> dict:
    """Parse a JSON string field in a dict, returning the dict with the field parsed."""
    val = d.get(field)
    if isinstance(val, str):
        try:
            d[field] = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def _json_hash(value: Any) -> str:
    import hashlib

    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def create_action_run(
    *,
    action_id: str,
    action_schema_version: int,
    actor_type: str,
    actor_id: str | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
    approval_id: int | None = None,
    parent_action_run_id: int | None = None,
    input_hash: str | None = None,
    input_payload: Any | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    input_json = json.dumps(input_payload, default=str) if input_payload is not None else None
    with _lock:
        cur = conn.execute(
            "INSERT INTO action_runs (action_id, action_schema_version, actor_type, actor_id, source_type, source_id, "
            "approval_id, parent_action_run_id, input_hash, input_json, status, started_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                action_id,
                action_schema_version,
                actor_type,
                actor_id,
                source_type,
                source_id,
                approval_id,
                parent_action_run_id,
                input_hash,
                input_json,
                "running",
                now,
            ),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "action_id": action_id,
        "action_schema_version": action_schema_version,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "source_type": source_type,
        "source_id": source_id,
        "approval_id": approval_id,
        "parent_action_run_id": parent_action_run_id,
        "input_hash": input_hash,
        "input_json": input_json,
        "status": "running",
        "started_at": now,
        "completed_at": None,
        "error": None,
    }


def record_action_event(
    action_run_id: int,
    event_type: str,
    *,
    message: str | None = None,
    payload: Any | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    payload_json = json.dumps(payload, default=str) if payload is not None else None
    with _lock:
        cur = conn.execute(
            "INSERT INTO action_events (action_run_id, event_type, message, payload_json, created_at) VALUES (?,?,?,?,?)",
            (action_run_id, event_type, message, payload_json, now),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "action_run_id": action_run_id,
        "event_type": event_type,
        "message": message,
        "payload": payload,
        "created_at": now,
    }


def complete_action_run(
    action_run_id: int,
    *,
    status: str,
    output_payload: Any | None = None,
    error: str | None = None,
) -> dict:
    if status not in {"succeeded", "failed", "rolled_back"}:
        raise ValueError(f"Invalid action run status: {status}")
    conn = _get_conn()
    now = _now()
    output_json = json.dumps(output_payload, default=str) if output_payload is not None else None
    with _lock:
        conn.execute(
            "UPDATE action_runs SET status = ?, output_json = ?, error = ?, completed_at = ? WHERE id = ?",
            (status, output_json, error, now, action_run_id),
        )
        updated = conn.execute("SELECT * FROM action_runs WHERE id = ?", (action_run_id,)).fetchone()
        conn.commit()
    return _require_row_dict(updated)


def get_action_run(action_run_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM action_runs WHERE id = ?", (action_run_id,)).fetchone()
    return _row_to_dict(row)


def get_action_runs(action_id: str | None = None, approval_id: int | None = None) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if action_id:
        clauses.append("action_id = ?")
        params.append(action_id)
    if approval_id is not None:
        clauses.append("approval_id = ?")
        params.append(approval_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(f"SELECT * FROM action_runs{where} ORDER BY id", params).fetchall()
    return _rows_to_list(rows)


def get_action_events(action_run_id: int) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT * FROM action_events WHERE action_run_id = ? ORDER BY id",
            (action_run_id,),
        ).fetchall()
    out = _rows_to_list(rows)
    for row in out:
        _parse_json_field(row, "payload_json")
        row["payload"] = row.get("payload_json")
    return out


# ---------------------------------------------------------------------------
# Catalysts
# ---------------------------------------------------------------------------


def create_catalyst(
    ticker: str,
    description: str,
    category: str = "fundamental",
    *,
    target_date: str | None = None,
    evidence: str | None = None,
    created_by: str = "user",
) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        cur = conn.execute(
            "INSERT INTO catalysts (ticker, description, category, target_date, evidence, created_at, updated_at, created_by) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (ticker.upper(), description, category, target_date, evidence, now, now, created_by),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "ticker": ticker.upper(),
        "description": description,
        "category": category,
        "status": "pending",
        "target_date": target_date,
        "evidence": evidence,
        "created_at": now,
        "updated_at": now,
        "created_by": created_by,
    }


def get_catalysts(ticker: str) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT * FROM catalysts WHERE ticker = ? ORDER BY created_at DESC",
            (ticker.upper(),),
        ).fetchall()
    return _rows_to_list(rows)


def update_catalyst_status(catalyst_id: int, status: str, evidence: str | None = None) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM catalysts WHERE id = ?", (catalyst_id,)).fetchone()
        if not row:
            raise ValueError(f"No catalyst with id {catalyst_id}")
        updates = {"status": status, "updated_at": now}
        if evidence is not None:
            updates["evidence"] = evidence
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(
            f"UPDATE catalysts SET {set_clause} WHERE id = ?",
            (*updates.values(), catalyst_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM catalysts WHERE id = ?", (catalyst_id,)).fetchone()
    return _require_row_dict(updated)


# ---------------------------------------------------------------------------
# Kill Conditions
# ---------------------------------------------------------------------------


def create_kill_condition(
    ticker: str,
    condition: str,
    *,
    metric: str | None = None,
    threshold: str | None = None,
    created_by: str = "user",
) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        cur = conn.execute(
            "INSERT INTO kill_conditions (ticker, condition, metric, threshold, created_at, updated_at, created_by) "
            "VALUES (?,?,?,?,?,?,?)",
            (ticker.upper(), condition, metric, threshold, now, now, created_by),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "ticker": ticker.upper(),
        "condition": condition,
        "metric": metric,
        "threshold": threshold,
        "status": "active",
        "triggered_at": None,
        "created_at": now,
        "updated_at": now,
        "created_by": created_by,
    }


def get_kill_conditions(ticker: str) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT * FROM kill_conditions WHERE ticker = ? ORDER BY created_at DESC",
            (ticker.upper(),),
        ).fetchall()
    return _rows_to_list(rows)


def update_kill_condition_status(kc_id: int, status: str) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM kill_conditions WHERE id = ?", (kc_id,)).fetchone()
        if not row:
            raise ValueError(f"No kill condition with id {kc_id}")
        triggered_at = now if status == "triggered" else _require_row_dict(row).get("triggered_at")
        conn.execute(
            "UPDATE kill_conditions SET status = ?, triggered_at = ?, updated_at = ? WHERE id = ?",
            (status, triggered_at, now, kc_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM kill_conditions WHERE id = ?", (kc_id,)).fetchone()
    return _require_row_dict(updated)


def delete_catalysts_by_ticker(ticker: str, *, created_by: str | None = None) -> int:
    """Delete catalysts for a ticker. Optionally filter by created_by."""
    conn = _get_conn()
    with _lock:
        if created_by:
            cur = conn.execute(
                "DELETE FROM catalysts WHERE ticker = ? AND created_by = ?",
                (ticker.upper(), created_by),
            )
        else:
            cur = conn.execute("DELETE FROM catalysts WHERE ticker = ?", (ticker.upper(),))
        conn.commit()
    return cur.rowcount


def delete_kill_conditions_by_ticker(ticker: str, *, created_by: str | None = None) -> int:
    """Delete kill conditions for a ticker. Optionally filter by created_by."""
    conn = _get_conn()
    with _lock:
        if created_by:
            cur = conn.execute(
                "DELETE FROM kill_conditions WHERE ticker = ? AND created_by = ?",
                (ticker.upper(), created_by),
            )
        else:
            cur = conn.execute("DELETE FROM kill_conditions WHERE ticker = ?", (ticker.upper(),))
        conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# Workflow Runs
# ---------------------------------------------------------------------------


def create_workflow_run(
    workflow_name: str,
    ticker: str | None = None,
    run_id: str | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    rid = run_id or uuid.uuid4().hex
    with _lock:
        conn.execute(
            "INSERT INTO workflow_runs (run_id, workflow_name, ticker, status, started_at) VALUES (?,?,?,?,?)",
            (rid, workflow_name, ticker.upper() if ticker else None, "running", now),
        )
        conn.commit()
    return {"run_id": rid, "workflow_name": workflow_name, "ticker": ticker, "status": "running", "started_at": now}


def complete_workflow_run(
    run_id: str,
    synthesis: str,
    artifacts: dict | list | None = None,
    tool_sections: list[dict] | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    artifacts_json = json.dumps(artifacts, default=str) if artifacts else None
    sections_json = json.dumps(tool_sections, default=str) if tool_sections else None
    with _lock:
        conn.execute(
            "UPDATE workflow_runs SET status = 'completed', completed_at = ?, synthesis = ?, artifacts = ?, tool_sections = ? WHERE run_id = ?",
            (now, synthesis, artifacts_json, sections_json, run_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        raise ValueError(f"No workflow run with id {run_id}")
    d = _require_row_dict(row)
    _parse_json_field(d, "artifacts")
    _parse_json_field(d, "tool_sections")
    return d


def fail_workflow_run(run_id: str, error: str) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        conn.execute(
            "UPDATE workflow_runs SET status = 'failed', completed_at = ?, error = ? WHERE run_id = ?",
            (now, error, run_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        raise ValueError(f"No workflow run with id {run_id}")
    return _require_row_dict(row)


def get_workflow_runs(
    workflow_name: str | None = None,
    ticker: str | None = None,
    limit: int = 20,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list = []
    if workflow_name:
        clauses.append("workflow_name = ?")
        params.append(workflow_name)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM workflow_runs{where} ORDER BY started_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    results = _rows_to_list(rows)
    for d in results:
        _parse_json_field(d, "artifacts")
        _parse_json_field(d, "tool_sections")
    return results


def get_workflow_run(run_id: str) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        return None
    d = _require_row_dict(row)
    _parse_json_field(d, "artifacts")
    _parse_json_field(d, "tool_sections")
    return d


# ---------------------------------------------------------------------------
# Report Runs
# ---------------------------------------------------------------------------

_REPORT_RUN_JSON_FIELDS = ("summary_json", "artifact_paths_json")


def _parse_report_run_json_fields(d: dict) -> dict:
    for field in _REPORT_RUN_JSON_FIELDS:
        _parse_json_field(d, field)
    return d


def upsert_report_run(record: dict) -> dict:
    conn = _get_conn()
    now = _now()
    report_type = str(record["report_type"])
    as_of = str(record["as_of"])
    report_id = str(record.get("report_id") or f"{report_type}:{as_of}")
    params = {
        "report_id": report_id,
        "report_type": report_type,
        "as_of": as_of,
        "source": record.get("source") or "github_actions",
        "source_run_id": record.get("source_run_id"),
        "source_url": record.get("source_url"),
        "status": record.get("status") or "completed",
        "report_hash": record.get("report_hash"),
        "input_hash": record.get("input_hash"),
        "summary_json": json.dumps(record.get("summary", {}), default=str),
        "artifact_paths_json": json.dumps(record.get("artifact_paths", {}), default=str),
        "issue_url": record.get("issue_url"),
        "created_at": record.get("created_at") or now,
        "updated_at": now,
        "synced_at": record.get("synced_at") or now,
        "error": record.get("error"),
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    updates = ", ".join(
        f"{column} = excluded.{column}" for column in params if column not in {"report_id", "created_at"}
    )
    with _lock:
        conn.execute(
            f"INSERT INTO report_runs ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(report_id) DO UPDATE SET {updates}",
            tuple(params.values()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM report_runs WHERE report_id = ?", (report_id,)).fetchone()
    return _parse_report_run_json_fields(_require_row_dict(row))


def get_report_runs(report_type: str | None = None, limit: int = 20) -> list[dict]:
    conn = _get_conn()
    safe_limit = max(1, min(int(limit), 100))
    if report_type:
        with _lock:
            rows = conn.execute(
                "SELECT * FROM report_runs WHERE report_type = ? ORDER BY as_of DESC, synced_at DESC LIMIT ?",
                (report_type, safe_limit),
            ).fetchall()
    else:
        with _lock:
            rows = conn.execute(
                "SELECT * FROM report_runs ORDER BY as_of DESC, synced_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
    return [_parse_report_run_json_fields(d) for d in _rows_to_list(rows)]


# ---------------------------------------------------------------------------
# Action Items
# ---------------------------------------------------------------------------


def create_action_item(
    description: str,
    action_type: str = "review",
    *,
    ticker: str | None = None,
    urgency: str = "normal",
    source_type: str = "user",
    source_id: str | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        cur = conn.execute(
            "INSERT INTO action_items (ticker, action_type, description, urgency, source_type, source_id, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (ticker.upper() if ticker else None, action_type, description, urgency, source_type, source_id, now),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "ticker": ticker.upper() if ticker else None,
        "action_type": action_type,
        "description": description,
        "urgency": urgency,
        "status": "open",
        "source_type": source_type,
        "source_id": source_id,
        "created_at": now,
        "completed_at": None,
        "resolution_note": None,
    }


def create_action_item_once(
    description: str,
    action_type: str = "review",
    *,
    ticker: str | None = None,
    urgency: str = "normal",
    source_type: str = "workflow",
    source_id: str | None = None,
) -> dict:
    conn = _get_conn()
    normalized_ticker = ticker.upper() if ticker else None
    if source_id:
        with _lock:
            row = conn.execute(
                "SELECT * FROM action_items WHERE source_type = ? AND source_id = ? AND COALESCE(ticker, '') = COALESCE(?, '') AND description = ? ORDER BY id DESC LIMIT 1",
                (source_type, source_id, normalized_ticker, description),
            ).fetchone()
        if row:
            return _require_row_dict(row)
    return create_action_item(
        description=description,
        action_type=action_type,
        ticker=normalized_ticker,
        urgency=urgency,
        source_type=source_type,
        source_id=source_id,
    )


def get_action_items(
    status: str | None = None,
    ticker: str | None = None,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM action_items{where} ORDER BY CASE urgency WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END, created_at DESC",
            params,
        ).fetchall()
    return _rows_to_list(rows)


def complete_action_item(item_id: int, resolution_note: str = "") -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM action_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise ValueError(f"No action item with id {item_id}")
        conn.execute(
            "UPDATE action_items SET status = 'completed', completed_at = ?, resolution_note = ? WHERE id = ?",
            (now, resolution_note or None, item_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM action_items WHERE id = ?", (item_id,)).fetchone()
    return _require_row_dict(updated)


def dismiss_action_item(item_id: int) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM action_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise ValueError(f"No action item with id {item_id}")
        conn.execute(
            "UPDATE action_items SET status = 'dismissed', completed_at = ? WHERE id = ?",
            (now, item_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM action_items WHERE id = ?", (item_id,)).fetchone()
    return _require_row_dict(updated)


# ---------------------------------------------------------------------------
# Watch Triggers
# ---------------------------------------------------------------------------


def create_watch_trigger(
    condition: str,
    trigger_type: str = "custom",
    *,
    ticker: str | None = None,
    source_type: str = "user",
    source_id: str | None = None,
    expires_at: str | None = None,
    definition: dict | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    trigger_type = str(trigger_type or "custom").strip().lower()
    if trigger_type not in WATCH_TRIGGER_TYPES:
        raise ValueError(f"Invalid watch trigger type: {trigger_type}")
    definition_json = json.dumps(definition, default=str) if definition else None
    with _lock:
        cur = conn.execute(
            "INSERT INTO watch_triggers (ticker, trigger_type, condition, source_type, source_id, created_at, expires_at, definition_json) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                ticker.upper() if ticker else None,
                trigger_type,
                condition,
                source_type,
                source_id,
                now,
                expires_at,
                definition_json,
            ),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "ticker": ticker.upper() if ticker else None,
        "trigger_type": trigger_type,
        "condition": condition,
        "status": "active",
        "source_type": source_type,
        "source_id": source_id,
        "created_at": now,
        "fired_at": None,
        "expires_at": expires_at,
        "definition_json": definition,
        "definition": definition,
        "last_checked_at": None,
        "last_result_json": None,
        "last_result": None,
        "last_evidence": None,
    }


def create_watch_trigger_once(
    condition: str,
    trigger_type: str = "custom",
    *,
    ticker: str | None = None,
    source_type: str = "workflow",
    source_id: str | None = None,
    expires_at: str | None = None,
    definition: dict | None = None,
) -> dict:
    conn = _get_conn()
    normalized_ticker = ticker.upper() if ticker else None
    if source_id:
        with _lock:
            row = conn.execute(
                "SELECT * FROM watch_triggers WHERE source_type = ? AND source_id = ? AND COALESCE(ticker, '') = COALESCE(?, '') AND condition = ? ORDER BY id DESC LIMIT 1",
                (source_type, source_id, normalized_ticker, condition),
            ).fetchone()
        if row:
            return _parse_watch_trigger_json_fields(_require_row_dict(row))
    return create_watch_trigger(
        condition=condition,
        trigger_type=trigger_type,
        ticker=normalized_ticker,
        source_type=source_type,
        source_id=source_id,
        expires_at=expires_at,
        definition=definition,
    )


def _parse_watch_trigger_json_fields(d: dict) -> dict:
    _parse_json_field(d, "definition_json")
    _parse_json_field(d, "last_result_json")
    d["definition"] = d.get("definition_json")
    d["last_result"] = d.get("last_result_json")
    return d


def get_watch_triggers(
    status: str | None = None,
    ticker: str | None = None,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM watch_triggers{where} ORDER BY created_at DESC",
            params,
        ).fetchall()
    return [_parse_watch_trigger_json_fields(d) for d in _rows_to_list(rows)]


def update_watch_trigger_check(
    trigger_id: int,
    *,
    result: dict | None = None,
    evidence: str | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    result_json = json.dumps(result or {}, default=str)
    with _lock:
        row = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
        if not row:
            raise ValueError(f"No watch trigger with id {trigger_id}")
        conn.execute(
            "UPDATE watch_triggers SET last_checked_at = ?, last_result_json = ?, last_evidence = ? WHERE id = ?",
            (now, result_json, evidence, trigger_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
    return _parse_watch_trigger_json_fields(_require_row_dict(updated))


def update_watch_trigger_definition(trigger_id: int, definition: dict) -> dict:
    conn = _get_conn()
    definition_json = json.dumps(definition, default=str)
    with _lock:
        row = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
        if not row:
            raise ValueError(f"No watch trigger with id {trigger_id}")
        conn.execute(
            "UPDATE watch_triggers SET definition_json = ? WHERE id = ?",
            (definition_json, trigger_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
    return _parse_watch_trigger_json_fields(_require_row_dict(updated))


def fire_watch_trigger(
    trigger_id: int,
    *,
    result: dict | None = None,
    evidence: str | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    result_json = json.dumps(result or {}, default=str)
    with _lock:
        row = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
        if not row:
            raise ValueError(f"No watch trigger with id {trigger_id}")
        conn.execute(
            "UPDATE watch_triggers SET status = 'fired', fired_at = ?, last_checked_at = ?, last_result_json = ?, last_evidence = ? WHERE id = ?",
            (now, now, result_json, evidence, trigger_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
    return _parse_watch_trigger_json_fields(_require_row_dict(updated))


def cancel_watch_trigger(trigger_id: int) -> dict:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
        if not row:
            raise ValueError(f"No watch trigger with id {trigger_id}")
        conn.execute(
            "UPDATE watch_triggers SET status = 'cancelled' WHERE id = ?",
            (trigger_id,),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
    return _parse_watch_trigger_json_fields(_require_row_dict(updated))


# ---------------------------------------------------------------------------
# Thesis Claims
# ---------------------------------------------------------------------------

_THESIS_CLAIM_JSON_FIELDS = (
    "source_requirements_json",
    "linked_catalyst_ids_json",
    "linked_kill_condition_ids_json",
)

_THESIS_CLAIM_STATUSES = {"active", "supported", "challenged", "disconfirmed", "retired"}


def normalize_source_requirements(value: Any) -> list[dict[str, Any]]:
    """Normalize legacy string source requirements into typed requirement objects."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            normalized.append(
                {
                    "type": "custom",
                    "description": text,
                    "required": True,
                    "freshness_days": None,
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        req_type = str(item.get("type") or "custom").strip() or "custom"
        description = str(item.get("description") or item.get("label") or req_type).strip()
        required_raw = item.get("required", True)
        if isinstance(required_raw, str):
            required = required_raw.strip().lower() not in {"false", "0", "no", "optional"}
        else:
            required = bool(required_raw)
        freshness_raw = item.get("freshness_days")
        freshness_days = None
        if freshness_raw not in (None, ""):
            try:
                freshness_days = max(0, int(freshness_raw))
            except (TypeError, ValueError):
                freshness_days = None
        normalized.append(
            {
                "type": req_type,
                "description": description,
                "required": required,
                "freshness_days": freshness_days,
            }
        )
    return normalized


def _normalize_thesis_claim_status(status: Any) -> str:
    normalized = str(status or "active").strip().lower()
    if normalized not in _THESIS_CLAIM_STATUSES:
        raise ValueError(f"Invalid thesis claim status: {normalized}")
    return normalized


def _normalize_claim_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    confidence = float(value)
    if confidence < 0 or confidence > 1:
        raise ValueError("Thesis claim confidence must be between 0 and 1")
    return confidence


def _normalize_claim_id_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    ids: list[int] = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def _parse_thesis_claim_json_fields(d: dict) -> dict:
    for field in _THESIS_CLAIM_JSON_FIELDS:
        _parse_json_field(d, field)
    d["source_requirements_json"] = normalize_source_requirements(d.get("source_requirements_json"))
    d["linked_catalyst_ids_json"] = _normalize_claim_id_list(d.get("linked_catalyst_ids_json"))
    d["linked_kill_condition_ids_json"] = _normalize_claim_id_list(d.get("linked_kill_condition_ids_json"))
    # Friendly aliases for API/UI callers while retaining the DB column-shaped keys.
    d["source_requirements"] = d["source_requirements_json"]
    d["linked_catalyst_ids"] = d["linked_catalyst_ids_json"]
    d["linked_kill_condition_ids"] = d["linked_kill_condition_ids_json"]
    return d


def create_thesis_claim(record: dict) -> dict:
    conn = _get_conn()
    now = _now()
    ticker = str(record["ticker"]).upper()
    params = {
        "ticker": ticker,
        "claim": record["claim"],
        "expected_evidence": record.get("expected_evidence"),
        "disconfirming_evidence": record.get("disconfirming_evidence"),
        "source_requirements_json": _encode_json(normalize_source_requirements(record.get("source_requirements", []))),
        "cadence": record.get("cadence"),
        "confidence": _normalize_claim_confidence(record.get("confidence")),
        "status": _normalize_thesis_claim_status(record.get("status")),
        "linked_catalyst_ids_json": _encode_json(_normalize_claim_id_list(record.get("linked_catalyst_ids", []))),
        "linked_kill_condition_ids_json": _encode_json(
            _normalize_claim_id_list(record.get("linked_kill_condition_ids", []))
        ),
        "source_type": record.get("source_type") or "user",
        "source_id": record.get("source_id"),
        "created_at": record.get("created_at") or now,
        "updated_at": now,
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    with _lock:
        cur = conn.execute(
            f"INSERT INTO thesis_claims ({columns}) VALUES ({placeholders})",
            tuple(params.values()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM thesis_claims WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _parse_thesis_claim_json_fields(_require_row_dict(row))


def create_thesis_claim_once(record: dict) -> dict:
    conn = _get_conn()
    ticker = str(record["ticker"]).upper()
    claim = str(record["claim"])
    source_type = record.get("source_type") or "workflow"
    source_id = record.get("source_id")
    if source_id:
        with _lock:
            row = conn.execute(
                "SELECT * FROM thesis_claims WHERE ticker = ? AND claim = ? AND source_type = ? AND source_id = ? ORDER BY id DESC LIMIT 1",
                (ticker, claim, source_type, source_id),
            ).fetchone()
        if row:
            return _parse_thesis_claim_json_fields(_require_row_dict(row))
    return create_thesis_claim({**record, "ticker": ticker, "source_type": source_type})


def get_thesis_claim(claim_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM thesis_claims WHERE id = ?", (claim_id,)).fetchone()
    if not row:
        return None
    return _parse_thesis_claim_json_fields(_require_row_dict(row))


def get_thesis_claims(
    ticker: str | None = None,
    status: str | None = None,
    limit: int = 100,
    *,
    source_type: str | None = None,
    source_id: str | None = None,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    if status:
        clauses.append("status = ?")
        params.append(status)
    if source_type:
        clauses.append("source_type = ?")
        params.append(source_type)
    if source_id:
        clauses.append("source_id = ?")
        params.append(source_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    safe_limit = max(1, min(int(limit), 500))
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM thesis_claims{where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*params, safe_limit),
        ).fetchall()
    return [_parse_thesis_claim_json_fields(d) for d in _rows_to_list(rows)]


def delete_thesis_claims_by_ticker(
    ticker: str,
    *,
    source_type: str | None = None,
    source_id: str | None = None,
    exclude_ids: list[int] | None = None,
) -> int:
    """Delete thesis claims for a ticker, optionally scoped to a source."""
    conn = _get_conn()
    clauses = ["ticker = ?"]
    params: list[Any] = [ticker.upper()]
    if source_type:
        clauses.append("source_type = ?")
        params.append(source_type)
    if source_id:
        clauses.append("source_id = ?")
        params.append(source_id)
    exclude_ids = _normalize_claim_id_list(exclude_ids or [])
    if exclude_ids:
        placeholders = ", ".join("?" for _ in exclude_ids)
        clauses.append(f"id NOT IN ({placeholders})")
        params.extend(exclude_ids)
    with _lock:
        cur = conn.execute(f"DELETE FROM thesis_claims WHERE {' AND '.join(clauses)}", params)
        conn.commit()
    return cur.rowcount


def update_thesis_claim(claim_id: int, updates: dict) -> dict:
    allowed = {
        "claim",
        "expected_evidence",
        "disconfirming_evidence",
        "cadence",
        "confidence",
        "status",
        "source_requirements",
        "linked_catalyst_ids",
        "linked_kill_condition_ids",
    }
    params: dict[str, Any] = {}
    for key, value in updates.items():
        if key not in allowed:
            continue
        if key == "source_requirements":
            params["source_requirements_json"] = _encode_json(normalize_source_requirements(value))
        elif key == "linked_catalyst_ids":
            params["linked_catalyst_ids_json"] = _encode_json(_normalize_claim_id_list(value))
        elif key == "linked_kill_condition_ids":
            params["linked_kill_condition_ids_json"] = _encode_json(_normalize_claim_id_list(value))
        elif key == "confidence":
            params[key] = _normalize_claim_confidence(value)
        elif key == "status":
            params[key] = _normalize_thesis_claim_status(value)
        else:
            params[key] = value
    if not params:
        raise ValueError("No valid thesis claim updates supplied")
    params["updated_at"] = _now()
    conn = _get_conn()
    set_clause = ", ".join(f"{key} = ?" for key in params)
    with _lock:
        row = conn.execute("SELECT * FROM thesis_claims WHERE id = ?", (claim_id,)).fetchone()
        if not row:
            raise ValueError(f"No thesis claim with id {claim_id}")
        conn.execute(
            f"UPDATE thesis_claims SET {set_clause} WHERE id = ?",
            (*params.values(), claim_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM thesis_claims WHERE id = ?", (claim_id,)).fetchone()
    return _parse_thesis_claim_json_fields(_require_row_dict(updated))


# ---------------------------------------------------------------------------
# Research Notes
# ---------------------------------------------------------------------------


def create_research_note(
    title: str,
    content: str,
    *,
    ticker: str | None = None,
    note_type: str = "general",
    source_type: str = "user",
    source_id: str | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        cur = conn.execute(
            "INSERT INTO research_notes (ticker, title, content, note_type, source_type, source_id, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (ticker.upper() if ticker else None, title, content, note_type, source_type, source_id, now),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "ticker": ticker.upper() if ticker else None,
        "title": title,
        "content": content,
        "note_type": note_type,
        "source_type": source_type,
        "source_id": source_id,
        "created_at": now,
    }


def create_research_note_once(
    title: str,
    content: str,
    *,
    ticker: str | None = None,
    note_type: str = "general",
    source_type: str = "workflow",
    source_id: str | None = None,
) -> dict:
    conn = _get_conn()
    normalized_ticker = ticker.upper() if ticker else None
    if source_id:
        with _lock:
            row = conn.execute(
                "SELECT * FROM research_notes WHERE source_type = ? AND source_id = ? AND COALESCE(ticker, '') = COALESCE(?, '') AND title = ? ORDER BY id DESC LIMIT 1",
                (source_type, source_id, normalized_ticker, title),
            ).fetchone()
        if row:
            return _require_row_dict(row)
    return create_research_note(
        title=title,
        content=content,
        ticker=normalized_ticker,
        note_type=note_type,
        source_type=source_type,
        source_id=source_id,
    )


def get_research_notes(
    ticker: str | None = None,
    limit: int = 20,
) -> list[dict]:
    conn = _get_conn()
    if ticker:
        with _lock:
            rows = conn.execute(
                "SELECT * FROM research_notes WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
                (ticker.upper(), limit),
            ).fetchall()
    else:
        with _lock:
            rows = conn.execute(
                "SELECT * FROM research_notes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return _rows_to_list(rows)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

_RECOMMENDATION_JSON_FIELDS = (
    "blocked_reasons_json",
    "what_changed_json",
    "evidence_json",
    "disconfirming_evidence_json",
    "alternatives_json",
    "opportunity_cost_json",
    "outcome_json",
    "source_quality_summary_json",
)


def _encode_json(value: Any) -> str:
    return json.dumps(value if value is not None else [], default=str)


def _parse_recommendation_json_fields(d: dict) -> dict:
    for field in _RECOMMENDATION_JSON_FIELDS:
        _parse_json_field(d, field)
    return d


def create_recommendation(record: dict) -> dict:
    conn = _get_conn()
    now = record.get("created_at") or _now()
    ticker = record.get("ticker")
    status = record.get("status") or (
        "blocked"
        if record.get("recommendation_status") == "blocked"
        else "error"
        if record.get("recommendation_status") == "error"
        else "open"
    )
    params = {
        "report_type": record["report_type"],
        "as_of": record["as_of"],
        "created_at": now,
        "source_report_path": record.get("source_report_path"),
        "source_json_path": record.get("source_json_path"),
        "stance": record["stance"],
        "recommendation_status": record.get("recommendation_status", "clear"),
        "critical_data_quality": record.get("critical_data_quality", "ok"),
        "blocked_reasons_json": _encode_json(record.get("blocked_reasons", [])),
        "what_changed_json": _encode_json(record.get("what_changed", [])),
        "do_nothing_rationale": record.get("do_nothing_rationale", ""),
        "action": record["action"],
        "ticker": ticker.upper() if isinstance(ticker, str) and ticker else None,
        "instrument": record.get("instrument") or "portfolio",
        "horizon": record.get("horizon"),
        "target_change": record.get("target_change"),
        "rationale": record.get("rationale") or "",
        "confidence": record.get("confidence"),
        "source_quality": record.get("source_quality", "ok"),
        "status": status,
        "evidence_json": _encode_json(record.get("evidence", [])),
        "disconfirming_evidence_json": _encode_json(record.get("disconfirming_evidence", [])),
        "catalyst": record.get("catalyst"),
        "invalidation": record.get("invalidation"),
        "expected_onset_window": record.get("expected_onset_window"),
        "alternatives_json": _encode_json(record.get("alternatives", [])),
        "opportunity_cost_json": _encode_json(record.get("opportunity_cost", [])),
        "approval_id": record.get("approval_id"),
        "approval_status": record.get("approval_status", "none"),
        "outcome_status": record.get("outcome_status", "pending"),
        "outcome_json": _encode_json(record.get("outcome", {})),
        "model": record.get("model"),
        "prompt_hash": record.get("prompt_hash"),
        "input_hash": record.get("input_hash"),
        "validation_status": record.get("validation_status"),
        "source_quality_summary_json": _encode_json(record.get("source_quality_summary", {})),
        "report_id": record.get("report_id"),
        "idempotency_key": record.get("idempotency_key"),
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    with _lock:
        cur = conn.execute(
            f"INSERT INTO recommendations ({columns}) VALUES ({placeholders})",
            tuple(params.values()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _parse_recommendation_json_fields(_require_row_dict(row))


def upsert_recommendation(record: dict) -> dict:
    if not record.get("idempotency_key"):
        return create_recommendation(record)

    conn = _get_conn()
    now = record.get("created_at") or _now()
    ticker = record.get("ticker")
    status = record.get("status") or (
        "blocked"
        if record.get("recommendation_status") == "blocked"
        else "error"
        if record.get("recommendation_status") == "error"
        else "open"
    )
    params = {
        "report_type": record["report_type"],
        "as_of": record["as_of"],
        "created_at": now,
        "source_report_path": record.get("source_report_path"),
        "source_json_path": record.get("source_json_path"),
        "stance": record["stance"],
        "recommendation_status": record.get("recommendation_status", "clear"),
        "critical_data_quality": record.get("critical_data_quality", "ok"),
        "blocked_reasons_json": _encode_json(record.get("blocked_reasons", [])),
        "what_changed_json": _encode_json(record.get("what_changed", [])),
        "do_nothing_rationale": record.get("do_nothing_rationale", ""),
        "action": record["action"],
        "ticker": ticker.upper() if isinstance(ticker, str) and ticker else None,
        "instrument": record.get("instrument") or "portfolio",
        "horizon": record.get("horizon"),
        "target_change": record.get("target_change"),
        "rationale": record.get("rationale") or "",
        "confidence": record.get("confidence"),
        "source_quality": record.get("source_quality", "ok"),
        "status": status,
        "evidence_json": _encode_json(record.get("evidence", [])),
        "disconfirming_evidence_json": _encode_json(record.get("disconfirming_evidence", [])),
        "catalyst": record.get("catalyst"),
        "invalidation": record.get("invalidation"),
        "expected_onset_window": record.get("expected_onset_window"),
        "alternatives_json": _encode_json(record.get("alternatives", [])),
        "opportunity_cost_json": _encode_json(record.get("opportunity_cost", [])),
        "approval_id": record.get("approval_id"),
        "approval_status": record.get("approval_status", "none"),
        "outcome_status": record.get("outcome_status", "pending"),
        "outcome_json": _encode_json(record.get("outcome", {})),
        "model": record.get("model"),
        "prompt_hash": record.get("prompt_hash"),
        "input_hash": record.get("input_hash"),
        "validation_status": record.get("validation_status"),
        "source_quality_summary_json": _encode_json(record.get("source_quality_summary", {})),
        "report_id": record.get("report_id"),
        "idempotency_key": record["idempotency_key"],
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    updates = ", ".join(
        f"{column} = COALESCE(excluded.{column}, {column})"
        if column in {"approval_id"}
        else f"{column} = excluded.{column}"
        for column in params
        if column
        not in {"created_at", "idempotency_key", "approval_status", "approval_id", "outcome_status", "outcome_json"}
    )
    updates = (
        f"{updates}, "
        "approval_id = COALESCE(recommendations.approval_id, excluded.approval_id), "
        "approval_status = CASE WHEN recommendations.approval_id IS NOT NULL THEN recommendations.approval_status ELSE excluded.approval_status END, "
        "outcome_status = recommendations.outcome_status, "
        "outcome_json = recommendations.outcome_json"
    )
    with _lock:
        conn.execute(
            f"INSERT INTO recommendations ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(idempotency_key) DO UPDATE SET {updates}",
            tuple(params.values()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM recommendations WHERE idempotency_key = ?",
            (record["idempotency_key"],),
        ).fetchone()
    return _parse_recommendation_json_fields(_require_row_dict(row))


def get_recommendations(
    report_type: str | None = None,
    status: str | None = None,
    ticker: str | None = None,
    approval_status: str | None = None,
    outcome_status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if report_type:
        clauses.append("report_type = ?")
        params.append(report_type)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    if approval_status:
        clauses.append("approval_status = ?")
        params.append(approval_status)
    if outcome_status:
        clauses.append("outcome_status = ?")
        params.append(outcome_status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM recommendations{where} ORDER BY as_of DESC, created_at DESC, id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [_parse_recommendation_json_fields(d) for d in _rows_to_list(rows)]


def get_recommendation(recommendation_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    if not row:
        return None
    return _parse_recommendation_json_fields(_require_row_dict(row))


def get_latest_recommendation(report_type: str | None = None) -> dict | None:
    results = get_recommendations(report_type=report_type, limit=1)
    return results[0] if results else None


def _update_recommendation_approval_tx(
    conn: sqlite3.Connection | PostgresCompatConnection,
    recommendation_id: int,
    approval_id: int | None,
    approval_status: str,
) -> dict:
    row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    if not row:
        raise ValueError(f"No recommendation with id {recommendation_id}")
    conn.execute(
        "UPDATE recommendations SET approval_id = COALESCE(?, approval_id), approval_status = ? WHERE id = ?",
        (approval_id, approval_status, recommendation_id),
    )
    updated = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    return _parse_recommendation_json_fields(_require_row_dict(updated))


def update_recommendation_approval(
    recommendation_id: int,
    approval_id: int | None,
    approval_status: str,
) -> dict:
    conn = _get_conn()
    with _lock:
        updated = _update_recommendation_approval_tx(conn, recommendation_id, approval_id, approval_status)
        conn.commit()
    return updated


def supersede_report_recommendations(report_id: str, active_idempotency_keys: list[str]) -> int:
    conn = _get_conn()
    if not report_id:
        return 0
    active = set(active_idempotency_keys)
    with _lock:
        rows = conn.execute(
            "SELECT id, idempotency_key FROM recommendations WHERE report_id = ? AND status IN ('open', 'blocked', 'error')",
            (report_id,),
        ).fetchall()
        superseded_ids = [int(row["id"]) for row in rows if str(row["idempotency_key"] or "") not in active]
        if not superseded_ids:
            return 0
        placeholders = ", ".join("?" for _ in superseded_ids)
        conn.execute(
            f"UPDATE recommendations SET status = 'superseded' WHERE id IN ({placeholders})",
            tuple(superseded_ids),
        )
        conn.commit()
    return len(superseded_ids)


def update_recommendation_outcome(
    recommendation_id: int,
    outcome_status: str,
    outcome: dict,
) -> dict:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
        if not row:
            raise ValueError(f"No recommendation with id {recommendation_id}")
        conn.execute(
            "UPDATE recommendations SET outcome_status = ?, outcome_json = ? WHERE id = ?",
            (outcome_status, json.dumps(outcome, default=str), recommendation_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    return _parse_recommendation_json_fields(_require_row_dict(updated))


# ---------------------------------------------------------------------------
# Pending Approvals
# ---------------------------------------------------------------------------

ApprovalPostCommitCallback = Callable[[], None]
ApprovalSideEffectHandler = Callable[
    [sqlite3.Connection | PostgresCompatConnection, dict, dict, list[ApprovalPostCommitCallback]],
    None,
]


class ApprovalApplicationError(RuntimeError):
    """Raised when an approval side effect fails and remains retryable."""

    def __init__(self, approval_id: int, error: str):
        self.approval_id = approval_id
        self.error = error
        super().__init__(f"Approval {approval_id} application failed: {error}")


def _parse_pending_approval_row(row: Any) -> dict:
    d = _require_row_dict(row)
    _parse_json_field(d, "proposed_change")
    if d.get("application_attempts") is None:
        d["application_attempts"] = 0
    if not d.get("application_status"):
        d["application_status"] = "pending"
    return d


def create_pending_approval(
    entity_type: str,
    proposed_change: dict,
    *,
    entity_id: int | None = None,
    ticker: str | None = None,
    reason: str | None = None,
    source_type: str = "workflow",
    source_id: str | None = None,
    action_id: str | None = None,
    action_schema_version: int | None = None,
    action_input_hash: str | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    change_json = json.dumps(proposed_change, default=str)
    with _lock:
        cur = conn.execute(
            "INSERT INTO pending_approvals (entity_type, entity_id, ticker, action_id, action_schema_version, "
            "action_input_hash, proposed_change, reason, source_type, source_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                entity_type,
                entity_id,
                ticker.upper() if ticker else None,
                action_id,
                action_schema_version,
                action_input_hash,
                change_json,
                reason,
                source_type,
                source_id,
                now,
            ),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "ticker": ticker.upper() if ticker else None,
        "action_id": action_id,
        "action_schema_version": action_schema_version,
        "action_input_hash": action_input_hash,
        "proposed_change": proposed_change,
        "reason": reason,
        "source_type": source_type,
        "source_id": source_id,
        "status": "pending",
        "created_at": now,
        "resolved_at": None,
        "resolved_note": None,
        "application_status": "pending",
        "application_attempts": 0,
        "application_started_at": None,
        "application_completed_at": None,
        "application_error": None,
    }


def create_pending_approval_once(
    entity_type: str,
    proposed_change: dict,
    *,
    entity_id: int | None = None,
    ticker: str | None = None,
    reason: str | None = None,
    source_type: str = "workflow",
    source_id: str | None = None,
    action_id: str | None = None,
    action_schema_version: int | None = None,
    action_input_hash: str | None = None,
) -> dict:
    proposed_hash = _json_hash(proposed_change)
    normalized_ticker = ticker.upper() if ticker else None
    if source_id:
        for approval in get_pending_approvals(status=None, ticker=normalized_ticker):
            if approval.get("entity_type") != entity_type:
                continue
            if action_id and approval.get("action_id") != action_id:
                continue
            if approval.get("source_type") != source_type or approval.get("source_id") != source_id:
                continue
            if _json_hash(approval.get("proposed_change")) == proposed_hash:
                return approval
    return create_pending_approval(
        entity_type=entity_type,
        proposed_change=proposed_change,
        entity_id=entity_id,
        ticker=normalized_ticker,
        reason=reason,
        source_type=source_type,
        source_id=source_id,
        action_id=action_id,
        action_schema_version=action_schema_version,
        action_input_hash=action_input_hash,
    )


def get_pending_approvals(
    status: str | None = "pending",
    ticker: str | None = None,
    application_status: str | None = None,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    if application_status:
        if application_status not in APPROVAL_APPLICATION_STATUSES:
            raise ValueError(f"Invalid application_status: {application_status}")
        clauses.append("application_status = ?")
        params.append(application_status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM pending_approvals{where} ORDER BY created_at DESC",
            params,
        ).fetchall()
    return [_parse_pending_approval_row(row) for row in rows]


def get_pending_approval(approval_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
    if not row:
        return None
    return _parse_pending_approval_row(row)


def resolve_approval(
    approval_id: int,
    status: str,
    resolved_note: str | None = None,
    *,
    actor_type: str = "user",
    actor_id: str | None = None,
    parent_action_run_id: int | None = None,
) -> dict:
    """Resolve a pending approval and apply approved side effects safely."""
    run = create_action_run(
        action_id="resolve_approval",
        action_schema_version=1,
        actor_type=actor_type,
        actor_id=actor_id,
        approval_id=approval_id,
        parent_action_run_id=parent_action_run_id,
        input_hash=_json_hash({"approval_id": approval_id, "status": status, "resolved_note": resolved_note}),
        input_payload={"approval_id": approval_id, "status": status, "resolved_note": resolved_note},
    )
    run_id = int(run["id"])
    record_action_event(run_id, "start", payload={"approval_id": approval_id, "status": status})
    try:
        result = _resolve_approval_impl(approval_id, status, resolved_note, parent_action_run_id=run_id)
    except Exception as exc:
        error = _approval_error_message(exc)
        record_action_event(run_id, "error", message=error)
        complete_action_run(run_id, status="failed", error=error)
        raise
    record_action_event(run_id, "complete", payload={"status": result.get("status")})
    complete_action_run(run_id, status="succeeded", output_payload=result)
    return result


def _resolve_approval_impl(
    approval_id: int,
    status: str,
    resolved_note: str | None = None,
    *,
    parent_action_run_id: int | None = None,
) -> dict:
    if status not in ("approved", "rejected"):
        raise ValueError(f"Resolution status must be 'approved' or 'rejected', got '{status}'")

    if status == "rejected":
        return _reject_approval(approval_id, resolved_note)

    conn = _get_conn()
    approval, should_apply = _claim_approval_for_application(conn, approval_id)
    if not should_apply:
        return approval

    callbacks: list[ApprovalPostCommitCallback] = []
    try:
        with _lock:
            try:
                _apply_approval_side_effect_tx(conn, approval, callbacks, parent_action_run_id=parent_action_run_id)
                _update_linked_recommendation_approval_tx(conn, approval, approval_id, "approved")
                now = _now()
                conn.execute(
                    "UPDATE pending_approvals "
                    "SET status = 'approved', resolved_at = ?, resolved_note = ?, "
                    "application_status = 'applied', application_completed_at = ?, application_error = NULL "
                    "WHERE id = ?",
                    (now, resolved_note, now, approval_id),
                )
                updated = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    except Exception as exc:
        _mark_approval_application_failed(approval_id, exc)
        error = _approval_error_message(exc)
        raise ApprovalApplicationError(approval_id, error) from exc

    _run_approval_post_commit_callbacks(callbacks)
    return _parse_pending_approval_row(updated)


def _reject_approval(approval_id: int, resolved_note: str | None) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
        if not row:
            raise ValueError(f"No pending approval with id {approval_id}")
        current = _parse_pending_approval_row(row)
        if current["status"] != "pending":
            raise ValueError(f"Approval {approval_id} is already {current['status']}")
        application_status = str(current.get("application_status") or "pending")
        if application_status == "applying" and not _approval_application_is_stale(current):
            raise ValueError(f"Approval {approval_id} application is already in progress")

        try:
            _update_linked_recommendation_approval_tx(conn, current, approval_id, "rejected")
            conn.execute(
                "UPDATE pending_approvals "
                "SET status = 'rejected', resolved_at = ?, resolved_note = ?, "
                "application_status = 'not_applicable', application_completed_at = ?, application_error = NULL "
                "WHERE id = ?",
                (now, resolved_note, now, approval_id),
            )
            updated = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return _parse_pending_approval_row(updated)


def _claim_approval_for_application(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval_id: int,
) -> tuple[dict, bool]:
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
        if not row:
            raise ValueError(f"No pending approval with id {approval_id}")
        current = _parse_pending_approval_row(row)
        application_status = str(current.get("application_status") or "pending")
        if current["status"] == "approved" and application_status == "applied":
            return current, False
        if current["status"] != "pending":
            raise ValueError(f"Approval {approval_id} is already {current['status']}")
        if application_status == "applying" and not _approval_application_is_stale(current):
            raise ValueError(f"Approval {approval_id} application is already in progress")
        if application_status not in {"pending", "failed", "applying"}:
            raise ValueError(f"Approval {approval_id} cannot be applied from state {application_status}")

        conn.execute(
            "UPDATE pending_approvals "
            "SET application_status = 'applying', application_attempts = COALESCE(application_attempts, 0) + 1, "
            "application_started_at = ?, application_completed_at = NULL, application_error = NULL "
            "WHERE id = ?",
            (now, approval_id),
        )
        updated = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
        conn.commit()
    return _parse_pending_approval_row(updated), True


def _approval_application_is_stale(approval: dict) -> bool:
    started_at = approval.get("application_started_at")
    if not started_at:
        return True
    try:
        started = datetime.fromisoformat(str(started_at))
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return datetime.now(UTC) - started >= APPROVAL_APPLICATION_LEASE


def _mark_approval_application_failed(approval_id: int, exc: Exception) -> None:
    conn = _get_conn()
    now = _now()
    error = _approval_error_message(exc)
    with _lock:
        conn.execute(
            "UPDATE pending_approvals "
            "SET application_status = 'failed', application_completed_at = ?, application_error = ? "
            "WHERE id = ? AND status = 'pending'",
            (now, error, approval_id),
        )
        conn.commit()


def _approval_error_message(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:1000]


def _approval_change(approval: dict) -> dict:
    change = approval.get("proposed_change")
    if isinstance(change, str):
        try:
            change = json.loads(change)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Approval proposed_change must be a JSON object") from exc
    if not isinstance(change, dict):
        raise ValueError("Approval proposed_change must be a JSON object")
    return change


def _update_linked_recommendation_approval_tx(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    approval_id: int,
    approval_status: str,
) -> None:
    change = approval.get("proposed_change")
    if not isinstance(change, dict) or change.get("recommendation_id") is None:
        return
    _update_recommendation_approval_tx(conn, int(change["recommendation_id"]), approval_id, approval_status)


def _apply_approval_side_effect_tx(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    callbacks: list[ApprovalPostCommitCallback],
    *,
    parent_action_run_id: int | None = None,
) -> None:
    action_id = str(approval.get("action_id") or "").strip()
    if action_id:
        from portfolio.action_registry import ActionContext, execute_action

        execute_action(
            action_id,
            _approval_change(approval),
            ActionContext(
                actor_type="approval_apply",
                source_type=approval.get("source_type"),
                source_id=approval.get("source_id"),
                approval_id=int(approval["id"]),
                parent_action_run_id=parent_action_run_id,
            ),
        )
        return
    entity_type = str(approval.get("entity_type") or "")
    handler = _APPROVAL_SIDE_EFFECT_HANDLERS.get(entity_type)
    if handler is None:
        raise ValueError(f"Unsupported approval entity_type: {entity_type}")
    handler(conn, approval, _approval_change(approval), callbacks)


def _run_approval_post_commit_callbacks(callbacks: list[ApprovalPostCommitCallback]) -> None:
    for callback in callbacks:
        try:
            callback()
        except Exception:
            logger.warning("Approval post-commit callback failed", exc_info=True)


def _optional_ticker(value: Any) -> str | None:
    ticker = str(value or "").strip().upper()
    return ticker or None


def _required_ticker(value: Any, entity_type: str) -> str:
    ticker = _optional_ticker(value)
    if not ticker:
        raise ValueError(f"{entity_type} approval requires ticker")
    return ticker


def _sync_markdown_callback(ticker: str) -> ApprovalPostCommitCallback:
    def _callback() -> None:
        from portfolio.thesis_sync import sync_markdown_from_entities

        sync_markdown_from_entities(ticker)

    return _callback


def _handle_thesis_status_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, callbacks
    ticker = _required_ticker(change.get("ticker") or approval.get("ticker"), "thesis_status")
    new_status = str(change.get("new_status") or "").strip().lower()
    if not new_status:
        raise ValueError("thesis_status approval requires new_status")
    reason = str(change.get("reason") or "")

    from portfolio.thesis_db import get_thesis_meta, update_thesis_status

    current = get_thesis_meta(ticker)
    if current and current.get("status") == new_status:
        return
    update_thesis_status(ticker, new_status, reason)


def _handle_evaluation_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, approval, callbacks
    from portfolio.thesis_db import save_evaluations

    evaluated_at = str(change.get("evaluated_at") or _now())
    evaluations = change.get("evaluations", [change])
    if not isinstance(evaluations, list):
        raise ValueError("evaluation approval requires evaluations to be a list")
    save_evaluations(evaluated_at, evaluations)


def _handle_catalyst_status_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del callbacks
    catalyst_id = change.get("catalyst_id") or approval.get("entity_id")
    if not catalyst_id:
        raise ValueError("catalyst_status approval requires catalyst_id")
    now = _now()
    row = conn.execute("SELECT * FROM catalysts WHERE id = ?", (int(catalyst_id),)).fetchone()
    if not row:
        raise ValueError(f"No catalyst with id {catalyst_id}")
    updates = {"status": change.get("status", "played_out"), "updated_at": now}
    if change.get("evidence") is not None:
        updates["evidence"] = change.get("evidence")
    set_clause = ", ".join(f"{key} = ?" for key in updates)
    conn.execute(f"UPDATE catalysts SET {set_clause} WHERE id = ?", (*updates.values(), int(catalyst_id)))


def _handle_kill_condition_status_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del callbacks
    kc_id = change.get("kill_condition_id") or approval.get("entity_id")
    if not kc_id:
        raise ValueError("kill_condition_status approval requires kill_condition_id")
    now = _now()
    row = conn.execute("SELECT * FROM kill_conditions WHERE id = ?", (int(kc_id),)).fetchone()
    if not row:
        raise ValueError(f"No kill condition with id {kc_id}")
    current = _require_row_dict(row)
    status = change.get("status", "triggered")
    triggered_at = now if status == "triggered" else current.get("triggered_at")
    conn.execute(
        "UPDATE kill_conditions SET status = ?, triggered_at = ?, updated_at = ? WHERE id = ?",
        (status, triggered_at, now, int(kc_id)),
    )


def _handle_action_item_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del callbacks
    now = _now()
    conn.execute(
        "INSERT INTO action_items (ticker, action_type, description, urgency, source_type, source_id, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            _optional_ticker(change.get("ticker") or approval.get("ticker")),
            change.get("action_type", "review"),
            change.get("description", ""),
            change.get("urgency", "normal"),
            approval.get("source_type", "workflow"),
            approval.get("source_id"),
            now,
        ),
    )


def _handle_watch_trigger_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del callbacks
    now = _now()
    trigger_type = str(change.get("trigger_type") or "custom").strip().lower()
    if trigger_type not in WATCH_TRIGGER_TYPES:
        raise ValueError(f"Invalid watch trigger type: {trigger_type}")
    definition = change.get("definition")
    definition_json = json.dumps(definition, default=str) if definition else None
    conn.execute(
        "INSERT INTO watch_triggers (ticker, trigger_type, condition, source_type, source_id, created_at, expires_at, definition_json) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            _optional_ticker(change.get("ticker") or approval.get("ticker")),
            trigger_type,
            change.get("condition", ""),
            approval.get("source_type", "workflow"),
            approval.get("source_id"),
            now,
            change.get("expires_at"),
            definition_json,
        ),
    )


def _handle_portfolio_positions_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, approval, callbacks
    from api.routers.portfolio_edit import PortfolioUpdateRequest, update_portfolio_positions

    update_portfolio_positions(PortfolioUpdateRequest(positions=change.get("positions") or []))


def _handle_hedge_positions_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, approval, callbacks
    from api.routers.portfolio_edit import HedgeUpdateRequest, update_hedge_positions

    update_hedge_positions(HedgeUpdateRequest(positions=change.get("positions") or []))


def _handle_thesis_content_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, callbacks
    from api.routers.thesis import SaveThesisRequest, save_thesis

    save_thesis(
        _required_ticker(change.get("ticker") or approval.get("ticker"), "thesis_content"),
        SaveThesisRequest(content=change.get("content", "")),
    )


def _handle_catalyst_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    now = _now()
    ticker = _required_ticker(change.get("ticker") or approval.get("ticker"), "catalyst")
    conn.execute(
        "INSERT INTO catalysts (ticker, description, category, target_date, evidence, created_at, updated_at, created_by) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            ticker,
            change.get("description", ""),
            change.get("category", "fundamental"),
            change.get("target_date"),
            change.get("evidence"),
            now,
            now,
            "agent",
        ),
    )
    callbacks.append(_sync_markdown_callback(ticker))


def _handle_kill_condition_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    now = _now()
    ticker = _required_ticker(change.get("ticker") or approval.get("ticker"), "kill_condition")
    conn.execute(
        "INSERT INTO kill_conditions (ticker, condition, metric, threshold, created_at, updated_at, created_by) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            ticker,
            change.get("condition", ""),
            change.get("metric"),
            change.get("threshold"),
            now,
            now,
            "agent",
        ),
    )
    callbacks.append(_sync_markdown_callback(ticker))


def _handle_research_note_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del callbacks
    now = _now()
    conn.execute(
        "INSERT INTO research_notes (ticker, title, content, note_type, source_type, source_id, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            _optional_ticker(change.get("ticker") or approval.get("ticker")),
            change.get("title", ""),
            change.get("content", ""),
            change.get("note_type", "general"),
            approval.get("source_type", "workflow"),
            approval.get("source_id"),
            now,
        ),
    )


def _handle_news_digest_delete_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, approval
    from api.routers.portfolio_news import _delete_digest_index_best_effort
    from portfolio.news_digests import delete_digest, validate_digest_id

    digest_id = validate_digest_id(str(change.get("digest_id") or ""))
    deleted = delete_digest(digest_id)
    if deleted:
        callbacks.append(lambda digest_id=digest_id: _delete_digest_index_best_effort(digest_id))


_APPROVAL_SIDE_EFFECT_HANDLERS: dict[str, ApprovalSideEffectHandler] = {
    "thesis_status": _handle_thesis_status_approval,
    "evaluation": _handle_evaluation_approval,
    "catalyst_status": _handle_catalyst_status_approval,
    "kill_condition_status": _handle_kill_condition_status_approval,
    "action_item": _handle_action_item_approval,
    "watch_trigger": _handle_watch_trigger_approval,
    "portfolio_positions": _handle_portfolio_positions_approval,
    "hedge_positions": _handle_hedge_positions_approval,
    "thesis_content": _handle_thesis_content_approval,
    "catalyst": _handle_catalyst_approval,
    "kill_condition": _handle_kill_condition_approval,
    "research_note": _handle_research_note_approval,
    "news_digest_delete": _handle_news_digest_delete_approval,
}
