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
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from api.postgres import use_postgres_state
from api.postgres_compat import PostgresCompatConnection

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "core.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

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
                 CHECK (trigger_type IN ('price_level', 'technical', 'fundamental', 'event', 'macro', 'custom')),
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
    proposed_change TEXT NOT NULL,
    reason          TEXT,
    source_type     TEXT NOT NULL DEFAULT 'workflow'
                    CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id       TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    resolved_note   TEXT
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

_lock = threading.Lock()
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
        "last_checked_at": None,
        "last_result_json": None,
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


def _parse_thesis_claim_json_fields(d: dict) -> dict:
    for field in _THESIS_CLAIM_JSON_FIELDS:
        _parse_json_field(d, field)
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
        "source_requirements_json": _encode_json(record.get("source_requirements", [])),
        "cadence": record.get("cadence"),
        "confidence": record.get("confidence"),
        "status": record.get("status") or "active",
        "linked_catalyst_ids_json": _encode_json(record.get("linked_catalyst_ids", [])),
        "linked_kill_condition_ids_json": _encode_json(record.get("linked_kill_condition_ids", [])),
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


def get_thesis_claims(
    ticker: str | None = None,
    status: str | None = None,
    limit: int = 100,
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
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    safe_limit = max(1, min(int(limit), 500))
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM thesis_claims{where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*params, safe_limit),
        ).fetchall()
    return [_parse_thesis_claim_json_fields(d) for d in _rows_to_list(rows)]


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
            params["source_requirements_json"] = _encode_json(value)
        elif key == "linked_catalyst_ids":
            params["linked_catalyst_ids_json"] = _encode_json(value)
        elif key == "linked_kill_condition_ids":
            params["linked_kill_condition_ids_json"] = _encode_json(value)
        else:
            params[key] = value
    params["updated_at"] = _now()
    if not params:
        raise ValueError("No valid thesis claim updates supplied")
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


def update_recommendation_approval(
    recommendation_id: int,
    approval_id: int | None,
    approval_status: str,
) -> dict:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
        if not row:
            raise ValueError(f"No recommendation with id {recommendation_id}")
        conn.execute(
            "UPDATE recommendations SET approval_id = COALESCE(?, approval_id), approval_status = ? WHERE id = ?",
            (approval_id, approval_status, recommendation_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    return _parse_recommendation_json_fields(_require_row_dict(updated))


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


def create_pending_approval(
    entity_type: str,
    proposed_change: dict,
    *,
    entity_id: int | None = None,
    ticker: str | None = None,
    reason: str | None = None,
    source_type: str = "workflow",
    source_id: str | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    change_json = json.dumps(proposed_change, default=str)
    with _lock:
        cur = conn.execute(
            "INSERT INTO pending_approvals (entity_type, entity_id, ticker, proposed_change, reason, source_type, source_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                entity_type,
                entity_id,
                ticker.upper() if ticker else None,
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
        "proposed_change": proposed_change,
        "reason": reason,
        "source_type": source_type,
        "source_id": source_id,
        "status": "pending",
        "created_at": now,
        "resolved_at": None,
        "resolved_note": None,
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
) -> dict:
    proposed_hash = _json_hash(proposed_change)
    normalized_ticker = ticker.upper() if ticker else None
    if source_id:
        for approval in get_pending_approvals(status=None, ticker=normalized_ticker):
            if approval.get("entity_type") != entity_type:
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
    )


def get_pending_approvals(
    status: str | None = "pending",
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
            f"SELECT * FROM pending_approvals{where} ORDER BY created_at DESC",
            params,
        ).fetchall()
    results = _rows_to_list(rows)
    for d in results:
        _parse_json_field(d, "proposed_change")
    return results


def get_pending_approval(approval_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
    if not row:
        return None
    d = _require_row_dict(row)
    _parse_json_field(d, "proposed_change")
    return d


def resolve_approval(approval_id: int, status: str, resolved_note: str | None = None) -> dict:
    """Resolve a pending approval (approve or reject).

    When approved, applies the side effect based on entity_type.
    """
    if status not in ("approved", "rejected"):
        raise ValueError(f"Resolution status must be 'approved' or 'rejected', got '{status}'")

    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
        if not row:
            raise ValueError(f"No pending approval with id {approval_id}")
        current = _require_row_dict(row)
        if current["status"] != "pending":
            raise ValueError(f"Approval {approval_id} is already {current['status']}")

        conn.execute(
            "UPDATE pending_approvals SET status = ?, resolved_at = ?, resolved_note = ? WHERE id = ?",
            (status, now, resolved_note, approval_id),
        )
        conn.commit()

    _parse_json_field(current, "proposed_change")
    change = current.get("proposed_change")
    if isinstance(change, dict) and change.get("recommendation_id") is not None:
        try:
            update_recommendation_approval(int(change["recommendation_id"]), approval_id, status)
        except Exception:
            logger.warning("Failed to update recommendation approval status", exc_info=True)

    # Apply side effect if approved
    if status == "approved":
        _apply_approval_side_effect(current)

    with _lock:
        updated = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
    d = _require_row_dict(updated)
    _parse_json_field(d, "proposed_change")
    return d


def _apply_approval_side_effect(approval: dict) -> None:
    """Apply the side effect of an approved change."""
    entity_type = approval["entity_type"]
    change = approval.get("proposed_change", {})
    if not isinstance(change, dict):
        try:
            change = json.loads(change)
        except Exception:
            return

    if entity_type == "thesis_status":
        from portfolio.thesis_db import update_thesis_status

        update_thesis_status(
            change.get("ticker", approval.get("ticker", "")),
            change.get("new_status", ""),
            change.get("reason", ""),
        )

    elif entity_type == "evaluation":
        from portfolio.thesis_db import save_evaluations

        evaluated_at = change.get("evaluated_at", _now())
        evaluations = change.get("evaluations", [change])
        save_evaluations(evaluated_at, evaluations)

    elif entity_type == "catalyst_status":
        catalyst_id = change.get("catalyst_id") or approval.get("entity_id")
        if catalyst_id:
            update_catalyst_status(
                int(catalyst_id),
                change.get("status", "played_out"),
                change.get("evidence"),
            )

    elif entity_type == "kill_condition_status":
        kc_id = change.get("kill_condition_id") or approval.get("entity_id")
        if kc_id:
            update_kill_condition_status(int(kc_id), change.get("status", "triggered"))

    elif entity_type == "action_item":
        create_action_item(
            description=change.get("description", ""),
            action_type=change.get("action_type", "review"),
            ticker=change.get("ticker", approval.get("ticker")),
            urgency=change.get("urgency", "normal"),
            source_type=approval.get("source_type", "workflow"),
            source_id=approval.get("source_id"),
        )

    elif entity_type == "watch_trigger":
        create_watch_trigger(
            condition=change.get("condition", ""),
            trigger_type=change.get("trigger_type", "custom"),
            ticker=change.get("ticker", approval.get("ticker")),
            source_type=approval.get("source_type", "workflow"),
            source_id=approval.get("source_id"),
            expires_at=change.get("expires_at"),
        )

    elif entity_type == "portfolio_positions":
        from api.routers.portfolio_edit import PortfolioUpdateRequest, update_portfolio_positions

        update_portfolio_positions(PortfolioUpdateRequest(positions=change.get("positions") or []))

    elif entity_type == "hedge_positions":
        from api.routers.portfolio_edit import HedgeUpdateRequest, update_hedge_positions

        update_hedge_positions(HedgeUpdateRequest(positions=change.get("positions") or []))

    elif entity_type == "thesis_content":
        from api.routers.thesis import SaveThesisRequest, save_thesis

        save_thesis(
            str(change.get("ticker") or approval.get("ticker") or ""),
            SaveThesisRequest(content=change.get("content", "")),
        )

    elif entity_type == "catalyst":
        result = create_catalyst(
            ticker=change.get("ticker", approval.get("ticker", "")),
            description=change.get("description", ""),
            category=change.get("category", "fundamental"),
            target_date=change.get("target_date"),
            created_by="agent",
        )
        try:
            from portfolio.thesis_sync import sync_markdown_from_entities

            sync_markdown_from_entities(result["ticker"])
        except Exception:
            pass

    elif entity_type == "kill_condition":
        result = create_kill_condition(
            ticker=change.get("ticker", approval.get("ticker", "")),
            condition=change.get("condition", ""),
            metric=change.get("metric"),
            threshold=change.get("threshold"),
            created_by="agent",
        )
        try:
            from portfolio.thesis_sync import sync_markdown_from_entities

            sync_markdown_from_entities(result["ticker"])
        except Exception:
            pass

    elif entity_type == "research_note":
        create_research_note(
            title=change.get("title", ""),
            content=change.get("content", ""),
            ticker=change.get("ticker", approval.get("ticker")),
            note_type=change.get("note_type", "general"),
            source_type=approval.get("source_type", "workflow"),
            source_id=approval.get("source_id"),
        )

    elif entity_type == "news_digest_delete":
        from api.routers.portfolio_news import delete_portfolio_news_digest

        delete_portfolio_news_digest(str(change.get("digest_id") or ""))
