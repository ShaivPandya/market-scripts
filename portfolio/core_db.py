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
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

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
    expires_at   TEXT
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

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_catalysts_ticker ON catalysts(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_kill_conditions_ticker ON kill_conditions(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_name ON workflow_runs(workflow_name)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_ticker ON workflow_runs(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_started ON workflow_runs(started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_action_items_status ON action_items(status)",
    "CREATE INDEX IF NOT EXISTS idx_action_items_ticker ON action_items(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_watch_triggers_status ON watch_triggers(status)",
    "CREATE INDEX IF NOT EXISTS idx_watch_triggers_ticker ON watch_triggers(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_research_notes_ticker ON research_notes(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_status ON pending_approvals(status)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_ticker ON pending_approvals(ticker)",
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
                            "action_items",
                            "watch_triggers",
                            "research_notes",
                            "pending_approvals",
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
        _CREATE_ACTION_ITEMS,
        _CREATE_WATCH_TRIGGERS,
        _CREATE_RESEARCH_NOTES,
        _CREATE_PENDING_APPROVALS,
    ]:
        conn.execute(stmt)
    for idx in _INDEXES:
        conn.execute(idx)
    conn.commit()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row else None


def _rows_to_list(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


def _parse_json_field(d: dict, field: str) -> dict:
    """Parse a JSON string field in a dict, returning the dict with the field parsed."""
    val = d.get(field)
    if isinstance(val, str):
        try:
            d[field] = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return d


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
    return dict(updated)


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
        triggered_at = now if status == "triggered" else dict(row).get("triggered_at")
        conn.execute(
            "UPDATE kill_conditions SET status = ?, triggered_at = ?, updated_at = ? WHERE id = ?",
            (status, triggered_at, now, kc_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM kill_conditions WHERE id = ?", (kc_id,)).fetchone()
    return dict(updated)


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
    d = dict(row)
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
    return dict(row)


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
    d = dict(row)
    _parse_json_field(d, "artifacts")
    _parse_json_field(d, "tool_sections")
    return d


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
    return dict(updated)


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
    return dict(updated)


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
) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        cur = conn.execute(
            "INSERT INTO watch_triggers (ticker, trigger_type, condition, source_type, source_id, created_at, expires_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (ticker.upper() if ticker else None, trigger_type, condition, source_type, source_id, now, expires_at),
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
    }


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
    return _rows_to_list(rows)


def fire_watch_trigger(trigger_id: int) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
        if not row:
            raise ValueError(f"No watch trigger with id {trigger_id}")
        conn.execute(
            "UPDATE watch_triggers SET status = 'fired', fired_at = ? WHERE id = ?",
            (now, trigger_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
    return dict(updated)


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
    return dict(updated)


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
    d = dict(row)
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
        current = dict(row)
        if current["status"] != "pending":
            raise ValueError(f"Approval {approval_id} is already {current['status']}")

        conn.execute(
            "UPDATE pending_approvals SET status = ?, resolved_at = ?, resolved_note = ? WHERE id = ?",
            (status, now, resolved_note, approval_id),
        )
        conn.commit()

    # Apply side effect if approved
    if status == "approved":
        _parse_json_field(current, "proposed_change")
        _apply_approval_side_effect(current)

    with _lock:
        updated = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
    d = dict(updated)
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
