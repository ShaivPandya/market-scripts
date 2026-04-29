"""
Persistent conversation memory for the AI agent.

Stores conversation sessions and summaries in SQLite so the agent can
reference past research across sessions.  Follows the same connection
pattern as portfolio/thesis_db.py (WAL mode, thread-safe, lazy init).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.postgres import use_postgres_state
from api.postgres_compat import PostgresCompatConnection

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = _REPO_ROOT / "data_cache" / "memory" / "memory.db"

_lock = threading.Lock()
_conn: sqlite3.Connection | PostgresCompatConnection | None = None

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id   TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    message_count INTEGER DEFAULT 0,
    key_tickers  TEXT,          -- JSON array
    key_topics   TEXT,          -- JSON array
    summary      TEXT,
    transcript   TEXT NOT NULL  -- JSON array of messages
)
"""

_CREATE_SESSIONS_IDX = """
CREATE INDEX IF NOT EXISTS idx_sessions_ended_at
ON conversation_sessions(ended_at DESC)
"""

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


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
                    _conn = PostgresCompatConnection()
                else:
                    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                    _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
                    _conn.execute("PRAGMA journal_mode=WAL")
                    _conn.row_factory = sqlite3.Row
                    _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_SESSIONS)
    conn.execute(_CREATE_SESSIONS_IDX)
    # Migrate: add columns for server-managed rolling memory
    for col, typedef in [
        ("rolling_summary", "TEXT"),
        ("server_messages", "TEXT NOT NULL DEFAULT '[]'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE conversation_sessions ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_session(
    messages: list[dict[str, Any]],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Save a conversation transcript. Returns the session record."""
    conn = _get_conn()
    sid = session_id or str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    timestamps: list[float] = [m["timestamp"] for m in messages if m.get("timestamp") is not None]
    started_at = datetime.fromtimestamp(min(timestamps) / 1000, tz=UTC).isoformat() if timestamps else now

    transcript_json = json.dumps(messages, default=str)

    with _lock:
        conn.execute(
            """
            INSERT INTO conversation_sessions
                (session_id, started_at, ended_at, message_count, transcript)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                ended_at = excluded.ended_at,
                message_count = excluded.message_count,
                transcript = excluded.transcript
            """,
            (sid, started_at, now, len(messages), transcript_json),
        )
        conn.commit()

    return {
        "session_id": sid,
        "started_at": started_at,
        "ended_at": now,
        "message_count": len(messages),
    }


def update_summary(
    session_id: str,
    summary: str,
    key_tickers: list[str] | None = None,
    key_topics: list[str] | None = None,
) -> bool:
    """Attach a summary to a session. Returns True if session exists."""
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            """
            UPDATE conversation_sessions
            SET summary = ?,
                key_tickers = ?,
                key_topics = ?
            WHERE session_id = ?
            """,
            (
                summary,
                json.dumps(key_tickers) if key_tickers else None,
                json.dumps(key_topics) if key_topics else None,
                session_id,
            ),
        )
        conn.commit()
        return cur.rowcount > 0


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """List recent sessions (without full transcripts)."""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            """
            SELECT session_id, started_at, ended_at, message_count,
                   key_tickers, key_topics, summary
            FROM conversation_sessions
            ORDER BY ended_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_session(session_id: str) -> dict[str, Any] | None:
    """Load a full session including transcript."""
    conn = _get_conn()
    with _lock:
        row = conn.execute(
            """
            SELECT session_id, started_at, ended_at, message_count,
                   key_tickers, key_topics, summary, transcript
            FROM conversation_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    result = _row_to_dict(row)
    try:
        result["transcript"] = json.loads(row["transcript"])
    except Exception:
        result["transcript"] = []
    return result


def delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if it existed."""
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "DELETE FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
        return cur.rowcount > 0


def get_recent_summaries(
    limit: int = 5,
    tickers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent session summaries for agent context injection.

    If *tickers* is provided, prefer sessions mentioning those tickers,
    but still fall back to the most recent if none match.
    """
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            """
            SELECT session_id, started_at, ended_at, message_count,
                   key_tickers, key_topics, summary
            FROM conversation_sessions
            WHERE summary IS NOT NULL AND summary != ''
            ORDER BY ended_at DESC
            LIMIT ?
            """,
            (limit * 3,),
        ).fetchall()

    results = [_row_to_dict(r) for r in rows]

    if tickers:
        upper = {t.upper() for t in tickers}
        relevant = [r for r in results if _tickers_overlap(r.get("key_tickers"), upper)]
        if relevant:
            return relevant[:limit]

    return results[:limit]


# ---------------------------------------------------------------------------
# Server-managed session state (rolling memory)
# ---------------------------------------------------------------------------


def get_or_create_session(session_id: str | None = None) -> dict[str, Any]:
    """Load an existing session or create a new one.

    Returns a dict with parsed ``server_messages`` (list) and
    ``rolling_summary`` (str | None).
    """
    conn = _get_conn()
    sid = session_id or str(uuid.uuid4())

    if session_id:
        with _lock:
            row = conn.execute(
                """
                SELECT session_id, started_at, ended_at, message_count,
                       rolling_summary, server_messages
                FROM conversation_sessions
                WHERE session_id = ?
                """,
                (sid,),
            ).fetchone()
        if row is not None:
            try:
                msgs = json.loads(row["server_messages"]) if row["server_messages"] else []
            except Exception:
                msgs = []
            return {
                "session_id": row["session_id"],
                "started_at": row["started_at"],
                "rolling_summary": row["rolling_summary"],
                "server_messages": msgs,
                "message_count": row["message_count"] or 0,
            }

    # Create a new session
    now = datetime.now(UTC).isoformat()
    with _lock:
        conn.execute(
            """
            INSERT OR IGNORE INTO conversation_sessions
                (session_id, started_at, ended_at, message_count, transcript, server_messages)
            VALUES (?, ?, ?, 0, '[]', '[]')
            """,
            (sid, now, now),
        )
        conn.commit()
    return {
        "session_id": sid,
        "started_at": now,
        "rolling_summary": None,
        "server_messages": [],
        "message_count": 0,
    }


def append_messages(session_id: str, messages: list[dict[str, Any]]) -> int:
    """Append messages to a session's server_messages. Returns new total count."""
    conn = _get_conn()
    now = datetime.now(UTC).isoformat()
    with _lock:
        row = conn.execute(
            "SELECT server_messages FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Session {session_id} not found")
        try:
            existing = json.loads(row["server_messages"]) if row["server_messages"] else []
        except Exception:
            existing = []
        existing.extend(messages)
        conn.execute(
            """
            UPDATE conversation_sessions
            SET server_messages = ?, message_count = ?, ended_at = ?
            WHERE session_id = ?
            """,
            (json.dumps(existing, default=str), len(existing), now, session_id),
        )
        conn.commit()
    return len(existing)


def update_rolling_summary(session_id: str, summary: str) -> bool:
    """Update the rolling summary for a session. Returns True if session exists."""
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "UPDATE conversation_sessions SET rolling_summary = ? WHERE session_id = ?",
            (summary, session_id),
        )
        conn.commit()
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("key_tickers", "key_topics"):
        raw = d.get(key)
        if isinstance(raw, str):
            try:
                d[key] = json.loads(raw)
            except Exception:
                d[key] = None
    d.pop("transcript", None)
    return d


def _tickers_overlap(
    stored: list[str] | str | None,
    wanted: set[str],
) -> bool:
    if stored is None:
        return False
    if isinstance(stored, str):
        try:
            stored = json.loads(stored)
        except Exception:
            return False
    if not isinstance(stored, list):
        return False
    return bool(wanted & {str(t).upper() for t in stored})
