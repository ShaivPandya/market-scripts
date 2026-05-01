"""
thesis_db.py -- SQLite-backed thesis metadata, status history, and evaluation store.

Tracks thesis lifecycle (active/under_review/invalidated) with full status
change history, and persists weekly monitoring evaluations from auto_report.

Public API:
  upsert_thesis_meta(ticker, status)  -> dict
  get_thesis_meta(ticker)             -> dict | None
  get_all_thesis_meta()               -> list[dict]
  update_thesis_status(ticker, new_status, reason) -> dict
  get_status_history(ticker)          -> list[dict]
  save_evaluations(evaluated_at, evaluations) -> int
  get_evaluations(ticker, limit)      -> list[dict]
  get_latest_evaluations()            -> list[dict]
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime, timezone
from pathlib import Path

from api.postgres import connect, use_postgres_state
from api.postgres_compat import PostgresCompatConnection

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "thesis.db"

_CREATE_THESIS_META = """
CREATE TABLE IF NOT EXISTS thesis_meta (
    ticker       TEXT PRIMARY KEY NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active'
                 CHECK (status IN ('active', 'under_review', 'invalidated')),
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
)
"""

_CREATE_STATUS_HISTORY = """
CREATE TABLE IF NOT EXISTS thesis_status_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    old_status   TEXT,
    new_status   TEXT NOT NULL,
    reason       TEXT,
    changed_at   TEXT NOT NULL,
    FOREIGN KEY (ticker) REFERENCES thesis_meta(ticker)
)
"""

_CREATE_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS thesis_evaluations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT NOT NULL,
    evaluated_at     TEXT NOT NULL,
    thesis_status    TEXT NOT NULL,
    technical_read   TEXT NOT NULL,
    fundamental_read TEXT NOT NULL,
    action           TEXT NOT NULL,
    confidence       TEXT NOT NULL,
    key_developments TEXT NOT NULL,
    earnings_note    TEXT,
    risk_flag        TEXT,
    FOREIGN KEY (ticker) REFERENCES thesis_meta(ticker)
)
"""

_CREATE_EVAL_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_ticker_date
ON thesis_evaluations(ticker, evaluated_at)
"""

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
                    _conn = PostgresCompatConnection(identity_tables={"thesis_status_history", "thesis_evaluations"})
                else:
                    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                    _conn.execute("PRAGMA journal_mode=WAL")
                    _conn.row_factory = sqlite3.Row
                    _conn.execute("PRAGMA foreign_keys = ON")
                    _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_THESIS_META)
    conn.execute(_CREATE_STATUS_HISTORY)
    conn.execute(_CREATE_EVALUATIONS)
    conn.execute(_CREATE_EVAL_INDEX)
    conn.commit()
    # Backfill from existing markdown files on first run
    count = conn.execute("SELECT COUNT(*) FROM thesis_meta").fetchone()[0]
    if count == 0:
        _backfill_from_markdown(conn)


def _backfill_from_markdown(conn: sqlite3.Connection) -> None:
    from paths import PROJECT_ROOT

    theses_dir = PROJECT_ROOT / "investment_theses"
    if not theses_dir.exists():
        return
    now = datetime.now(UTC).isoformat()
    inserted = 0
    for md_file in sorted(theses_dir.glob("*.md")):
        ticker = md_file.stem.upper()
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO thesis_meta (ticker, status, created_at, updated_at) VALUES (?,?,?,?)",
            (ticker, "active", now, now),
        )
        conn.execute(
            "INSERT INTO thesis_status_history (ticker, old_status, new_status, reason, changed_at) VALUES (?,?,?,?,?)",
            (ticker, None, "active", "Backfilled from existing markdown file", now),
        )
        inserted += 1
    conn.commit()
    if inserted:
        logger.info("thesis_db: backfilled %d theses from markdown files", inserted)


# ---------------------------------------------------------------------------
# thesis_meta CRUD
# ---------------------------------------------------------------------------


def upsert_thesis_meta(ticker: str, status: str = "active") -> dict:
    """Create thesis_meta row if missing, or update updated_at if it exists."""
    if use_postgres_state():
        return _pg_upsert_thesis_meta(ticker, status=status)

    conn = _get_conn()
    now = datetime.now(UTC).isoformat()
    with _lock:
        existing = conn.execute(
            "SELECT ticker, status, created_at, updated_at FROM thesis_meta WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE thesis_meta SET updated_at = ? WHERE ticker = ?",
                (now, ticker),
            )
            conn.commit()
            return {
                "ticker": ticker,
                "status": existing["status"],
                "created_at": existing["created_at"],
                "updated_at": now,
            }
        conn.execute(
            "INSERT INTO thesis_meta (ticker, status, created_at, updated_at) VALUES (?,?,?,?)",
            (ticker, status, now, now),
        )
        conn.execute(
            "INSERT INTO thesis_status_history (ticker, old_status, new_status, reason, changed_at) VALUES (?,?,?,?,?)",
            (ticker, None, status, "Thesis created", now),
        )
        conn.commit()
    return {"ticker": ticker, "status": status, "created_at": now, "updated_at": now}


def get_thesis_meta(ticker: str) -> dict | None:
    if use_postgres_state():
        with connect() as conn:
            row = conn.execute(
                "SELECT ticker, status, created_at, updated_at FROM thesis_meta WHERE ticker = %s",
                (ticker,),
            ).fetchone()
        return dict(row) if row else None

    conn = _get_conn()
    with _lock:
        row = conn.execute(
            "SELECT ticker, status, created_at, updated_at FROM thesis_meta WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    return dict(row) if row else None


def get_all_thesis_meta() -> list[dict]:
    if use_postgres_state():
        with connect() as conn:
            rows = conn.execute(
                "SELECT ticker, status, created_at, updated_at FROM thesis_meta ORDER BY ticker"
            ).fetchall()
        return [dict(r) for r in rows]

    conn = _get_conn()
    with _lock:
        rows = conn.execute("SELECT ticker, status, created_at, updated_at FROM thesis_meta ORDER BY ticker").fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Status history
# ---------------------------------------------------------------------------


def update_thesis_status(ticker: str, new_status: str, reason: str = "") -> dict:
    """Change thesis status and append a history row. Returns updated meta."""
    if use_postgres_state():
        return _pg_update_thesis_status(ticker, new_status, reason=reason)

    conn = _get_conn()
    now = datetime.now(UTC).isoformat()
    with _lock:
        current = conn.execute("SELECT status FROM thesis_meta WHERE ticker = ?", (ticker,)).fetchone()
        if not current:
            raise ValueError(f"No thesis_meta row for ticker '{ticker}'")
        old_status = current["status"]
        conn.execute(
            "UPDATE thesis_meta SET status = ?, updated_at = ? WHERE ticker = ?",
            (new_status, now, ticker),
        )
        conn.execute(
            "INSERT INTO thesis_status_history (ticker, old_status, new_status, reason, changed_at) VALUES (?,?,?,?,?)",
            (ticker, old_status, new_status, reason or None, now),
        )
        conn.commit()
    return {"ticker": ticker, "old_status": old_status, "new_status": new_status, "updated_at": now}


def get_status_history(ticker: str) -> list[dict]:
    if use_postgres_state():
        with connect() as conn:
            rows = conn.execute(
                "SELECT id, ticker, old_status, new_status, reason, changed_at "
                "FROM thesis_status_history WHERE ticker = %s ORDER BY changed_at DESC",
                (ticker,),
            ).fetchall()
        return [dict(r) for r in rows]

    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT id, ticker, old_status, new_status, reason, changed_at "
            "FROM thesis_status_history WHERE ticker = ? ORDER BY changed_at DESC",
            (ticker,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------


def save_evaluations(evaluated_at: str, evaluations: list[dict]) -> int:
    """Bulk insert weekly monitoring evaluations. Returns count of rows saved."""
    if use_postgres_state():
        return _pg_save_evaluations(evaluated_at, evaluations)

    conn = _get_conn()
    inserted = 0
    with _lock:
        for ev in evaluations:
            ticker = ev.get("ticker")
            if not ticker:
                continue
            key_devs = ev.get("key_developments", [])
            if isinstance(key_devs, list):
                key_devs = json.dumps(key_devs)
            conn.execute(
                "INSERT OR REPLACE INTO thesis_evaluations "
                "(ticker, evaluated_at, thesis_status, technical_read, fundamental_read, "
                "action, confidence, key_developments, earnings_note, risk_flag) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    ticker,
                    evaluated_at,
                    ev.get("thesis_status", ""),
                    ev.get("technical_read", ""),
                    ev.get("fundamental_read", ""),
                    ev.get("action", ""),
                    str(ev.get("confidence", "")),
                    key_devs,
                    ev.get("earnings_note"),
                    str(ev.get("risk_flag", "")) if ev.get("risk_flag") is not None else None,
                ),
            )
            inserted += 1
        conn.commit()
    return inserted


def get_evaluations(ticker: str, limit: int = 20) -> list[dict]:
    if use_postgres_state():
        with connect() as conn:
            rows = conn.execute(
                "SELECT id, ticker, evaluated_at, thesis_status, technical_read, fundamental_read, "
                "action, confidence, key_developments, earnings_note, risk_flag "
                "FROM thesis_evaluations WHERE ticker = %s ORDER BY evaluated_at DESC LIMIT %s",
                (ticker, limit),
            ).fetchall()
        return _parse_evaluation_rows(rows)

    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT id, ticker, evaluated_at, thesis_status, technical_read, fundamental_read, "
            "action, confidence, key_developments, earnings_note, risk_flag "
            "FROM thesis_evaluations WHERE ticker = ? ORDER BY evaluated_at DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["key_developments"] = json.loads(d["key_developments"])
        except (json.JSONDecodeError, TypeError):
            d["key_developments"] = []
        results.append(d)
    return results


def get_latest_evaluations() -> list[dict]:
    """Return the most recent evaluation for each ticker."""
    if use_postgres_state():
        with connect() as conn:
            rows = conn.execute(
                "SELECT e.id, e.ticker, e.evaluated_at, e.thesis_status, e.technical_read, "
                "e.fundamental_read, e.action, e.confidence, e.key_developments, "
                "e.earnings_note, e.risk_flag "
                "FROM thesis_evaluations e "
                "INNER JOIN (SELECT ticker, MAX(evaluated_at) AS max_date FROM thesis_evaluations GROUP BY ticker) latest "
                "ON e.ticker = latest.ticker AND e.evaluated_at = latest.max_date "
                "ORDER BY e.ticker"
            ).fetchall()
        return _parse_evaluation_rows(rows)

    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT e.id, e.ticker, e.evaluated_at, e.thesis_status, e.technical_read, "
            "e.fundamental_read, e.action, e.confidence, e.key_developments, "
            "e.earnings_note, e.risk_flag "
            "FROM thesis_evaluations e "
            "INNER JOIN (SELECT ticker, MAX(evaluated_at) AS max_date FROM thesis_evaluations GROUP BY ticker) latest "
            "ON e.ticker = latest.ticker AND e.evaluated_at = latest.max_date "
            "ORDER BY e.ticker"
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["key_developments"] = json.loads(d["key_developments"])
        except (json.JSONDecodeError, TypeError):
            d["key_developments"] = []
        results.append(d)
    return results


def _pg_upsert_thesis_meta(ticker: str, status: str = "active") -> dict:
    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        existing = conn.execute(
            "SELECT ticker, status, created_at, updated_at FROM thesis_meta WHERE ticker = %s",
            (ticker,),
        ).fetchone()
        if existing:
            conn.execute("UPDATE thesis_meta SET updated_at = %s WHERE ticker = %s", (now, ticker))
            conn.commit()
            return {
                "ticker": ticker,
                "status": existing["status"],
                "created_at": existing["created_at"],
                "updated_at": now,
            }
        conn.execute(
            "INSERT INTO thesis_meta (ticker, status, created_at, updated_at) VALUES (%s, %s, %s, %s)",
            (ticker, status, now, now),
        )
        conn.execute(
            "INSERT INTO thesis_status_history (ticker, old_status, new_status, reason, changed_at) VALUES (%s, %s, %s, %s, %s)",
            (ticker, None, status, "Thesis created", now),
        )
        conn.commit()
    return {"ticker": ticker, "status": status, "created_at": now, "updated_at": now}


def _pg_update_thesis_status(ticker: str, new_status: str, reason: str = "") -> dict:
    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        current = conn.execute("SELECT status FROM thesis_meta WHERE ticker = %s", (ticker,)).fetchone()
        if not current:
            raise ValueError(f"No thesis_meta row for ticker '{ticker}'")
        old_status = current["status"]
        conn.execute(
            "UPDATE thesis_meta SET status = %s, updated_at = %s WHERE ticker = %s",
            (new_status, now, ticker),
        )
        conn.execute(
            "INSERT INTO thesis_status_history (ticker, old_status, new_status, reason, changed_at) VALUES (%s, %s, %s, %s, %s)",
            (ticker, old_status, new_status, reason or None, now),
        )
        conn.commit()
    return {"ticker": ticker, "old_status": old_status, "new_status": new_status, "updated_at": now}


def _pg_save_evaluations(evaluated_at: str, evaluations: list[dict]) -> int:
    inserted = 0
    rows = []
    for ev in evaluations:
        ticker = ev.get("ticker")
        if not ticker:
            continue
        key_devs = ev.get("key_developments", [])
        if isinstance(key_devs, list):
            key_devs = json.dumps(key_devs)
        rows.append(
            (
                ticker,
                evaluated_at,
                ev.get("thesis_status", ""),
                ev.get("technical_read", ""),
                ev.get("fundamental_read", ""),
                ev.get("action", ""),
                str(ev.get("confidence", "")),
                key_devs,
                ev.get("earnings_note"),
                str(ev.get("risk_flag", "")) if ev.get("risk_flag") is not None else None,
            )
        )
        inserted += 1
    if not rows:
        return 0
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO thesis_evaluations
                    (ticker, evaluated_at, thesis_status, technical_read, fundamental_read,
                     action, confidence, key_developments, earnings_note, risk_flag)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, evaluated_at) DO UPDATE SET
                    thesis_status = EXCLUDED.thesis_status,
                    technical_read = EXCLUDED.technical_read,
                    fundamental_read = EXCLUDED.fundamental_read,
                    action = EXCLUDED.action,
                    confidence = EXCLUDED.confidence,
                    key_developments = EXCLUDED.key_developments,
                    earnings_note = EXCLUDED.earnings_note,
                    risk_flag = EXCLUDED.risk_flag
                """,
                rows,
            )
        conn.commit()
    return inserted


def _parse_evaluation_rows(rows) -> list[dict]:
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["key_developments"] = json.loads(d["key_developments"])
        except (json.JSONDecodeError, TypeError):
            d["key_developments"] = []
        results.append(d)
    return results
