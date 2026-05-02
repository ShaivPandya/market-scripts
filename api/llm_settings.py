"""Persistent live-app LLM settings."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.postgres import use_postgres_state
from api.postgres_compat import PostgresCompatConnection

DB_PATH = Path(__file__).parent / "app_settings.db"

LLM_PROVIDER_KEY = "llm.provider"
ALLOWED_LLM_PROVIDERS = {"anthropic", "openai"}

_lock = threading.Lock()
_conn: sqlite3.Connection | PostgresCompatConnection | None = None

_CREATE_APP_SETTINGS = """
CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


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
                    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                    _conn.row_factory = sqlite3.Row
                    _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection | PostgresCompatConnection) -> None:
    conn.execute(_CREATE_APP_SETTINGS)
    conn.commit()


def _row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return {key: row[key] for key in row.keys()}


def get_setting(key: str) -> dict[str, Any] | None:
    if not use_postgres_state() and not DB_PATH.exists():
        return None
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT key, value, updated_at FROM app_settings WHERE key = ?", (key,)).fetchone()
    return _row_to_dict(row) if row else None


def set_setting(key: str, value: str) -> dict[str, Any]:
    conn = _get_conn()
    updated_at = _now()
    with _lock:
        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value, updated_at),
        )
        conn.commit()
        row = conn.execute("SELECT key, value, updated_at FROM app_settings WHERE key = ?", (key,)).fetchone()
    return _row_to_dict(row)


def get_llm_provider_setting() -> str | None:
    row = get_setting(LLM_PROVIDER_KEY)
    if not row:
        return None
    provider = str(row.get("value") or "").strip().lower()
    return provider if provider in ALLOWED_LLM_PROVIDERS else None


def set_llm_provider_setting(provider: str) -> dict[str, Any]:
    normalized = (provider or "").strip().lower()
    if normalized not in ALLOWED_LLM_PROVIDERS:
        raise ValueError("LLM provider must be 'anthropic' or 'openai'")
    return set_setting(LLM_PROVIDER_KEY, normalized)
