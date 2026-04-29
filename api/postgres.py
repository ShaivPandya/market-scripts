"""Postgres connection helpers for Cloud SQL."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


def database_url(env_var: str = "DATABASE_URL") -> str | None:
    return (os.getenv(env_var) or "").strip() or None


def psycopg_database_url(env_var: str = "DATABASE_URL") -> str | None:
    url = database_url(env_var)
    if not url:
        return None
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def require_database_url(env_var: str = "DATABASE_URL") -> str:
    url = psycopg_database_url(env_var)
    if not url:
        raise RuntimeError(f"{env_var} is required for Postgres-backed state.")
    return url


@contextmanager
def connect(env_var: str = "DATABASE_URL") -> Iterator:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("psycopg is required for Postgres-backed state.") from exc

    conn = psycopg.connect(require_database_url(env_var), row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()
