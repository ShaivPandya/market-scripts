"""Postgres connection helpers for Cloud SQL."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


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


def use_postgres_state() -> bool:
    backend = (os.getenv("STATE_DB_BACKEND") or "").strip().lower()
    if backend:
        return backend == "postgres"
    return os.getenv("ENVIRONMENT", "development").strip().lower() == "production"


def open_connection(env_var: str = "DATABASE_URL", *, register_pgvector: bool = False) -> Any:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("psycopg is required for Postgres-backed state.") from exc

    conn = psycopg.connect(require_database_url(env_var), row_factory=dict_row)
    if register_pgvector:
        try:
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            conn.close()
            raise RuntimeError("pgvector is required for retrieval Postgres state.") from exc
        register_vector(conn)
    return conn


@contextmanager
def connect(env_var: str = "DATABASE_URL") -> Iterator:
    conn = open_connection(env_var)
    try:
        yield conn
    finally:
        conn.close()
