"""Postgres connection helpers for Cloud SQL."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_POOLS: dict[tuple[str, bool, str], Any] = {}
_POOLS_LOCK = threading.Lock()


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


def _env_bool(name: str, *, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return default


def _env_int(name: str, *, default: int, minimum: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, *, default: float, minimum: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _pool_enabled() -> bool:
    return _env_bool("POSTGRES_POOL_ENABLED", default=True)


def _pool_min_size() -> int:
    return _env_int("POSTGRES_POOL_MIN_SIZE", default=0, minimum=0)


def _pool_max_size() -> int:
    return _env_int("POSTGRES_POOL_MAX_SIZE", default=4, minimum=1)


def _pool_timeout() -> float:
    return _env_float("POSTGRES_POOL_TIMEOUT_SECONDS", default=5.0, minimum=0.1)


def _new_direct_connection(env_var: str = "DATABASE_URL", *, register_pgvector: bool = False) -> Any:
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


def open_connection(env_var: str = "DATABASE_URL", *, register_pgvector: bool = False) -> Any:
    return _new_direct_connection(env_var, register_pgvector=register_pgvector)


def _new_pool(env_var: str = "DATABASE_URL", *, register_pgvector: bool = False) -> Any:
    try:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ImportError as exc:
        raise RuntimeError("psycopg_pool is required for pooled Postgres state.") from exc

    configure = None
    if register_pgvector:
        try:
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError("pgvector is required for retrieval Postgres state.") from exc

        def configure(conn: Any) -> None:
            register_vector(conn)

    min_size = _pool_min_size()
    max_size = max(_pool_max_size(), min_size)
    return ConnectionPool(
        conninfo=require_database_url(env_var),
        kwargs={"row_factory": dict_row},
        min_size=min_size,
        max_size=max_size,
        timeout=_pool_timeout(),
        configure=configure,
        open=True,
    )


def _get_pool(env_var: str = "DATABASE_URL", *, register_pgvector: bool = False) -> Any:
    conninfo = require_database_url(env_var)
    key = (env_var, register_pgvector, conninfo)
    pool = _POOLS.get(key)
    if pool is not None:
        return pool
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = _new_pool(env_var, register_pgvector=register_pgvector)
            _POOLS[key] = pool
        return pool


def close_pools() -> None:
    """Close process-local Postgres pools. Intended for tests and shutdown hooks."""
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.close()


@contextmanager
def connect(env_var: str = "DATABASE_URL", *, register_pgvector: bool = False) -> Iterator:
    if not _pool_enabled():
        conn = open_connection(env_var, register_pgvector=register_pgvector)
        try:
            yield conn
        finally:
            conn.close()
        return

    pool = _get_pool(env_var, register_pgvector=register_pgvector)
    conn = pool.getconn(timeout=_pool_timeout())
    try:
        yield conn
    finally:
        try:
            conn.rollback()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
        pool.putconn(conn)
