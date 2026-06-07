from __future__ import annotations

import pytest


def test_use_postgres_state_rejects_sqlite_without_explicit_test_override(monkeypatch):
    from api.postgres import use_postgres_state

    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.delenv("TALISMAN_ALLOW_SQLITE_STATE", raising=False)

    with pytest.raises(RuntimeError, match="Postgres-only"):
        use_postgres_state()


def test_use_postgres_state_allows_sqlite_with_explicit_test_override(monkeypatch):
    from api.postgres import use_postgres_state

    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setenv("TALISMAN_ALLOW_SQLITE_STATE", "true")

    assert use_postgres_state() is False


def test_new_pool_checks_connections_before_reuse(monkeypatch):
    import sys
    import types

    from api import postgres

    calls: dict[str, object] = {}

    class _ConnectionPool:
        @staticmethod
        def check_connection(_conn):
            return None

        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setattr(postgres, "require_database_url", lambda _env_var="DATABASE_URL": "postgresql://db")
    monkeypatch.setitem(sys.modules, "psycopg.rows", types.SimpleNamespace(dict_row=object()))
    monkeypatch.setitem(sys.modules, "psycopg_pool", types.SimpleNamespace(ConnectionPool=_ConnectionPool))

    postgres._new_pool()

    assert calls["check"] is _ConnectionPool.check_connection
