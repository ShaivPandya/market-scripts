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
