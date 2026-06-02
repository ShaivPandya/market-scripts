"""Tests for auth user/session persistence."""

import bcrypt

import api.auth_store as auth_store
from api.auth_store import (
    create_session,
    ensure_auth_users_seeded,
    get_user_by_username,
    lookup_session,
    reset_auth_store_for_tests,
    revoke_session,
    verify_csrf,
    verify_password,
)


class _Cursor:
    def __init__(self, rows=None):
        self._rows = list(rows or [])

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _RecordingConn:
    def __init__(self, *, existing_user=False):
        self.calls = []
        self.existing_user = existing_user

    def execute(self, sql, params=None):
        params = tuple(params or ())
        self.calls.append((sql, params))
        if sql == "SELECT 1":
            return _Cursor([{"?column?": 1}])
        if "SELECT id, username, email FROM auth_users" in sql and self.existing_user:
            return _Cursor([{"id": "user-1", "username": "admin", "email": None}])
        if "SELECT * FROM auth_users WHERE id" in sql:
            return _Cursor([{"id": "user-1", "username": "admin", "email": None}])
        if "SELECT role FROM auth_user_roles" in sql:
            return _Cursor([{"role": "admin"}])
        return _Cursor()

    def commit(self):
        pass

    def close(self):
        pass


def test_seed_admin_user_from_env(monkeypatch):
    reset_auth_store_for_tests()
    monkeypatch.setenv("AUTH_PASSWORD_HASH", bcrypt.hashpw(b"secret", bcrypt.gensalt(12)).decode())
    monkeypatch.setenv("AUTH_DEFAULT_USERNAME", "admin")
    ensure_auth_users_seeded()
    user = get_user_by_username("admin")
    assert user is not None
    assert verify_password(user, "secret")
    assert "admin" in user.roles


def test_opaque_session_lifecycle(monkeypatch):
    reset_auth_store_for_tests()
    monkeypatch.setenv("AUTH_PASSWORD_HASH", bcrypt.hashpw(b"secret", bcrypt.gensalt(12)).decode())
    ensure_auth_users_seeded()
    user = get_user_by_username("admin")
    assert user is not None
    session = create_session(user)
    loaded = lookup_session(session.session_token)
    assert loaded is not None
    assert loaded.user.username == "admin"
    assert verify_csrf(session.session_token, session.csrf_token)
    revoke_session(session.session_token)
    assert lookup_session(session.session_token) is None


def test_auth_user_active_uses_boolean_parameters_for_postgres():
    for existing_user in (False, True):
        conn = _RecordingConn(existing_user=existing_user)
        auth_store._upsert_user(
            conn,
            username="admin",
            password_hash="hash",
            email=None,
            roles=("admin",),
        )

        active_user_writes = [
            params
            for sql, params in conn.calls
            if "auth_users" in sql and ("INSERT INTO" in sql or "UPDATE auth_users" in sql)
        ]
        assert active_user_writes
        assert any(any(param is True for param in params) for params in active_user_writes)
        assert all(not any(type(param) is int and param == 1 for param in params) for params in active_user_writes)
        assert all("active = 1" not in sql for sql, _params in conn.calls)


def test_auth_user_active_read_filters_use_boolean_parameters(monkeypatch):
    reset_auth_store_for_tests()
    conn = _RecordingConn()
    monkeypatch.setattr(auth_store, "_conn", conn)
    monkeypatch.setattr(auth_store, "_seeded", True)

    get_user_by_username("admin")
    auth_store.get_user_by_email("admin@example.com")
    lookup_session("session-token")

    active_reads = [params for sql, params in conn.calls if "active = ?" in sql or "u.active = ?" in sql]
    assert len(active_reads) == 3
    assert all(any(param is True for param in params) for params in active_reads)
    assert all("active = 1" not in sql for sql, _params in conn.calls)
