"""Tests for auth user/session persistence."""

import bcrypt

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
