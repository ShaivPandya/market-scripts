"""Tests for authentication flow — login, logout, token validation."""

import os
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.responses import JSONResponse
from jose import jwt


@pytest.fixture(autouse=True)
def _reset_login_attempt_state():
    from api.routers import auth as auth_router

    auth_router._reset_login_attempt_state()
    yield
    auth_router._reset_login_attempt_state()


def test_login_success(client):
    resp = client.post("/api/auth/login", json={"password": "testpass"})
    assert resp.status_code == 200
    assert resp.json() == {"detail": "ok"}
    assert "__session" in resp.cookies


def test_login_wrong_password(client):
    resp = client.post("/api/auth/login", json={"password": "wrongpassword"})
    assert resp.status_code == 401


def test_login_missing_password(client):
    resp = client.post("/api/auth/login", json={})
    assert resp.status_code == 422  # Pydantic validation error


def test_logout(client):
    # Login first
    client.post("/api/auth/login", json={"password": "testpass"})
    # Logout
    resp = client.post("/api/auth/logout")
    assert resp.status_code == 200
    assert resp.json() == {"detail": "ok"}


def test_me_unauthenticated(client):
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_me_authenticated(auth_client):
    resp = auth_client.get("/api/auth/me")
    assert resp.status_code == 200
    assert resp.json() == {"username": "admin"}


def test_health_no_auth_required(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_public_health_is_diagnostic_safe(client):
    resp = client.get("/api/health")

    assert resp.status_code == 200
    text = resp.text.lower()
    for forbidden in ("fred", "sqlite", "traceback", "/users/", "error:"):
        assert forbidden not in text


def test_admin_health_requires_auth(client, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_detailed_health_response", lambda: JSONResponse({"status": "ok", "checks": {}}))

    unauthenticated = client.get("/api/admin/health")
    assert unauthenticated.status_code == 401

    login = client.post("/api/auth/login", json={"password": "testpass"})
    assert login.status_code == 200
    authenticated = client.get("/api/admin/health")
    assert authenticated.status_code in {200, 503}
    assert "checks" in authenticated.json()


def test_openapi_moved_under_api_prefix_in_development(client):
    assert client.get("/api/openapi.json").status_code == 200
    assert client.get("/openapi.json").status_code == 404


def test_production_docs_and_schema_are_disabled(client, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")

    for path in ("/api/openapi.json", "/api/docs", "/api/redoc"):
        resp = client.get(path)
        assert resp.status_code == 404


def test_global_request_body_limit_rejects_large_json_before_login_parse(client, monkeypatch):
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "64")

    resp = client.post("/api/auth/login", json={"password": "x" * 100})

    assert resp.status_code == 413


def test_login_password_length_is_limited(client):
    resp = client.post("/api/auth/login", json={"password": "x" * 513})

    assert resp.status_code == 422


def test_repeated_bad_logins_lock_out_client(client, monkeypatch):
    from api.routers import auth as auth_router

    now = [1_000.0]
    monkeypatch.setenv("AUTH_LOGIN_FAILURE_LIMIT", "2")
    monkeypatch.setenv("AUTH_LOGIN_FAILURE_WINDOW_SECONDS", "300")
    monkeypatch.setenv("AUTH_LOGIN_LOCKOUT_SECONDS", "60")
    monkeypatch.setattr(auth_router.time, "time", lambda: now[0])

    first = client.post("/api/auth/login", json={"password": "wrong"})
    second = client.post("/api/auth/login", json={"password": "wrong"})
    correct_while_locked = client.post("/api/auth/login", json={"password": "testpass"})

    assert first.status_code == 401
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"
    assert correct_while_locked.status_code == 429

    now[0] += 61
    allowed_after_lockout = client.post("/api/auth/login", json={"password": "testpass"})
    assert allowed_after_lockout.status_code == 200


def test_successful_login_clears_failed_login_counter(client, monkeypatch):
    monkeypatch.setenv("AUTH_LOGIN_FAILURE_LIMIT", "2")
    monkeypatch.setenv("AUTH_LOGIN_FAILURE_WINDOW_SECONDS", "300")
    monkeypatch.setenv("AUTH_LOGIN_LOCKOUT_SECONDS", "60")

    failed = client.post("/api/auth/login", json={"password": "wrong"})
    success = client.post("/api/auth/login", json={"password": "testpass"})
    failed_after_success = client.post("/api/auth/login", json={"password": "wrong"})

    assert failed.status_code == 401
    assert success.status_code == 200
    assert failed_after_success.status_code == 401


def test_password_mode_does_not_require_proxy_secret_for_login(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "password")
    monkeypatch.setenv("API_PROXY_SECRET", "proxy-secret")
    monkeypatch.delenv("REQUIRE_API_PROXY_SECRET", raising=False)

    resp = client.post("/api/auth/login", json={"password": "testpass"})
    assert resp.status_code == 200


def test_explicit_proxy_secret_requirement_blocks_missing_header(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "password")
    monkeypatch.setenv("API_PROXY_SECRET", "proxy-secret")
    monkeypatch.setenv("REQUIRE_API_PROXY_SECRET", "true")

    missing = client.post("/api/auth/login", json={"password": "testpass"})
    assert missing.status_code == 403

    allowed = client.post(
        "/api/auth/login",
        json={"password": "testpass"},
        headers={"X-Api-Proxy-Secret": "proxy-secret"},
    )
    assert allowed.status_code == 200


def test_cloudflare_mode_requires_backend_proxy_secret(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "cloudflare")
    monkeypatch.delenv("API_PROXY_SECRET", raising=False)

    protected = client.get("/api/agent/workflows")
    assert protected.status_code == 403
    assert protected.json() == {"detail": "API proxy secret is required for this auth mode."}

    health = client.get("/api/health")
    assert health.status_code == 200


def test_cloudflare_mode_enforces_proxy_secret_header(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "cloudflare")
    monkeypatch.setenv("API_PROXY_SECRET", "proxy-secret")

    missing = client.get("/api/agent/workflows")
    assert missing.status_code == 403

    allowed = client.get("/api/agent/workflows", headers={"X-Api-Proxy-Secret": "proxy-secret"})
    assert allowed.status_code == 200


# ---------------------------------------------------------------------------
# SHA-34: Smoke password auth
# ---------------------------------------------------------------------------


def test_smoke_password_login_succeeds_when_hash_configured(client, monkeypatch):
    """Smoke password should authenticate successfully when AUTH_SMOKE_PASSWORD_HASH is set."""
    import bcrypt

    smoke_pw = "smoke-test-password"
    smoke_hash = bcrypt.hashpw(smoke_pw.encode(), bcrypt.gensalt(12)).decode()
    monkeypatch.setenv("AUTH_SMOKE_PASSWORD_HASH", smoke_hash)

    resp = client.post("/api/auth/login", json={"password": smoke_pw})
    assert resp.status_code == 200
    assert resp.json() == {"detail": "ok"}
    assert "__session" in resp.cookies


def test_smoke_password_creates_smoke_subject(client, monkeypatch):
    """Smoke login should issue a token with subject 'smoke', not 'admin'."""
    import bcrypt
    from jose import jwt as jose_jwt

    smoke_pw = "smoke-test-password"
    smoke_hash = bcrypt.hashpw(smoke_pw.encode(), bcrypt.gensalt(12)).decode()
    monkeypatch.setenv("AUTH_SMOKE_PASSWORD_HASH", smoke_hash)

    resp = client.post("/api/auth/login", json={"password": smoke_pw})
    assert resp.status_code == 200

    token = resp.cookies.get("__session")
    assert token
    payload = jose_jwt.decode(token, os.environ["JWT_SECRET"], algorithms=["HS256"])
    assert payload["sub"] == "smoke"


def test_admin_login_still_works_with_smoke_hash_configured(client, monkeypatch):
    """Normal admin login must still work when smoke hash is configured."""
    import bcrypt

    smoke_pw = "different-smoke-password"
    smoke_hash = bcrypt.hashpw(smoke_pw.encode(), bcrypt.gensalt(12)).decode()
    monkeypatch.setenv("AUTH_SMOKE_PASSWORD_HASH", smoke_hash)

    resp = client.post("/api/auth/login", json={"password": "testpass"})
    assert resp.status_code == 200
    assert resp.json() == {"detail": "ok"}


def test_admin_login_subject_is_admin(client, monkeypatch):
    """When logging in with admin password, subject should still be 'admin'."""
    from jose import jwt as jose_jwt

    monkeypatch.delenv("AUTH_SMOKE_PASSWORD_HASH", raising=False)

    resp = client.post("/api/auth/login", json={"password": "testpass"})
    assert resp.status_code == 200

    token = resp.cookies.get("__session")
    payload = jose_jwt.decode(token, os.environ["JWT_SECRET"], algorithms=["HS256"])
    assert payload["sub"] == "admin"


def test_no_smoke_login_when_hash_absent(client, monkeypatch):
    """Without AUTH_SMOKE_PASSWORD_HASH, any non-admin password should be rejected."""
    monkeypatch.delenv("AUTH_SMOKE_PASSWORD_HASH", raising=False)

    resp = client.post("/api/auth/login", json={"password": "some-random-password"})
    assert resp.status_code == 401


def test_wrong_smoke_password_rejected(client, monkeypatch):
    """Wrong password should still be rejected even when smoke hash is configured."""
    import bcrypt

    smoke_pw = "smoke-test-password"
    smoke_hash = bcrypt.hashpw(smoke_pw.encode(), bcrypt.gensalt(12)).decode()
    monkeypatch.setenv("AUTH_SMOKE_PASSWORD_HASH", smoke_hash)

    resp = client.post("/api/auth/login", json={"password": "totally-wrong"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Security hardening tests
# ---------------------------------------------------------------------------


def test_jwt_requires_iat_and_exp(client):
    # Login to get a valid token
    resp = client.post("/api/auth/login", json={"password": "testpass"})
    assert resp.status_code == 200
    token = resp.cookies.get("__session")

    # Verify the token has iat and exp
    secret = os.environ["JWT_SECRET"]
    payload = jwt.decode(token, secret, algorithms=["HS256"])
    assert "iat" in payload
    assert "exp" in payload

    # Manually create a token without iat and try to use it
    bad_token = jwt.encode(
        {"sub": "admin", "exp": datetime.now(UTC) + timedelta(hours=1)},
        secret,
        algorithm="HS256",
    )
    client.cookies.set("__session", bad_token, domain="testserver.local", path="/")
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid or expired token"


def test_production_config_validation(monkeypatch):
    from api.routers import auth

    # Mock ENVIRONMENT to production
    monkeypatch.setenv("ENVIRONMENT", "production")

    # Short JWT_SECRET
    monkeypatch.setenv("JWT_SECRET", "too-short")
    monkeypatch.setenv(
        "AUTH_PASSWORD_HASH",
        "$2b$12$43F.9axQmqL0Owf7Hsp4tub0wukaMzCmz8JlTz.UJD8emjTZUVy0C",
    )

    with pytest.raises(RuntimeError, match="JWT_SECRET must be at least 32 characters"):
        auth._verify_production_config()

    # Valid secret, invalid hash
    monkeypatch.setenv("JWT_SECRET", "a" * 32)
    monkeypatch.setenv("AUTH_PASSWORD_HASH", "not-a-bcrypt-hash")

    with pytest.raises(RuntimeError, match="AUTH_PASSWORD_HASH must be a valid bcrypt hash"):
        auth._verify_production_config()

    # Both valid
    monkeypatch.setenv("AUTH_PASSWORD_HASH", "$2b$12$" + "a" * 50)
    auth._verify_production_config()  # Should not raise
