"""Tests for authentication flow — login, logout, token validation."""

import os

import pytest
from fastapi.responses import JSONResponse


@pytest.fixture(autouse=True)
def _reset_login_attempt_state():
    from api.routers import auth as auth_router

    auth_router._reset_login_attempt_state()
    yield
    auth_router._reset_login_attempt_state()


def test_login_success(client):
    resp = client.post("/api/v1/auth/login", json={"password": "testpass"})
    assert resp.status_code == 200
    assert resp.json() == {"detail": "ok"}
    assert "__session" in resp.cookies


def test_login_wrong_password(client):
    resp = client.post("/api/v1/auth/login", json={"password": "wrongpassword"})
    assert resp.status_code == 401


def test_login_missing_password(client):
    resp = client.post("/api/v1/auth/login", json={})
    assert resp.status_code == 422  # Pydantic validation error


def test_logout(client):
    # Login first
    client.post("/api/v1/auth/login", json={"password": "testpass"})
    # Logout
    resp = client.post("/api/v1/auth/logout")
    assert resp.status_code == 200
    assert resp.json() == {"detail": "ok"}


def test_me_unauthenticated(client):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401


def test_me_authenticated(auth_client):
    resp = auth_client.get("/api/v1/auth/me")
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
    for forbidden in ("portfolio_db", "thesis_db", "core_db", "fred", "sqlite", "traceback", "/users/", "error:"):
        assert forbidden not in text


def test_admin_health_requires_auth(client, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_detailed_health_response", lambda: JSONResponse({"status": "ok", "checks": {}}))

    unauthenticated = client.get("/api/v1/admin/health")
    assert unauthenticated.status_code == 401

    login = client.post("/api/v1/auth/login", json={"password": "testpass"})
    assert login.status_code == 200
    authenticated = client.get("/api/v1/admin/health")
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

    resp = client.post("/api/v1/auth/login", json={"password": "x" * 100})

    assert resp.status_code == 413


def test_login_password_length_is_limited(client):
    resp = client.post("/api/v1/auth/login", json={"password": "x" * 513})

    assert resp.status_code == 422


def test_repeated_bad_logins_lock_out_client(client, monkeypatch):
    from api.routers import auth as auth_router

    now = [1_000.0]
    monkeypatch.setenv("AUTH_LOGIN_FAILURE_LIMIT", "2")
    monkeypatch.setenv("AUTH_LOGIN_FAILURE_WINDOW_SECONDS", "300")
    monkeypatch.setenv("AUTH_LOGIN_LOCKOUT_SECONDS", "60")
    monkeypatch.setattr(auth_router.time, "time", lambda: now[0])

    first = client.post("/api/v1/auth/login", json={"password": "wrong"})
    second = client.post("/api/v1/auth/login", json={"password": "wrong"})
    correct_while_locked = client.post("/api/v1/auth/login", json={"password": "testpass"})

    assert first.status_code == 401
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"
    assert correct_while_locked.status_code == 429

    now[0] += 61
    allowed_after_lockout = client.post("/api/v1/auth/login", json={"password": "testpass"})
    assert allowed_after_lockout.status_code == 200


def test_successful_login_clears_failed_login_counter(client, monkeypatch):
    monkeypatch.setenv("AUTH_LOGIN_FAILURE_LIMIT", "2")
    monkeypatch.setenv("AUTH_LOGIN_FAILURE_WINDOW_SECONDS", "300")
    monkeypatch.setenv("AUTH_LOGIN_LOCKOUT_SECONDS", "60")

    failed = client.post("/api/v1/auth/login", json={"password": "wrong"})
    success = client.post("/api/v1/auth/login", json={"password": "testpass"})
    failed_after_success = client.post("/api/v1/auth/login", json={"password": "wrong"})

    assert failed.status_code == 401
    assert success.status_code == 200
    assert failed_after_success.status_code == 401


def test_password_mode_does_not_require_proxy_secret_for_login(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "password")
    monkeypatch.setenv("API_PROXY_SECRET", "proxy-secret")
    monkeypatch.delenv("REQUIRE_API_PROXY_SECRET", raising=False)

    resp = client.post("/api/v1/auth/login", json={"password": "testpass"})
    assert resp.status_code == 200


def test_explicit_proxy_secret_requirement_blocks_missing_header(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "password")
    monkeypatch.setenv("API_PROXY_SECRET", "proxy-secret")
    monkeypatch.setenv("REQUIRE_API_PROXY_SECRET", "true")

    missing = client.post("/api/v1/auth/login", json={"password": "testpass"})
    assert missing.status_code == 403

    allowed = client.post(
        "/api/v1/auth/login",
        json={"password": "testpass"},
        headers={"X-Api-Proxy-Secret": "proxy-secret"},
    )
    assert allowed.status_code == 200


def test_cloudflare_mode_requires_backend_proxy_secret(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "cloudflare")
    monkeypatch.delenv("API_PROXY_SECRET", raising=False)

    protected = client.get("/api/v1/agent/workflows")
    assert protected.status_code == 403
    assert protected.json() == {"detail": "API proxy secret is required for this auth mode."}

    health = client.get("/api/health")
    assert health.status_code == 200


def test_cloudflare_mode_enforces_proxy_secret_header(client, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "cloudflare")
    monkeypatch.setenv("API_PROXY_SECRET", "proxy-secret")

    missing = client.get("/api/v1/agent/workflows")
    assert missing.status_code == 403

    allowed = client.get("/api/v1/agent/workflows", headers={"X-Api-Proxy-Secret": "proxy-secret"})
    assert allowed.status_code == 200
