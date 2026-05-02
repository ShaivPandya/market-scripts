"""Tests for authentication flow — login, logout, token validation."""

import os

import pytest


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
    data = resp.json()
    assert "status" in data


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
