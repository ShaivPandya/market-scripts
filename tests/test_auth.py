"""Tests for authentication flow — login, logout, token validation."""

import os

import pytest


def test_login_success(client):
    resp = client.post("/api/v1/auth/login", json={"password": "testpass"})
    assert resp.status_code == 200
    assert resp.json() == {"detail": "ok"}
    assert "access_token" in resp.cookies


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
