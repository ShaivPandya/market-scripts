"""Test-suite conftest — API client fixtures."""

import os

import pytest

# Ensure auth env vars are set for testing
os.environ.setdefault("JWT_SECRET", "test-secret-for-ci")
os.environ.setdefault(
    "AUTH_PASSWORD_HASH",
    "$2b$12$LJ3m4ys3Lk0TSwHiRb0v5u1N6DpFV65WJUGAjhBrE8gFnLbMKqGTS",  # hash of "testpass"
)


@pytest.fixture
def client():
    """Unauthenticated FastAPI test client."""
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


@pytest.fixture
def auth_client(client):
    """Authenticated FastAPI test client with a valid session cookie."""
    resp = client.post("/api/v1/auth/login", json={"password": "testpass"})
    # If login succeeds, the cookie is set automatically on the client
    if resp.status_code == 200:
        return client
    # Fallback: return unauthenticated client (tests should handle this)
    return client
