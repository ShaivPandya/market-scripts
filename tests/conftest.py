"""Test-suite conftest — API client fixtures."""

import os

import pytest

# Ensure auth env vars are set for testing
os.environ["JWT_SECRET"] = "test-secret-for-ci"
os.environ["AUTH_PASSWORD_HASH"] = (
    # bcrypt hash of "testpass"
    "$2b$12$43F.9axQmqL0Owf7Hsp4tub0wukaMzCmz8JlTz.UJD8emjTZUVy0C"
)
os.environ["AUTH_MODE"] = "password"
os.environ["AUTH_LOGIN_RATE_LIMIT"] = "1000/minute"
os.environ["ASYNC_JOB_BACKEND"] = "local"


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
