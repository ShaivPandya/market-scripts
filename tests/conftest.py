"""Test-suite conftest — API client fixtures."""

import os

import pytest
from fastapi.testclient import TestClient

# Ensure auth env vars are set for testing
os.environ["JWT_SECRET"] = "test-secret-for-ci"
os.environ["AUTH_PASSWORD_HASH"] = (
    # bcrypt hash of "testpass"
    "$2b$12$43F.9axQmqL0Owf7Hsp4tub0wukaMzCmz8JlTz.UJD8emjTZUVy0C"
)
os.environ["AUTH_MODE"] = "password"
os.environ["AUTH_LOGIN_RATE_LIMIT"] = "1000/minute"
os.environ["ASYNC_JOB_BACKEND"] = "local"
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ["ENVIRONMENT"] = "development"
os.environ["STATE_STORAGE_BACKEND"] = "local"


@pytest.fixture
def client():
    """Unauthenticated FastAPI test client."""
    os.environ["JWT_SECRET"] = "test-secret-for-ci"
    os.environ["AUTH_PASSWORD_HASH"] = "$2b$12$43F.9axQmqL0Owf7Hsp4tub0wukaMzCmz8JlTz.UJD8emjTZUVy0C"
    os.environ["AUTH_MODE"] = "password"
    os.environ["AUTH_LOGIN_RATE_LIMIT"] = "1000/minute"
    os.environ["ENVIRONMENT"] = "development"
    os.environ["STATE_STORAGE_BACKEND"] = "local"
    from api.main import app

    class SchemaAwareTestClient(TestClient):
        def request(self, method, url, **kwargs):  # noqa: ANN001, ANN201 - test helper mirrors TestClient.
            headers = dict(kwargs.pop("headers", {}) or {})
            if not any(key.lower() in {"x-request-schema-name", "x-request-schema-version"} for key in headers):
                from api.request_schema import schema_headers_for_path

                path = str(url).split("?", 1)[0]
                headers.update(schema_headers_for_path(app, str(method), path))
            kwargs["headers"] = headers
            return super().request(method, url, **kwargs)

    return SchemaAwareTestClient(app)


@pytest.fixture
def auth_client(client):
    """Authenticated FastAPI test client with a valid session cookie."""
    resp = client.post("/api/v1/auth/login", json={"password": "testpass"})
    # If login succeeds, the cookie is set automatically on the client
    if resp.status_code == 200:
        session_cookie = resp.cookies.get("__session")
        if session_cookie:
            client.cookies.set("__session", session_cookie, domain="testserver.local", path="/")
        return client
    # Fallback: return unauthenticated client (tests should handle this)
    return client


@pytest.fixture(autouse=True)
def _isolate_app_settings(tmp_path, monkeypatch):
    """Keep ignored local app settings out of test provider selection."""
    from api import llm_settings

    if llm_settings._conn is not None:
        llm_settings._conn.close()
    monkeypatch.setattr(llm_settings, "_conn", None)
    monkeypatch.setattr(llm_settings, "DB_PATH", tmp_path / "app_settings.db")
    yield
    if llm_settings._conn is not None:
        llm_settings._conn.close()
    monkeypatch.setattr(llm_settings, "_conn", None)
