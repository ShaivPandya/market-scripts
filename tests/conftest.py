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
os.environ["AUTH_DEFAULT_USERNAME"] = "admin"
os.environ["AUTH_LOGIN_RATE_LIMIT"] = "1000/minute"
os.environ["ASYNC_JOB_BACKEND"] = "local"
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ["ENVIRONMENT"] = "development"
os.environ["TALISMAN_ALLOW_SQLITE_STATE"] = "true"
os.environ["STATE_DB_BACKEND"] = "sqlite"
os.environ["DATABASE_URL"] = ""
os.environ["STATE_STORAGE_BACKEND"] = "local"

collect_ignore = []
if os.environ.get("RUN_ONTOLOGY_PERF") != "1":
    collect_ignore.append("test_ontology_query_perf.py")


def _login_client(client: TestClient, password: str = "testpass", username: str | None = None) -> None:
    body: dict[str, str] = {"password": password}
    if username:
        body["username"] = username
    resp = client.post("/api/auth/login", json=body)
    if resp.status_code != 200:
        return
    data = resp.json()
    csrf = data.get("csrfToken")
    if csrf:
        client._csrf_token = csrf  # type: ignore[attr-defined]
    session_cookie = resp.cookies.get("__session")
    if session_cookie:
        client.cookies.set("__session", session_cookie, domain="testserver.local", path="/")


@pytest.fixture(autouse=True)
def _reset_auth_state():
    from api.auth_store import reset_auth_store_for_tests
    from api.routers import auth as auth_router

    reset_auth_store_for_tests()
    auth_router._reset_login_attempt_state()
    yield
    reset_auth_store_for_tests()
    auth_router._reset_login_attempt_state()


@pytest.fixture
def client():
    """Unauthenticated FastAPI test client."""
    os.environ["JWT_SECRET"] = "test-secret-for-ci"
    os.environ["AUTH_PASSWORD_HASH"] = "$2b$12$43F.9axQmqL0Owf7Hsp4tub0wukaMzCmz8JlTz.UJD8emjTZUVy0C"
    os.environ["AUTH_MODE"] = "password"
    os.environ["AUTH_DEFAULT_USERNAME"] = "admin"
    os.environ["AUTH_LOGIN_RATE_LIMIT"] = "1000/minute"
    os.environ["ENVIRONMENT"] = "development"
    os.environ["TALISMAN_ALLOW_SQLITE_STATE"] = "true"
    os.environ["STATE_DB_BACKEND"] = "sqlite"
    os.environ["DATABASE_URL"] = ""
    os.environ["STATE_STORAGE_BACKEND"] = "local"
    from api.main import app

    class SchemaAwareTestClient(TestClient):
        def request(self, method, url, **kwargs):  # noqa: ANN001, ANN201 - test helper mirrors TestClient.
            headers = dict(kwargs.pop("headers", {}) or {})
            if not any(key.lower() in {"x-request-schema-name", "x-request-schema-version"} for key in headers):
                from api.request_schema import schema_headers_for_path

                path = str(url).split("?", 1)[0]
                headers.update(schema_headers_for_path(app, str(method), path))
            csrf = getattr(self, "_csrf_token", None)
            if csrf and str(method).upper() in {"POST", "PUT", "PATCH", "DELETE"}:
                headers.setdefault("X-CSRF-Token", csrf)
            kwargs["headers"] = headers
            return super().request(method, url, **kwargs)

    test_client = SchemaAwareTestClient(app)
    test_client._csrf_token = None  # type: ignore[attr-defined]
    return test_client


@pytest.fixture
def auth_client(client):
    """Authenticated FastAPI test client with a valid session cookie and CSRF token."""
    _login_client(client)
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
