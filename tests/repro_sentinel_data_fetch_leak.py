import os
import pytest
from api.exceptions import DataFetchError
from api.main import app
from fastapi.testclient import TestClient

def test_data_fetch_error_leak_repro(monkeypatch):
    """Verify that DataFetchError leaks the detail field in development."""
    # Ensure environment is set to development
    monkeypatch.setenv("ENVIRONMENT", "development")

    @app.get("/api/test-leak")
    def leak_endpoint():
        raise DataFetchError(source="test_source", detail="sensitive_internal_detail")

    client = TestClient(app)

    response = client.get("/api/test-leak")
    assert response.status_code == 424
    data = response.json()
    assert data["error"] == "Data fetch failed: test_source"
    assert data["source"] == "test_source"
    assert data["detail"] == "sensitive_internal_detail"

def test_data_fetch_error_redaction_repro(monkeypatch):
    """Verify that DataFetchError detail is redacted in production (after fix)."""
    monkeypatch.setenv("ENVIRONMENT", "production")

    @app.get("/api/test-redact")
    def redact_endpoint():
        raise DataFetchError(source="test_source", detail="sensitive_internal_detail")

    client = TestClient(app)

    response = client.get("/api/test-redact")
    assert response.status_code == 424
    data = response.json()
    assert data["error"] == "Data fetch failed: test_source"
    assert data["source"] == "test_source"
    assert "detail" not in data
