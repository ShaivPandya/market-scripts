
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.exceptions import DataFetchError, AsyncJobDispatchError
import os

def test_error_redaction_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")

    @app.get("/test-data-fetch-error")
    async def route_data_fetch():
        raise DataFetchError(source="test-source", detail="sensitive internal detail")

    @app.get("/test-async-dispatch-error")
    async def route_async_dispatch():
        raise AsyncJobDispatchError(detail="connection_string=secret")

    client = TestClient(app)

    # Test DataFetchError redaction
    resp = client.get("/test-data-fetch-error")
    assert resp.status_code == 424
    data = resp.json()
    assert data["error"] == "Data fetch failed: test-source"
    assert "detail" not in data  # Should be redacted

    # Test AsyncJobDispatchError redaction
    resp = client.get("/test-async-dispatch-error")
    assert resp.status_code == 503
    data = resp.json()
    assert "secret" not in data["error"] # Should not be in message
    assert "detail" not in data # Should be redacted

def test_error_detail_in_development(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")

    @app.get("/test-dev-error")
    async def route_dev_error():
        raise DataFetchError(source="test-source", detail="sensitive internal detail")

    client = TestClient(app)

    resp = client.get("/test-dev-error")
    assert resp.status_code == 424
    data = resp.json()
    assert data["detail"] == "sensitive internal detail" # Should be present in dev
