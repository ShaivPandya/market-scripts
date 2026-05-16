import os
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.exceptions import DataFetchError, AppError
from api.main import _app_error_handler

def test_data_fetch_error_redaction(monkeypatch):
    """
    Verify that DataFetchError details are redacted in production
    but visible in development.
    """
    test_app = FastAPI()
    # Register the same handler used in the real app
    test_app.add_exception_handler(AppError, _app_error_handler)

    @test_app.get("/trigger-data-fetch-error")
    async def trigger_error():
        raise DataFetchError(source="external_api", detail="secret_api_key_leaked_here")

    client = TestClient(test_app)

    # 1. Test Development Mode (Default)
    monkeypatch.setenv("ENVIRONMENT", "development")
    resp = client.get("/trigger-data-fetch-error")
    assert resp.status_code == 424
    data = resp.json()
    assert data["error"] == "Data fetch failed: external_api"
    assert data["source"] == "external_api"
    assert data["detail"] == "secret_api_key_leaked_here"
    assert data["type"] == "DataFetchError"

    # 2. Test Production Mode (Redacted)
    monkeypatch.setenv("ENVIRONMENT", "production")
    resp = client.get("/trigger-data-fetch-error")
    assert resp.status_code == 424
    data = resp.json()
    assert data["error"] == "Data fetch failed: external_api"
    assert data["source"] == "external_api"
    assert "detail" not in data  # CRITICAL: detail must be redacted
    assert data["type"] == "DataFetchError"
