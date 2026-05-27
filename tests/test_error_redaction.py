import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.main import _app_error_handler
from api.exceptions import AppError, DataFetchError, AsyncJobDispatchError

# Create a dedicated test app to avoid side effects on the main app
test_app = FastAPI()
test_app.add_exception_handler(AppError, _app_error_handler)

@test_app.get("/data-fetch-error")
async def raise_data_fetch_error():
    raise DataFetchError(source="test-source", detail="sensitive technical detail")

@test_app.get("/async-dispatch-error")
async def raise_async_dispatch_error():
    raise AsyncJobDispatchError(detail="connection string with password")

def test_data_fetch_error_leaks_detail_in_development(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    client = TestClient(test_app)
    response = client.get("/data-fetch-error")
    assert response.status_code == 424
    data = response.json()
    assert data["detail"] == "sensitive technical detail"

def test_data_fetch_error_redacts_detail_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    client = TestClient(test_app)
    response = client.get("/data-fetch-error")
    assert response.status_code == 424
    data = response.json()
    # This is expected to FAIL before the fix as it currently doesn't redact
    assert "detail" not in data

def test_async_dispatch_error_redacts_detail_in_production(monkeypatch):
    # This test verifies that technical details are NOT leaked in the error message
    monkeypatch.setenv("ENVIRONMENT", "production")
    client = TestClient(test_app)
    response = client.get("/async-dispatch-error")
    assert response.status_code == 503
    data = response.json()
    # Verify that the technical detail passed to the exception is not in the public error message
    assert "connection string" not in data["error"]
    assert data["error"] == "Async job dispatch failed"
