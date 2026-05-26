import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.exceptions import AppError, DataFetchError, AsyncJobDispatchError
from api.main import _app_error_handler

# Create a minimal app for testing the exception handler
app = FastAPI()
app.add_exception_handler(AppError, _app_error_handler)

@app.get("/data-fetch-error")
async def raise_data_fetch_error():
    raise DataFetchError(source="yfinance", detail="Technical connection string or stack trace")

@app.get("/async-dispatch-error")
async def raise_async_dispatch_error():
    raise AsyncJobDispatchError(detail="Internal queue connection failed: redis://secrets@localhost")

@pytest.fixture
def prod_client(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    return TestClient(app)

@pytest.fixture
def dev_client(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    return TestClient(app)

def test_data_fetch_error_redaction_in_production(prod_client):
    response = prod_client.get("/data-fetch-error")
    assert response.status_code == 424
    data = response.json()
    # SECURE behavior: detail should NOT be in the response
    assert "detail" not in data

def test_async_dispatch_error_redaction_in_production(prod_client):
    response = prod_client.get("/async-dispatch-error")
    assert response.status_code == 503
    data = response.json()
    # SECURE behavior: message should be generic, NOT containing technical details
    assert data["error"] == "Async job dispatch failed"

def test_data_fetch_error_detail_in_development(dev_client):
    response = dev_client.get("/data-fetch-error")
    assert response.status_code == 424
    data = response.json()
    # In development, we want the details
    assert data.get("detail") == "Technical connection string or stack trace"
