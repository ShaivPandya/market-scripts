"""
Reproduction tests for technical error detail leakage.
"""

import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient
from api.exceptions import AsyncJobDispatchError, DataFetchError, AppError
from api.main import app

# Create a temporary router to trigger specific errors
reproduction_router = APIRouter(prefix="/reproduction")

@reproduction_router.get("/async-job-error")
def trigger_async_job_error():
    raise AsyncJobDispatchError(detail="connection string: postgresql://secret@localhost:5432")

@reproduction_router.get("/data-fetch-error")
def trigger_data_fetch_error():
    raise DataFetchError(source="yfinance", detail="Technical timeout after 30s")

@reproduction_router.get("/generic-app-error")
def trigger_generic_app_error():
    raise AppError("Something went wrong", status_code=500)

app.include_router(reproduction_router)

def test_async_job_dispatch_error_is_generic_in_message(monkeypatch):
    """
    Test that AsyncJobDispatchError message is generic.
    Technical details are moved to 'detail'.
    """
    monkeypatch.setenv("ENVIRONMENT", "development")
    client = TestClient(app)

    resp = client.get("/reproduction/async-job-error")
    assert resp.status_code == 503
    data = resp.json()

    # Message should be generic
    assert data["error"] == "Async job dispatch failed"
    # Detail should be available in development
    assert data["detail"] == "connection string: postgresql://secret@localhost:5432"

def test_data_fetch_error_includes_detail_in_dev(monkeypatch):
    """
    Test that DataFetchError includes 'detail' in dev.
    """
    monkeypatch.setenv("ENVIRONMENT", "development")
    client = TestClient(app)

    resp = client.get("/reproduction/data-fetch-error")
    assert resp.status_code == 424
    data = resp.json()
    assert data["detail"] == "Technical timeout after 30s"

def test_error_redaction_in_production(monkeypatch):
    """
    Test that 'detail' is redacted when ENVIRONMENT=production.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    client = TestClient(app)

    # Check DataFetchError
    resp = client.get("/reproduction/data-fetch-error")
    data = resp.json()
    assert "detail" not in data
    assert data["error"] == "Data fetch failed: yfinance"

    # Check AsyncJobDispatchError
    resp = client.get("/reproduction/async-job-error")
    data = resp.json()
    assert "detail" not in data
    assert data["error"] == "Async job dispatch failed"
