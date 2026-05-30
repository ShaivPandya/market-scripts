"""Tests to ensure technical error details are redacted in production."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.exceptions import AppError, AsyncJobDispatchError, DataFetchError
from api.main import app


def test_app_error_detail_redaction_in_production(monkeypatch):
    """
    Ensures that any AppError with a 'detail' attribute has that detail redacted
    when ENVIRONMENT=production.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")

    class TechnicalError(AppError):
        def __init__(self, detail: str):
            super().__init__("A technical error occurred", status_code=500)
            self.detail = detail

    @app.get("/test-technical-error")
    async def route():
        raise TechnicalError(detail="SECRET_TOKEN_123")

    client = TestClient(app)
    response = client.get("/test-technical-error")

    assert response.status_code == 500
    data = response.json()
    assert "SECRET_TOKEN_123" not in str(data)
    assert "detail" not in data
    assert data["error"] == "A technical error occurred"


def test_data_fetch_error_redaction_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")

    @app.get("/test-data-fetch-redaction")
    async def route():
        raise DataFetchError(source="external-api", detail="key=highly-sensitive")

    client = TestClient(app)
    response = client.get("/test-data-fetch-redaction")

    assert response.status_code == 424
    data = response.json()
    assert "highly-sensitive" not in str(data)
    assert "detail" not in data
    assert data["source"] == "external-api"


def test_async_job_error_redaction_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")

    @app.get("/test-async-redaction")
    async def route():
        raise AsyncJobDispatchError(detail="connection_string=redis://pass@host")

    client = TestClient(app)
    response = client.get("/test-async-redaction")

    assert response.status_code == 503
    data = response.json()
    assert "redis://" not in data["error"]
    assert "detail" not in data
    assert data["error"] == "Async job dispatch failed"


def test_detail_available_in_development(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")

    @app.get("/test-dev-detail")
    async def route():
        raise DataFetchError(source="dev", detail="debug-info")

    client = TestClient(app)
    response = client.get("/test-dev-detail")

    assert response.status_code == 424
    data = response.json()
    assert data["detail"] == "debug-info"
