import os

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from api.exceptions import AppError, DataFetchError


def test_data_fetch_error_redaction_development(monkeypatch):
    """Verify that DataFetchError details are exposed in development."""
    from api.main import _app_error_handler

    app = FastAPI()
    app.add_exception_handler(AppError, _app_error_handler)

    @app.get("/error")
    def raise_error():
        raise DataFetchError(source="test", detail="Secret internal error details")

    # Mock IS_PRODUCTION to False
    monkeypatch.setattr("api.main.IS_PRODUCTION", False)

    client = TestClient(app)
    response = client.get("/error")

    assert response.status_code == 424
    data = response.json()
    assert data["error"] == "Data fetch failed: test"
    assert data["detail"] == "Secret internal error details"


def test_data_fetch_error_redaction_production(monkeypatch):
    """Verify that DataFetchError details are redacted in production."""
    from api.main import _app_error_handler

    app = FastAPI()
    app.add_exception_handler(AppError, _app_error_handler)

    @app.get("/error")
    def raise_error():
        raise DataFetchError(source="test", detail="Secret internal error details")

    # Mock IS_PRODUCTION to True
    monkeypatch.setattr("api.main.IS_PRODUCTION", True)

    client = TestClient(app)
    response = client.get("/error")

    assert response.status_code == 424
    data = response.json()
    assert data["error"] == "Data fetch failed: test"

    # After fix, this should pass
    assert "detail" not in data
