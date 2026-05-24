import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient
from unittest.mock import patch

from api.exceptions import AppError, DataFetchError, AsyncJobDispatchError
from api.main import app

router = APIRouter()

@router.get("/test-error/app-error")
def trigger_app_error():
    raise AppError("Generic message", status_code=400)

@router.get("/test-error/data-fetch-error")
def trigger_data_fetch_error():
    raise DataFetchError(source="TestService", detail="Sensitive technical detail")

@router.get("/test-error/async-dispatch-error")
def trigger_async_dispatch_error():
    # After refactor, this should use the new detail field
    raise AsyncJobDispatchError(detail="Internal connection string")

app.include_router(router)

def test_error_redaction_in_production():
    client = TestClient(app)
    with patch("api.main._is_production_runtime", return_value=True):
        # Test DataFetchError
        response = client.get("/test-error/data-fetch-error")
        assert response.status_code == 424
        data = response.json()
        assert "detail" not in data
        assert data["error"] == "Data fetch failed: TestService"

        # Test AsyncJobDispatchError
        response = client.get("/test-error/async-dispatch-error")
        assert response.status_code == 503
        data = response.json()
        assert "detail" not in data
        assert "Internal connection string" not in data["error"]

def test_error_exposure_in_development():
    client = TestClient(app)
    with patch("api.main._is_production_runtime", return_value=False):
        # Test DataFetchError
        response = client.get("/test-error/data-fetch-error")
        assert response.status_code == 424
        data = response.json()
        assert data.get("detail") == "Sensitive technical detail"

        # Test AsyncJobDispatchError
        response = client.get("/test-error/async-dispatch-error")
        assert response.status_code == 503
        data = response.json()
        # This part might fail BEFORE refactor if detail is appended to message
        # But our goal is to ensure it is available in DEV after refactor.
        assert "Internal connection string" in (data.get("detail") or data.get("error") or "")
