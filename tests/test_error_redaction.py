import pytest
import os
from api.exceptions import DataFetchError, AsyncJobDispatchError
from fastapi import Request, APIRouter

def test_data_fetch_error_redacts_detail_in_production(auth_client, monkeypatch):
    """
    Verify that DataFetchError technical 'detail' is REDACTED when ENVIRONMENT=production.
    """
    from api import main

    # Create a temporary router to avoid messing with existing ones too much
    test_router = APIRouter()

    @test_router.get("/api/test-error")
    async def trigger_error():
        raise DataFetchError(source="SecretService", detail="connection_string=postgresql://user:password@internal-db:5432/db")

    main.app.include_router(test_router)

    # Force production mode
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(main, "ENVIRONMENT", "production")
    monkeypatch.setattr(main, "IS_PRODUCTION", True)

    resp = auth_client.get("/api/test-error")

    assert resp.status_code == 424
    data = resp.json()
    # Currently this fails because it's NOT redacting, it returns the details.
    assert "detail" not in data or data["detail"] == ""

def test_async_job_dispatch_error_redacts_detail_in_production(auth_client, monkeypatch):
    """
    Verify that AsyncJobDispatchError does NOT leak 'detail' in the 'message' when ENVIRONMENT=production.
    """
    from api import main

    test_router = APIRouter()

    @test_router.get("/api/test-error-async")
    async def trigger_error_async():
        raise AsyncJobDispatchError(detail="Redis connection failed at redis://:super-secret@localhost:6379")

    main.app.include_router(test_router)

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(main, "ENVIRONMENT", "production")
    monkeypatch.setattr(main, "IS_PRODUCTION", True)

    resp = auth_client.get("/api/test-error-async")
    assert resp.status_code == 503
    data = resp.json()

    # Should fail currently because it leaks
    assert "super-secret" not in data["error"]
    assert data["error"] == "Async job dispatch failed"
