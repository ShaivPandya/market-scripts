import pytest
from api.exceptions import DataFetchError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.main import _app_error_handler
from api.exceptions import AppError

def test_data_fetch_error_leakage(monkeypatch):
    from api import main

    # Force production mode
    monkeypatch.setattr(main, "IS_PRODUCTION", True)
    monkeypatch.setattr(main, "ENVIRONMENT", "production")

    # Mock _is_production_runtime to return True
    monkeypatch.setattr(main, "_is_production_runtime", lambda: True)

    app = FastAPI()
    app.add_exception_handler(AppError, _app_error_handler)

    @app.get("/error")
    def trigger_error():
        raise DataFetchError(source="test_source", detail="SECRET_INTERNAL_DETAIL")

    client = TestClient(app)
    resp = client.get("/error")

    assert resp.status_code == 424
    data = resp.json()
    assert data["error"] == "Data fetch failed: test_source"
    assert data["source"] == "test_source"

    # Verify it is NO LONGER leaked in production
    assert "detail" not in data

if __name__ == "__main__":
    # If run directly, run the test
    import sys
    pytest.main([__file__])
