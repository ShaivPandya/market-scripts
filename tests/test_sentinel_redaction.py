import pytest
from fastapi.testclient import TestClient
from api.main import app, ENVIRONMENT
from api.exceptions import DataFetchError

def test_data_fetch_error_redaction_in_production(monkeypatch):
    # Set ENVIRONMENT to production
    monkeypatch.setenv("ENVIRONMENT", "production")

    # We need to re-import or reload app if ENVIRONMENT is evaluated at module level
    # In api/main.py, ENVIRONMENT and IS_PRODUCTION are set at module level.

    from api import main
    monkeypatch.setattr(main, "ENVIRONMENT", "production")
    monkeypatch.setattr(main, "IS_PRODUCTION", True)

    client = TestClient(app)

    @app.get("/api/test-data-fetch-error")
    def trigger_error():
        raise DataFetchError(source="test_source", detail="sensitive_detail")

    response = client.get("/api/test-data-fetch-error")
    assert response.status_code == 424
    data = response.json()
    assert data["source"] == "test_source"
    assert "detail" not in data

def test_data_fetch_error_no_redaction_in_development(monkeypatch):
    # Set ENVIRONMENT to development
    monkeypatch.setenv("ENVIRONMENT", "development")

    from api import main
    monkeypatch.setattr(main, "ENVIRONMENT", "development")
    monkeypatch.setattr(main, "IS_PRODUCTION", False)

    client = TestClient(app)

    @app.get("/api/test-data-fetch-error-dev")
    def trigger_error():
        raise DataFetchError(source="test_source", detail="sensitive_detail")

    response = client.get("/api/test-data-fetch-error-dev")
    assert response.status_code == 424
    data = response.json()
    assert data["source"] == "test_source"
    assert data["detail"] == "sensitive_detail"
