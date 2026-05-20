import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.exceptions import DataFetchError

def test_data_fetch_error_redaction_in_production(monkeypatch):
    # Mock IS_PRODUCTION to True
    monkeypatch.setattr("api.main.IS_PRODUCTION", True)

    client = TestClient(app)

    # We need a route that raises DataFetchError to test the handler
    @app.get("/test-data-fetch-error")
    async def raise_error():
        raise DataFetchError(source="test_source", detail="sensitive internal info")

    response = client.get("/test-data-fetch-error")
    assert response.status_code == 424
    data = response.json()
    assert data["error"] == "Data fetch failed: test_source"
    assert data["source"] == "test_source"
    assert "detail" not in data

def test_data_fetch_error_leaks_in_development(monkeypatch):
    # Mock IS_PRODUCTION to False
    monkeypatch.setattr("api.main.IS_PRODUCTION", False)

    client = TestClient(app)

    # The route might already be added from previous test, but FastAPI allows it for testing
    @app.get("/test-data-fetch-error-dev")
    async def raise_error_dev():
        raise DataFetchError(source="test_source_dev", detail="sensitive internal info")

    response = client.get("/test-data-fetch-error-dev")
    assert response.status_code == 424
    data = response.json()
    assert data["error"] == "Data fetch failed: test_source_dev"
    assert data["source"] == "test_source_dev"
    assert data["detail"] == "sensitive internal info"
