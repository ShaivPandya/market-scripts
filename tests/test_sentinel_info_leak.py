import pytest
from unittest.mock import patch
from api.exceptions import DataFetchError

def test_data_fetch_error_redacts_detail_in_production(auth_client, monkeypatch):
    # Set environment to production
    monkeypatch.setenv("ENVIRONMENT", "production")

    # Ensure IS_PRODUCTION in api.main is True for this test
    from api import main
    monkeypatch.setattr(main, "IS_PRODUCTION", True)

    # Mock audit to avoid DB dependencies in test
    monkeypatch.setattr("api.audit.emit_audit_event", lambda *a, **k: None)

    # Trigger a DataFetchError
    with patch("api.routers.portfolio._current_holdings") as mock_holdings:
        mock_holdings.side_effect = DataFetchError(source="test_source", detail="SECRET_INTERNAL_DETAIL")

        resp = auth_client.get("/api/portfolio")

        assert resp.status_code == 424
        data = resp.json()

        # VERIFY REDACTION: Detail should be replaced by generic error message in production
        assert "detail" in data
        assert data["detail"] == "Data fetch failed: test_source" # exc.message
        assert data["source"] == "test_source"
        assert data["error"] == "Data fetch failed: test_source"

def test_data_fetch_error_shows_detail_in_development(auth_client, monkeypatch):
    # Set environment to development
    monkeypatch.setenv("ENVIRONMENT", "development")

    # Ensure IS_PRODUCTION in api.main is False for this test
    from api import main
    monkeypatch.setattr(main, "IS_PRODUCTION", False)

    # Mock audit to avoid DB dependencies in test
    monkeypatch.setattr("api.audit.emit_audit_event", lambda *a, **k: None)

    # Trigger a DataFetchError
    with patch("api.routers.portfolio._current_holdings") as mock_holdings:
        mock_holdings.side_effect = DataFetchError(source="test_source", detail="SECRET_INTERNAL_DETAIL")

        resp = auth_client.get("/api/portfolio")

        assert resp.status_code == 424
        data = resp.json()

        # VERIFY NO REDACTION: Detail should be present in development
        assert "detail" in data
        assert data["detail"] == "SECRET_INTERNAL_DETAIL"
        assert data["source"] == "test_source"
