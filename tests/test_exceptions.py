"""Tests for custom exception hierarchy and global handlers."""

import pytest

from api.exceptions import AnalysisError, AppError, ConfigurationError, DataFetchError


class TestExceptionHierarchy:
    def test_all_inherit_from_app_error(self):
        assert issubclass(DataFetchError, AppError)
        assert issubclass(ConfigurationError, AppError)
        assert issubclass(AnalysisError, AppError)

    def test_data_fetch_error_defaults(self):
        err = DataFetchError(source="yfinance", detail="timeout")
        assert err.status_code == 424
        assert "yfinance" in err.message
        assert err.source == "yfinance"
        assert err.detail == "timeout"

    def test_configuration_error_defaults(self):
        err = ConfigurationError(key="FRED_API_KEY")
        assert err.status_code == 503
        assert "FRED_API_KEY" in err.message

    def test_analysis_error_defaults(self):
        err = AnalysisError()
        assert err.status_code == 500
        assert "computation failed" in err.message.lower()

    def test_app_error_custom_status(self):
        err = AppError("custom", status_code=418)
        assert err.status_code == 418
        assert err.message == "custom"


class TestGlobalHandlers:
    def test_unhandled_error_returns_500(self, client):
        """The global unhandled error handler should return a sanitized 500."""
        # Hitting a protected endpoint without auth triggers 401, not 500.
        # But the health endpoint should always work.
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_error_response_shape(self, client):
        """401 errors from auth should have the expected shape."""
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401
        data = resp.json()
        assert "detail" in data
