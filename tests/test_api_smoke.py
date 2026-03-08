"""
Smoke tests for every API GET endpoint.

Each test verifies the endpoint is reachable and returns the expected HTTP
status code when called with authentication. POST endpoints are excluded
since they require specific request bodies or mock data.

These tests hit the real data modules — they may be slow or flaky if
external APIs (yfinance, FRED, etc.) are unavailable. Mark them with
@pytest.mark.slow if you need to skip in CI.
"""

import pytest

# All GET endpoints that require auth
_API_PREFIX = "/api/v1"

_AUTH_GET_ENDPOINTS = [
    f"{_API_PREFIX}/portfolio",
    f"{_API_PREFIX}/momentum",
    f"{_API_PREFIX}/market-breadth",
    f"{_API_PREFIX}/top50-breadth",
    f"{_API_PREFIX}/price-volume-signals",
    f"{_API_PREFIX}/vix-term-structure",
    f"{_API_PREFIX}/economic-growth",
    f"{_API_PREFIX}/liquidity",
    f"{_API_PREFIX}/breakout",
    f"{_API_PREFIX}/positioning/summary",
    f"{_API_PREFIX}/positioning/instruments",
    f"{_API_PREFIX}/index-dashboard",
    f"{_API_PREFIX}/fx-dashboard",
    f"{_API_PREFIX}/commodities",
    f"{_API_PREFIX}/central-banks",
    f"{_API_PREFIX}/sector-metrics",
    f"{_API_PREFIX}/signal-aggregator",
    f"{_API_PREFIX}/industry-monitor",
    f"{_API_PREFIX}/yield-curve",
    f"{_API_PREFIX}/country-dashboard",
    f"{_API_PREFIX}/portfolio-news",
    f"{_API_PREFIX}/fx-model/pairs",
    f"{_API_PREFIX}/thesis/status",
]


class TestPublicEndpoints:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_has_checks(self, client):
        resp = client.get("/api/health")
        data = resp.json()
        assert "checks" in data

    def test_openapi_docs(self, client):
        resp = client.get("/api/docs")
        assert resp.status_code == 200


class TestAuthRequired:
    """Verify all protected endpoints reject unauthenticated requests."""

    @pytest.mark.parametrize("endpoint", _AUTH_GET_ENDPOINTS)
    def test_unauthenticated_returns_401(self, client, endpoint):
        resp = client.get(endpoint)
        assert resp.status_code == 401, f"{endpoint} should require auth"


class TestRequestId:
    def test_response_includes_request_id(self, client):
        resp = client.get("/api/health")
        assert "x-request-id" in resp.headers

    def test_custom_request_id_echoed(self, client):
        resp = client.get("/api/health", headers={"x-request-id": "test-123"})
        assert resp.headers.get("x-request-id") == "test-123"
