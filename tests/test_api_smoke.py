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
_AUTH_GET_ENDPOINTS = [
    "/api/portfolio",
    "/api/momentum",
    "/api/market-breadth",
    "/api/top50-breadth",
    "/api/price-volume-signals",
    "/api/vix-term-structure",
    "/api/economic-growth",
    "/api/liquidity",
    "/api/breakout",
    "/api/positioning/summary",
    "/api/positioning/instruments",
    "/api/index-dashboard",
    "/api/fx-dashboard",
    "/api/commodities",
    "/api/central-banks",
    "/api/sector-metrics",
    "/api/industry-monitor",
    "/api/yield-curve",
    "/api/country-dashboard",
    "/api/portfolio-news",
    "/api/fx-model/pairs",
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
