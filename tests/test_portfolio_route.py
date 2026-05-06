import pytest


def _portfolio_payload(timeframe: str) -> dict:
    return {
        "positions": {},
        "metadata": {},
        "position_order": [],
        "timeframe": timeframe,
    }


@pytest.mark.parametrize(
    ("timeframe", "expected_cache_name"),
    [
        ("This Week", "short_cache"),
        ("Daily", "long_cache"),
        ("Weekly", "long_cache"),
        ("Monthly", "long_cache"),
    ],
)
def test_portfolio_single_timeframe_uses_freshness_cache(auth_client, monkeypatch, timeframe, expected_cache_name):
    from api.routers import portfolio as portfolio_router
    from portfolio import portfolio_dashboard

    cache_calls = []
    loader_calls = []

    def fake_get_or_set_cached(cache, key, loader):
        cache_calls.append((cache, key))
        return loader()

    def fake_get_data(*, timeframe: str = "Daily", all_timeframes: bool = False):
        loader_calls.append((timeframe, all_timeframes))
        return _portfolio_payload(timeframe)

    monkeypatch.setattr(portfolio_router, "_current_holdings", lambda: [{"ticker": "MU", "role": "position"}])
    monkeypatch.setattr(portfolio_router, "get_or_set_cached", fake_get_or_set_cached)
    monkeypatch.setattr(portfolio_dashboard, "get_data", fake_get_data)

    resp = auth_client.get("/api/v1/portfolio", params={"timeframe": timeframe})

    assert resp.status_code == 200
    expected_cache = getattr(portfolio_router, expected_cache_name)
    assert len(cache_calls) == 1
    cache, key = cache_calls[0]
    assert cache is expected_cache
    assert f"portfolio:v3:{timeframe}:" in key
    assert loader_calls == [(timeframe, False)]
    assert resp.json()["timeframe"] == timeframe
    assert resp.json()["holdings"] == [{"ticker": "MU", "role": "position"}]


def test_portfolio_all_timeframes_is_deprecated_but_compatible(auth_client, monkeypatch):
    from api.routers import portfolio as portfolio_router
    from portfolio import portfolio_dashboard

    cache_calls = []
    loader_calls = []

    def fake_get_or_set_cached(cache, key, loader):
        cache_calls.append((cache, key))
        return loader()

    def fake_get_data(*, timeframe: str = "Daily", all_timeframes: bool = False):
        loader_calls.append((timeframe, all_timeframes))
        return {"timeframes": {"Daily": _portfolio_payload("Daily")}}

    monkeypatch.setattr(portfolio_router, "_current_holdings", lambda: [{"ticker": "MU", "role": "position"}])
    monkeypatch.setattr(portfolio_router, "get_or_set_cached", fake_get_or_set_cached)
    monkeypatch.setattr(portfolio_dashboard, "get_data", fake_get_data)

    resp = auth_client.get("/api/v1/portfolio", params={"all_timeframes": "true"})

    assert resp.status_code == 200
    assert resp.headers["Deprecation"] == "true"
    assert len(cache_calls) == 1
    cache, key = cache_calls[0]
    assert cache is portfolio_router.short_cache
    assert "portfolio:all_timeframes:v3:" in key
    assert loader_calls == [("Daily", True)]

    payload = resp.json()
    assert "timeframes" in payload
    assert payload["holdings"] == [{"ticker": "MU", "role": "position"}]
    assert payload["_meta"]["deprecated_endpoint"] is True
    assert payload["_meta"]["replacement"] == "/api/v1/portfolio?timeframe={timeframe}"
