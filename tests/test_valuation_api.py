from api.routers import valuation as valuation_router
from equities.valuation import multiples


def test_get_position_valuation_endpoint(auth_client, monkeypatch):
    monkeypatch.setattr(multiples, "read_profile_override", lambda ticker: None)
    monkeypatch.setattr(
        multiples,
        "get_position_valuation",
        lambda ticker: {
            "ticker": ticker,
            "company_name": "Test Co",
            "market_data": {},
            "profile": {
                "id": "general_equity",
                "label": "General Equity",
                "selection_mode": "auto",
                "weights": {},
                "effective_weights": {},
                "options": [],
            },
            "metrics": {},
            "peer_context": {"source": "mock", "peer_count": 0, "peers": [], "metric_stats": {}},
            "composite_score": {"value": None, "status": "missing", "components": {}},
            "data_quality": {"status": "ok", "usable_metric_count": 0, "warnings": [], "metric_statuses": {}},
            "value_range": {
                "saved": False,
                "metric": "price_sales",
                "scenarios": {
                    "bear": {"multiple": 1.0, "denominator": 100.0, "expected_price": 10.0, "percent_change": 0.0}
                },
            },
        },
    )

    resp = auth_client.get("/api/v1/valuation/ZZVALUATION")

    assert resp.status_code == 200
    assert resp.json()["ticker"] == "ZZVALUATION"
    assert resp.json()["value_range"]["metric"] == "price_sales"


def test_update_position_valuation_profile_override_endpoint(auth_client, monkeypatch):
    deleted_keys = []

    monkeypatch.setattr(multiples, "read_profile_override", lambda ticker: None)
    monkeypatch.setattr(
        multiples,
        "write_profile_override",
        lambda ticker, profile_id: {"ticker": ticker, "profile_override": profile_id},
    )
    monkeypatch.setattr(valuation_router, "delete_cached", lambda cache, key: deleted_keys.append(key))

    resp = auth_client.put("/api/v1/valuation/ZZVALUATION/profile-override", json={"profile_id": "bank_financial"})

    assert resp.status_code == 200
    assert resp.json()["profile_override"] == "bank_financial"
    assert deleted_keys
    assert all(key.startswith("position_valuation:") for key in deleted_keys)
    assert all(not key.startswith(("valuation_current:", "valuation_peer_row:")) for key in deleted_keys)


def test_update_position_valuation_value_range_endpoint(auth_client, monkeypatch):
    deleted_keys = []

    monkeypatch.setattr(multiples, "read_profile_override", lambda ticker: None)
    monkeypatch.setattr(valuation_router, "delete_cached", lambda cache, key: deleted_keys.append(key))

    def _write(ticker, payload):
        return {"ticker": ticker, "value_range": payload}

    monkeypatch.setattr(multiples, "write_value_range_assumption", _write)

    resp = auth_client.put(
        "/api/v1/valuation/ZZVALUATION/value-range",
        json={
            "metric": "price_sales",
            "scenarios": {
                "bear": {"multiple": 4.0, "denominator": 1000000000},
                "base": {"multiple": 6.0, "denominator": 1200000000},
                "bull": {"multiple": 8.0, "denominator": 1400000000},
            },
        },
    )

    assert resp.status_code == 200
    assert resp.json()["ticker"] == "ZZVALUATION"
    assert resp.json()["value_range"]["metric"] == "price_sales"
    assert resp.json()["value_range"]["scenarios"]["bull"]["multiple"] == 8.0
    assert deleted_keys
    assert all(key.startswith("position_valuation:") for key in deleted_keys)
    assert all(not key.startswith(("valuation_current:", "valuation_peer_row:")) for key in deleted_keys)


def test_delete_position_valuation_value_range_endpoint(auth_client, monkeypatch):
    deleted_keys = []

    monkeypatch.setattr(multiples, "read_profile_override", lambda ticker: None)
    monkeypatch.setattr(valuation_router, "delete_cached", lambda cache, key: deleted_keys.append(key))

    def _delete(ticker, metric):
        return {"ticker": ticker, "value_range": {"selected_metric": "price_sales", "metric_assumptions": {}}}

    monkeypatch.setattr(multiples, "delete_value_range_assumption", _delete)

    resp = auth_client.delete("/api/v1/valuation/ZZVALUATION/value-range/price_sales")

    assert resp.status_code == 200
    assert resp.json()["ticker"] == "ZZVALUATION"
    assert resp.json()["value_range"]["metric_assumptions"] == {}
    assert deleted_keys
    assert all(key.startswith("position_valuation:") for key in deleted_keys)
    assert all(not key.startswith(("valuation_current:", "valuation_peer_row:")) for key in deleted_keys)
