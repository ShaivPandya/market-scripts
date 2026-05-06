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
            "historical_bands": {},
            "composite_score": {"value": None, "status": "missing", "components": {}},
            "data_quality": {"status": "ok", "usable_metric_count": 0, "warnings": [], "metric_statuses": {}},
        },
    )

    resp = auth_client.get("/api/v1/valuation/ZZVALUATION")

    assert resp.status_code == 200
    assert resp.json()["ticker"] == "ZZVALUATION"


def test_update_position_valuation_profile_override_endpoint(auth_client, monkeypatch):
    monkeypatch.setattr(multiples, "read_profile_override", lambda ticker: None)
    monkeypatch.setattr(
        multiples,
        "write_profile_override",
        lambda ticker, profile_id: {"ticker": ticker, "profile_override": profile_id},
    )

    resp = auth_client.put("/api/v1/valuation/ZZVALUATION/profile-override", json={"profile_id": "bank_financial"})

    assert resp.status_code == 200
    assert resp.json()["profile_override"] == "bank_financial"
