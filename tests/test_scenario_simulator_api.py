from __future__ import annotations


def _api_payload() -> dict:
    return {
        "portfolio": {
            "portfolio_id": "default-portfolio",
            "account_id": "default-account",
            "base_currency": "USD",
            "book_value": 10000,
            "positions": [{"ticker": "MU", "direction": "long", "notional_base": 1000}],
        },
        "position": {
            "ticker": "MU",
            "direction": "long",
            "quantity": 10,
            "current_price": 100,
            "notional_base": 1000,
            "average_daily_volume_notional": 500,
            "position_uid": "position:MU",
        },
        "candidates": [{"action": "add", "delta": {"notional_base": 500}}],
        "scenarios": [{"scenario_id": "base", "name": "Base", "price_move_pct": 5, "probability": 1}],
    }


def test_scenario_simulator_requires_auth(client):
    response = client.post("/api/scenario-simulator/evaluate", json=_api_payload())

    assert response.status_code == 401


def test_scenario_simulator_preview_default_does_not_persist(auth_client, monkeypatch):
    import api.routers.scenario_simulator as route

    class UnexpectedWriteback:
        def __init__(self):
            raise AssertionError("writeback should not be constructed for preview requests")

    monkeypatch.setattr(route, "DecisionOntologyWriteback", UnexpectedWriteback)

    response = auth_client.post("/api/scenario-simulator/evaluate", json=_api_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["persisted"] is False
    assert body["outcomes"][0]["artifact_ids"] == {}
    assert body["outcomes"][0]["course_of_action_id"].startswith("course_of_action:")


def test_scenario_simulator_rejects_hedge(auth_client):
    payload = _api_payload()
    payload["candidates"] = [{"action": "hedge"}]

    response = auth_client.post("/api/scenario-simulator/evaluate", json=payload)

    assert response.status_code == 422
    assert "Hedging is out of scope" in response.text


def test_scenario_simulator_persist_returns_artifact_ids(auth_client, monkeypatch):
    import api.routers.scenario_simulator as route

    class FakeWriteback:
        def record_scenario_simulation(self, **kwargs):
            assert kwargs["simulation"]["persisted"] is False
            return {
                "artifact_ids": {"comparison_id": "course_of_action_comparison:sim"},
                "outcome_artifact_ids": {
                    "candidate:1": {
                        "course_of_action_id": "course_of_action:sim_add",
                        "simulated_outcome_ids": ["simulated_outcome:sim_add_base"],
                    }
                },
            }

    monkeypatch.setattr(route, "DecisionOntologyWriteback", lambda: FakeWriteback())
    payload = _api_payload()
    payload["persist"] = True

    response = auth_client.post("/api/scenario-simulator/evaluate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["persisted"] is True
    assert body["artifact_ids"]["comparison_id"] == "course_of_action_comparison:sim"
    assert body["outcomes"][0]["artifact_ids"]["course_of_action_id"] == "course_of_action:sim_add"


def test_scenario_simulator_enrich_from_risk_snapshot(auth_client, monkeypatch):
    import api.routers.scenario_simulator as route

    def fake_latest(ticker: str):
        assert ticker == "MU"
        return {
            "result_id": "position-risk:MU:unit",
            "risk_score": 0.42,
            "component_scores": {
                "volatility_cluster": 0.12,
                "breadth_stress": 0.18,
                "sector_stress": 0.22,
                "macro_regime": 0.15,
            },
        }

    monkeypatch.setattr(route, "get_latest_position_risk", fake_latest)
    monkeypatch.setattr(route, "get_latest_portfolio_risk", lambda: None)

    payload = _api_payload()
    payload["scenarios"] = [{"scenario_id": "base", "name": "Base", "price_move_pct": 5, "probability": 1}]
    payload["enrich_from_risk_snapshot"] = True

    response = auth_client.post("/api/scenario-simulator/evaluate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["risk_provenance"]["position_risk_snapshot_id"] == "position-risk:MU:unit"
    assert body["outcomes"][0]["scenario_outcomes"][0]["thesis_pressure"] == 0.42
