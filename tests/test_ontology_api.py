from __future__ import annotations


class _FakeService:
    def __init__(self, payload):
        self.payload = payload

    def query(self, query, intent, filters, timeframe, include_graph):
        out = dict(self.payload)
        out.setdefault("intent", intent or "portfolio_risk_exposure")
        out.setdefault(
            "interpreted_query",
            {
                "source": "structured" if intent else "deterministic_fallback",
                "query": query,
                "entity": None,
                "filters": filters,
            },
        )
        out.setdefault("as_of", "2026-03-08T00:00:00Z")
        out.setdefault("source_status", {"portfolio": {"status": "ok"}})
        out.setdefault("results", [])
        out.setdefault(
            "aggregate",
            {
                "position_count": 0,
                "risk_buckets": {"high": 0, "medium": 0, "low": 0},
                "asset_exposure_counts": {},
                "average_risk_score": 0.0,
                "confidence": 1.0,
            },
        )
        return out


def test_ontology_query_requires_auth(client):
    resp = client.post("/api/v1/ontology/query", json={"intent": "portfolio_risk_exposure"})
    assert resp.status_code == 401


def test_ontology_query_structured_returns_schema(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    fake = _FakeService(
        {
            "results": [
                {
                    "ticker": "MU",
                    "asset": "equity",
                    "direction": "long",
                    "sector": "Information Technology",
                    "risk_score": 0.72,
                    "risk_level": "medium",
                    "evidence": [],
                }
            ]
        }
    )
    monkeypatch.setattr(ontology_router, "_service", fake)

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={
            "intent": "portfolio_risk_exposure",
            "filters": {"tickers": ["MU"], "min_risk_score": 0.5},
            "timeframe": "Daily",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "portfolio_risk_exposure"
    assert data["interpreted_query"]["source"] == "structured"
    assert isinstance(data["source_status"], dict)
    assert isinstance(data["results"], list)
    assert "aggregate" in data


def test_ontology_query_nl_path_returns_interpreted_source(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    fake = _FakeService(
        {
            "intent": "positions_in_deteriorating_macro",
            "interpreted_query": {
                "source": "deterministic_fallback",
                "query": "Which positions are in deteriorating macro conditions?",
                "entity": None,
                "filters": {},
            },
        }
    )
    monkeypatch.setattr(ontology_router, "_service", fake)

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"query": "Which positions are in deteriorating macro conditions?"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "positions_in_deteriorating_macro"
    assert data["interpreted_query"]["source"] in {"llm", "deterministic_fallback", "structured"}


def test_ontology_query_partial_failure_returns_200_with_degraded_confidence(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    fake = _FakeService(
        {
            "source_status": {
                "portfolio": {"status": "ok"},
                "vix_term_structure": {"status": "error", "detail": "timeout"},
            },
            "aggregate": {
                "position_count": 1,
                "risk_buckets": {"high": 1, "medium": 0, "low": 0},
                "asset_exposure_counts": {"equity": 1},
                "average_risk_score": 0.81,
                "confidence": 0.62,
            },
        }
    )
    monkeypatch.setattr(ontology_router, "_service", fake)

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["aggregate"]["confidence"] < 1.0
    assert data["source_status"]["vix_term_structure"]["status"] in {"error", "partial"}
