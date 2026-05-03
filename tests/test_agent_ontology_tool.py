from __future__ import annotations

import json

from api.agent_tools import TOOL_DEFINITIONS, execute_tool
from ontology.action_registry import get_tool_exposure
from ontology.policy import PolicyDenied, admin_actor, agent_actor


def test_query_ontology_tool_registered():
    names = {tool.get("name") for tool in TOOL_DEFINITIONS}
    assert "query_ontology" in names
    assert "get_signal_aggregator" in names


def test_query_ontology_tool_policy_spec_registered():
    exposure = get_tool_exposure("query_ontology")

    assert exposure.policy_spec is not None
    assert exposure.policy_spec.ontology_actions == ("query",)
    assert set(exposure.policy_spec.dynamic_ontology_actions({"include_graph": True, "refresh_snapshot": True})) == {
        "graph.read",
        "snapshot.refresh",
    }


def test_query_ontology_tool_dispatch(monkeypatch):
    def fake_query(
        self,
        query,
        intent,
        filters,
        timeframe,
        include_graph,
        run_id,
        refresh_snapshot=False,
        page=1,
        page_size=25,
    ):
        return {
            "run_id": run_id or "run-1",
            "intent": intent or "portfolio_risk_exposure",
            "interpreted_query": {
                "source": "structured" if intent else "deterministic_fallback",
                "query": query,
                "filters": filters,
            },
            "as_of": "2026-03-08T00:00:00Z",
            "source_status": {"portfolio": {"status": "ok"}},
            "results": [],
            "aggregate": {
                "position_count": 0,
                "risk_buckets": {"high": 0, "medium": 0, "low": 0},
                "asset_exposure_counts": {},
                "average_risk_score": 0.0,
                "confidence": 1.0,
            },
        }

    monkeypatch.setattr("ontology.service.OntologyQueryService.query", fake_query)

    raw = execute_tool(
        "query_ontology",
        {
            "query": "Show me my portfolio risk exposure",
            "timeframe": "Daily",
            "page_size": 5,
        },
    )

    payload = json.loads(raw)
    assert payload["run_id"] == "run-1"
    assert payload["intent"] == "portfolio_risk_exposure"
    assert "source_status" in payload
    assert "aggregate" in payload


def test_query_ontology_string_filters(monkeypatch):
    """Filters passed as a JSON string should be parsed into a dict."""
    captured: dict = {}
    monkeypatch.setattr("api.agent_tools.get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr("api.agent_tools.set_cached", lambda *args, **kwargs: None)

    def fake_query(
        self,
        query,
        intent,
        filters,
        timeframe,
        include_graph,
        run_id,
        refresh_snapshot=False,
        page=1,
        page_size=25,
    ):
        captured["filters"] = filters
        return {
            "run_id": run_id or "run-1",
            "intent": intent or "portfolio_risk_exposure",
            "interpreted_query": {"source": "structured", "query": query, "filters": filters},
            "as_of": "2026-03-08T00:00:00Z",
            "source_status": {"portfolio": {"status": "ok"}},
            "results": [],
            "aggregate": {
                "position_count": 0,
                "risk_buckets": {"high": 0, "medium": 0, "low": 0},
                "asset_exposure_counts": {},
                "average_risk_score": 0.0,
                "confidence": 1.0,
            },
        }

    monkeypatch.setattr("ontology.service.OntologyQueryService.query", fake_query)

    raw = execute_tool(
        "query_ontology",
        {
            "query": "Show risk for AAPL",
            "filters": json.dumps({"tickers": ["AAPL"]}),
        },
    )

    payload = json.loads(raw)
    assert payload["run_id"] == "run-1"
    # The handler should have parsed the JSON string into a dict
    assert captured["filters"] == {"tickers": ["AAPL"]}


def test_query_ontology_tool_fails_closed_on_invalid_filters_json(monkeypatch):
    monkeypatch.setattr("api.agent_tools.get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr("api.agent_tools.set_cached", lambda *args, **kwargs: None)

    called = {"query": False}

    def fake_query(
        self,
        query,
        intent,
        filters,
        timeframe,
        include_graph,
        run_id,
        refresh_snapshot=False,
        page=1,
        page_size=25,
    ):
        called["query"] = True
        return {"run_id": "run-1", "intent": "portfolio_risk_exposure", "results": []}

    monkeypatch.setattr("ontology.service.OntologyQueryService.query", fake_query)

    payload = json.loads(
        execute_tool(
            "query_ontology",
            {
                "query": "Show risk for AAPL",
                "filters": "{not-json",
            },
        )
    )

    assert called["query"] is False
    assert payload["_meta"]["status"] == "error"
    assert "query_ontology.filters must be a valid JSON object" in payload["error"]


def test_query_ontology_tool_returns_structured_policy_denial_for_graph_or_refresh(monkeypatch):
    monkeypatch.setattr("api.agent_tools.get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr("api.agent_tools.set_cached", lambda *args, **kwargs: None)

    def fake_query(
        self,
        query,
        intent,
        filters,
        timeframe,
        include_graph,
        run_id,
        refresh_snapshot=False,
        page=1,
        page_size=25,
    ):
        if include_graph:
            raise PolicyDenied("graph denied")
        if refresh_snapshot:
            raise PolicyDenied("refresh denied")
        return {"run_id": "run-1", "intent": "portfolio_risk_exposure", "results": []}

    monkeypatch.setattr("ontology.service.OntologyQueryService.query", fake_query)

    graph_payload = json.loads(
        execute_tool("query_ontology", {"include_graph": True}, actor=agent_actor(admin_actor()))
    )
    refresh_payload = json.loads(
        execute_tool("query_ontology", {"refresh_snapshot": True}, actor=agent_actor(admin_actor("owner")))
    )

    assert graph_payload["type"] == "PolicyDenied"
    assert graph_payload["detail"] == "graph denied"
    assert graph_payload["_meta"]["status"] == "denied"

    assert refresh_payload["type"] == "PolicyDenied"
    assert refresh_payload["detail"] == "refresh denied"
    assert refresh_payload["_meta"]["status"] == "denied"


def test_query_ontology_tool_cache_is_actor_scoped(monkeypatch):
    store: dict[tuple[int, str], dict] = {}

    monkeypatch.setattr("api.agent_tools.get_cached", lambda cache, key: store.get((id(cache), key)))
    monkeypatch.setattr(
        "api.agent_tools.set_cached",
        lambda cache, key, value: store.__setitem__((id(cache), key), value),
    )

    calls: list[str] = []

    def fake_query(
        self,
        query,
        intent,
        filters,
        timeframe,
        include_graph,
        run_id,
        refresh_snapshot=False,
        page=1,
        page_size=25,
        actor=None,
    ):
        calls.append(actor.actor_id if actor is not None else "missing")
        return {
            "run_id": run_id or "run-1",
            "intent": intent or "portfolio_risk_exposure",
            "interpreted_query": {"source": "structured", "query": query, "filters": filters},
            "as_of": "2026-03-08T00:00:00Z",
            "source_status": {"portfolio": {"status": "ok"}},
            "results": [],
            "aggregate": {
                "position_count": 0,
                "risk_buckets": {"high": 0, "medium": 0, "low": 0},
                "asset_exposure_counts": {},
                "average_risk_score": 0.0,
                "confidence": 1.0,
            },
        }

    monkeypatch.setattr("ontology.service.OntologyQueryService.query", fake_query)

    actor_a = agent_actor(admin_actor("alice"))
    actor_b = agent_actor(admin_actor("bob"))

    execute_tool("query_ontology", {"query": "Show risk"}, actor=actor_a)
    execute_tool("query_ontology", {"query": "Show risk"}, actor=actor_a)
    execute_tool("query_ontology", {"query": "Show risk"}, actor=actor_b)

    assert calls == ["agent:alice", "agent:bob"]


def test_query_ontology_tool_preserves_run_pagination_and_graph_meta_for_large_payloads(monkeypatch):
    monkeypatch.setattr("api.agent_tools.get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr("api.agent_tools.set_cached", lambda *args, **kwargs: None)

    def fake_query(
        self,
        query,
        intent,
        filters,
        timeframe,
        include_graph,
        run_id,
        refresh_snapshot=False,
        page=1,
        page_size=25,
    ):
        return {
            "run_id": "run-large",
            "intent": "portfolio_risk_exposure",
            "interpreted_query": {"source": "structured", "query": query, "entity": None, "filters": filters},
            "as_of": "2026-03-08T00:00:00Z",
            "source_status": {"portfolio": {"status": "ok"}},
            "results": [
                {
                    "ticker": f"TICK{i:02d}",
                    "asset": "equity",
                    "direction": "long",
                    "sector": "Information Technology",
                    "risk_score": 0.9,
                    "risk_level": "high",
                    "evidence": [{"name": f"Signal {i}", "contribution": 0.2}],
                }
                for i in range(40)
            ],
            "aggregate": {
                "position_count": 40,
                "risk_buckets": {"high": 40, "medium": 0, "low": 0},
                "asset_exposure_counts": {"equity": 40},
                "average_risk_score": 0.9,
                "confidence": 1.0,
            },
            "graph": {
                "nodes": [{"id": f"position:TICK{i:02d}", "type": "Position"} for i in range(12)],
                "edges": [
                    {
                        "source_id": f"position:TICK{i:02d}",
                        "target_id": f"asset:TICK{i:02d}",
                        "relation_type": "references_asset",
                    }
                    for i in range(12)
                ],
            },
            "_meta": {
                "pagination": {
                    "page": 2,
                    "page_size": 7,
                    "returned_results": 7,
                    "total_results": 40,
                    "total_pages": 6,
                    "has_prev": True,
                    "has_next": True,
                    "sort": "risk_score_desc_then_position_id_asc",
                    "exact_total": True,
                },
                "graph": {"scope": "page", "node_count": 12, "edge_count": 12},
            },
        }

    monkeypatch.setattr("ontology.service.OntologyQueryService.query", fake_query)

    payload = json.loads(execute_tool("query_ontology", {"query": "Show risk", "include_graph": True}))

    assert payload["run_id"] == "run-large"
    assert payload["_meta"]["pagination"]["page"] == 2
    assert payload["_meta"]["pagination"]["page_size"] == 7
    assert payload["_meta"]["graph"]["scope"] == "page"
    assert payload["_meta"]["graph"]["node_count"] == 12
    assert payload["graph"]["node_count"] == 12
    assert len(payload["results"]) == 25


def test_signal_aggregator_tool_dispatch(monkeypatch):
    def fake_build(lookback_weeks, positioning_instruments, include_raw_modules, include_history=False):
        assert lookback_weeks == 104
        assert positioning_instruments == "SP500,EUR"
        assert include_raw_modules is False
        return {
            "status": "ok",
            "as_of": "2026-03-08",
            "regime": {
                "label": "transitional",
                "score": 51.2,
                "confidence": 1.0,
                "history_percentile": 58.4,
            },
            "weights": {"configured": {}, "effective": {}},
            "factors": [],
            "module_status": {},
            "failed_modules": [],
            "history": {
                "frequency": "weekly",
                "lookback_weeks": 104,
                "coverage": {},
                "series": [],
                "episodes": [],
            },
        }

    monkeypatch.setattr("api.signal_aggregator.build_signal_aggregator", fake_build)
    monkeypatch.setattr("api.agent_tools.get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr("api.agent_tools.set_cached", lambda *args, **kwargs: None)

    raw = execute_tool(
        "get_signal_aggregator",
        {
            "lookback_weeks": 104,
            "positioning_instruments": "SP500,EUR",
        },
    )
    payload = json.loads(raw)
    assert payload["status"] == "ok"
    assert payload["regime"]["label"] == "transitional"
