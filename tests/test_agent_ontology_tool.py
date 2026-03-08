from __future__ import annotations

import json

from api.agent_tools import TOOL_DEFINITIONS, execute_tool


def test_query_ontology_tool_registered():
    names = {tool.get("name") for tool in TOOL_DEFINITIONS}
    assert "query_ontology" in names
    assert "get_signal_aggregator" in names


def test_query_ontology_tool_dispatch(monkeypatch):
    def fake_query(self, query, intent, filters, timeframe, include_graph, run_id, refresh_snapshot=False):
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
            "filters": {"max_results": 5},
        },
    )

    payload = json.loads(raw)
    assert payload["run_id"] == "run-1"
    assert payload["intent"] == "portfolio_risk_exposure"
    assert "source_status" in payload
    assert "aggregate" in payload


def test_signal_aggregator_tool_dispatch(monkeypatch):
    def fake_build(lookback_weeks, positioning_instruments, include_raw_modules):
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
