from __future__ import annotations

from ontology.models import InterpretedQuery
from ontology.parser import parse_hybrid_query


def test_parser_uses_structured_intent_when_provided():
    parsed = parse_hybrid_query(
        query="Show me risk",
        intent="portfolio_risk_exposure",
        filters={"tickers": ["MU"]},
    )
    assert isinstance(parsed, InterpretedQuery)
    assert parsed.source == "structured"
    assert parsed.intent == "portfolio_risk_exposure"
    assert parsed.filters.get("tickers") == ["MU"]


def test_parser_falls_back_to_deterministic_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr("ontology.parser._parse_with_llm", lambda q: None)

    parsed = parse_hybrid_query(
        query="Which positions are in sectors with deteriorating macro conditions?",
        intent=None,
        filters=None,
    )

    assert parsed.source == "deterministic_fallback"
    assert parsed.intent == "positions_in_deteriorating_macro"


def test_parser_uses_llm_parse_when_available(monkeypatch):
    def fake_llm(_query: str):
        return {
            "intent": "portfolio_risk_exposure",
            "filters": {"tickers": ["CRWD"], "min_risk_score": 0.55},
            "entity": "CRWD",
        }

    monkeypatch.setattr("ontology.parser._parse_with_llm", fake_llm)

    parsed = parse_hybrid_query(
        query="Show my CRWD risk exposure",
        intent=None,
        filters=None,
    )

    assert parsed.source == "llm"
    assert parsed.intent == "portfolio_risk_exposure"
    assert parsed.entity == "CRWD"
    assert parsed.filters.get("tickers") == ["CRWD"]
    assert parsed.filters.get("min_risk_score") == 0.55
