from __future__ import annotations

import pytest


def test_idea_create_defaults_to_equity_security():
    from api.routers.ideas import IdeaCreateRequest

    req = IdeaCreateRequest(ticker="aapl")

    assert req.ticker == "AAPL"
    assert req.asset == "equity"
    assert req.instrument_type == "security"
    assert req.price_symbol == "AAPL"
    assert req.contract_multiplier == 1.0


def test_idea_create_canonicalizes_spot_fx():
    from api.routers.ideas import IdeaCreateRequest

    req = IdeaCreateRequest(ticker="eur/usd", instrument_type="spot_fx")

    assert req.ticker == "EURUSD=X"
    assert req.asset == "fx"
    assert req.instrument_type == "spot_fx"
    assert req.price_symbol == "EURUSD=X"
    assert req.fx_base_currency == "EUR"
    assert req.fx_quote_currency == "USD"
    assert req.currency == "USD"
    assert req.exchange == "FX"


def test_idea_create_known_future_infers_asset_and_multiplier():
    from api.routers.ideas import IdeaCreateRequest

    req = IdeaCreateRequest(ticker="cl=f", instrument_type="future")

    assert req.ticker == "CL=F"
    assert req.asset == "commodity"
    assert req.instrument_type == "future"
    assert req.price_symbol == "CL=F"
    assert req.contract_multiplier == 1000.0


def test_idea_create_unknown_future_requires_multiplier():
    from api.routers.ideas import IdeaCreateRequest

    with pytest.raises(ValueError, match="contract multiplier"):
        IdeaCreateRequest(ticker="ABC=F", instrument_type="future")

    req = IdeaCreateRequest(ticker="ABC=F", instrument_type="future", contract_multiplier=25, asset="commodity")
    assert req.contract_multiplier == 25
    assert req.asset == "commodity"


def test_legacy_idea_instrument_defaults():
    from api.routers.ideas import _with_default_idea_instrument_fields

    idea = _with_default_idea_instrument_fields({"id": "investment_idea:TLT", "ticker": "TLT"})

    assert idea["asset"] == "equity"
    assert idea["instrument_type"] == "security"
    assert idea["price_symbol"] == "TLT"
    assert idea["contract_multiplier"] == 1.0


def test_ontology_investment_idea_accepts_instrument_fields_and_rejects_unknown():
    from pydantic import ValidationError

    from ontology.schemas.objects import InvestmentIdea

    idea = InvestmentIdea(
        idea_id="investment_idea:CL",
        ticker="CL=F",
        asset="commodity",
        instrument_type="future",
        price_symbol="CL=F",
        contract_multiplier=1000,
        exchange="NYMEX",
    )

    assert idea.asset == "commodity"
    assert idea.instrument_type == "future"
    assert idea.price_symbol == "CL=F"
    with pytest.raises(ValidationError):
        InvestmentIdea(idea_id="investment_idea:CL", ticker="CL=F", unknown_field=True)


def test_analyzer_synthetic_idea_rows_preserve_multi_asset_metadata(monkeypatch):
    from portfolio.portfolio_optimizer import portfolio_analyzer

    class FakeReadService:
        def list_objects(self, object_type, limit=500):
            assert object_type == "InvestmentIdea"
            return [
                {
                    "id": "investment_idea:EURUSD",
                    "ticker": "EURUSD=X",
                    "status": "watching",
                    "metadata": {"analyzer_direction": "long"},
                    "asset": "fx",
                    "instrument_type": "spot_fx",
                    "price_symbol": "EURUSD=X",
                    "contract_multiplier": 1.0,
                    "fx_base_currency": "EUR",
                    "fx_quote_currency": "USD",
                    "currency": "USD",
                    "exchange": "FX",
                }
            ]

    import ontology.runtime_read_service as runtime_read_service

    monkeypatch.setattr(runtime_read_service, "OntologyRuntimeReadService", FakeReadService)

    rows = portfolio_analyzer._synthetic_idea_rows(set())

    assert rows == [
        {
            "ticker": "EURUSD=X",
            "asset": "fx",
            "instrument_type": "spot_fx",
            "direction": "long",
            "quantity": 0,
            "contract_multiplier": 1.0,
            "price_symbol": "EURUSD=X",
            "fx_base_currency": "EUR",
            "fx_quote_currency": "USD",
            "currency": "USD",
            "country": "",
            "exchange": "FX",
            "source_type": "idea",
            "source_id": "investment_idea:EURUSD",
            "idea_id": "investment_idea:EURUSD",
            "company_name": "",
        }
    ]


def test_non_equity_deterministic_evaluation_skips_equity_only_missing_inputs():
    from api.routers import ideas as ideas_router

    context = {
        "idea": {
            "id": "investment_idea:EURUSD",
            "ticker": "EURUSD=X",
            "asset": "fx",
            "instrument_type": "spot_fx",
            "price_symbol": "EURUSD=X",
            "metadata": {"analyzer_direction": "long"},
            "user_notes": "Dollar downside if growth slows and carry narrows.",
        },
        "ticker": "EURUSD=X",
        "instrument": {
            "ticker": "EURUSD=X",
            "asset": "fx",
            "instrument_type": "spot_fx",
            "price_symbol": "EURUSD=X",
            "contract_multiplier": 1.0,
            "fx_base_currency": "EUR",
            "fx_quote_currency": "USD",
            "currency": "USD",
            "exchange": "FX",
        },
        "asset": "fx",
        "instrument_type": "spot_fx",
        "overview_content": "",
        "thesis_content": "",
        "management_quality_content": "",
        "signal_aggregator": {"ok": True, "data": {"regime": "risk-on"}},
        "industry_monitor": {"ok": True, "skipped": True},
        "dossier": {"ok": True, "skipped": True},
        "tool_errors": [],
        "analyzer_context": {"status": "inactive", "ticker": "EURUSD=X"},
        "use_portfolio_context": True,
        "evaluated_at": "2026-05-15T00:00:00+00:00",
    }

    result = ideas_router._deterministic_evaluation(context)
    missing_fields = {row["field"] for row in result["missing_information"]}

    assert "overview" not in missing_fields
    assert "management_quality" not in missing_fields
    assert set(result["factor_scores"]) == set(ideas_router.CANONICAL_IDEA_FACTORS)
    assert result["factor_scores"]["management_quality"]["status"] == "not_applicable"
    assert result["data_quality"]["instrument"]["asset"] == "fx"
