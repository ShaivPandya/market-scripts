from __future__ import annotations

import pytest

from ontology.sources.source_registry import (
    SourceRegistryEntry,
    all_source_registry_entries,
    source_registry_metadata,
    source_registry_metadata_for_snapshot,
    validate_source_registry,
)


def test_source_registry_entries_validate_and_include_core_domains():
    validate_source_registry()
    entries = all_source_registry_entries()

    assert entries["market_breadth"].dataset_domain == "market"
    assert entries["market_breadth"].authority_rank == 1
    assert entries["financials"].fallback_source_id == "yfinance_fundamentals"
    assert entries["portfolio_news_digest"].freshness_sla_seconds is None
    assert entries["source_ingestion_document"].vendor_name == "user_upload"


def test_source_registry_derives_reliability_tiers():
    entries = all_source_registry_entries()

    assert source_registry_metadata("market_breadth")["reliability_tier"] == "critical"
    assert source_registry_metadata("momentum")["reliability_tier"] == "supplemental"
    assert source_registry_metadata("market_regime")["reliability_tier"] == "standard"
    assert source_registry_metadata("portfolio_news_digest")["reliability_tier"] == "ad_hoc"
    assert entries["market_breadth"].reliability_tier is None


def test_source_registry_rejects_invalid_fallbacks():
    bad = {
        "primary": SourceRegistryEntry(
            source_id="primary",
            vendor_name="vendor",
            dataset_domain="market",
            authority_rank=1,
            freshness_sla_seconds=0,
            required=True,
            fallback_source_id="missing",
            reliability_tier=None,
        )
    }

    with pytest.raises(ValueError, match="fallback_source_id missing"):
        validate_source_registry(bad)


def test_source_registry_snapshot_lookup():
    from api.snapshot_keys import SNAPSHOT_MARKET_BREADTH

    metadata = source_registry_metadata_for_snapshot(SNAPSHOT_MARKET_BREADTH)

    assert metadata is not None
    assert metadata["source_id"] == "market_breadth"
    assert source_registry_metadata("financials")["dataset_domain"] == "fundamental"


def test_fundamental_momentum_result_includes_source_registry(monkeypatch):
    from api.routers import fundamental_momentum as fm
    from portfolio.momentum.fundamental_momentum import eps_screen

    monkeypatch.setattr(fm, "_resolve_tickers", lambda _req: ["AAA"])
    monkeypatch.setattr(fm, "_resolve_benchmark", lambda _req: "self")
    monkeypatch.setattr(eps_screen, "get_data", lambda **_kwargs: {"results_df": []})

    result = fm._compute_fundamental_momentum(
        fm.FMRequest(screen_type="EPS", input_mode="Custom Tickers", tickers="AAA")
    )

    assert result["_meta"]["source_registry"]["source_id"] == "fundamental_momentum"


def test_financials_route_includes_source_registry(monkeypatch):
    from api.cache import invalidate_all
    from api.routers.financials import FinancialsRequest, run_financials
    from portfolio.momentum.fundamental_momentum import financials_single

    invalidate_all()
    monkeypatch.setattr(financials_single, "get_data", lambda _ticker: {"ticker": "AAA", "data_source": "sec_edgar"})

    result = run_financials(FinancialsRequest(ticker="AAA"))

    assert result["_meta"]["source_registry"]["source_id"] == "financials"


def test_portfolio_news_list_includes_source_registry(monkeypatch):
    from api.routers import portfolio_news
    from portfolio import news_digests

    monkeypatch.setattr(news_digests, "list_digests", lambda: {"digests": []})

    result = portfolio_news.list_portfolio_news()

    assert result["_meta"]["source_registry"]["source_id"] == "portfolio_news_digest"
