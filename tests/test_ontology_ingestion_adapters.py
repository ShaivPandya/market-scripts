from __future__ import annotations

from pathlib import Path

import pytest

import ontology.ingestion as ingestion
import portfolio.core_db as core_db
from ontology.sources.base import LineageMetadata, SourceResult
from ontology.sources.dtos import (
    LaborMarketSnapshot,
    LiquiditySnapshot,
    MarketBreadthSnapshot,
    PortfolioMetadata,
    PortfolioPosition,
    PortfolioSnapshot,
    SectorMetricRow,
    SectorMetricsSnapshot,
    Top50BreadthSnapshot,
    VixTermStructureSnapshot,
)


@pytest.fixture(autouse=True)
def _temp_core_db(tmp_path, monkeypatch):
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    yield
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "_conn", None)


class _Repo:
    def __init__(self):
        self.saved = None

    def save_snapshot(self, **kwargs):
        self.saved = kwargs

    def prune_runs_older_than(self, *, days: int):
        return 0


class _Adapter:
    def __init__(self, name: str):
        self.source_name = name
        self.source_version = "test"
        self.required = True
        self.raw_module = "tests"
        self.raw_function = name
        self.parameters = {}


def _result(name, data, status="ok"):
    return SourceResult(
        data=data,
        status=status,
        quality="ok" if status == "ok" else "missing",
        fetched_at="2026-05-01T20:00:00+00:00",
        as_of="2026-05-01T20:00:00+00:00",
        lineage=LineageMetadata(
            raw_module="tests",
            raw_function=name,
            adapter=name,
            adapter_version="test",
            payload_fingerprint="abc",
            provenance_event_id=f"pv:source_adapter_run:test:{name}",
        ),
    )


def test_ingestion_uses_adapter_results(monkeypatch):
    metadata = PortfolioMetadata(ticker="MU", asset="equity", direction="long")
    portfolio = PortfolioSnapshot(
        positions={
            "MU": PortfolioPosition(
                ticker="MU",
                asset="equity",
                direction="long",
                latest_price=100.0,
                series_points=1,
                as_of="2026-05-01T20:00:00+00:00",
                metadata=metadata,
            )
        },
        timeframe="Daily",
        timestamp="2026-05-01T20:00:00+00:00",
    )
    results = {
        "portfolio": _result("portfolio", portfolio),
        "market_breadth": _result(
            "market_breadth",
            MarketBreadthSnapshot(500, 50, 45, 25, 11, "2026-05-01"),
        ),
        "top50_breadth": _result("top50_breadth", Top50BreadthSnapshot(40, 30, 20, 50)),
        "vix_term_structure": _result(
            "vix_term_structure",
            VixTermStructureSnapshot("2026-05-01", 20, 22, 1.1, "Neutral"),
        ),
        "sector_metrics": _result(
            "sector_metrics",
            SectorMetricsSnapshot(
                rows=[
                    SectorMetricRow(
                        "Information Technology",
                        30,
                        None,
                        -1,
                        None,
                        -4,
                        None,
                        -3,
                    )
                ],
                timestamp="2026-05-01T20:00:00+00:00",
            ),
        ),
        "liquidity": _result("liquidity", LiquiditySnapshot(-0.2, "normal", "2026-05-01")),
        "sentiment": _result("sentiment", None, status="partial"),
        "positioning_summary": _result("positioning_summary", None, status="partial"),
        "economic_growth": _result("economic_growth", None, status="partial"),
        "labor_market": _result(
            "labor_market",
            LaborMarketSnapshot(latest={}, timestamp="2026-05-01T20:00:00+00:00", initial_claims_change=5),
        ),
    }

    required = {name: _Adapter(name) for name in list(results)[:6]}
    optional = {name: _Adapter(name) for name in list(results)[6:]}
    monkeypatch.setattr(ingestion, "build_adapter_registry", lambda timeframe: (required, optional, {}))
    monkeypatch.setattr(ingestion, "run_adapters", lambda adapters: {name: results[name] for name in adapters})
    monkeypatch.setattr(ingestion, "_ingest_thesis_entities", lambda *args, **kwargs: None)

    repo = _Repo()
    out = ingestion.ingest_into_repository(repo=repo, timeframe="Daily", include_deep_modules=False)

    assert out.source_status["portfolio"]["source_version"] == "test"
    assert out.required_modules == list(required.keys())
    trace = core_db.get_provenance_trace(ontology_run_id=out.run_id)
    assert any(
        event["id"] == out.provenance_event_id and event["event_type"] == "ontology_run" for event in trace["events"]
    )
    record_kinds = {record["record_kind"] for record in trace["source_records"]}
    assert {"portfolio_position", "sector_metric", "snapshot"}.issubset(record_kinds)
    assert all(record["retention_class"] == "source_ref_90d" for record in trace["source_records"])
    assert repo.saved is not None
    positions = [node for node in repo.saved["nodes"] if node.id == "position:MU"]
    assert positions
    assert positions[0].properties["latest_price"] == 100.0


def test_ingestion_no_longer_imports_api_routers():
    text = Path(ingestion.__file__).read_text(encoding="utf-8")
    assert "api.routers" not in text
