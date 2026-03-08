from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

import ontology.service as svc
from ontology.models import InterpretedQuery
from ontology.service import OntologyQueryService, OntologyRunNotFoundError


@dataclass
class _FakeIngestion:
    run_id: str
    as_of: str
    source_status: dict
    required_modules: list[str]
    optional_modules: list[str]
    component_scores: dict


class _FakeRepo:
    def __init__(self):
        self.run_requests: list[str] = []
        self.latest_run_requests = 0
        self.latest_run: dict | None = None
        self.rows_by_run = {
            "run-live": [
                {
                    "position_id": "position:MU",
                    "position_props": {
                        "ticker": "MU",
                        "asset": "equity",
                        "direction": "long",
                        "risk_score": 0.72,
                        "risk_level": "medium",
                        "ontology_run_id": "run-live",
                    },
                    "sector_props": {"name": "Information Technology"},
                }
            ],
            "run-historical": [
                {
                    "position_id": "position:MU",
                    "position_props": {
                        "ticker": "MU",
                        "asset": "equity",
                        "direction": "long",
                        "risk_score": 0.81,
                        "risk_level": "high",
                        "ontology_run_id": "run-historical",
                    },
                    "sector_props": {"name": "Information Technology"},
                }
            ],
        }

    def get_run(self, run_id: str):
        self.run_requests.append(run_id)
        if run_id != "run-historical":
            return None
        return {
            "run_id": "run-historical",
            "as_of": "2026-03-08T00:00:00Z",
            "source_status": {"portfolio": {"status": "ok"}},
            "required_modules": ["portfolio"],
            "created_at": "2026-03-08 00:00:00",
        }

    def get_latest_run(self):
        self.latest_run_requests += 1
        return self.latest_run

    def fetch_snapshot_position_asset_sector_rows(self, run_id: str):
        return self.rows_by_run.get(run_id, [])

    def fetch_snapshot_position_signal_evidence(self, run_id: str, position_id: str):
        if run_id == "run-historical":
            return [{"signal_id": "signal:test", "edge_props": {"name": "signal", "contribution": 0.5}}]
        return [{"signal_id": "signal:test", "edge_props": {"name": "signal", "contribution": 0.2}}]

    def fetch_snapshot_all_position_signal_evidence(self, run_id: str):
        evidence = self.fetch_snapshot_position_signal_evidence(run_id, "")
        return {
            row["position_id"]: list(evidence) for row in (self.rows_by_run.get(run_id) or []) if row.get("position_id")
        }

    def fetch_snapshot_graph(self, run_id: str):
        return {"nodes": [{"id": "position:MU"}], "edges": []}


def _stub_parse(*_args, **_kwargs):
    return InterpretedQuery(
        intent="portfolio_risk_exposure",
        source="structured",
        filters={},
        entity=None,
        original_query=None,
    )


def test_query_without_run_id_ingests_and_returns_run(monkeypatch):
    repo = _FakeRepo()
    service = OntologyQueryService(repository=repo)

    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)
    monkeypatch.setattr(
        svc,
        "ingest_into_repository",
        lambda repo, timeframe, include_deep_modules: _FakeIngestion(
            run_id="run-live",
            as_of="2026-03-09T00:00:00Z",
            source_status={"portfolio": {"status": "ok"}},
            required_modules=["portfolio"],
            optional_modules=[],
            component_scores={},
        ),
    )

    resp = service.query(
        query="show risk",
        intent=None,
        filters={},
        timeframe="Daily",
        include_graph=False,
        run_id=None,
    )

    assert resp["run_id"] == "run-live"
    assert resp["as_of"] == "2026-03-09T00:00:00Z"
    assert len(resp["results"]) == 1
    assert repo.run_requests == []


def test_query_without_run_id_reuses_fresh_latest_snapshot(monkeypatch):
    repo = _FakeRepo()
    repo.latest_run = {
        "run_id": "run-live",
        "as_of": "2026-03-09T00:00:00Z",
        "source_status": {"portfolio": {"status": "ok"}},
        "required_modules": ["portfolio"],
        "created_at": (datetime.now(UTC) - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    service = OntologyQueryService(repository=repo)

    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)

    def _unexpected_ingest(*_args, **_kwargs):
        raise AssertionError("ingest should not run when a fresh latest run exists")

    monkeypatch.setattr(svc, "ingest_into_repository", _unexpected_ingest)

    resp = service.query(
        query="show risk",
        intent=None,
        filters={},
        timeframe="Daily",
        include_graph=False,
        run_id=None,
        refresh_snapshot=False,
    )

    assert resp["run_id"] == "run-live"
    assert repo.latest_run_requests == 1


def test_query_without_run_id_ingests_when_latest_is_stale(monkeypatch):
    repo = _FakeRepo()
    repo.latest_run = {
        "run_id": "run-live",
        "as_of": "2026-03-09T00:00:00Z",
        "source_status": {"portfolio": {"status": "ok"}},
        "required_modules": ["portfolio"],
        "created_at": (datetime.now(UTC) - timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    service = OntologyQueryService(repository=repo)

    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)

    called = {"ingested": False}

    def _fake_ingest(*_args, **_kwargs):
        called["ingested"] = True
        return _FakeIngestion(
            run_id="run-historical",
            as_of="2026-03-10T00:00:00Z",
            source_status={"portfolio": {"status": "ok"}},
            required_modules=["portfolio"],
            optional_modules=[],
            component_scores={},
        )

    monkeypatch.setattr(svc, "ingest_into_repository", _fake_ingest)

    resp = service.query(
        query="show risk",
        intent=None,
        filters={},
        timeframe="Daily",
        include_graph=False,
        run_id=None,
        refresh_snapshot=False,
    )

    assert called["ingested"] is True
    assert resp["run_id"] == "run-historical"


def test_query_with_refresh_snapshot_forces_ingestion(monkeypatch):
    repo = _FakeRepo()
    repo.latest_run = {
        "run_id": "run-live",
        "as_of": "2026-03-09T00:00:00Z",
        "source_status": {"portfolio": {"status": "ok"}},
        "required_modules": ["portfolio"],
        "created_at": (datetime.now(UTC) - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    service = OntologyQueryService(repository=repo)

    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)

    called = {"ingested": False}

    def _fake_ingest(*_args, **_kwargs):
        called["ingested"] = True
        return _FakeIngestion(
            run_id="run-historical",
            as_of="2026-03-10T00:00:00Z",
            source_status={"portfolio": {"status": "ok"}},
            required_modules=["portfolio"],
            optional_modules=[],
            component_scores={},
        )

    monkeypatch.setattr(svc, "ingest_into_repository", _fake_ingest)

    resp = service.query(
        query="show risk",
        intent=None,
        filters={},
        timeframe="Daily",
        include_graph=False,
        run_id=None,
        refresh_snapshot=True,
    )

    assert called["ingested"] is True
    assert resp["run_id"] == "run-historical"
    assert repo.latest_run_requests == 0


def test_query_without_run_id_ingests_when_latest_required_module_is_error(monkeypatch):
    repo = _FakeRepo()
    repo.latest_run = {
        "run_id": "run-live",
        "as_of": "2026-03-09T00:00:00Z",
        "source_status": {"portfolio": {"status": "error", "detail": "failed"}},
        "required_modules": ["portfolio"],
        "created_at": (datetime.now(UTC) - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    service = OntologyQueryService(repository=repo)

    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)
    called = {"ingested": False}

    def _fake_ingest(*_args, **_kwargs):
        called["ingested"] = True
        return _FakeIngestion(
            run_id="run-historical",
            as_of="2026-03-10T00:00:00Z",
            source_status={"portfolio": {"status": "ok"}},
            required_modules=["portfolio"],
            optional_modules=[],
            component_scores={},
        )

    monkeypatch.setattr(svc, "ingest_into_repository", _fake_ingest)

    resp = service.query(
        query="show risk",
        intent=None,
        filters={},
        timeframe="Daily",
        include_graph=False,
        run_id=None,
        refresh_snapshot=False,
    )

    assert called["ingested"] is True
    assert resp["run_id"] == "run-historical"


def test_query_with_run_id_replays_without_ingestion(monkeypatch):
    repo = _FakeRepo()
    service = OntologyQueryService(repository=repo)

    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)

    def _unexpected_ingest(*_args, **_kwargs):
        raise AssertionError("ingest should not run when run_id is provided")

    monkeypatch.setattr(svc, "ingest_into_repository", _unexpected_ingest)

    resp = service.query(
        query="show risk",
        intent=None,
        filters={},
        timeframe="Daily",
        include_graph=True,
        run_id="run-historical",
    )

    assert resp["run_id"] == "run-historical"
    assert resp["aggregate"]["position_count"] == 1
    assert "graph" in resp
    assert repo.run_requests == ["run-historical"]


def test_query_with_unknown_run_id_raises():
    repo = _FakeRepo()
    service = OntologyQueryService(repository=repo)

    with pytest.raises(OntologyRunNotFoundError):
        service.query(
            query="show risk",
            intent=None,
            filters={},
            timeframe="Daily",
            include_graph=False,
            run_id="missing",
        )
