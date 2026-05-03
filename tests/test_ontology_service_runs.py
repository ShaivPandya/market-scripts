from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

import ontology.service as svc
import portfolio.core_db as core_db
from ontology.models import InterpretedQuery
from ontology.policy import DefaultOntologyPolicy, OntologyAction, PolicyDecision, PolicyDenied, admin_actor
from ontology.service import OntologyQueryService, OntologyRunNotFoundError


@dataclass
class _FakeIngestion:
    run_id: str
    as_of: str
    source_status: dict
    required_modules: list[str]
    optional_modules: list[str]
    component_scores: dict


@pytest.fixture(autouse=True)
def _use_temp_audit_db(tmp_path, monkeypatch):
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "test_core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    yield
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "_conn", None)


class _Policy(DefaultOntologyPolicy):
    def __init__(
        self,
        *,
        denied_actions: set[str] | None = None,
        denied_objects: set[str] | None = None,
        fields: dict[str, set[str]] | None = None,
    ):
        self.denied_actions = denied_actions or set()
        self.denied_objects = denied_objects or set()
        self.fields = fields or {}

    def check_action(self, actor, action: str, context=None):
        if action in self.denied_actions:
            return PolicyDecision(False, f"denied action: {action}")
        return PolicyDecision(True)

    def check_object(self, actor, node, action: str = "read"):
        if node.id in self.denied_objects:
            return PolicyDecision(False, f"denied object: {node.id}")
        return PolicyDecision(True)

    def check_relationship(self, actor, edge, source=None, target=None, action: str = "read"):
        return PolicyDecision(True)

    def allowed_fields(self, actor, resource):
        key = getattr(resource, "type", None) or getattr(resource, "owner_type", None) or ""
        return self.fields.get(str(key))


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

    def fetch_snapshot_position_asset_sector_rows(self, run_id: str, *, schema_mode="upgraded"):
        return self.rows_by_run.get(run_id, [])

    def query_snapshot_positions_page(self, run_id: str, *, filters=None, page=1, page_size=25, schema_mode="upgraded"):
        rows = list(self.rows_by_run.get(run_id, []))
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        return {
            "rows": rows[start:end],
            "total_results": len(rows),
            "page": page,
            "page_size": page_size,
        }

    def aggregate_snapshot_positions(self, run_id: str, *, filters=None):
        rows = self.rows_by_run.get(run_id, [])
        scores = [float(row["position_props"].get("risk_score") or 0.0) for row in rows]
        assets: dict[str, int] = {}
        for row in rows:
            asset = str(row["position_props"].get("asset") or "unknown")
            assets[asset] = assets.get(asset, 0) + 1
        return {
            "position_count": len(rows),
            "risk_buckets": {
                "high": sum(1 for score in scores if score >= 0.75),
                "medium": sum(1 for score in scores if 0.5 <= score < 0.75),
                "low": sum(1 for score in scores if score < 0.5),
            },
            "asset_exposure_counts": assets,
            "average_risk_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }

    def fetch_snapshot_position_signal_evidence(self, run_id: str, position_id: str, *, schema_mode="upgraded"):
        if run_id == "run-historical":
            return [{"signal_id": "signal:test", "edge_props": {"name": "signal", "contribution": 0.5}}]
        return [{"signal_id": "signal:test", "edge_props": {"name": "signal", "contribution": 0.2}}]

    def fetch_snapshot_all_position_signal_evidence(self, run_id: str, *, schema_mode="upgraded"):
        evidence = self.fetch_snapshot_position_signal_evidence(run_id, "")
        return {
            row["position_id"]: list(evidence) for row in (self.rows_by_run.get(run_id) or []) if row.get("position_id")
        }

    def fetch_snapshot_position_signal_evidence_batch(self, run_id: str, position_ids, *, schema_mode="upgraded"):
        grouped = self.fetch_snapshot_all_position_signal_evidence(run_id, schema_mode=schema_mode)
        return {position_id: list(grouped.get(position_id, [])) for position_id in position_ids}

    def fetch_snapshot_position_thesis_context_batch(self, run_id: str, position_ids, *, schema_mode="upgraded"):
        return {position_id: {} for position_id in position_ids}

    def snapshot_has_positions(self, run_id: str) -> bool:
        return bool(self.rows_by_run.get(run_id))

    def fetch_snapshot_graph(self, run_id: str, *, schema_mode="upgraded"):
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


def test_query_without_run_id_ingests_when_latest_required_module_is_partial(monkeypatch):
    repo = _FakeRepo()
    repo.latest_run = {
        "run_id": "run-live",
        "as_of": "2026-03-09T00:00:00Z",
        "source_status": {
            "portfolio": {
                "status": "partial",
                "quality": "schema_drift",
                "source_name": "portfolio",
                "source_version": "1",
            }
        },
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
            source_status={"portfolio": {"status": "ok", "quality": "ok"}},
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


def test_query_uses_bounded_batches_for_graph_reads(monkeypatch):
    class _BoundedRepo(_FakeRepo):
        def __init__(self):
            super().__init__()
            self.rows_by_run["run-historical"] = [
                {
                    "position_id": "position:NVDA",
                    "position_label": "NVDA",
                    "position_props": {
                        "ticker": "NVDA",
                        "asset": "equity",
                        "direction": "long",
                        "risk_score": 0.81,
                        "risk_level": "high",
                        "ontology_run_id": "run-historical",
                    },
                    "asset_id": "asset:NVDA",
                    "asset_label": "NVDA",
                    "asset_props": {"ticker": "NVDA", "asset": "equity"},
                    "sector_id": "sector:information_technology",
                    "sector_label": "Information Technology",
                    "sector_props": {"name": "Information Technology"},
                    "position_asset_edge_props": {"ontology_run_id": "run-historical"},
                    "asset_sector_edge_props": {"ontology_run_id": "run-historical", "source": "test"},
                },
                {
                    "position_id": "position:MU",
                    "position_label": "MU",
                    "position_props": {
                        "ticker": "MU",
                        "asset": "equity",
                        "direction": "long",
                        "risk_score": 0.72,
                        "risk_level": "medium",
                        "ontology_run_id": "run-historical",
                    },
                    "asset_id": "asset:MU",
                    "asset_label": "MU",
                    "asset_props": {"ticker": "MU", "asset": "equity"},
                    "sector_id": "sector:information_technology",
                    "sector_label": "Information Technology",
                    "sector_props": {"name": "Information Technology"},
                    "position_asset_edge_props": {"ontology_run_id": "run-historical"},
                    "asset_sector_edge_props": {"ontology_run_id": "run-historical", "source": "test"},
                },
            ]

        def fetch_snapshot_all_position_signal_evidence(self, run_id: str, *, schema_mode="upgraded"):
            raise AssertionError("legacy full-snapshot signal evidence fetch should not be used")

        def fetch_snapshot_graph(self, run_id: str, *, schema_mode="upgraded"):
            raise AssertionError("legacy full-snapshot graph fetch should not be used")

        def fetch_snapshot_position_signal_evidence_batch(self, run_id: str, position_ids, *, schema_mode="upgraded"):
            out = {}
            for position_id in position_ids:
                ticker = position_id.split(":")[-1].lower()
                out[position_id] = [
                    {
                        "position_id": position_id,
                        "signal_id": f"signal:test:{ticker}",
                        "signal_label": f"{ticker.upper()} signal",
                        "signal_props": {"source": "test"},
                        "edge_props": {"source": "test", "name": f"{ticker.upper()} signal", "contribution": 0.3},
                    }
                ]
            return out

        def fetch_snapshot_position_thesis_context_batch(self, run_id: str, position_ids, *, schema_mode="upgraded"):
            return {position_id: {} for position_id in position_ids}

    repo = _BoundedRepo()
    service = OntologyQueryService(repository=repo)
    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)

    resp = service.query(
        query="show risk",
        intent=None,
        filters={},
        timeframe="Daily",
        include_graph=True,
        run_id="run-historical",
        page=1,
        page_size=1,
    )

    assert len(resp["results"]) == 1
    assert resp["_meta"]["pagination"]["page"] == 1
    assert resp["_meta"]["pagination"]["page_size"] == 1
    assert resp["_meta"]["pagination"]["total_results"] == 2
    assert resp["_meta"]["graph"]["scope"] == "page"
    node_ids = {node["id"] for node in resp["graph"]["nodes"]}
    assert "position:NVDA" in node_ids
    assert "position:MU" not in node_ids


def test_query_returns_empty_past_end_page_with_pagination(monkeypatch):
    repo = _FakeRepo()
    repo.rows_by_run["run-historical"] = [
        {
            "position_id": "position:NVDA",
            "position_props": {
                "ticker": "NVDA",
                "asset": "equity",
                "direction": "long",
                "risk_score": 0.81,
                "risk_level": "high",
                "ontology_run_id": "run-historical",
            },
            "sector_props": {"name": "Information Technology"},
        },
        {
            "position_id": "position:MU",
            "position_props": {
                "ticker": "MU",
                "asset": "equity",
                "direction": "long",
                "risk_score": 0.72,
                "risk_level": "medium",
                "ontology_run_id": "run-historical",
            },
            "sector_props": {"name": "Information Technology"},
        },
    ]
    service = OntologyQueryService(repository=repo)

    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)

    resp = service.query(
        query="show risk",
        intent=None,
        filters={},
        timeframe="Daily",
        include_graph=False,
        run_id="run-historical",
        page=5,
        page_size=1,
    )

    assert resp["results"] == []
    assert resp["_meta"]["pagination"]["page"] == 5
    assert resp["_meta"]["pagination"]["page_size"] == 1
    assert resp["_meta"]["pagination"]["total_results"] == 2
    assert resp["_meta"]["pagination"]["returned_results"] == 0
    assert resp["_meta"]["pagination"]["has_prev"] is True
    assert resp["_meta"]["pagination"]["has_next"] is False


def test_build_page_graph_reports_page_scope_and_node_truncation():
    rows = []
    evidence_by_position = {}
    for index in range(200):
        position_id = f"position:T{index:03d}"
        asset_id = f"asset:T{index:03d}"
        rows.append(
            {
                "position_id": position_id,
                "position_label": f"T{index:03d}",
                "position_props": {"ticker": f"T{index:03d}", "asset": "equity", "risk_score": 0.7},
                "position_schema_name": "Position",
                "position_schema_version": 1,
                "position_updated_at": "2026-03-08T00:00:00Z",
                "asset_id": asset_id,
                "asset_label": f"T{index:03d}",
                "asset_props": {"ticker": f"T{index:03d}", "asset": "equity"},
                "asset_schema_name": "Asset",
                "asset_schema_version": 1,
                "asset_updated_at": "2026-03-08T00:00:00Z",
                "sector_id": "sector:information_technology",
                "sector_label": "Information Technology",
                "sector_props": {"name": "Information Technology"},
                "sector_schema_name": "Sector",
                "sector_schema_version": 1,
                "sector_updated_at": "2026-03-08T00:00:00Z",
                "position_asset_edge_props": {"ontology_run_id": "run-historical"},
                "position_asset_edge_schema_name": "references_asset",
                "position_asset_edge_schema_version": 1,
                "position_asset_edge_relation_schema_name": "references_asset",
                "position_asset_edge_relation_schema_version": 1,
                "position_asset_edge_updated_at": "2026-03-08T00:00:00Z",
                "asset_sector_edge_props": {"ontology_run_id": "run-historical", "source": "test"},
                "asset_sector_edge_schema_name": "belongs_to_sector",
                "asset_sector_edge_schema_version": 1,
                "asset_sector_edge_relation_schema_name": "belongs_to_sector",
                "asset_sector_edge_relation_schema_version": 1,
                "asset_sector_edge_updated_at": "2026-03-08T00:00:00Z",
            }
        )
        evidence_by_position[position_id] = [
            {
                "signal_id": f"signal:test:t{index:03d}",
                "signal_label": f"T{index:03d}",
                "signal_props": {"source": "test"},
                "signal_schema_name": "Signal",
                "signal_schema_version": 1,
                "signal_updated_at": "2026-03-08T00:00:00Z",
                "edge_props": {"source": "test", "name": f"T{index:03d}", "contribution": 0.1},
                "edge_schema_name": "exposed_to_signal",
                "edge_schema_version": 1,
                "edge_relation_schema_name": "exposed_to_signal",
                "edge_relation_schema_version": 1,
                "edge_updated_at": "2026-03-08T00:00:00Z",
            }
        ]

    graph, meta = svc._build_page_graph(rows, evidence_by_position, {}, run_id="run-historical")

    assert meta["scope"] == "page"
    assert meta["max_nodes"] == svc.GRAPH_PAGE_NODE_LIMIT
    assert meta["max_edges"] == svc.GRAPH_PAGE_EDGE_LIMIT
    assert meta["truncated"] is True
    assert len(graph["nodes"]) == svc.GRAPH_PAGE_NODE_LIMIT
    assert len(graph["edges"]) <= svc.GRAPH_PAGE_EDGE_LIMIT


def test_page_graph_builder_caps_edges():
    builder = svc._PageGraphBuilder(max_nodes=6, max_edges=2)

    assert builder.add_node({"id": "position:MU", "type": "Position", "label": "MU", "properties": {}})
    assert builder.add_node({"id": "asset:MU", "type": "Asset", "label": "MU", "properties": {}})
    assert builder.add_node(
        {"id": "sector:information_technology", "type": "Sector", "label": "Information Technology", "properties": {}}
    )
    assert builder.add_edge({"source_id": "position:MU", "target_id": "asset:MU", "relation_type": "references_asset"})
    assert builder.add_edge(
        {
            "source_id": "asset:MU",
            "target_id": "sector:information_technology",
            "relation_type": "belongs_to_sector",
        }
    )
    assert (
        builder.add_edge(
            {"source_id": "position:MU", "target_id": "sector:information_technology", "relation_type": "synthetic"}
        )
        is False
    )
    assert builder.truncated is True
    assert len(builder.edges) == 2


def test_entity_context_auto_scopes_ticker_without_broadening_user_filters(monkeypatch):
    class _CaptureRepo(_FakeRepo):
        def __init__(self):
            super().__init__()
            self.filters_seen: list[dict] = []

        def query_snapshot_positions_page(
            self,
            run_id: str,
            *,
            filters=None,
            page=1,
            page_size=25,
            schema_mode="upgraded",
        ):
            self.filters_seen.append(dict(filters or {}))
            return super().query_snapshot_positions_page(
                run_id,
                filters=filters,
                page=page,
                page_size=page_size,
                schema_mode=schema_mode,
            )

    repo = _CaptureRepo()
    service = OntologyQueryService(repository=repo)

    monkeypatch.setattr(
        svc,
        "parse_hybrid_query",
        lambda **_kwargs: InterpretedQuery(
            intent="entity_context",
            source="structured",
            filters={"min_risk_score": 0.8},
            entity="MU",
            original_query="Show MU context",
        ),
    )
    service.query(
        query="Show MU context",
        intent=None,
        filters={},
        run_id="run-historical",
        include_graph=False,
    )

    monkeypatch.setattr(
        svc,
        "parse_hybrid_query",
        lambda **_kwargs: InterpretedQuery(
            intent="entity_context",
            source="structured",
            filters={"sectors": ["Information Technology"], "min_risk_score": 0.8},
            entity="MU",
            original_query="Show MU in its sector",
        ),
    )
    service.query(
        query="Show MU in its sector",
        intent=None,
        filters={},
        run_id="run-historical",
        include_graph=False,
    )

    assert repo.filters_seen[0] == {"tickers": ["MU"], "min_risk_score": 0.8}
    assert repo.filters_seen[1] == {"sectors": ["Information Technology"], "min_risk_score": 0.8}


def test_thesis_review_enriches_only_visible_page_positions(monkeypatch):
    class _ThesisRepo(_FakeRepo):
        def __init__(self):
            super().__init__()
            self.rows_by_run["run-historical"] = [
                {
                    "position_id": "position:NVDA",
                    "position_label": "NVDA",
                    "position_props": {
                        "ticker": "NVDA",
                        "asset": "equity",
                        "direction": "long",
                        "risk_score": 0.91,
                        "risk_level": "high",
                        "ontology_run_id": "run-historical",
                    },
                    "asset_id": "asset:NVDA",
                    "asset_label": "NVDA",
                    "asset_props": {"ticker": "NVDA", "asset": "equity"},
                    "sector_id": "sector:information_technology",
                    "sector_label": "Information Technology",
                    "sector_props": {"name": "Information Technology"},
                    "position_asset_edge_props": {"ontology_run_id": "run-historical"},
                    "asset_sector_edge_props": {"ontology_run_id": "run-historical", "source": "test"},
                },
                {
                    "position_id": "position:MU",
                    "position_label": "MU",
                    "position_props": {
                        "ticker": "MU",
                        "asset": "equity",
                        "direction": "long",
                        "risk_score": 0.72,
                        "risk_level": "medium",
                        "ontology_run_id": "run-historical",
                    },
                    "asset_id": "asset:MU",
                    "asset_label": "MU",
                    "asset_props": {"ticker": "MU", "asset": "equity"},
                    "sector_id": "sector:information_technology",
                    "sector_label": "Information Technology",
                    "sector_props": {"name": "Information Technology"},
                    "position_asset_edge_props": {"ontology_run_id": "run-historical"},
                    "asset_sector_edge_props": {"ontology_run_id": "run-historical", "source": "test"},
                },
            ]
            self.position_ids_seen: list[str] = []

        def fetch_snapshot_position_thesis_context_batch(self, run_id: str, position_ids, *, schema_mode="upgraded"):
            self.position_ids_seen = list(position_ids)
            return {
                "position:NVDA": {
                    "thesis": {
                        "node": {
                            "id": "thesis:NVDA",
                            "type": "Thesis",
                            "label": "Thesis: NVDA",
                            "properties": {
                                "ticker": "NVDA",
                                "status": "active",
                                "created_at": "2026-03-01T00:00:00Z",
                                "updated_at": "2026-03-08T00:00:00Z",
                            },
                        },
                        "edge": {
                            "source_id": "position:NVDA",
                            "target_id": "thesis:NVDA",
                            "relation_type": "has_thesis",
                            "properties": {"ontology_run_id": run_id},
                        },
                    },
                    "evaluations": [
                        {
                            "node": {
                                "id": "evaluation:NVDA:2026-03-08T00:00:00Z",
                                "type": "Evaluation",
                                "label": "Eval: NVDA",
                                "properties": {
                                    "evaluated_at": "2026-03-08T00:00:00Z",
                                    "thesis_status": "active",
                                    "technical_read": "supportive",
                                    "fundamental_read": "supportive",
                                    "action": "hold",
                                    "confidence": "high",
                                },
                            },
                            "edge": {
                                "source_id": "thesis:NVDA",
                                "target_id": "evaluation:NVDA:2026-03-08T00:00:00Z",
                                "relation_type": "evaluated_by",
                                "properties": {"ontology_run_id": run_id},
                            },
                        }
                    ],
                    "catalysts": [],
                }
            }

    repo = _ThesisRepo()
    service = OntologyQueryService(repository=repo)
    monkeypatch.setattr(
        svc,
        "parse_hybrid_query",
        lambda **_kwargs: InterpretedQuery(
            intent="thesis_review",
            source="structured",
            filters={},
            entity=None,
            original_query="Review the thesis page",
        ),
    )

    resp = service.query(
        query="Review the thesis page",
        intent=None,
        filters={},
        run_id="run-historical",
        include_graph=False,
        page=1,
        page_size=1,
    )

    assert repo.position_ids_seen == ["position:NVDA"]
    assert len(resp["results"]) == 1
    assert resp["results"][0]["ticker"] == "NVDA"
    assert resp["results"][0]["thesis"] == {
        "status": "active",
        "created_at": "2026-03-01T00:00:00Z",
        "updated_at": "2026-03-08T00:00:00Z",
    }
    assert resp["results"][0]["latest_evaluation"]["action"] == "hold"


def test_compare_snapshots_reports_added_removed_risk_and_signal_transitions():
    class _CompareRepo(_FakeRepo):
        def __init__(self):
            super().__init__()
            self.rows_by_run = {
                "run-0": [
                    {
                        "position_id": "position:MU",
                        "position_props": {
                            "ticker": "MU",
                            "asset": "equity",
                            "direction": "long",
                            "risk_score": 0.4,
                            "risk_level": "low",
                            "volatility_cluster": 0.4,
                            "breadth_stress": 0.3,
                            "sector_stress": 0.4,
                            "macro_regime": 0.35,
                            "ontology_run_id": "run-0",
                        },
                        "sector_props": {"name": "Information Technology"},
                    }
                ],
                "run-1": [
                    {
                        "position_id": "position:MU",
                        "position_props": {
                            "ticker": "MU",
                            "asset": "equity",
                            "direction": "long",
                            "risk_score": 0.81,
                            "risk_level": "high",
                            "volatility_cluster": 0.72,
                            "breadth_stress": 0.3,
                            "sector_stress": 0.4,
                            "macro_regime": 0.8,
                            "ontology_run_id": "run-1",
                        },
                        "sector_props": {"name": "Information Technology"},
                    },
                    {
                        "position_id": "position:NVDA",
                        "position_props": {
                            "ticker": "NVDA",
                            "asset": "equity",
                            "direction": "long",
                            "risk_score": 0.77,
                            "risk_level": "high",
                            "volatility_cluster": 0.7,
                            "breadth_stress": 0.65,
                            "sector_stress": 0.6,
                            "macro_regime": 0.7,
                            "ontology_run_id": "run-1",
                        },
                        "sector_props": {"name": "Information Technology"},
                    },
                ],
            }

        def get_run(self, run_id: str):
            if run_id not in {"run-0", "run-1"}:
                return None
            return {
                "run_id": run_id,
                "as_of": f"2026-03-0{1 if run_id == 'run-0' else 8}T00:00:00Z",
                "source_status": {"portfolio": {"status": "ok"}},
                "required_modules": ["portfolio"],
                "component_scores": {"macro_regime": 0.3 if run_id == "run-0" else 0.8},
                "created_at": "2026-03-08 00:00:00",
            }

    diff = OntologyQueryService(repository=_CompareRepo()).compare_snapshots("run-0", "run-1")

    assert diff["positions_added"] == ["NVDA"]
    assert diff["positions_removed"] == []
    assert diff["risk_changes"][0]["ticker"] == "MU"
    assert diff["risk_changes"][0]["delta"] == 0.41
    assert any(row["component"] == "volatility_cluster" for row in diff["signal_transitions"])
    assert diff["component_diffs"]["macro_regime"]["delta"] == 0.5


@pytest.mark.parametrize(
    ("query_kwargs", "denied_action"),
    [
        (
            {"run_id": "run-historical", "include_graph": True, "refresh_snapshot": False},
            OntologyAction.GRAPH_READ,
        ),
        (
            {"run_id": None, "include_graph": False, "refresh_snapshot": True},
            OntologyAction.SNAPSHOT_REFRESH,
        ),
    ],
)
def test_query_denied_graph_or_refresh_permission_short_circuits_before_repo_fetch(
    monkeypatch,
    query_kwargs,
    denied_action,
):
    class _TrapRepo(_FakeRepo):
        def get_run(self, run_id: str):
            raise AssertionError("repo should not be touched when dynamic action is denied")

        def get_latest_run(self):
            raise AssertionError("latest run lookup should not happen when refresh is denied")

    service = OntologyQueryService(repository=_TrapRepo(), policy=_Policy(denied_actions={denied_action}))
    monkeypatch.setattr(svc, "parse_hybrid_query", lambda **_kwargs: pytest.fail("parse should not run"))

    with pytest.raises(PolicyDenied, match=str(denied_action)):
        service.query(
            query="show risk",
            intent="portfolio_risk_exposure",
            filters={},
            timeframe="Daily",
            schema_mode="upgraded",
            actor=admin_actor(),
            **query_kwargs,
        )


def test_query_success_and_denial_emit_ontology_read_audits(monkeypatch):
    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)
    ok_service = OntologyQueryService(repository=_FakeRepo())
    ok_service.query(
        query="show risk",
        intent=None,
        filters={},
        run_id="run-historical",
        include_graph=False,
        actor=admin_actor("auditor"),
    )

    denied_service = OntologyQueryService(repository=_FakeRepo(), policy=_Policy(denied_actions={OntologyAction.QUERY}))
    with pytest.raises(PolicyDenied):
        denied_service.query(
            query="show risk",
            intent=None,
            filters={},
            run_id="run-historical",
            include_graph=False,
            actor=admin_actor("auditor"),
        )

    rows = core_db.get_audit_events(action_name="ontology.query", limit=10)
    statuses = {row["status"] for row in rows}
    assert {"succeeded", "denied"} <= statuses
    succeeded = [row for row in rows if row["status"] == "succeeded"][0]
    denied = [row for row in rows if row["status"] == "denied"][0]
    assert succeeded["object_refs"] == [{"type": "ontology_run", "id": "run-historical"}]
    assert succeeded["after_summary"]["page"] == 1
    assert denied["metadata"]["include_graph"] is False


def test_filtered_objects_make_aggregate_and_temporal_diff_inexact(monkeypatch):
    class _CompareRepo(_FakeRepo):
        def __init__(self):
            super().__init__()
            self.rows_by_run = {
                "run-0": [
                    {
                        "position_id": "position:MU",
                        "position_props": {
                            "ticker": "MU",
                            "asset": "equity",
                            "direction": "long",
                            "risk_score": 0.42,
                            "risk_level": "low",
                            "ontology_run_id": "run-0",
                        },
                        "sector_props": {"name": "Information Technology"},
                    }
                ],
                "run-1": [
                    {
                        "position_id": "position:MU",
                        "position_props": {
                            "ticker": "MU",
                            "asset": "equity",
                            "direction": "long",
                            "risk_score": 0.82,
                            "risk_level": "high",
                            "ontology_run_id": "run-1",
                        },
                        "sector_props": {"name": "Information Technology"},
                    }
                ],
            }

        def get_run(self, run_id: str):
            if run_id not in self.rows_by_run:
                return None
            return {
                "run_id": run_id,
                "as_of": "2026-03-08T00:00:00Z",
                "source_status": {"portfolio": {"status": "ok"}},
                "required_modules": ["portfolio"],
                "component_scores": {},
                "created_at": "2026-03-08 00:00:00",
            }

    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)
    service = OntologyQueryService(
        repository=_CompareRepo(),
        policy=_Policy(fields={"Position": {"ticker", "asset", "direction", "risk_level", "ontology_run_id"}}),
    )

    resp = service.query(
        query="show risk",
        intent=None,
        filters={},
        run_id="run-1",
        include_graph=False,
        actor=admin_actor(),
    )
    diff = service.compare_snapshots("run-0", "run-1", actor=admin_actor())

    assert resp["_meta"]["pagination"]["exact_total"] is False
    assert resp["aggregate"]["average_risk_score"] == 0.0
    assert diff["risk_changes"] == []
    assert diff["_meta"]["authorization"]["redacted_fields"] >= 1
