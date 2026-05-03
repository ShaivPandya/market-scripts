from __future__ import annotations

import json
import sqlite3
from typing import Any, Literal

import pytest

import ontology.ingestion as ingestion
import portfolio.core_db as core_db
from ontology.models import OntologyEdge, OntologyNode
from ontology.repository import OntologyRepository, _build_snapshot_position_query_parts
from ontology.schemas.identity import catalyst_id, evaluation_id, signal_id
from ontology.schemas.registry import OntologySchemaValidationError
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


@pytest.fixture
def source_status_factory():
    def _build(*, required_status: str = "ok", optional_status: str = "partial") -> dict[str, Any]:
        return {
            "portfolio": {
                "status": required_status,
                "quality": "ok" if required_status == "ok" else "schema_drift",
                "source_name": "portfolio",
                "source_version": "1",
                "fetched_at": "2026-05-01T20:00:00+00:00",
                "lineage": {
                    "raw_module": "portfolio.portfolio_dashboard",
                    "raw_function": "get_data",
                    "adapter": "portfolio",
                    "adapter_version": "1",
                    "payload_fingerprint": "portfolio-abc",
                },
                "schema_drift": [
                    {
                        "severity": "info",
                        "path": "$.extra",
                        "expected": "known field",
                        "actual": "str",
                        "action": "ignored",
                    }
                ],
            },
            "sentiment": {
                "status": optional_status,
                "quality": "missing" if optional_status != "ok" else "ok",
                "source_name": "sentiment",
                "source_version": "1",
            },
        }

    return _build


@pytest.fixture
def semantic_graph_factory():
    def _build(
        run_id: str = "run-1",
        *,
        position_id: str = "position:MU",
        include_optional: bool = False,
        invalid_exposure: bool = False,
    ) -> tuple[list[OntologyNode], list[OntologyEdge]]:
        position_props: dict[str, Any] = {
            "schema_version": 1,
            "ticker": "MU",
            "asset": "equity",
            "direction": "long",
            "timeframe": "Daily",
            "latest_price": 100.0,
            "as_of": "2026-03-08T00:00:00Z",
            "risk_score": 0.72,
            "risk_level": "medium",
            "volatility_cluster": 0.4,
            "breadth_stress": 0.5,
            "sector_stress": 0.6,
            "macro_regime": 0.7,
            "ontology_run_id": run_id,
        }
        nodes = [
            OntologyNode(
                id=position_id,
                type="Position",
                label="MU",
                properties=position_props,
                schema_name="Position",
                schema_version=1,
            ),
            OntologyNode(id="asset:MU", type="Asset", label="MU", properties={"ticker": "MU", "asset": "equity"}),
            OntologyNode(
                id="sector:information_technology",
                type="Sector",
                label="Information Technology",
                properties={"name": "Information Technology"},
            ),
            OntologyNode(id="signal:test", type="Signal", label="Signal Alpha", properties={"source": "test"}),
        ]
        edges = [
            OntologyEdge(position_id, "asset:MU", "references_asset", {"ontology_run_id": run_id}),
            OntologyEdge(
                "asset:MU",
                "sector:information_technology",
                "belongs_to_sector",
                {"ontology_run_id": run_id, "source": "test"},
            ),
            OntologyEdge(
                position_id,
                "signal:test",
                "exposed_to_signal",
                {
                    "schema_version": 1,
                    "component": "macro_regime",
                    "source": "test",
                    "name": "Signal Alpha",
                    "value": 1.2,
                    "threshold": "higher is worse",
                    "direction": "deteriorating",
                    "contribution": 1.2 if invalid_exposure else 0.4,
                    "ontology_run_id": run_id,
                },
                schema_name="PositionSignalExposure",
                schema_version=1,
            ),
        ]
        if include_optional:
            nodes.extend(
                [
                    OntologyNode(
                        id="thesis:MU",
                        type="Thesis",
                        label="Thesis: MU",
                        properties={"ticker": "MU", "status": "active", "ontology_run_id": run_id},
                    ),
                    OntologyNode(
                        id="evaluation:MU:2026-03-08T00:00:00Z",
                        type="Evaluation",
                        label="Eval: MU",
                        properties={
                            "ticker": "MU",
                            "evaluated_at": "2026-03-08T00:00:00Z",
                            "thesis_status": "active",
                            "technical_read": "supportive",
                            "fundamental_read": "supportive",
                            "action": "hold",
                            "confidence": "high",
                            "ontology_run_id": run_id,
                        },
                    ),
                    OntologyNode(
                        id="catalyst:MU:0",
                        type="Catalyst",
                        label="Demand recovery",
                        properties={
                            "ticker": "MU",
                            "name": "Demand recovery",
                            "description": "Demand improves",
                            "ontology_run_id": run_id,
                        },
                    ),
                ]
            )
            edges.extend(
                [
                    OntologyEdge(position_id, "thesis:MU", "has_thesis", {"ontology_run_id": run_id}),
                    OntologyEdge(
                        "thesis:MU",
                        "evaluation:MU:2026-03-08T00:00:00Z",
                        "evaluated_by",
                        {"ontology_run_id": run_id},
                    ),
                    OntologyEdge("thesis:MU", "catalyst:MU:0", "has_catalyst", {"ontology_run_id": run_id}),
                ]
            )
        return nodes, edges

    return _build


class _Adapter:
    def __init__(self, name: str):
        self.source_name = name
        self.source_version = "test"
        self.required = True
        self.raw_module = "tests"
        self.raw_function = name
        self.parameters = {}


def _result(name: str, data: Any, status: str = "ok") -> SourceResult:
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
            payload_fingerprint=f"{name}-abc",
        ),
    )


def test_repository_upsert_and_join_queries(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)

    nodes = [
        OntologyNode(
            id="position:MU",
            type="Position",
            label="MU",
            properties={"ticker": "MU", "ontology_run_id": "run-1", "risk_score": 0.7, "risk_level": "medium"},
        ),
        OntologyNode(
            id="asset:MU",
            type="Asset",
            label="MU",
            properties={"ticker": "MU", "asset": "equity"},
        ),
        OntologyNode(
            id="sector:information_technology",
            type="Sector",
            label="Information Technology",
            properties={"name": "Information Technology"},
        ),
        OntologyNode(
            id="signal:test",
            type="Signal",
            label="Test Signal",
            properties={"source": "test"},
        ),
    ]
    edges = [
        OntologyEdge("position:MU", "asset:MU", "references_asset", {}),
        OntologyEdge("asset:MU", "sector:information_technology", "belongs_to_sector", {}),
        OntologyEdge(
            "position:MU",
            "signal:test",
            "exposed_to_signal",
            {
                "source": "test",
                "name": "Test Signal",
                "contribution": 0.3,
                "ontology_run_id": "run-1",
            },
        ),
    ]

    repo.upsert_graph(nodes, edges)

    rows = repo.fetch_position_asset_sector_rows()
    assert len(rows) == 1
    assert rows[0]["position_id"] == "position:MU"
    assert rows[0]["asset_id"] == "asset:MU"
    assert rows[0]["sector_props"]["name"] == "Information Technology"

    evidence = repo.fetch_position_signal_evidence("position:MU")
    assert len(evidence) == 1
    assert evidence[0]["signal_id"] == "signal:test:test_signal"
    assert evidence[0]["edge_props"]["contribution"] == 0.3

    graph = repo.fetch_graph()
    assert len(graph["nodes"]) == 4
    assert len(graph["edges"]) == 3
    assert all(n["schema_version"] == 1 for n in graph["nodes"])
    assert all(e["schema_version"] == 1 for e in graph["edges"])

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(nodes)").fetchall()}
        assert {"schema_name", "schema_version"} <= columns
        row = conn.execute(
            "SELECT schema_name, schema_version, properties_json FROM nodes WHERE id = ?", ("position:MU",)
        ).fetchone()
        assert row is not None
        assert row[0] == "Position"
        assert row[1] == 1
        assert json.loads(row[2])["schema_version"] == 1


def test_snapshot_rows_are_run_scoped(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)

    nodes_run_1 = [
        OntologyNode(
            id="position:MU",
            type="Position",
            label="MU",
            properties={"ticker": "MU", "risk_score": 0.7, "risk_level": "medium", "ontology_run_id": "run-1"},
        ),
        OntologyNode(id="asset:MU", type="Asset", label="MU", properties={"ticker": "MU", "asset": "equity"}),
        OntologyNode(
            id="sector:information_technology",
            type="Sector",
            label="Information Technology",
            properties={"name": "Information Technology"},
        ),
        OntologyNode(id="signal:s1", type="Signal", label="Signal 1", properties={"source": "test"}),
    ]
    edges_run_1 = [
        OntologyEdge("position:MU", "asset:MU", "references_asset", {"ontology_run_id": "run-1"}),
        OntologyEdge("asset:MU", "sector:information_technology", "belongs_to_sector", {"ontology_run_id": "run-1"}),
        OntologyEdge(
            "position:MU",
            "signal:s1",
            "exposed_to_signal",
            {"name": "Signal 1", "contribution": 0.2, "ontology_run_id": "run-1"},
        ),
    ]

    nodes_run_2 = [
        OntologyNode(
            id="position:MU",
            type="Position",
            label="MU",
            properties={"ticker": "MU", "risk_score": 0.9, "risk_level": "high", "ontology_run_id": "run-2"},
        ),
        OntologyNode(id="asset:MU", type="Asset", label="MU", properties={"ticker": "MU", "asset": "equity"}),
        OntologyNode(
            id="sector:information_technology",
            type="Sector",
            label="Information Technology",
            properties={"name": "Information Technology"},
        ),
        OntologyNode(id="signal:s2", type="Signal", label="Signal 2", properties={"source": "test"}),
    ]
    edges_run_2 = [
        OntologyEdge("position:MU", "asset:MU", "references_asset", {"ontology_run_id": "run-2"}),
        OntologyEdge("asset:MU", "sector:information_technology", "belongs_to_sector", {"ontology_run_id": "run-2"}),
        OntologyEdge(
            "position:MU",
            "signal:s2",
            "exposed_to_signal",
            {"name": "Signal 2", "contribution": 0.5, "ontology_run_id": "run-2"},
        ),
    ]

    repo.save_snapshot(
        run_id="run-1",
        as_of="2026-03-08T00:00:00Z",
        source_status={"portfolio": {"status": "ok"}},
        required_modules=["portfolio"],
        optional_modules=[],
        component_scores={"macro_regime": 0.5},
        nodes=nodes_run_1,
        edges=edges_run_1,
    )
    repo.save_snapshot(
        run_id="run-2",
        as_of="2026-03-09T00:00:00Z",
        source_status={"portfolio": {"status": "ok"}},
        required_modules=["portfolio"],
        optional_modules=[],
        component_scores={"macro_regime": 0.7},
        nodes=nodes_run_2,
        edges=edges_run_2,
    )

    rows_run_1 = repo.fetch_snapshot_position_asset_sector_rows("run-1", schema_mode="upgraded")
    rows_run_2 = repo.fetch_snapshot_position_asset_sector_rows("run-2", schema_mode="upgraded")
    assert rows_run_1[0]["position_props"]["risk_level"] == "medium"
    assert rows_run_2[0]["position_props"]["risk_level"] == "high"

    ev_run_1 = repo.fetch_snapshot_position_signal_evidence("run-1", "position:MU", schema_mode="upgraded")
    ev_run_2 = repo.fetch_snapshot_position_signal_evidence("run-2", "position:MU", schema_mode="upgraded")
    assert ev_run_1[0]["signal_id"] == "signal:test:signal_1"
    assert ev_run_2[0]["signal_id"] == "signal:test:signal_2"

    meta = repo.get_run("run-2")
    assert meta is not None
    assert meta["run_id"] == "run-2"
    assert meta["as_of"] == "2026-03-09T00:00:00Z"


def test_snapshot_prune_removes_old_runs(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)

    nodes = _core_nodes()
    edges = _core_edges()

    repo.save_snapshot(
        run_id="old-run",
        as_of="2026-03-08T00:00:00Z",
        source_status={},
        required_modules=[],
        optional_modules=[],
        component_scores={},
        nodes=nodes,
        edges=edges,
    )
    repo.save_snapshot(
        run_id="new-run",
        as_of="2026-03-09T00:00:00Z",
        source_status={},
        required_modules=[],
        optional_modules=[],
        component_scores={},
        nodes=nodes,
        edges=edges,
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE ontology_runs SET created_at = datetime('now', '-120 days') WHERE run_id = 'old-run'")

    deleted = repo.prune_runs_older_than(days=90)
    assert deleted == 1
    assert repo.get_run("old-run") is None
    assert repo.get_run("new-run") is not None


def test_get_latest_run_returns_most_recent(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)

    nodes = _core_nodes()
    edges = _core_edges()

    repo.save_snapshot(
        run_id="run-old",
        as_of="2026-03-08T00:00:00Z",
        source_status={},
        required_modules=[],
        optional_modules=[],
        component_scores={},
        nodes=nodes,
        edges=edges,
    )
    repo.save_snapshot(
        run_id="run-new",
        as_of="2026-03-09T00:00:00Z",
        source_status={},
        required_modules=[],
        optional_modules=[],
        component_scores={},
        nodes=nodes,
        edges=edges,
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE ontology_runs SET created_at = datetime('now', '-20 minutes') WHERE run_id = 'run-old'")
        conn.execute("UPDATE ontology_runs SET created_at = datetime('now', '-1 minutes') WHERE run_id = 'run-new'")

    latest = repo.get_latest_run()
    assert latest is not None
    assert latest["run_id"] == "run-new"


def test_snapshot_preserves_expanded_source_status_metadata(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)

    source_status = {
        "portfolio": {
            "status": "ok",
            "quality": "ok",
            "source_name": "portfolio",
            "source_version": "1",
            "fetched_at": "2026-05-01T20:00:00+00:00",
            "lineage": {
                "raw_module": "portfolio.portfolio_dashboard",
                "raw_function": "get_data",
                "adapter": "portfolio",
                "adapter_version": "1",
                "payload_fingerprint": "abc",
            },
            "schema_drift": [
                {
                    "severity": "info",
                    "path": "$.extra",
                    "expected": "known field",
                    "actual": "str",
                    "action": "ignored",
                }
            ],
        }
    }

    repo.save_snapshot(
        run_id="run-source-meta",
        as_of="2026-05-01T20:00:00+00:00",
        source_status=source_status,
        required_modules=["portfolio"],
        optional_modules=[],
        component_scores={},
        nodes=_core_nodes(),
        edges=_core_edges(),
    )

    saved = repo.get_run("run-source-meta")
    assert saved is not None
    assert saved["source_status"]["portfolio"]["quality"] == "ok"
    assert saved["source_status"]["portfolio"]["lineage"]["payload_fingerprint"] == "abc"
    assert saved["source_status"]["portfolio"]["schema_drift"][0]["severity"] == "info"


def test_backfill_schema_versions_rewrites_legacy_optional_ids(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ontology_runs(
                run_id,
                as_of,
                source_status_json,
                required_modules_json,
                optional_modules_json,
                component_scores_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            ("run-legacy", "2026-03-08T00:00:00Z", "{}", "[]", "[]", "{}"),
        )
        conn.executemany(
            """
            INSERT INTO snapshot_nodes(run_id, id, type, label, properties_json, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    "run-legacy",
                    "position:MU",
                    "Position",
                    "MU",
                    json.dumps(
                        {
                            "ticker": "MU",
                            "asset": "equity",
                            "direction": "long",
                            "risk_score": 0.72,
                            "risk_level": "medium",
                            "ontology_run_id": "run-legacy",
                        }
                    ),
                ),
                (
                    "run-legacy",
                    "asset:MU",
                    "Asset",
                    "MU",
                    json.dumps({"ticker": "MU", "asset": "equity"}),
                ),
                (
                    "run-legacy",
                    "sector:information_technology",
                    "Sector",
                    "Information Technology",
                    json.dumps({"name": "Information Technology", "source": "test"}),
                ),
                (
                    "run-legacy",
                    "thesis:MU",
                    "Thesis",
                    "Thesis: MU",
                    json.dumps({"ticker": "MU", "status": "active", "ontology_run_id": "run-legacy"}),
                ),
                (
                    "run-legacy",
                    "evaluation:MU:2026-03-08T00:00:00Z",
                    "Evaluation",
                    "Eval: MU",
                    json.dumps(
                        {
                            "ticker": "MU",
                            "evaluated_at": "2026-03-08T00:00:00Z",
                            "thesis_status": "strengthen",
                            "technical_read": "supportive",
                            "fundamental_read": "supportive",
                            "action": "hold",
                            "confidence": "high",
                            "ontology_run_id": "run-legacy",
                        }
                    ),
                ),
                (
                    "run-legacy",
                    "catalyst:MU:0",
                    "Catalyst",
                    "Demand recovery",
                    json.dumps(
                        {
                            "ticker": "MU",
                            "name": "Demand recovery",
                            "description": "Demand improves",
                            "ontology_run_id": "run-legacy",
                        }
                    ),
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO snapshot_edges(run_id, source_id, target_id, relation_type, properties_json, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                ("run-legacy", "position:MU", "asset:MU", "references_asset", "{}"),
                (
                    "run-legacy",
                    "asset:MU",
                    "sector:information_technology",
                    "belongs_to_sector",
                    json.dumps({"source": "test"}),
                ),
                ("run-legacy", "position:MU", "thesis:MU", "has_thesis", "{}"),
                ("run-legacy", "thesis:MU", "evaluation:MU:2026-03-08T00:00:00Z", "evaluated_by", "{}"),
                ("run-legacy", "thesis:MU", "catalyst:MU:0", "has_catalyst", "{}"),
            ],
        )

    dry_run = repo.backfill_schema_versions(dry_run=True)
    snapshot_report = [s for s in dry_run["scopes"] if s.get("run_id") == "run-legacy"][0]
    assert snapshot_report["rewritten_ids"] == 2

    repo.backfill_schema_versions(dry_run=False)
    graph = repo.fetch_snapshot_graph("run-legacy", schema_mode="upgraded")
    ids = {node["id"] for node in graph["nodes"]}
    assert evaluation_id("MU", "2026-03-08T00:00:00Z") in ids
    assert catalyst_id("MU", "Demand recovery", "Demand improves") in ids
    assert all(edge["properties"]["ontology_run_id"] == "run-legacy" for edge in graph["edges"])
    belongs = [edge for edge in graph["edges"] if edge["relation_type"] == "belongs_to_sector"][0]
    assert belongs["properties"]["source"] == "test"
    assert all(node["schema_version"] == 1 for node in graph["nodes"])
    assert all(edge["schema_version"] == 1 for edge in graph["edges"])


def test_legacy_snapshot_stored_mode_survives_test_schema_evolution(tmp_path, monkeypatch):
    from ontology.schemas import registry
    from ontology.schemas.objects import PositionV1

    class PositionV2(PositionV1):
        schema_version: Literal[2] = 2
        risk_bucket: str

    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    _insert_minimal_legacy_snapshot(db_path, run_id="run-legacy")

    monkeypatch.setitem(registry.NODE_SCHEMAS, "Position", PositionV2)
    monkeypatch.setitem(
        registry.NODE_UPGRADE_ADAPTERS,
        ("Position", 1, 2),
        lambda payload: {**payload, "schema_version": 2, "risk_bucket": "evolved"},
    )

    stored = repo.fetch_snapshot_graph("run-legacy", schema_mode="stored")
    upgraded = repo.fetch_snapshot_graph("run-legacy", schema_mode="upgraded")

    stored_position = [node for node in stored["nodes"] if node["type"] == "Position"][0]
    upgraded_position = [node for node in upgraded["nodes"] if node["type"] == "Position"][0]
    assert "risk_bucket" not in stored_position["properties"]
    assert upgraded_position["properties"]["schema_version"] == 2
    assert upgraded_position["properties"]["risk_bucket"] == "evolved"


def test_stored_snapshot_read_ignores_current_relation_cardinality_change(tmp_path, monkeypatch):
    from ontology.schemas.relations import (
        EXPOSED_TO_SIGNAL,
        RELATION_REGISTRY,
        RelationCardinality,
        RelationDefinition,
    )

    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    _insert_minimal_legacy_snapshot(db_path, run_id="run-relation")
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO snapshot_nodes(run_id, id, type, label, properties_json, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                ("run-relation", "signal:one", "Signal", "One", json.dumps({"source": "test"})),
                ("run-relation", "signal:two", "Signal", "Two", json.dumps({"source": "test"})),
            ],
        )
        conn.executemany(
            """
            INSERT INTO snapshot_edges(run_id, source_id, target_id, relation_type, properties_json, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                ("run-relation", "position:MU", "signal:one", "exposed_to_signal", "{}"),
                ("run-relation", "position:MU", "signal:two", "exposed_to_signal", "{}"),
            ],
        )

    current = RELATION_REGISTRY[EXPOSED_TO_SIGNAL]
    monkeypatch.setitem(
        RELATION_REGISTRY,
        EXPOSED_TO_SIGNAL,
        RelationDefinition(
            name=current.name,
            source_type=current.source_type,
            target_type=current.target_type,
            cardinality=RelationCardinality.SOURCE_UNIQUE,
            required_properties=current.required_properties,
            optional=current.optional,
        ),
    )

    stored = repo.fetch_snapshot_graph("run-relation", schema_mode="stored")
    exposures = [edge for edge in stored["edges"] if edge["relation_type"] == "exposed_to_signal"]
    assert len(exposures) == 2


def test_query_snapshot_positions_page_filters_and_paginates(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    _insert_snapshot_query_fixture(db_path, run_id="run-query")

    page_1 = repo.query_snapshot_positions_page(
        "run-query",
        filters={},
        page=1,
        page_size=2,
        schema_mode="upgraded",
    )
    assert page_1["total_results"] == 3
    assert [row["position_id"] for row in page_1["rows"]] == ["position:NVDA", "position:MU"]

    page_2 = repo.query_snapshot_positions_page(
        "run-query",
        filters={},
        page=2,
        page_size=2,
        schema_mode="upgraded",
    )
    assert [row["position_id"] for row in page_2["rows"]] == ["position:GLD"]

    page_9 = repo.query_snapshot_positions_page(
        "run-query",
        filters={},
        page=9,
        page_size=2,
        schema_mode="upgraded",
    )
    assert page_9["rows"] == []
    assert page_9["total_results"] == 3

    by_ticker = repo.query_snapshot_positions_page(
        "run-query",
        filters={"tickers": ["mu"]},
        page=1,
        page_size=10,
        schema_mode="upgraded",
    )
    assert [row["position_id"] for row in by_ticker["rows"]] == ["position:MU"]

    by_sector = repo.query_snapshot_positions_page(
        "run-query",
        filters={"sectors": ["Information Technology"]},
        page=1,
        page_size=10,
        schema_mode="upgraded",
    )
    assert [row["position_id"] for row in by_sector["rows"]] == ["position:NVDA", "position:MU"]

    by_asset = repo.query_snapshot_positions_page(
        "run-query",
        filters={"assets": ["commodity"]},
        page=1,
        page_size=10,
        schema_mode="upgraded",
    )
    assert [row["position_id"] for row in by_asset["rows"]] == ["position:GLD"]

    by_risk = repo.query_snapshot_positions_page(
        "run-query",
        filters={"min_risk_score": 0.75},
        page=1,
        page_size=10,
        schema_mode="upgraded",
    )
    assert [row["position_id"] for row in by_risk["rows"]] == ["position:NVDA"]


def test_aggregate_snapshot_positions_respects_filters(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    _insert_snapshot_query_fixture(db_path, run_id="run-query")

    aggregate = repo.aggregate_snapshot_positions(
        "run-query",
        filters={"sectors": ["Information Technology"]},
    )
    assert aggregate["position_count"] == 2
    assert aggregate["risk_buckets"] == {"high": 1, "medium": 1, "low": 0}
    assert aggregate["asset_exposure_counts"] == {"equity": 2}
    assert aggregate["average_risk_score"] == 0.765


def test_snapshot_batch_traversal_methods_are_scoped_to_requested_positions(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    _insert_snapshot_query_fixture(db_path, run_id="run-query")

    evidence = repo.fetch_snapshot_position_signal_evidence_batch(
        "run-query",
        ["position:MU", "position:GLD"],
        schema_mode="upgraded",
    )
    assert sorted(evidence) == ["position:GLD", "position:MU"]
    assert evidence["position:MU"][0]["signal_id"] == "signal:test:mu"
    assert evidence["position:GLD"][0]["edge_props"]["contribution"] == 0.11

    thesis = repo.fetch_snapshot_position_thesis_context_batch(
        "run-query",
        ["position:MU", "position:GLD"],
        schema_mode="upgraded",
    )
    assert thesis["position:MU"]["thesis"]["node"]["id"] == "thesis:MU"
    assert thesis["position:MU"]["evaluations"][0]["node"]["id"] == evaluation_id("MU", "2026-03-08T00:00:00Z")
    assert thesis["position:MU"]["catalysts"][0]["node"]["id"] == catalyst_id(
        "MU", "Memory recovery", "Demand improves"
    )
    assert thesis["position:GLD"] == {}


def test_snapshot_has_positions_reports_presence_without_full_row_load(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    _insert_snapshot_query_fixture(db_path, run_id="run-query")

    assert repo.snapshot_has_positions("run-query") is True
    assert repo.snapshot_has_positions("missing-run") is False


def test_snapshot_query_plans_use_paginated_query_indexes(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    OntologyRepository(db_path=db_path)
    _insert_snapshot_query_fixture(db_path, run_id="run-query")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        index_names = {row["name"] for row in conn.execute("PRAGMA index_list('snapshot_nodes')").fetchall()}
        assert "idx_snapshot_nodes_run_type_id" in index_names
        assert "idx_snapshot_nodes_position_risk_sort" in index_names

        parts = _build_snapshot_position_query_parts("run-query", {"min_risk_score": 0.5}, use_postgres=False)
        plan_rows = conn.execute(
            f"""
            EXPLAIN QUERY PLAN
            SELECT p.id
            {parts["from_sql"]}
            WHERE {parts["where_sql"]}
            ORDER BY {parts["risk_score_sort_expr"]} DESC, p.id ASC
            LIMIT 2 OFFSET 0
            """,
            tuple(parts["params"]),
        ).fetchall()
        plan_details = " | ".join(str(row["detail"]) for row in plan_rows)
        assert "idx_snapshot_nodes_run_type" in plan_details or "idx_snapshot_nodes_run_type_id" in plan_details

        signal_plan = conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT ps.source_id, s.id
            FROM snapshot_edges ps
            JOIN snapshot_nodes s
              ON s.run_id = ps.run_id
             AND s.id = ps.target_id
             AND s.type = 'Signal'
            WHERE ps.run_id = ?
              AND ps.relation_type = 'exposed_to_signal'
              AND ps.source_id IN (?, ?)
            ORDER BY ps.source_id, s.id
            """,
            ("run-query", "position:MU", "position:GLD"),
        ).fetchall()
        signal_details = " | ".join(str(row["detail"]) for row in signal_plan)
        assert "idx_snapshot_edges_run_relation_source_target" in signal_details

        thesis_plan = conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT ht.source_id, eb.target_id
            FROM snapshot_edges ht
            LEFT JOIN snapshot_edges eb
              ON eb.run_id = ht.run_id
             AND eb.source_id = ht.target_id
             AND eb.relation_type = 'evaluated_by'
            WHERE ht.run_id = ?
              AND ht.relation_type = 'has_thesis'
              AND ht.source_id IN (?, ?)
            ORDER BY ht.source_id, eb.target_id
            """,
            ("run-query", "position:MU", "position:GLD"),
        ).fetchall()
        thesis_details = " | ".join(str(row["detail"]) for row in thesis_plan)
        assert "idx_snapshot_edges_run_relation_source_target" in thesis_details


def test_direct_sqlite_edge_insert_rejects_missing_endpoint(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    OntologyRepository(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO edges(source_id, target_id, relation_type, properties_json, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                ("position:missing", "asset:missing", "references_asset", "{}"),
            )


def test_upsert_edges_rejects_cardinality_conflict(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    repo.upsert_graph(_core_nodes(), _core_edges())
    repo.upsert_nodes(
        [OntologyNode(id="asset:NVDA", type="Asset", label="NVDA", properties={"ticker": "NVDA", "asset": "equity"})]
    )

    with pytest.raises(OntologySchemaValidationError, match="only one target"):
        repo.upsert_edges([OntologyEdge("position:MU", "asset:NVDA", "references_asset", {})])


def test_backfill_dry_run_reports_relation_errors(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO nodes(id, type, label, properties_json, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            ("position:MU", "Position", "MU", json.dumps({"ticker": "MU"})),
        )

    report = repo.backfill_schema_versions(dry_run=True)

    assert report["errors"]
    assert "references_asset" in report["errors"][0]["error"]


def test_backfill_write_rejects_cardinality_conflict(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_edges_unique_source_relation")
        conn.executemany(
            """
            INSERT INTO nodes(id, type, label, properties_json, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            [
                ("position:MU", "Position", "MU", json.dumps({"ticker": "MU"})),
                ("asset:MU", "Asset", "MU", json.dumps({"ticker": "MU", "asset": "equity"})),
                ("asset:NVDA", "Asset", "NVDA", json.dumps({"ticker": "NVDA", "asset": "equity"})),
                (
                    "sector:information_technology",
                    "Sector",
                    "Information Technology",
                    json.dumps({"name": "Information Technology"}),
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO edges(source_id, target_id, relation_type, properties_json, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            [
                ("position:MU", "asset:MU", "references_asset", "{}"),
                ("position:MU", "asset:NVDA", "references_asset", "{}"),
                ("asset:MU", "sector:information_technology", "belongs_to_sector", json.dumps({"source": "test"})),
                ("asset:NVDA", "sector:information_technology", "belongs_to_sector", json.dumps({"source": "test"})),
            ],
        )

    with pytest.raises(OntologySchemaValidationError, match="only one target"):
        repo.backfill_schema_versions(dry_run=False)


def test_upsert_graph_rejects_noncanonical_position_and_does_not_mutate_live_graph(tmp_path, semantic_graph_factory):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    repo.upsert_graph(_core_nodes(), _core_edges())
    before = repo.fetch_graph()

    nodes, edges = semantic_graph_factory(position_id="position:WRONG")

    with pytest.raises(OntologySchemaValidationError, match="non-canonical identity"):
        repo.upsert_graph(nodes, edges)

    after = repo.fetch_graph()
    assert after == before
    assert {node["id"] for node in after["nodes"]} == {
        "position:MU",
        "asset:MU",
        "sector:information_technology",
    }


def test_save_snapshot_rejects_invalid_signal_exposure_and_does_not_persist_partial_run(
    tmp_path, semantic_graph_factory
):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    nodes, edges = semantic_graph_factory(run_id="run-invalid", invalid_exposure=True)

    with pytest.raises(OntologySchemaValidationError, match="Invalid exposed_to_signal"):
        repo.save_snapshot(
            run_id="run-invalid",
            as_of="2026-03-08T00:00:00Z",
            source_status={"portfolio": {"status": "ok"}},
            required_modules=["portfolio"],
            optional_modules=[],
            component_scores={"macro_regime": 0.5},
            nodes=nodes,
            edges=edges,
        )

    assert repo.get_run("run-invalid") is None
    with sqlite3.connect(db_path) as conn:
        run_count = conn.execute("SELECT COUNT(*) FROM ontology_runs WHERE run_id = 'run-invalid'").fetchone()[0]
        node_count = conn.execute("SELECT COUNT(*) FROM snapshot_nodes WHERE run_id = 'run-invalid'").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM snapshot_edges WHERE run_id = 'run-invalid'").fetchone()[0]
    assert (run_count, node_count, edge_count) == (0, 0, 0)


def test_save_snapshot_emits_success_and_failure_audits_with_lineage_and_counts(
    tmp_path,
    monkeypatch,
    semantic_graph_factory,
    source_status_factory,
):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    nodes, edges = semantic_graph_factory(run_id="run-audit", include_optional=True)
    source_status = source_status_factory(required_status="ok", optional_status="partial")

    repo.save_snapshot(
        run_id="run-audit",
        as_of="2026-05-01T20:00:00+00:00",
        source_status=source_status,
        required_modules=["portfolio"],
        optional_modules=["sentiment"],
        component_scores={"macro_regime": 0.55},
        nodes=nodes,
        edges=edges,
    )

    real_connect = repo._connect

    class _FailingConnection:
        def __init__(self, conn):
            self._conn = conn

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

        def execute(self, sql, *args, **kwargs):
            return self._conn.execute(sql, *args, **kwargs)

        def executemany(self, sql, *args, **kwargs):
            if "INSERT INTO snapshot_nodes" in str(sql):
                raise sqlite3.IntegrityError("forced snapshot failure")
            return self._conn.executemany(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(repo, "_connect", lambda: _FailingConnection(real_connect()))

    with pytest.raises(OntologySchemaValidationError, match="forced snapshot failure"):
        repo.save_snapshot(
            run_id="run-failed-audit",
            as_of="2026-05-02T20:00:00+00:00",
            source_status=source_status,
            required_modules=["portfolio"],
            optional_modules=["sentiment"],
            component_scores={"macro_regime": 0.65},
            nodes=nodes,
            edges=edges,
        )

    rows = core_db.get_audit_events(action_name="ontology.snapshot.saved", limit=10)
    succeeded = [row for row in rows if row["status"] == "succeeded"][0]
    failed = [row for row in rows if row["status"] == "failed"][0]

    assert succeeded["object_refs"] == [{"type": "ontology_run", "id": "run-audit"}]
    assert succeeded["after_summary"]["run_id"] == "run-audit"
    assert succeeded["after_summary"]["node_count"] == len(nodes)
    assert succeeded["source_lineage"]["source_status_counts"]["ok"] == 1
    assert succeeded["source_lineage"]["source_status_counts"]["partial"] == 1

    assert failed["object_refs"] == [{"type": "ontology_run", "id": "run-failed-audit"}]
    assert failed["after_summary"]["run_id"] == "run-failed-audit"
    assert "forced snapshot failure" in str(failed["error"])
    assert failed["source_lineage"]["run_id"] == "run-failed-audit"
    assert repo.get_run("run-failed-audit") is None


def test_backfill_failure_reports_exact_run_and_leaves_rows_unchanged(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    _insert_minimal_legacy_snapshot(db_path, run_id="run-a")
    _insert_minimal_legacy_snapshot(db_path, run_id="run-b")

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_snapshot_edges_unique_run_source_relation")
        conn.execute(
            """
            INSERT INTO snapshot_nodes(run_id, id, type, label, properties_json, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            ("run-b", "asset:NVDA", "Asset", "NVDA", json.dumps({"ticker": "NVDA", "asset": "equity"})),
        )
        conn.execute(
            """
            INSERT INTO snapshot_edges(run_id, source_id, target_id, relation_type, properties_json, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            ("run-b", "position:MU", "asset:NVDA", "references_asset", "{}"),
        )
        conn.commit()

    with pytest.raises(OntologySchemaValidationError, match="run-b"):
        repo.backfill_schema_versions(dry_run=False)

    report = repo.backfill_schema_versions(dry_run=True)
    assert any(
        scope.get("run_id") == "run-b" and "only one target" in str(scope.get("error")) for scope in report["scopes"]
    )

    with sqlite3.connect(db_path) as conn:
        run_a_version = conn.execute(
            "SELECT schema_version FROM snapshot_nodes WHERE run_id = ? AND id = ?",
            ("run-a", "position:MU"),
        ).fetchone()[0]
        run_b_edge_count = conn.execute(
            "SELECT COUNT(*) FROM snapshot_edges WHERE run_id = ? AND source_id = ? AND relation_type = ?",
            ("run-b", "position:MU", "references_asset"),
        ).fetchone()[0]
    assert run_a_version == 0
    assert run_b_edge_count == 2


def test_schema_bindings_preserve_stored_vs_upgraded_read_contract_after_schema_evolution(tmp_path, monkeypatch):
    from ontology.schemas import registry
    from ontology.schemas.objects import PositionV1

    class PositionV2(PositionV1):
        schema_version: Literal[2] = 2
        risk_bucket: str

    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    repo.save_snapshot(
        run_id="run-bindings",
        as_of="2026-03-08T00:00:00Z",
        source_status={"portfolio": {"status": "ok"}},
        required_modules=["portfolio"],
        optional_modules=[],
        component_scores={"macro_regime": 0.5},
        nodes=_core_nodes(),
        edges=_core_edges(),
    )

    run_meta = repo.get_run("run-bindings")
    assert run_meta is not None
    assert any(
        binding["schema_kind"] == "ontology_object"
        and binding["schema_name"] == "Position"
        and binding["schema_version"] == 1
        for binding in run_meta["schema_bindings"]
    )

    monkeypatch.setitem(registry.NODE_SCHEMAS, "Position", PositionV2)
    monkeypatch.setitem(
        registry.NODE_UPGRADE_ADAPTERS,
        ("Position", 1, 2),
        lambda payload: {**payload, "schema_version": 2, "risk_bucket": "evolved"},
    )

    stored = repo.fetch_snapshot_graph("run-bindings", schema_mode="stored")
    upgraded = repo.fetch_snapshot_graph("run-bindings", schema_mode="upgraded")

    stored_position = [node for node in stored["nodes"] if node["type"] == "Position"][0]
    upgraded_position = [node for node in upgraded["nodes"] if node["type"] == "Position"][0]
    assert stored_position["schema_version"] == 1
    assert "risk_bucket" not in stored_position["properties"]
    assert upgraded_position["properties"]["schema_version"] == 2
    assert upgraded_position["properties"]["risk_bucket"] == "evolved"


def test_ingestion_snapshot_persists_canonical_thesis_evaluation_catalyst_signal_mapping(tmp_path, monkeypatch):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)

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
    monkeypatch.setattr(
        ingestion,
        "compute_volatility_cluster",
        lambda *_args, **_kwargs: (
            0.4,
            [
                {
                    "source": "macro",
                    "name": "Vol Signal",
                    "threshold": "higher is worse",
                    "direction": "stable",
                }
            ],
        ),
    )
    monkeypatch.setattr(
        ingestion,
        "compute_breadth_stress",
        lambda *_args, **_kwargs: (
            0.5,
            [
                {
                    "source": "market_breadth",
                    "name": "Breadth Signal",
                    "threshold": "higher is worse",
                    "direction": "stable",
                }
            ],
        ),
    )
    monkeypatch.setattr(ingestion, "compute_sector_stress_map", lambda *_args, **_kwargs: ({"MU": 0.6}, []))
    monkeypatch.setattr(
        ingestion,
        "compute_macro_regime",
        lambda *_args, **_kwargs: (
            0.7,
            [
                {
                    "source": "liquidity",
                    "name": "Macro Regime",
                    "threshold": "higher is worse",
                    "direction": "deteriorating",
                }
            ],
        ),
    )
    monkeypatch.setattr(
        "portfolio.thesis_db.get_all_thesis_meta",
        lambda: [{"ticker": "MU", "status": "active", "created_at": "2026-05-01", "updated_at": "2026-05-02"}],
    )
    monkeypatch.setattr(
        "portfolio.thesis_db.get_latest_evaluations",
        lambda: [
            {
                "ticker": "MU",
                "evaluated_at": "2026-05-02T00:00:00Z",
                "thesis_status": "active",
                "technical_read": "supportive",
                "fundamental_read": "supportive",
                "action": "hold",
                "confidence": "high",
            }
        ],
    )
    monkeypatch.setattr(
        ingestion,
        "_parse_catalysts_from_thesis",
        lambda ticker: [{"name": "Memory recovery", "description": "Demand improves"}] if ticker == "MU" else [],
    )

    out = ingestion.ingest_into_repository(repo=repo, timeframe="Daily", include_deep_modules=False)
    graph = repo.fetch_snapshot_graph(out.run_id, schema_mode="upgraded")
    ids = {node["id"] for node in graph["nodes"]}

    assert "thesis:MU" in ids
    assert evaluation_id("MU", "2026-05-02T00:00:00Z") in ids
    assert catalyst_id("MU", "Memory recovery", "Demand improves") in ids
    assert signal_id("macro", "Vol Signal") in ids
    assert signal_id("market_breadth", "Breadth Signal") in ids
    assert signal_id("liquidity", "Macro Regime") in ids
    assert "evaluation:MU:2026-05-02T00:00:00Z" not in ids
    assert "catalyst:MU:0" not in ids


def _core_nodes() -> list[OntologyNode]:
    return [
        OntologyNode(id="position:MU", type="Position", label="MU", properties={"ticker": "MU"}),
        OntologyNode(id="asset:MU", type="Asset", label="MU", properties={"ticker": "MU", "asset": "equity"}),
        OntologyNode(
            id="sector:information_technology",
            type="Sector",
            label="Information Technology",
            properties={"name": "Information Technology"},
        ),
    ]


def _core_edges() -> list[OntologyEdge]:
    return [
        OntologyEdge("position:MU", "asset:MU", "references_asset", {}),
        OntologyEdge("asset:MU", "sector:information_technology", "belongs_to_sector", {"source": "test"}),
    ]


def _insert_minimal_legacy_snapshot(db_path, *, run_id: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ontology_runs(
                run_id,
                as_of,
                source_status_json,
                required_modules_json,
                optional_modules_json,
                component_scores_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (run_id, "2026-03-08T00:00:00Z", "{}", "[]", "[]", "{}"),
        )
        conn.executemany(
            """
            INSERT INTO snapshot_nodes(run_id, id, type, label, properties_json, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    run_id,
                    "position:MU",
                    "Position",
                    "MU",
                    json.dumps(
                        {
                            "ticker": "MU",
                            "asset": "equity",
                            "direction": "long",
                            "risk_score": 0.72,
                            "risk_level": "medium",
                            "ontology_run_id": run_id,
                        }
                    ),
                ),
                (run_id, "asset:MU", "Asset", "MU", json.dumps({"ticker": "MU", "asset": "equity"})),
                (
                    run_id,
                    "sector:information_technology",
                    "Sector",
                    "Information Technology",
                    json.dumps({"name": "Information Technology", "source": "test"}),
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO snapshot_edges(run_id, source_id, target_id, relation_type, properties_json, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (run_id, "position:MU", "asset:MU", "references_asset", "{}"),
                (
                    run_id,
                    "asset:MU",
                    "sector:information_technology",
                    "belongs_to_sector",
                    json.dumps({"source": "test"}),
                ),
            ],
        )


def _insert_snapshot_query_fixture(db_path, *, run_id: str) -> None:
    repo = OntologyRepository(db_path=db_path)
    nodes = [
        OntologyNode(
            id="position:MU",
            type="Position",
            label="MU",
            properties={
                "ticker": "MU",
                "asset": "equity",
                "direction": "long",
                "risk_score": 0.72,
                "risk_level": "medium",
                "ontology_run_id": run_id,
            },
        ),
        OntologyNode(
            id="position:NVDA",
            type="Position",
            label="NVDA",
            properties={
                "ticker": "NVDA",
                "asset": "equity",
                "direction": "long",
                "risk_score": 0.81,
                "risk_level": "high",
                "ontology_run_id": run_id,
            },
        ),
        OntologyNode(
            id="position:GLD",
            type="Position",
            label="GLD",
            properties={
                "ticker": "GLD",
                "asset": "commodity",
                "direction": "long",
                "risk_score": 0.31,
                "risk_level": "low",
                "ontology_run_id": run_id,
            },
        ),
        OntologyNode(id="asset:MU", type="Asset", label="MU", properties={"ticker": "MU", "asset": "equity"}),
        OntologyNode(id="asset:NVDA", type="Asset", label="NVDA", properties={"ticker": "NVDA", "asset": "equity"}),
        OntologyNode(id="asset:GLD", type="Asset", label="GLD", properties={"ticker": "GLD", "asset": "commodity"}),
        OntologyNode(
            id="sector:information_technology",
            type="Sector",
            label="Information Technology",
            properties={"name": "Information Technology"},
        ),
        OntologyNode(
            id="sector:commodities",
            type="Sector",
            label="Commodities",
            properties={"name": "Commodities"},
        ),
        OntologyNode(id="signal:test:mu", type="Signal", label="MU", properties={"source": "test"}),
        OntologyNode(id="signal:test:nvda", type="Signal", label="NVDA", properties={"source": "test"}),
        OntologyNode(id="signal:test:gld", type="Signal", label="GLD", properties={"source": "test"}),
        OntologyNode(
            id="thesis:MU",
            type="Thesis",
            label="Thesis: MU",
            properties={"ticker": "MU", "status": "active", "ontology_run_id": run_id},
        ),
        OntologyNode(
            id=evaluation_id("MU", "2026-03-08T00:00:00Z"),
            type="Evaluation",
            label="Evaluation: MU",
            properties={
                "ticker": "MU",
                "evaluated_at": "2026-03-08T00:00:00Z",
                "thesis_status": "strengthen",
                "technical_read": "supportive",
                "fundamental_read": "supportive",
                "action": "hold",
                "confidence": "high",
                "ontology_run_id": run_id,
            },
        ),
        OntologyNode(
            id=catalyst_id("MU", "Memory recovery", "Demand improves"),
            type="Catalyst",
            label="Memory recovery",
            properties={
                "ticker": "MU",
                "name": "Memory recovery",
                "description": "Demand improves",
                "ontology_run_id": run_id,
            },
        ),
    ]
    edges = [
        OntologyEdge("position:MU", "asset:MU", "references_asset", {"ontology_run_id": run_id}),
        OntologyEdge("position:NVDA", "asset:NVDA", "references_asset", {"ontology_run_id": run_id}),
        OntologyEdge("position:GLD", "asset:GLD", "references_asset", {"ontology_run_id": run_id}),
        OntologyEdge(
            "asset:MU",
            "sector:information_technology",
            "belongs_to_sector",
            {"source": "test", "ontology_run_id": run_id},
        ),
        OntologyEdge(
            "asset:NVDA",
            "sector:information_technology",
            "belongs_to_sector",
            {"source": "test", "ontology_run_id": run_id},
        ),
        OntologyEdge(
            "asset:GLD",
            "sector:commodities",
            "belongs_to_sector",
            {"source": "test", "ontology_run_id": run_id},
        ),
        OntologyEdge(
            "position:MU",
            "signal:test:mu",
            "exposed_to_signal",
            {"source": "test", "name": "MU", "contribution": 0.22, "ontology_run_id": run_id},
        ),
        OntologyEdge(
            "position:NVDA",
            "signal:test:nvda",
            "exposed_to_signal",
            {"source": "test", "name": "NVDA", "contribution": 0.41, "ontology_run_id": run_id},
        ),
        OntologyEdge(
            "position:GLD",
            "signal:test:gld",
            "exposed_to_signal",
            {"source": "test", "name": "GLD", "contribution": 0.11, "ontology_run_id": run_id},
        ),
        OntologyEdge("position:MU", "thesis:MU", "has_thesis", {"ontology_run_id": run_id}),
        OntologyEdge(
            "thesis:MU",
            evaluation_id("MU", "2026-03-08T00:00:00Z"),
            "evaluated_by",
            {"ontology_run_id": run_id},
        ),
        OntologyEdge(
            "thesis:MU",
            catalyst_id("MU", "Memory recovery", "Demand improves"),
            "has_catalyst",
            {"ontology_run_id": run_id},
        ),
    ]
    repo.save_snapshot(
        run_id=run_id,
        as_of="2026-03-08T00:00:00Z",
        source_status={"portfolio": {"status": "ok"}},
        required_modules=["portfolio"],
        optional_modules=[],
        component_scores={"macro_regime": 0.5},
        nodes=nodes,
        edges=edges,
    )
