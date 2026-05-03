from __future__ import annotations

import json
import sqlite3

import pytest

from ontology.models import OntologyEdge, OntologyNode
from ontology.repository import OntologyRepository
from ontology.schemas.identity import catalyst_id, evaluation_id
from ontology.schemas.registry import OntologySchemaValidationError


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

    rows_run_1 = repo.fetch_snapshot_position_asset_sector_rows("run-1")
    rows_run_2 = repo.fetch_snapshot_position_asset_sector_rows("run-2")
    assert rows_run_1[0]["position_props"]["risk_level"] == "medium"
    assert rows_run_2[0]["position_props"]["risk_level"] == "high"

    ev_run_1 = repo.fetch_snapshot_position_signal_evidence("run-1", "position:MU")
    ev_run_2 = repo.fetch_snapshot_position_signal_evidence("run-2", "position:MU")
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
    graph = repo.fetch_snapshot_graph("run-legacy")
    ids = {node["id"] for node in graph["nodes"]}
    assert evaluation_id("MU", "2026-03-08T00:00:00Z") in ids
    assert catalyst_id("MU", "Demand recovery", "Demand improves") in ids
    assert all(edge["properties"]["ontology_run_id"] == "run-legacy" for edge in graph["edges"])
    belongs = [edge for edge in graph["edges"] if edge["relation_type"] == "belongs_to_sector"][0]
    assert belongs["properties"]["source"] == "test"
    assert all(node["schema_version"] == 1 for node in graph["nodes"])
    assert all(edge["schema_version"] == 1 for edge in graph["edges"])


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
