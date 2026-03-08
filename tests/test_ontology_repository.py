from __future__ import annotations

import sqlite3

from ontology.models import OntologyEdge, OntologyNode
from ontology.repository import OntologyRepository


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
    assert evidence[0]["signal_id"] == "signal:test"
    assert evidence[0]["edge_props"]["contribution"] == 0.3

    graph = repo.fetch_graph()
    assert len(graph["nodes"]) == 4
    assert len(graph["edges"]) == 3


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
    assert ev_run_1[0]["signal_id"] == "signal:s1"
    assert ev_run_2[0]["signal_id"] == "signal:s2"

    meta = repo.get_run("run-2")
    assert meta is not None
    assert meta["run_id"] == "run-2"
    assert meta["as_of"] == "2026-03-09T00:00:00Z"


def test_snapshot_prune_removes_old_runs(tmp_path):
    db_path = tmp_path / "ontology.sqlite3"
    repo = OntologyRepository(db_path=db_path)

    nodes = [OntologyNode(id="position:MU", type="Position", label="MU", properties={"ticker": "MU"})]
    edges: list[OntologyEdge] = []

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
