from __future__ import annotations

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
