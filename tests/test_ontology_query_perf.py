from __future__ import annotations

import os
from time import perf_counter

import pytest

import ontology.service as svc
from ontology.models import OntologyEdge, OntologyNode
from ontology.repository import OntologyRepository


def _timed(fn):
    start = perf_counter()
    result = fn()
    return perf_counter() - start, result


def _seed_perf_snapshot(repo: OntologyRepository, *, run_id: str, size: int) -> None:
    nodes: list[OntologyNode] = [
        OntologyNode(
            id="sector:information_technology",
            type="Sector",
            label="Information Technology",
            properties={"name": "Information Technology"},
        )
    ]
    edges: list[OntologyEdge] = []

    for index in range(size):
        ticker = f"T{index:05d}"
        position = f"position:{ticker}"
        asset = f"asset:{ticker}"
        signal = f"signal:test:{ticker.lower()}"
        risk_score = 0.9 if index % 5 == 0 else 0.55 if index % 2 == 0 else 0.25
        risk_level = "high" if risk_score >= 0.75 else "medium" if risk_score >= 0.5 else "low"
        nodes.extend(
            [
                OntologyNode(
                    id=position,
                    type="Position",
                    label=ticker,
                    properties={
                        "ticker": ticker,
                        "asset": "equity",
                        "direction": "long",
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "ontology_run_id": run_id,
                    },
                ),
                OntologyNode(id=asset, type="Asset", label=ticker, properties={"ticker": ticker, "asset": "equity"}),
                OntologyNode(id=signal, type="Signal", label=ticker, properties={"source": "test"}),
            ]
        )
        edges.extend(
            [
                OntologyEdge(position, asset, "references_asset", {"ontology_run_id": run_id}),
                OntologyEdge(
                    asset,
                    "sector:information_technology",
                    "belongs_to_sector",
                    {"source": "test", "ontology_run_id": run_id},
                ),
                OntologyEdge(
                    position,
                    signal,
                    "exposed_to_signal",
                    {
                        "source": "test",
                        "name": ticker,
                        "contribution": round(risk_score / 2, 4),
                        "ontology_run_id": run_id,
                    },
                ),
            ]
        )

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


@pytest.mark.skipif(
    os.getenv("RUN_ONTOLOGY_PERF") != "1", reason="set RUN_ONTOLOGY_PERF=1 to run local ontology perf benchmark"
)
@pytest.mark.parametrize("size", [1000, 5000, 20000])
def test_ontology_query_perf_benchmark(tmp_path, size: int):
    db_path = tmp_path / f"ontology-perf-{size}.sqlite3"
    repo = OntologyRepository(db_path=db_path)
    run_id = f"run-perf-{size}"
    _seed_perf_snapshot(repo, run_id=run_id, size=size)

    old_rows_s, old_rows = _timed(
        lambda: repo.fetch_snapshot_position_asset_sector_rows(run_id, schema_mode="upgraded")
    )
    old_evidence_s, old_evidence = _timed(
        lambda: repo.fetch_snapshot_all_position_signal_evidence(run_id, schema_mode="upgraded")
    )
    old_graph_s, old_graph = _timed(lambda: repo.fetch_snapshot_graph(run_id, schema_mode="upgraded"))

    page_s, page_data = _timed(
        lambda: repo.query_snapshot_positions_page(
            run_id,
            filters={},
            page=1,
            page_size=25,
            schema_mode="upgraded",
        )
    )
    filtered_page_s, filtered_page = _timed(
        lambda: repo.query_snapshot_positions_page(
            run_id,
            filters={"min_risk_score": 0.75},
            page=1,
            page_size=25,
            schema_mode="upgraded",
        )
    )
    position_ids = [str(row["position_id"]) for row in page_data["rows"]]
    batch_s, batch_evidence = _timed(
        lambda: repo.fetch_snapshot_position_signal_evidence_batch(
            run_id,
            position_ids,
            schema_mode="upgraded",
        )
    )
    graph_s, page_graph = _timed(lambda: svc._build_page_graph(page_data["rows"], batch_evidence, {}, run_id=run_id))

    print(
        f"size={size} old_rows={old_rows_s:.4f}s old_evidence={old_evidence_s:.4f}s "
        f"old_graph={old_graph_s:.4f}s page={page_s:.4f}s filtered_page={filtered_page_s:.4f}s "
        f"batch_evidence={batch_s:.4f}s page_graph={graph_s:.4f}s"
    )

    assert len(old_rows) == size
    assert len(old_evidence) == size
    assert len(old_graph["nodes"]) > len(page_graph[0]["nodes"])
    assert page_data["total_results"] == size
    assert len(page_data["rows"]) <= 25
    assert len(filtered_page["rows"]) <= 25
    assert len(batch_evidence) <= 25
    assert page_graph[1]["scope"] == "page"
    assert page_graph[1]["max_nodes"] == svc.GRAPH_PAGE_NODE_LIMIT
