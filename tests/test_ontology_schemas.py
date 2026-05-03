from __future__ import annotations

import pytest
from pydantic import ValidationError

from ontology.models import OntologyEdge, OntologyNode
from ontology.schemas.identity import evaluation_id, signal_id
from ontology.schemas.objects import PositionV1
from ontology.schemas.registry import OntologySchemaValidationError, normalize_graph, normalize_node


def test_position_schema_normalizes_and_checks_risk_level():
    model = PositionV1(
        ticker=" mu ",
        asset=" Equity ",
        direction=" Long ",
        timeframe="Daily",
        latest_price=72.4,
        as_of="2026-03-08T00:00:00Z",
        risk_score=0.72,
        risk_level="medium",
        volatility_cluster=0.6,
        breadth_stress=0.7,
        sector_stress=0.5,
        macro_regime=0.4,
        ontology_run_id="run-1",
    )

    assert model.ticker == "MU"
    assert model.asset == "equity"
    assert model.direction == "long"

    with pytest.raises(ValidationError):
        PositionV1(
            ticker="MU",
            asset="equity",
            direction="long",
            timeframe="Daily",
            risk_score=0.81,
            risk_level="medium",
            volatility_cluster=0.6,
            breadth_stress=0.7,
            sector_stress=0.5,
            macro_regime=0.4,
            ontology_run_id="run-1",
        )


def test_legacy_signal_node_is_canonicalized_to_stable_identity():
    node = normalize_node(
        OntologyNode(
            id="signal:test",
            type="Signal",
            label="Test Signal",
            properties={"source": "test", "ontology_run_id": "run-1"},
        ),
        run_id="run-1",
    )

    assert node.id == signal_id("test", "Test Signal")
    assert node.schema_name == "Signal"
    assert node.schema_version == 1
    assert node.properties["schema_version"] == 1


def test_graph_validation_rejects_missing_core_edge_endpoint():
    nodes = [
        OntologyNode(
            id="position:MU",
            type="Position",
            label="MU",
            properties={
                "ticker": "MU",
                "asset": "equity",
                "direction": "long",
                "risk_score": 0.2,
                "risk_level": "low",
                "ontology_run_id": "run-1",
            },
        )
    ]
    edges = [OntologyEdge("position:MU", "signal:missing", "exposed_to_signal", {"ontology_run_id": "run-1"})]

    with pytest.raises(OntologySchemaValidationError, match="missing target"):
        normalize_graph(nodes, edges, run_id="run-1")


def test_graph_validation_can_skip_invalid_optional_thesis_node():
    nodes = [
        OntologyNode(
            id="position:MU",
            type="Position",
            label="MU",
            properties={
                "ticker": "MU",
                "asset": "equity",
                "direction": "long",
                "risk_score": 0.2,
                "risk_level": "low",
                "ontology_run_id": "run-1",
            },
        ),
        OntologyNode(
            id="thesis:MU",
            type="Thesis",
            label="Thesis: MU",
            properties={"schema_version": 1, "ticker": "MU"},
            schema_name="Thesis",
            schema_version=1,
        ),
    ]
    edges = [OntologyEdge("position:MU", "thesis:MU", "has_thesis", {"ontology_run_id": "run-1"})]

    graph = normalize_graph(nodes, edges, run_id="run-1", skip_optional_invalid=True)

    assert [node.id for node in graph.nodes] == ["position:MU"]
    assert graph.edges == []
    assert graph.warnings


def test_evaluation_identity_uses_canonical_timestamp_key():
    assert evaluation_id("mu", "2026-03-08T00:00:00Z") == "evaluation:MU:2026_03_08t00_00_00_00_00"
