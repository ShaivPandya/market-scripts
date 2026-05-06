from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from ontology.models import EntityType, OntologyEdge, OntologyNode
from ontology.schema_definitions import SCHEMA_KIND_ONTOLOGY_OBJECT, ontology_schema_definitions
from ontology.schemas.identity import action_item_id, evaluation_id, hedge_position_id, signal_id
from ontology.schemas.objects import ActionItemV1, HedgePositionV1, InvestmentIdeaV1, PositionV1
from ontology.schemas.registry import NODE_SCHEMAS, OntologySchemaValidationError, normalize_graph, normalize_node

ROOT = Path(__file__).resolve().parents[1]


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


def test_operational_object_schemas_have_stable_identities():
    hedge = normalize_node(
        OntologyNode(
            id=hedge_position_id("MU"),
            type="HedgePosition",
            label="MU hedge",
            properties=HedgePositionV1(ticker="mu", direction="short").model_dump(mode="json"),
            schema_name="HedgePosition",
            schema_version=1,
        ),
        allow_legacy=False,
    )
    action_item = normalize_node(
        OntologyNode(
            id=action_item_id(42),
            type="ActionItem",
            label="Review MU",
            properties=ActionItemV1(legacy_id=42, description="Review MU").model_dump(mode="json"),
            schema_name="ActionItem",
            schema_version=1,
        ),
        allow_legacy=False,
    )

    assert hedge.id == "hedge_position:MU"
    assert hedge.properties["ticker"] == "MU"
    assert action_item.id == "action_item:42"
    assert action_item.properties["status"] == "open"


def test_account_schema_drops_deprecated_tax_lot_field():
    account = normalize_node(
        OntologyNode(
            id="account:default_account",
            type="Account",
            label="Default Account",
            properties={
                "schema_version": 1,
                "account_id": "default-account",
                "investor_id": "default-investor",
                "account_type": "unspecified",
                "tax_status": "unknown",
                "tax_lot_data_available": None,
                "ontology_run_id": "operational",
            },
            schema_name="Account",
            schema_version=1,
        ),
        allow_legacy=False,
    )

    assert account.schema_version == 1
    assert account.properties["account_id"] == "default_account"
    assert "tax_lot_data_available" not in account.properties


def test_runtime_migration_schema_rejects_unregistered_fields():
    with pytest.raises(ValidationError):
        InvestmentIdeaV1(
            idea_id="investment_idea:MU",
            ticker="MU",
            status="watching",
            unregistered_field=True,
        )


def test_every_entity_type_has_pydantic_schema_and_definition():
    entity_types = set(get_args(EntityType))
    definitions = {
        definition.schema_name
        for definition in ontology_schema_definitions()
        if definition.schema_kind == SCHEMA_KIND_ONTOLOGY_OBJECT and definition.schema_version == 1
    }

    assert entity_types <= set(NODE_SCHEMAS)
    assert entity_types <= definitions


def test_literal_write_object_calls_use_registered_object_types():
    registered = set(NODE_SCHEMAS)
    offenders: list[str] = []
    files = subprocess.check_output(
        [
            "rg",
            "--files",
            "-g",
            "*.py",
            "-g",
            "!.venv/**",
            "-g",
            "!frontend/node_modules/**",
            "-g",
            "!**/__pycache__/**",
            "-g",
            "!tests/**",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for rel_path in files:
        path = ROOT / rel_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_object"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value not in registered
            ):
                offenders.append(f"{rel_path}:{node.lineno}:{node.args[0].value}")

    assert offenders == []


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

    graph = normalize_graph(nodes, edges, run_id="run-1", skip_optional_invalid=True, require_core_edges=False)

    assert [node.id for node in graph.nodes] == ["position:MU"]
    assert graph.edges == []
    assert graph.warnings


def test_evaluation_identity_uses_canonical_timestamp_key():
    assert evaluation_id("mu", "2026-03-08T00:00:00Z") == "evaluation:MU:2026_03_08t00_00_00_00_00"


def test_relation_registry_rejects_unsupported_relation_type():
    with pytest.raises(OntologySchemaValidationError, match="Unsupported relation type"):
        normalize_graph(_core_nodes(), [OntologyEdge("position:MU", "asset:MU", "owns", {})])


def test_relation_registry_rejects_wrong_endpoint_types():
    with pytest.raises(OntologySchemaValidationError, match="must connect Position->Asset"):
        normalize_graph(_core_nodes(), [OntologyEdge("asset:MU", "position:MU", "references_asset", {})])


def test_relation_registry_rejects_missing_v1_properties():
    edge = OntologyEdge(
        "position:MU",
        "asset:MU",
        "references_asset",
        {"schema_version": 1},
        schema_name="Relation",
        schema_version=1,
    )

    with pytest.raises(OntologySchemaValidationError, match="Invalid references_asset"):
        normalize_graph(_core_nodes(), [edge])


def test_relation_registry_rejects_belongs_to_sector_without_source():
    edges = [
        OntologyEdge("position:MU", "asset:MU", "references_asset", {"ontology_run_id": "run-1"}),
        OntologyEdge(
            "asset:MU",
            "sector:information_technology",
            "belongs_to_sector",
            {"schema_version": 1, "ontology_run_id": "run-1"},
            schema_name="Relation",
            schema_version=1,
        ),
    ]

    with pytest.raises(OntologySchemaValidationError, match="source"):
        normalize_graph(_core_nodes(), edges, run_id="run-1")


def test_relation_registry_rejects_invalid_exposure_contribution():
    nodes = _core_nodes() + [
        OntologyNode(id="signal:test", type="Signal", label="Test Signal", properties={"source": "test"}),
    ]
    edges = _core_edges() + [
        OntologyEdge(
            "position:MU",
            "signal:test",
            "exposed_to_signal",
            {
                "schema_version": 1,
                "component": "test",
                "source": "test",
                "name": "Test Signal",
                "threshold": "higher is worse",
                "direction": "stable",
                "contribution": 1.2,
                "ontology_run_id": "run-1",
            },
            schema_name="PositionSignalExposure",
            schema_version=1,
        )
    ]

    with pytest.raises(OntologySchemaValidationError, match="Invalid exposed_to_signal"):
        normalize_graph(nodes, edges, run_id="run-1")


def test_relation_registry_rejects_position_referencing_two_assets():
    nodes = _core_nodes() + [
        OntologyNode(id="asset:NVDA", type="Asset", label="NVDA", properties={"ticker": "NVDA", "asset": "equity"}),
    ]
    edges = _core_edges() + [
        OntologyEdge("position:MU", "asset:NVDA", "references_asset", {"ontology_run_id": "run-1"})
    ]

    with pytest.raises(OntologySchemaValidationError, match="only one target"):
        normalize_graph(nodes, edges, run_id="run-1")


def test_relation_registry_rejects_asset_belonging_to_two_sectors():
    nodes = _core_nodes() + [
        OntologyNode(
            id="sector:semiconductors",
            type="Sector",
            label="Semiconductors",
            properties={"name": "Semiconductors"},
        ),
    ]
    edges = _core_edges() + [
        OntologyEdge("asset:MU", "sector:semiconductors", "belongs_to_sector", {"source": "override"})
    ]

    with pytest.raises(OntologySchemaValidationError, match="only one target"):
        normalize_graph(nodes, edges, run_id="run-1")


def test_relation_registry_rejects_multiple_theses_for_position():
    nodes = _core_nodes() + [
        _thesis_node("MU"),
        _thesis_node("NVDA"),
    ]
    edges = _core_edges() + [
        OntologyEdge("position:MU", "thesis:MU", "has_thesis", {"ontology_run_id": "run-1"}),
        OntologyEdge("position:MU", "thesis:NVDA", "has_thesis", {"ontology_run_id": "run-1"}),
    ]

    with pytest.raises(OntologySchemaValidationError, match="only one target"):
        normalize_graph(nodes, edges, run_id="run-1")


def test_relation_registry_rejects_two_owners_for_thesis_evaluation_catalyst_and_signal():
    nodes = _two_position_core_nodes() + [
        _thesis_node("MU"),
        _thesis_node("NVDA"),
        OntologyNode(
            id="evaluation:MU:2026-03-08T00:00:00Z",
            type="Evaluation",
            label="Eval: MU",
            properties={
                "ticker": "MU",
                "evaluated_at": "2026-03-08T00:00:00Z",
                "thesis_status": "strengthen",
                "technical_read": "supportive",
                "fundamental_read": "supportive",
                "action": "hold",
                "confidence": "high",
                "ontology_run_id": "run-1",
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
                "ontology_run_id": "run-1",
            },
        ),
        OntologyNode(
            id="macro_indicator:vol",
            type="MacroIndicator",
            label="Vol",
            properties={
                "indicator_key": "vol",
                "name": "Vol",
                "source": "test",
                "as_of": "run-1",
                "ontology_run_id": "run-1",
            },
        ),
        OntologyNode(
            id="macro_indicator:breadth",
            type="MacroIndicator",
            label="Breadth",
            properties={
                "indicator_key": "breadth",
                "name": "Breadth",
                "source": "test",
                "as_of": "run-1",
                "ontology_run_id": "run-1",
            },
        ),
        OntologyNode(id="signal:test", type="Signal", label="Shared Signal", properties={"source": "test"}),
    ]
    owner_edges = _two_position_core_edges() + [
        OntologyEdge("position:MU", "thesis:MU", "has_thesis", {"ontology_run_id": "run-1"}),
        OntologyEdge("position:NVDA", "thesis:NVDA", "has_thesis", {"ontology_run_id": "run-1"}),
    ]

    cases = [
        owner_edges
        + [
            OntologyEdge("position:NVDA", "thesis:MU", "has_thesis", {"ontology_run_id": "run-1"}),
        ],
        owner_edges
        + [
            OntologyEdge(
                "thesis:MU",
                "evaluation:MU:2026-03-08T00:00:00Z",
                "evaluated_by",
                {"ontology_run_id": "run-1"},
            ),
            OntologyEdge(
                "thesis:NVDA",
                "evaluation:MU:2026-03-08T00:00:00Z",
                "evaluated_by",
                {"ontology_run_id": "run-1"},
            ),
        ],
        owner_edges
        + [
            OntologyEdge("thesis:MU", "catalyst:MU:0", "has_catalyst", {"ontology_run_id": "run-1"}),
            OntologyEdge("thesis:NVDA", "catalyst:MU:0", "has_catalyst", {"ontology_run_id": "run-1"}),
        ],
        owner_edges
        + [
            OntologyEdge("macro_indicator:vol", "signal:test", "emits_signal", {"ontology_run_id": "run-1"}),
            OntologyEdge("macro_indicator:breadth", "signal:test", "emits_signal", {"ontology_run_id": "run-1"}),
        ],
    ]

    for edges in cases:
        with pytest.raises(OntologySchemaValidationError, match="only one source"):
            normalize_graph(nodes, edges, run_id="run-1")


def test_relation_registry_allows_many_to_many_relations():
    nodes = _core_nodes() + [
        OntologyNode(
            id="macro_indicator:vol",
            type="MacroIndicator",
            label="Vol",
            properties={
                "indicator_key": "vol",
                "name": "Vol",
                "source": "test",
                "as_of": "run-1",
                "ontology_run_id": "run-1",
            },
        ),
        OntologyNode(
            id="macro_indicator:breadth",
            type="MacroIndicator",
            label="Breadth",
            properties={
                "indicator_key": "breadth",
                "name": "Breadth",
                "source": "test",
                "as_of": "run-1",
                "ontology_run_id": "run-1",
            },
        ),
        OntologyNode(id="signal:one", type="Signal", label="One", properties={"source": "test"}),
        OntologyNode(id="signal:two", type="Signal", label="Two", properties={"source": "test"}),
    ]
    edges = _core_edges() + [
        OntologyEdge(
            "sector:information_technology", "macro_indicator:vol", "affected_by", {"ontology_run_id": "run-1"}
        ),
        OntologyEdge(
            "sector:information_technology",
            "macro_indicator:breadth",
            "affected_by",
            {"ontology_run_id": "run-1"},
        ),
        OntologyEdge(
            "position:MU",
            "signal:one",
            "exposed_to_signal",
            _exposure_props("One"),
        ),
        OntologyEdge(
            "position:MU",
            "signal:two",
            "exposed_to_signal",
            _exposure_props("Two"),
        ),
    ]

    graph = normalize_graph(nodes, edges, run_id="run-1")

    assert len(graph.edges) == 6


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
        OntologyEdge("position:MU", "asset:MU", "references_asset", {"ontology_run_id": "run-1"}),
        OntologyEdge(
            "asset:MU",
            "sector:information_technology",
            "belongs_to_sector",
            {"ontology_run_id": "run-1", "source": "test"},
        ),
    ]


def _two_position_core_nodes() -> list[OntologyNode]:
    return _core_nodes() + [
        OntologyNode(id="position:NVDA", type="Position", label="NVDA", properties={"ticker": "NVDA"}),
        OntologyNode(id="asset:NVDA", type="Asset", label="NVDA", properties={"ticker": "NVDA", "asset": "equity"}),
    ]


def _two_position_core_edges() -> list[OntologyEdge]:
    return _core_edges() + [
        OntologyEdge("position:NVDA", "asset:NVDA", "references_asset", {"ontology_run_id": "run-1"}),
        OntologyEdge(
            "asset:NVDA",
            "sector:information_technology",
            "belongs_to_sector",
            {"ontology_run_id": "run-1", "source": "test"},
        ),
    ]


def _thesis_node(ticker: str) -> OntologyNode:
    return OntologyNode(
        id=f"thesis:{ticker}",
        type="Thesis",
        label=f"Thesis: {ticker}",
        properties={
            "ticker": ticker,
            "status": "active",
            "created_at": "run-1",
            "updated_at": "run-1",
            "ontology_run_id": "run-1",
        },
    )


def _exposure_props(name: str) -> dict[str, object]:
    return {
        "component": "test",
        "source": "test",
        "name": name,
        "threshold": "higher is worse",
        "direction": "stable",
        "contribution": 0.2,
        "ontology_run_id": "run-1",
    }
