from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from ontology.models import EntityType, OntologyEdge, OntologyNode
from ontology.schema_definitions import SCHEMA_KIND_ONTOLOGY_OBJECT, ontology_schema_definitions
from ontology.schemas.identity import action_item_id, evaluation_id, hedge_position_id, signal_id
from ontology.schemas.objects import (
    ActionItemV1,
    FactorScoreV1,
    HedgePositionV1,
    IdeaComparisonRankingV1,
    InvestmentIdeaV1,
    ManagementQualityAssessmentV1,
    MissingInformationRequirementV1,
    OptimizationRunV1,
    PositionV1,
    ProvenanceEventV1,
    SourceFreshnessV1,
)
from ontology.schemas.registry import NODE_SCHEMAS, OntologySchemaValidationError, normalize_graph, normalize_node
from ontology.schemas.relations import (
    PROVENANCE_RELATION_TYPES,
    PROVENANCE_REQUIRED_PROPERTIES,
    get_relation_definition,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCAN_EXCLUDED_PARTS = {
    ".venv",
    "__pycache__",
    "node_modules",
    "tests",
}


def _iter_source_python_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.py")
        if path.is_file() and SOURCE_SCAN_EXCLUDED_PARTS.isdisjoint(path.relative_to(ROOT).parts)
    )


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


def test_workflow_run_schema_accepts_persisted_timestamps():
    run = normalize_node(
        OntologyNode(
            id="workflow_run:workflow_thesis_review_unit",
            type="WorkflowRun",
            label="thesis_review",
            properties={
                "schema_version": 1,
                "run_id": "workflow:thesis_review:unit",
                "workflow_name": "thesis_review",
                "ticker": "mu",
                "status": "running",
                "started_at": "2026-05-13T16:08:59.403662+00:00",
                "created_at": "2026-05-13T16:08:59.403662+00:00",
                "updated_at": "2026-05-13T16:08:59.403662+00:00",
                "ontology_run_id": "operational",
            },
            schema_name="WorkflowRun",
            schema_version=1,
        ),
        allow_legacy=False,
    )

    assert run.properties["ticker"] == "MU"
    assert run.properties["created_at"] == "2026-05-13T16:08:59.403662+00:00"
    assert run.properties["updated_at"] == "2026-05-13T16:08:59.403662+00:00"


def test_runtime_migration_schema_rejects_unregistered_fields():
    with pytest.raises(ValidationError):
        InvestmentIdeaV1(
            idea_id="investment_idea:MU",
            ticker="MU",
            status="watching",
            unregistered_field=True,
        )


def test_first_class_research_optimizer_and_management_quality_objects_accept_uid_links():
    ranking = IdeaComparisonRankingV1(
        ranking_id="idea_comparison_ranking:run_1_rank_1",
        comparison_run_id="idea_comparison_run:run_1",
        idea_id="investment_idea:mu",
        evaluation_id="idea_evaluation:eval_1",
        ticker="mu",
        rank=1,
        action="buy",
    )
    factor = FactorScoreV1(
        factor_score_id="factor_score:eval_1_management",
        parent_uid="idea_evaluation:eval_1",
        parent_type="IdeaEvaluation",
        factor_name="management_quality",
        score=82,
    )
    missing = MissingInformationRequirementV1(
        requirement_id="missing_information_requirement:eval_1_valuation",
        parent_uid="idea_evaluation:eval_1",
        parent_type="IdeaEvaluation",
        field="valuation",
    )
    run = OptimizationRunV1(run_id="optimization_run:run_1", mission_id="optimization_mission:default")
    freshness = SourceFreshnessV1(
        freshness_id="source_freshness:run_1_reports",
        parent_uid=run.run_id,
        parent_type="OptimizationRun",
        source_name="reports",
        status="ok",
    )
    assessment = ManagementQualityAssessmentV1(
        assessment_id="management_quality_assessment:issuer_mu",
        issuer_id="issuer:mu",
        ticker="mu",
    )

    assert ranking.ticker == "MU"
    assert factor.score == 82
    assert missing.status == "open"
    assert run.status == "running"
    assert freshness.freshness_category is None
    assert assessment.ticker == "MU"


def test_every_entity_type_has_pydantic_schema_and_definition():
    entity_types = set(get_args(EntityType))
    definitions = {
        definition.schema_name
        for definition in ontology_schema_definitions()
        if definition.schema_kind == SCHEMA_KIND_ONTOLOGY_OBJECT and definition.schema_version == 1
    }

    assert entity_types <= set(NODE_SCHEMAS)
    assert entity_types <= definitions


def test_provenance_event_requires_lifecycle_redaction_retention_and_context():
    event = ProvenanceEventV1(
        event_id="pv:unit",
        event_type="unit",
        event_name="test",
        status="started",
        actor_id="alice",
        redaction_policy="audit_summary_v1",
        retention_class="provenance_365d",
    )

    assert event.status == "started"

    with pytest.raises(ValidationError, match="at least one"):
        ProvenanceEventV1(
            event_id="pv:no-context",
            event_type="unit",
            event_name="test",
            status="started",
            redaction_policy="audit_summary_v1",
            retention_class="provenance_365d",
        )

    with pytest.raises(ValidationError):
        ProvenanceEventV1(
            event_id="pv:bad-status",
            event_type="unit",
            event_name="test",
            status="running",
            actor_id="alice",
            redaction_policy="audit_summary_v1",
            retention_class="provenance_365d",
        )

    with pytest.raises(ValidationError):
        ProvenanceEventV1(
            event_id="pv:no-retention",
            event_type="unit",
            event_name="test",
            status="started",
            actor_id="alice",
            redaction_policy="audit_summary_v1",
            retention_class=" ",
        )


def test_literal_write_object_calls_use_registered_object_types():
    registered = set(NODE_SCHEMAS)
    offenders: list[str] = []
    for path in _iter_source_python_files():
        rel_path = path.relative_to(ROOT)
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


@pytest.mark.parametrize("relation_type", sorted(PROVENANCE_RELATION_TYPES))
def test_relation_registry_accepts_typed_provenance_relation_verbs(relation_type):
    definition = get_relation_definition(relation_type)
    nodes = [
        OntologyNode(
            id="provenance_event:pv_unit",
            type="ProvenanceEvent",
            label="Unit event",
            properties={
                "event_id": "pv:unit",
                "event_type": "unit",
                "event_name": "test",
                "status": "started",
                "actor_id": "alice",
                "redaction_policy": "audit_summary_v1",
                "retention_class": "provenance_365d",
            },
            schema_name="ProvenanceEvent",
            schema_version=1,
        ),
        OntologyNode(
            id="object_version_ref:version_1",
            type="ObjectVersionRef",
            label="version 1",
            properties={
                "ref_id": "version:1",
                "object_uid": "position:MU",
                "version_id": "version:1",
                "ontology_run_id": "operational",
            },
            schema_name="ObjectVersionRef",
            schema_version=1,
        ),
    ]
    edge = OntologyEdge(
        "provenance_event:pv_unit",
        "object_version_ref:version_1",
        relation_type,
        {
            "event_id": "pv:unit",
            "ontology_run_id": "operational",
            "source_ref_type": "producer_event",
            "source_ref_id": "pv:unit",
            "target_ref_type": "ontology_object_version",
            "target_ref_id": "version:1",
            "redaction_policy": "audit_summary_v1",
            "retention_class": "provenance_365d",
        },
        schema_name=relation_type,
        schema_version=1,
        relation_schema_name=relation_type,
        relation_schema_version=1,
    )

    graph = normalize_graph(nodes, [edge], require_core_edges=False)

    assert graph.edges[0].relation_type == relation_type
    assert definition.required_properties == PROVENANCE_REQUIRED_PROPERTIES


def test_relation_registry_rejects_unregistered_legacy_provenance_link_relation():
    nodes = [
        OntologyNode(
            id="provenance_event:pv_unit",
            type="ProvenanceEvent",
            label="Unit event",
            properties={
                "event_id": "pv:unit",
                "event_type": "unit",
                "event_name": "test",
                "status": "started",
                "actor_id": "alice",
                "redaction_policy": "audit_summary_v1",
                "retention_class": "provenance_365d",
            },
            schema_name="ProvenanceEvent",
            schema_version=1,
        ),
        OntologyNode(
            id="object_version_ref:version_1",
            type="ObjectVersionRef",
            label="version 1",
            properties={
                "ref_id": "version:1",
                "object_uid": "position:MU",
                "version_id": "version:1",
                "ontology_run_id": "operational",
            },
            schema_name="ObjectVersionRef",
            schema_version=1,
        ),
    ]

    with pytest.raises(OntologySchemaValidationError, match="Unsupported relation type"):
        normalize_graph(
            nodes,
            [
                OntologyEdge(
                    "provenance_event:pv_unit",
                    "object_version_ref:version_1",
                    "provenance_event_records_link",
                    {"ontology_run_id": "operational"},
                    schema_name="provenance_event_records_link",
                    schema_version=1,
                    relation_schema_name="provenance_event_records_link",
                    relation_schema_version=1,
                )
            ],
            require_core_edges=False,
        )


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
