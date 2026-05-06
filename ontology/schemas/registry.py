from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import ValidationError

from ontology.models import EntityType, OntologyEdge, OntologyNode
from ontology.schemas.base import OntologySchemaBase
from ontology.schemas.identity import (
    account_id,
    action_event_id,
    action_item_id,
    action_run_id,
    agent_session_ref_id,
    approval_id,
    asset_id,
    audit_event_id,
    catalyst_id,
    citation_id,
    computed_snapshot_ref_id,
    document_artifact_id,
    evaluation_id,
    evidence_id,
    executed_action_id,
    executed_decision_record_id,
    factor_score_id,
    hedge_position_id,
    idea_comparison_ranking_id,
    idea_comparison_run_id,
    idea_evaluation_id,
    instrument_id,
    investment_idea_id,
    investment_policy_id,
    investor_id,
    issuer_id,
    kill_condition_id,
    macro_indicator_id,
    management_quality_accomplishment_id,
    management_quality_assessment_id,
    management_quality_scorecard_row_id,
    management_quality_setback_id,
    mandate_id,
    missing_information_requirement_id,
    model_call_ref_id,
    object_version_ref_id,
    ontology_run_ref_id,
    optimization_action_snapshot_id,
    optimization_alert_id,
    optimization_mission_id,
    optimization_run_id,
    policy_gate_result_id,
    portfolio_id,
    position_id,
    provenance_event_id,
    recommendation_id,
    relation_version_ref_id,
    report_run_id,
    risk_limit_id,
    risk_metric_id,
    scenario_id,
    schema_definition_ref_id,
    sector_id,
    signal_id,
    source_freshness_id,
    source_record_object_id,
    thesis_claim_id,
    thesis_id,
    tool_call_ref_id,
    trade_proposal_id,
    watch_trigger_id,
    workflow_artifact_id,
    workflow_run_id,
)
from ontology.schemas.legacy import adapt_edge_payload, adapt_node_payload
from ontology.schemas.objects import (
    AccountV1,
    ActionEventV1,
    ActionItemV1,
    ActionRunV1,
    AgentSessionRefV1,
    ApprovalV1,
    AssetV1,
    AuditEventV1,
    CatalystV1,
    CitationV1,
    ComputedSnapshotRefV1,
    DocumentArtifactV1,
    EvaluationV1,
    EvidenceV1,
    ExecutedActionV1,
    ExecutedDecisionRecordV1,
    FactorScoreV1,
    HedgePositionV1,
    IdeaComparisonRankingV1,
    IdeaComparisonRunV1,
    IdeaEvaluationV1,
    InstrumentV1,
    InvestmentIdeaV1,
    InvestmentPolicyV1,
    InvestorV1,
    IssuerV1,
    KillConditionV1,
    MacroIndicatorV1,
    ManagementQualityAccomplishmentV1,
    ManagementQualityAssessmentV1,
    ManagementQualityScorecardRowV1,
    ManagementQualitySetbackV1,
    MandateV1,
    MissingInformationRequirementV1,
    ModelCallRefV1,
    ObjectVersionRefV1,
    OntologyObjectV1,
    OntologyRunRefV1,
    OptimizationActionSnapshotV1,
    OptimizationAlertV1,
    OptimizationMissionV1,
    OptimizationRunV1,
    PolicyGateResultV1,
    PortfolioV1,
    PositionV1,
    ProvenanceEventV1,
    RecommendationV1,
    RelationVersionRefV1,
    ReportRunV1,
    RiskLimitV1,
    RiskMetricV1,
    ScenarioV1,
    SchemaDefinitionRefV1,
    SectorV1,
    SignalV1,
    SourceFreshnessV1,
    SourceRecordV1,
    ThesisClaimV1,
    ThesisV1,
    ToolCallRefV1,
    TradeProposalV1,
    WatchTriggerV1,
    WorkflowArtifactV1,
    WorkflowRunV1,
)
from ontology.schemas.relations import (
    BELONGS_TO_SECTOR,
    EVALUATED_BY,
    HAS_CATALYST,
    HAS_THESIS,
    OPTIONAL_RELATIONS,
    REFERENCES_ASSET,
    RelationCardinality,
    dump_edge_properties,
    edge_schema_for_relation,
    edge_schema_name,
    get_relation_definition,
)

NODE_SCHEMAS: dict[EntityType, type[OntologySchemaBase]] = {
    "Position": PositionV1,
    "HedgePosition": HedgePositionV1,
    "Asset": AssetV1,
    "Instrument": InstrumentV1,
    "Issuer": IssuerV1,
    "Investor": InvestorV1,
    "Account": AccountV1,
    "Portfolio": PortfolioV1,
    "Mandate": MandateV1,
    "InvestmentPolicy": InvestmentPolicyV1,
    "RiskLimit": RiskLimitV1,
    "RiskMetric": RiskMetricV1,
    "Scenario": ScenarioV1,
    "PolicyGateResult": PolicyGateResultV1,
    "TradeProposal": TradeProposalV1,
    "SourceRecord": SourceRecordV1,
    "ObjectVersionRef": ObjectVersionRefV1,
    "ExecutedAction": ExecutedActionV1,
    "ExecutedDecisionRecord": ExecutedDecisionRecordV1,
    "AuditEvent": AuditEventV1,
    "Sector": SectorV1,
    "MacroIndicator": MacroIndicatorV1,
    "Signal": SignalV1,
    "Thesis": ThesisV1,
    "Evaluation": EvaluationV1,
    "Catalyst": CatalystV1,
    "KillCondition": KillConditionV1,
    "ThesisClaim": ThesisClaimV1,
    "Evidence": EvidenceV1,
    "Citation": CitationV1,
    "ActionItem": ActionItemV1,
    "WatchTrigger": WatchTriggerV1,
    "Approval": ApprovalV1,
    "ActionRun": ActionRunV1,
    "ActionEvent": ActionEventV1,
    "ProvenanceEvent": ProvenanceEventV1,
    "RelationVersionRef": RelationVersionRefV1,
    "SchemaDefinitionRef": SchemaDefinitionRefV1,
    "OntologyRunRef": OntologyRunRefV1,
    "AgentSessionRef": AgentSessionRefV1,
    "ModelCallRef": ModelCallRefV1,
    "ToolCallRef": ToolCallRefV1,
    "ComputedSnapshotRef": ComputedSnapshotRefV1,
    "WorkflowRun": WorkflowRunV1,
    "WorkflowArtifact": WorkflowArtifactV1,
    "Recommendation": RecommendationV1,
    "ReportRun": ReportRunV1,
    "DocumentArtifact": DocumentArtifactV1,
    "InvestmentIdea": InvestmentIdeaV1,
    "IdeaEvaluation": IdeaEvaluationV1,
    "IdeaComparisonRun": IdeaComparisonRunV1,
    "IdeaComparisonRanking": IdeaComparisonRankingV1,
    "FactorScore": FactorScoreV1,
    "MissingInformationRequirement": MissingInformationRequirementV1,
    "OptimizationMission": OptimizationMissionV1,
    "OptimizationRun": OptimizationRunV1,
    "OptimizationActionSnapshot": OptimizationActionSnapshotV1,
    "OptimizationAlert": OptimizationAlertV1,
    "SourceFreshness": SourceFreshnessV1,
    "ManagementQualityAssessment": ManagementQualityAssessmentV1,
    "ManagementQualityScorecardRow": ManagementQualityScorecardRowV1,
    "ManagementQualityAccomplishment": ManagementQualityAccomplishmentV1,
    "ManagementQualitySetback": ManagementQualitySetbackV1,
}
OPTIONAL_NODE_TYPES = {
    "Thesis",
    "Evaluation",
    "Catalyst",
    "KillCondition",
    "ThesisClaim",
    "ActionItem",
    "WatchTrigger",
    "Approval",
    "ActionRun",
    "ActionEvent",
    "WorkflowRun",
    "WorkflowArtifact",
    "Recommendation",
    "ReportRun",
    "DocumentArtifact",
    "ProvenanceEvent",
    "RelationVersionRef",
    "SchemaDefinitionRef",
    "OntologyRunRef",
    "AgentSessionRef",
    "ModelCallRef",
    "ToolCallRef",
    "ComputedSnapshotRef",
    "InvestmentIdea",
    "IdeaEvaluation",
    "IdeaComparisonRun",
    "IdeaComparisonRanking",
    "FactorScore",
    "MissingInformationRequirement",
    "OptimizationMission",
    "OptimizationRun",
    "OptimizationActionSnapshot",
    "OptimizationAlert",
    "SourceFreshness",
    "ManagementQualityAssessment",
    "ManagementQualityScorecardRow",
    "ManagementQualityAccomplishment",
    "ManagementQualitySetback",
    "Investor",
    "Account",
    "Portfolio",
    "Mandate",
    "InvestmentPolicy",
    "RiskLimit",
    "RiskMetric",
    "Scenario",
    "PolicyGateResult",
    "TradeProposal",
    "SourceRecord",
    "ObjectVersionRef",
    "ExecutedAction",
    "AuditEvent",
}
NodeUpgradeAdapter = Any
NODE_UPGRADE_ADAPTERS: dict[tuple[str, int, int], NodeUpgradeAdapter] = {}


class OntologySchemaValidationError(ValueError):
    pass


@dataclass(slots=True)
class NormalizedGraph:
    nodes: list[OntologyNode]
    edges: list[OntologyEdge]
    node_id_map: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RelationValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise OntologySchemaValidationError("; ".join(self.errors))


def normalize_node(
    node: OntologyNode,
    *,
    run_id: str | None = None,
    allow_legacy: bool = True,
) -> OntologyNode:
    try:
        schema_cls = NODE_SCHEMAS[node.type]
    except KeyError as exc:
        raise OntologySchemaValidationError(f"Unsupported node type: {node.type}") from exc

    legacy_payload = _is_legacy_payload(node.properties, node.schema_version)
    node_id = node.id
    label = node.label
    payload = dict(node.properties or {})
    if legacy_payload:
        if not allow_legacy:
            raise OntologySchemaValidationError(f"Legacy ontology node is not allowed: {node.id}")
        try:
            node_id, label, payload = adapt_node_payload(
                node_id=node.id,
                node_type=node.type,
                label=node.label,
                properties=payload,
                run_id=run_id,
            )
        except Exception as exc:
            raise OntologySchemaValidationError(f"Invalid legacy node {node.id}: {exc}") from exc
        payload_version = 1
    else:
        payload_version = int(payload.get("schema_version") or node.schema_version or 0)

    current_version = _schema_version_for(schema_cls)
    try:
        payload = _upgrade_node_payload(node.type, payload, from_version=payload_version, to_version=current_version)
    except Exception as exc:
        raise OntologySchemaValidationError(
            f"Missing compatible upgrade for {node.type} node {node.id}: {exc}"
        ) from exc

    try:
        model = cast(OntologyObjectV1, schema_cls.model_validate(payload))
    except ValidationError as exc:
        raise OntologySchemaValidationError(f"Invalid {node.type} node {node.id}: {exc}") from exc

    expected_id = expected_node_id(node.type, model)
    if node_id != expected_id:
        if legacy_payload:
            node_id = expected_id
        else:
            raise OntologySchemaValidationError(f"Node {node.id} has non-canonical identity; expected {expected_id}")

    return OntologyNode(
        id=node_id,
        type=node.type,
        label=_label_for(node.type, label, model),
        properties=model.model_dump(mode="json"),
        schema_name=node.type,
        schema_version=current_version,
    )


def normalize_edge(
    edge: OntologyEdge,
    *,
    run_id: str | None = None,
    allow_legacy: bool = True,
    source_id: str | None = None,
    target_id: str | None = None,
) -> OntologyEdge:
    payload = dict(edge.properties or {})
    legacy_payload = _is_legacy_payload(payload, edge.schema_version)
    if legacy_payload:
        if not allow_legacy:
            raise OntologySchemaValidationError(
                f"Legacy ontology edge is not allowed: {edge.source_id}->{edge.target_id}:{edge.relation_type}"
            )
        try:
            payload = adapt_edge_payload(relation_type=edge.relation_type, properties=payload, run_id=run_id)
        except Exception as exc:
            raise OntologySchemaValidationError(f"Invalid legacy edge {edge.relation_type}: {exc}") from exc

    try:
        schema_cls = edge_schema_for_relation(edge.relation_type)
    except ValueError as exc:
        raise OntologySchemaValidationError(str(exc)) from exc
    try:
        model = schema_cls.model_validate(payload)
    except ValidationError as exc:
        raise OntologySchemaValidationError(
            f"Invalid {edge.relation_type} edge {edge.source_id}->{edge.target_id}: {exc}"
        ) from exc

    return OntologyEdge(
        source_id=source_id or edge.source_id,
        target_id=target_id or edge.target_id,
        relation_type=edge.relation_type,
        properties=dump_edge_properties(model),
        schema_name=edge_schema_name(edge.relation_type),
        schema_version=1,
        relation_schema_name=edge.relation_type,
        relation_schema_version=1,
    )


def validate_edge_relation(
    edge: OntologyEdge,
    node_types: Mapping[str, str],
    *,
    run_id: str | None = None,
    allow_legacy: bool = True,
    source_id: str | None = None,
    target_id: str | None = None,
) -> OntologyEdge:
    relation_source_id = source_id or edge.source_id
    relation_target_id = target_id or edge.target_id
    _validate_relation(edge.relation_type, relation_source_id, relation_target_id, node_types)
    normalized = normalize_edge(
        edge,
        run_id=run_id,
        allow_legacy=allow_legacy,
        source_id=relation_source_id,
        target_id=relation_target_id,
    )
    _validate_required_relation_properties(normalized)
    return normalized


def validate_graph_relations(
    nodes: list[OntologyNode],
    edges: list[OntologyEdge],
    *,
    require_core_edges: bool = True,
    skip_optional_invalid: bool = False,
) -> RelationValidationReport:
    report = RelationValidationReport()
    node_types = {node.id: node.type for node in nodes}
    valid_edges: list[OntologyEdge] = []

    for edge in edges:
        try:
            valid_edges.append(validate_edge_relation(edge, node_types, allow_legacy=True))
        except OntologySchemaValidationError as exc:
            if skip_optional_invalid and edge.relation_type in OPTIONAL_RELATIONS:
                report.warnings.append(str(exc))
            else:
                report.errors.append(str(exc))

    report.errors.extend(_cardinality_errors(valid_edges))
    if require_core_edges:
        report.errors.extend(_core_edge_errors(nodes, valid_edges))
    report.errors.extend(_optional_owner_errors(nodes, valid_edges))
    return report


def normalize_graph(
    nodes: list[OntologyNode],
    edges: list[OntologyEdge],
    *,
    run_id: str | None = None,
    allow_legacy: bool = True,
    skip_optional_invalid: bool = False,
    require_core_edges: bool = True,
) -> NormalizedGraph:
    normalized_nodes: dict[str, OntologyNode] = {}
    id_map: dict[str, str] = {}
    skipped_old_ids: set[str] = set()
    warnings: list[str] = []

    for node in nodes:
        try:
            normalized_node = normalize_node(node, run_id=run_id, allow_legacy=allow_legacy)
        except OntologySchemaValidationError as exc:
            if skip_optional_invalid and node.type in OPTIONAL_NODE_TYPES:
                skipped_old_ids.add(node.id)
                warnings.append(str(exc))
                continue
            raise

        if normalized_node.id in normalized_nodes and normalized_nodes[normalized_node.id] != normalized_node:
            raise OntologySchemaValidationError(
                f"Duplicate canonical node id after normalization: {normalized_node.id}"
            )
        normalized_nodes[normalized_node.id] = normalized_node
        id_map[node.id] = normalized_node.id

    normalized_edges: dict[tuple[str, str, str], OntologyEdge] = {}
    node_types = {node_id: node.type for node_id, node in normalized_nodes.items()}

    for edge in edges:
        if edge.source_id in skipped_old_ids or edge.target_id in skipped_old_ids:
            continue
        source_id = id_map.get(edge.source_id, edge.source_id)
        target_id = id_map.get(edge.target_id, edge.target_id)
        try:
            normalized_edge = validate_edge_relation(
                edge,
                node_types,
                run_id=run_id,
                allow_legacy=allow_legacy,
                source_id=source_id,
                target_id=target_id,
            )
        except OntologySchemaValidationError as exc:
            if skip_optional_invalid and edge.relation_type in OPTIONAL_RELATIONS:
                warnings.append(str(exc))
                continue
            raise

        normalized_edges[(normalized_edge.source_id, normalized_edge.target_id, normalized_edge.relation_type)] = (
            normalized_edge
        )

    relation_report = validate_graph_relations(
        list(normalized_nodes.values()),
        list(normalized_edges.values()),
        require_core_edges=require_core_edges,
        skip_optional_invalid=skip_optional_invalid,
    )
    warnings.extend(relation_report.warnings)
    relation_report.raise_for_errors()

    return NormalizedGraph(
        nodes=list(normalized_nodes.values()),
        edges=list(normalized_edges.values()),
        node_id_map=id_map,
        warnings=warnings,
    )


def expected_node_id(node_type: str, model: OntologyObjectV1) -> str:
    if isinstance(model, PositionV1):
        return position_id(model.ticker)
    if isinstance(model, HedgePositionV1):
        return hedge_position_id(model.ticker)
    if isinstance(model, AssetV1):
        return asset_id(model.ticker)
    if isinstance(model, InstrumentV1):
        return instrument_id(model.instrument_id)
    if isinstance(model, IssuerV1):
        return issuer_id(model.issuer_id)
    if isinstance(model, InvestorV1):
        return investor_id(model.investor_id)
    if isinstance(model, AccountV1):
        return account_id(model.account_id)
    if isinstance(model, PortfolioV1):
        return portfolio_id(model.portfolio_id)
    if isinstance(model, MandateV1):
        return mandate_id(model.mandate_id)
    if isinstance(model, InvestmentPolicyV1):
        return investment_policy_id(model.policy_id)
    if isinstance(model, RiskLimitV1):
        return risk_limit_id(model.limit_id)
    if isinstance(model, RiskMetricV1):
        return risk_metric_id(model.metric_id)
    if isinstance(model, ScenarioV1):
        return scenario_id(model.scenario_id)
    if isinstance(model, PolicyGateResultV1):
        return policy_gate_result_id(model.gate_result_id)
    if isinstance(model, TradeProposalV1):
        return trade_proposal_id(model.proposal_id)
    if isinstance(model, SourceRecordV1):
        return source_record_object_id(model.source_record_id)
    if isinstance(model, ObjectVersionRefV1):
        return object_version_ref_id(model.ref_id)
    if isinstance(model, ExecutedActionV1):
        return executed_action_id(model.executed_action_id)
    if isinstance(model, ExecutedDecisionRecordV1):
        return executed_decision_record_id(model.decision_record_id)
    if isinstance(model, AuditEventV1):
        return audit_event_id(model.event_id)
    if isinstance(model, SectorV1):
        return sector_id(model.name)
    if isinstance(model, MacroIndicatorV1):
        return macro_indicator_id(model.indicator_key)
    if isinstance(model, SignalV1):
        return signal_id(model.source, model.name)
    if isinstance(model, ThesisV1):
        return thesis_id(model.ticker)
    if isinstance(model, EvaluationV1):
        return evaluation_id(model.ticker, model.evaluated_at)
    if isinstance(model, CatalystV1):
        return catalyst_id(model.ticker, model.name, model.description)
    if isinstance(model, KillConditionV1):
        return kill_condition_id(model.ticker, model.legacy_id or model.condition)
    if isinstance(model, ThesisClaimV1):
        return thesis_claim_id(model.ticker, model.legacy_id or model.claim)
    if isinstance(model, EvidenceV1):
        return evidence_id(model.evidence_id)
    if isinstance(model, CitationV1):
        return citation_id(model.citation_id)
    if isinstance(model, ActionItemV1):
        return action_item_id(model.legacy_id or model.description)
    if isinstance(model, WatchTriggerV1):
        return watch_trigger_id(model.legacy_id or model.condition)
    if isinstance(model, ApprovalV1):
        return approval_id(model.legacy_id or f"{model.entity_type}:{model.action_input_hash or model.created_at}")
    if isinstance(model, ActionRunV1):
        return action_run_id(model.legacy_id or f"{model.action_id}:{model.started_at}")
    if isinstance(model, ActionEventV1):
        return action_event_id(model.legacy_id or f"{model.action_run_id}:{model.event_type}:{model.created_at}")
    if isinstance(model, ProvenanceEventV1):
        return provenance_event_id(model.event_id)
    if isinstance(model, RelationVersionRefV1):
        return relation_version_ref_id(model.ref_id)
    if isinstance(model, SchemaDefinitionRefV1):
        return schema_definition_ref_id(model.ref_id)
    if isinstance(model, OntologyRunRefV1):
        return ontology_run_ref_id(model.run_id)
    if isinstance(model, AgentSessionRefV1):
        return agent_session_ref_id(model.session_id)
    if isinstance(model, ModelCallRefV1):
        return model_call_ref_id(model.call_id)
    if isinstance(model, ToolCallRefV1):
        return tool_call_ref_id(model.call_id)
    if isinstance(model, ComputedSnapshotRefV1):
        return computed_snapshot_ref_id(model.snapshot_key)
    if isinstance(model, WorkflowRunV1):
        return workflow_run_id(model.run_id)
    if isinstance(model, WorkflowArtifactV1):
        return workflow_artifact_id(model.artifact_id)
    if isinstance(model, RecommendationV1):
        return recommendation_id(
            model.legacy_id
            or model.recommendation_id
            or model.idempotency_key
            or f"{model.report_type}:{model.as_of}:{model.action}:{model.ticker}"
        )
    if isinstance(model, ReportRunV1):
        return report_run_id(model.report_id)
    if isinstance(model, DocumentArtifactV1):
        return document_artifact_id(model.document_type, model.document_id)
    if isinstance(model, InvestmentIdeaV1):
        return investment_idea_id(model.idea_id)
    if isinstance(model, IdeaEvaluationV1):
        return idea_evaluation_id(model.evaluation_id)
    if isinstance(model, IdeaComparisonRunV1):
        return idea_comparison_run_id(model.comparison_run_id)
    if isinstance(model, IdeaComparisonRankingV1):
        return idea_comparison_ranking_id(model.ranking_id)
    if isinstance(model, FactorScoreV1):
        return factor_score_id(model.factor_score_id)
    if isinstance(model, MissingInformationRequirementV1):
        return missing_information_requirement_id(model.requirement_id)
    if isinstance(model, OptimizationMissionV1):
        return optimization_mission_id(model.mission_id)
    if isinstance(model, OptimizationRunV1):
        return optimization_run_id(model.run_id)
    if isinstance(model, OptimizationActionSnapshotV1):
        return optimization_action_snapshot_id(model.snapshot_id)
    if isinstance(model, OptimizationAlertV1):
        return optimization_alert_id(model.alert_id)
    if isinstance(model, SourceFreshnessV1):
        return source_freshness_id(model.freshness_id)
    if isinstance(model, ManagementQualityAssessmentV1):
        return management_quality_assessment_id(model.assessment_id)
    if isinstance(model, ManagementQualityScorecardRowV1):
        return management_quality_scorecard_row_id(model.row_id)
    if isinstance(model, ManagementQualityAccomplishmentV1):
        return management_quality_accomplishment_id(model.accomplishment_id)
    if isinstance(model, ManagementQualitySetbackV1):
        return management_quality_setback_id(model.setback_id)
    raise OntologySchemaValidationError(f"Unsupported node schema for type {node_type}")


def node_from_schema(
    *,
    node_id: str,
    node_type: EntityType,
    label: str,
    model: OntologyObjectV1,
) -> OntologyNode:
    return normalize_node(
        OntologyNode(
            id=node_id,
            type=node_type,
            label=label,
            properties=model.model_dump(mode="json"),
            schema_name=node_type,
            schema_version=1,
        ),
        allow_legacy=False,
    )


def register_node_schema_upgrade_adapter(
    node_type: str,
    from_version: int,
    to_version: int,
    adapter: NodeUpgradeAdapter,
) -> None:
    NODE_UPGRADE_ADAPTERS[(node_type, int(from_version), int(to_version))] = adapter


def _validate_relation(
    relation_type: str,
    source_id: str,
    target_id: str,
    node_types: Mapping[str, str],
) -> None:
    try:
        definition = get_relation_definition(relation_type)
    except ValueError as exc:
        raise OntologySchemaValidationError(str(exc)) from exc
    expected_source_types = definition.allowed_source_types or frozenset({definition.source_type})
    expected_target_types = definition.allowed_target_types or frozenset({definition.target_type})
    source_type = node_types.get(source_id)
    target_type = node_types.get(target_id)
    if source_type is None:
        raise OntologySchemaValidationError(f"Edge {relation_type} has missing source node: {source_id}")
    if target_type is None:
        raise OntologySchemaValidationError(f"Edge {relation_type} has missing target node: {target_id}")
    if source_type not in expected_source_types or target_type not in expected_target_types:
        expected_source = "|".join(sorted(expected_source_types))
        expected_target = "|".join(sorted(expected_target_types))
        raise OntologySchemaValidationError(
            f"Edge {relation_type} must connect {expected_source}->{expected_target}, got {source_type}->{target_type}"
        )


def _validate_required_relation_properties(edge: OntologyEdge) -> None:
    try:
        definition = get_relation_definition(edge.relation_type)
    except ValueError as exc:
        raise OntologySchemaValidationError(str(exc)) from exc
    missing = [name for name in sorted(definition.required_properties) if _missing_property(edge.properties.get(name))]
    if missing:
        fields = ", ".join(missing)
        raise OntologySchemaValidationError(
            f"Edge {edge.relation_type} {edge.source_id}->{edge.target_id} is missing required properties: {fields}"
        )


def _missing_property(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _schema_version_for(schema_cls: type[OntologySchemaBase]) -> int:
    field = getattr(schema_cls, "model_fields", {}).get("schema_version")
    default = getattr(field, "default", 1)
    try:
        return int(default)
    except (TypeError, ValueError):
        return 1


def _upgrade_node_payload(
    node_type: str,
    payload: dict[str, Any],
    *,
    from_version: int,
    to_version: int,
) -> dict[str, Any]:
    current_version = int(from_version)
    upgraded = dict(payload)
    if current_version == to_version:
        return upgraded
    if current_version > to_version:
        raise ValueError(f"future schema version {current_version} cannot be read as v{to_version}")
    while current_version < to_version:
        adapter = NODE_UPGRADE_ADAPTERS.get((node_type, current_version, current_version + 1))
        if adapter is None:
            raise ValueError(f"{node_type} v{current_version}->v{current_version + 1}")
        upgraded = adapter(upgraded)
        current_version += 1
    return upgraded


def _cardinality_errors(edges: list[OntologyEdge]) -> list[str]:
    errors: list[str] = []
    unique_edges = {(edge.source_id, edge.target_id, edge.relation_type): edge for edge in edges}
    by_source: dict[tuple[str, str], set[str]] = {}
    by_target: dict[tuple[str, str], set[str]] = {}

    for edge in unique_edges.values():
        definition = get_relation_definition(edge.relation_type)
        if definition.cardinality in {RelationCardinality.SOURCE_UNIQUE, RelationCardinality.SOURCE_AND_TARGET_UNIQUE}:
            by_source.setdefault((edge.relation_type, edge.source_id), set()).add(edge.target_id)
        if definition.cardinality in {RelationCardinality.TARGET_UNIQUE, RelationCardinality.SOURCE_AND_TARGET_UNIQUE}:
            by_target.setdefault((edge.relation_type, edge.target_id), set()).add(edge.source_id)

    for (relation_type, source_id), target_ids in sorted(by_source.items()):
        if len(target_ids) > 1:
            errors.append(
                f"Edge {relation_type} allows only one target for source {source_id}, got {sorted(target_ids)}"
            )
    for (relation_type, target_id), source_ids in sorted(by_target.items()):
        if len(source_ids) > 1:
            errors.append(
                f"Edge {relation_type} allows only one source for target {target_id}, got {sorted(source_ids)}"
            )
    return errors


def _core_edge_errors(nodes: list[OntologyNode], edges: list[OntologyEdge]) -> list[str]:
    errors: list[str] = []
    positions = sorted(node.id for node in nodes if node.type == "Position")
    refs_by_position: dict[str, list[OntologyEdge]] = {node_id: [] for node_id in positions}
    sectors_by_asset: dict[str, list[OntologyEdge]] = {}

    for edge in edges:
        if edge.relation_type == REFERENCES_ASSET and edge.source_id in refs_by_position:
            refs_by_position[edge.source_id].append(edge)
        if edge.relation_type == BELONGS_TO_SECTOR:
            sectors_by_asset.setdefault(edge.source_id, []).append(edge)

    referenced_assets: set[str] = set()
    for ontology_position_id, references in refs_by_position.items():
        if len(references) != 1:
            errors.append(f"Position {ontology_position_id} must have exactly one {REFERENCES_ASSET} edge")
            continue
        referenced_assets.add(references[0].target_id)

    for ontology_asset_id in sorted(referenced_assets):
        if len(sectors_by_asset.get(ontology_asset_id, [])) != 1:
            errors.append(f"Referenced asset {ontology_asset_id} must have exactly one {BELONGS_TO_SECTOR} edge")

    return errors


def _optional_owner_errors(nodes: list[OntologyNode], edges: list[OntologyEdge]) -> list[str]:
    required_incoming = {
        "Thesis": HAS_THESIS,
        "Evaluation": EVALUATED_BY,
        "Catalyst": HAS_CATALYST,
    }
    optional_nodes = {node.id: required_incoming[node.type] for node in nodes if node.type in required_incoming}
    incoming: dict[tuple[str, str], int] = {
        (relation_type, node_id): 0 for node_id, relation_type in optional_nodes.items()
    }

    for edge in edges:
        key = (edge.relation_type, edge.target_id)
        if key in incoming:
            incoming[key] += 1

    errors: list[str] = []
    for node_id, relation_type in sorted(optional_nodes.items()):
        if incoming[(relation_type, node_id)] != 1:
            errors.append(f"{node_id} must have exactly one incoming {relation_type} owner edge")
    return errors


def _is_legacy_payload(properties: dict[str, Any], schema_version: int) -> bool:
    return schema_version != 1 and int(properties.get("schema_version") or 0) != 1


def _label_for(node_type: str, label: str, model: OntologyObjectV1) -> str:
    if isinstance(model, (PositionV1, AssetV1)):
        return model.ticker
    if isinstance(model, SectorV1):
        return model.name
    if isinstance(model, MacroIndicatorV1):
        return model.name
    if isinstance(model, SignalV1):
        return model.name
    if isinstance(model, ThesisV1):
        return f"Thesis: {model.ticker}"
    if isinstance(model, EvaluationV1):
        return f"Eval: {model.ticker}"
    if isinstance(model, CatalystV1):
        return model.name
    if isinstance(model, InvestmentIdeaV1):
        return model.ticker
    if isinstance(model, IdeaComparisonRankingV1):
        return f"{model.ticker} rank {model.rank}"
    if isinstance(model, FactorScoreV1):
        return model.factor_name
    if isinstance(model, MissingInformationRequirementV1):
        return model.field
    if isinstance(model, OptimizationMissionV1):
        return model.name
    if isinstance(model, OptimizationAlertV1):
        return model.change_summary
    if isinstance(model, SourceFreshnessV1):
        return f"{model.source_name}: {model.status}"
    if isinstance(model, ManagementQualityAssessmentV1):
        return f"Management quality: {model.ticker or model.issuer_id}"
    if isinstance(model, ManagementQualityScorecardRowV1):
        return model.question
    if isinstance(model, ManagementQualityAccomplishmentV1):
        return model.title or model.text[:80]
    if isinstance(model, ManagementQualitySetbackV1):
        return model.title or model.text[:80]
    return label
