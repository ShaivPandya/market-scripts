from __future__ import annotations

import hashlib
import json
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
    analyst_feedback_id,
    approval_id,
    asset_id,
    audit_event_id,
    catalyst_id,
    citation_id,
    classification_id,
    company_financial_profile_id,
    computed_snapshot_ref_id,
    course_of_action_comparison_id,
    course_of_action_dissent_id,
    course_of_action_id,
    course_of_action_rationale_id,
    decision_outcome_id,
    document_artifact_id,
    equity_overview_id,
    evaluation_id,
    evidence_id,
    executed_action_id,
    executed_decision_record_id,
    extraction_run_id,
    extrinsic_sensitivity_id,
    factor_score_id,
    forward_outlook_id,
    hedge_position_id,
    hedge_position_uid,
    idea_comparison_ranking_id,
    idea_comparison_run_id,
    idea_evaluation_id,
    idea_lifecycle_event_id,
    industry_force_assessment_id,
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
    market_regime_snapshot_id,
    media_artifact_id,
    missing_information_requirement_id,
    mission_definition_id,
    model_call_ref_id,
    monitor_definition_id,
    monitor_hit_id,
    object_version_ref_id,
    observation_id,
    ontology_run_ref_id,
    opportunity_candidate_id,
    optimization_action_snapshot_id,
    optimization_alert_id,
    optimization_mission_id,
    optimization_run_id,
    pattern_detection_id,
    policy_gate_result_id,
    portfolio_id,
    portfolio_position_uid,
    portfolio_risk_snapshot_id,
    position_id,
    position_risk_snapshot_id,
    provenance_event_id,
    recommendation_id,
    regime_episode_id,
    relation_version_ref_id,
    report_run_id,
    risk_limit_id,
    risk_metric_id,
    scenario_assumption_id,
    scenario_id,
    schema_definition_ref_id,
    sector_id,
    signal_factor_score_id,
    signal_id,
    simulated_outcome_id,
    source_freshness_id,
    source_manifest_id,
    source_record_object_id,
    supply_chain_relationship_id,
    supply_demand_outlook_id,
    thesis_claim_id,
    thesis_document_id,
    thesis_id,
    thesis_section_id,
    tool_call_ref_id,
    trade_proposal_id,
    watch_trigger_id,
    workflow_artifact_id,
    workflow_run_id,
)
from ontology.schemas.objects import (
    Account,
    ActionEvent,
    ActionItem,
    ActionRun,
    AgentSessionRef,
    AnalystFeedback,
    Approval,
    Asset,
    AuditEvent,
    Catalyst,
    Citation,
    Classification,
    CompanyFinancialProfile,
    ComputedSnapshotRef,
    CourseOfAction,
    CourseOfActionComparison,
    CourseOfActionDissent,
    CourseOfActionRationale,
    DecisionOutcome,
    DocumentArtifact,
    EquityOverview,
    Evaluation,
    Evidence,
    ExecutedAction,
    ExecutedDecisionRecord,
    ExtractionRun,
    ExtrinsicSensitivity,
    FactorScore,
    ForwardOutlook,
    HedgePosition,
    IdeaComparisonRanking,
    IdeaComparisonRun,
    IdeaEvaluation,
    IdeaLifecycleEvent,
    IndustryForceAssessment,
    Instrument,
    InvestmentIdea,
    InvestmentPolicy,
    Investor,
    Issuer,
    KillCondition,
    MacroIndicator,
    ManagementQualityAccomplishment,
    ManagementQualityAssessment,
    ManagementQualityScorecardRow,
    ManagementQualitySetback,
    MarketRegimeSnapshot,
    MediaArtifact,
    MissingInformationRequirement,
    MissionDefinition,
    ModelCallRef,
    MonitorDefinition,
    MonitorHit,
    ObjectVersionRef,
    Observation,
    OntologyObject,
    OntologyRunRef,
    OpportunityCandidate,
    OptimizationActionSnapshot,
    OptimizationAlert,
    OptimizationMission,
    OptimizationRun,
    PatternDetection,
    PolicyGateResult,
    Portfolio,
    PortfolioRiskSnapshot,
    Position,
    PositionRiskSnapshot,
    ProvenanceEvent,
    Recommendation,
    RegimeEpisode,
    RelationVersionRef,
    ReportRun,
    RiskLimit,
    RiskMetric,
    Scenario,
    ScenarioAssumption,
    SchemaDefinitionRef,
    Sector,
    Signal,
    SignalFactorScore,
    SimulatedOutcome,
    SourceFreshness,
    SourceManifest,
    SourceRecord,
    SupplyChainRelationship,
    SupplyDemandOutlook,
    Thesis,
    ThesisClaim,
    ThesisDocument,
    ThesisSection,
    ToolCallRef,
    TradeProposal,
    WatchTrigger,
    WorkflowArtifact,
    WorkflowRun,
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
    "Position": Position,
    "HedgePosition": HedgePosition,
    "Asset": Asset,
    "Instrument": Instrument,
    "Issuer": Issuer,
    "Investor": Investor,
    "Account": Account,
    "Portfolio": Portfolio,
    "InvestmentPolicy": InvestmentPolicy,
    "RiskLimit": RiskLimit,
    "RiskMetric": RiskMetric,
    "Scenario": Scenario,
    "PolicyGateResult": PolicyGateResult,
    "TradeProposal": TradeProposal,
    "SourceRecord": SourceRecord,
    "ObjectVersionRef": ObjectVersionRef,
    "ExecutedAction": ExecutedAction,
    "ExecutedDecisionRecord": ExecutedDecisionRecord,
    "AuditEvent": AuditEvent,
    "Sector": Sector,
    "MacroIndicator": MacroIndicator,
    "Signal": Signal,
    "Thesis": Thesis,
    "Evaluation": Evaluation,
    "Catalyst": Catalyst,
    "KillCondition": KillCondition,
    "ThesisClaim": ThesisClaim,
    "Evidence": Evidence,
    "Citation": Citation,
    "ActionItem": ActionItem,
    "MonitorDefinition": MonitorDefinition,
    "MissionDefinition": MissionDefinition,
    "MonitorHit": MonitorHit,
    "WatchTrigger": WatchTrigger,
    "Approval": Approval,
    "ActionRun": ActionRun,
    "ActionEvent": ActionEvent,
    "ProvenanceEvent": ProvenanceEvent,
    "RelationVersionRef": RelationVersionRef,
    "SchemaDefinitionRef": SchemaDefinitionRef,
    "OntologyRunRef": OntologyRunRef,
    "AgentSessionRef": AgentSessionRef,
    "ModelCallRef": ModelCallRef,
    "ToolCallRef": ToolCallRef,
    "ComputedSnapshotRef": ComputedSnapshotRef,
    "MarketRegimeSnapshot": MarketRegimeSnapshot,
    "SignalFactorScore": SignalFactorScore,
    "ForwardOutlook": ForwardOutlook,
    "RegimeEpisode": RegimeEpisode,
    "PositionRiskSnapshot": PositionRiskSnapshot,
    "PortfolioRiskSnapshot": PortfolioRiskSnapshot,
    "WorkflowRun": WorkflowRun,
    "WorkflowArtifact": WorkflowArtifact,
    "Recommendation": Recommendation,
    "CourseOfAction": CourseOfAction,
    "CourseOfActionComparison": CourseOfActionComparison,
    "ScenarioAssumption": ScenarioAssumption,
    "SimulatedOutcome": SimulatedOutcome,
    "DecisionOutcome": DecisionOutcome,
    "CourseOfActionRationale": CourseOfActionRationale,
    "CourseOfActionDissent": CourseOfActionDissent,
    "ReportRun": ReportRun,
    "SourceManifest": SourceManifest,
    "DocumentArtifact": DocumentArtifact,
    "MediaArtifact": MediaArtifact,
    "ExtractionRun": ExtractionRun,
    "Observation": Observation,
    "Classification": Classification,
    "PatternDetection": PatternDetection,
    "AnalystFeedback": AnalystFeedback,
    "EquityOverview": EquityOverview,
    "CompanyFinancialProfile": CompanyFinancialProfile,
    "ExtrinsicSensitivity": ExtrinsicSensitivity,
    "IndustryForceAssessment": IndustryForceAssessment,
    "SupplyDemandOutlook": SupplyDemandOutlook,
    "SupplyChainRelationship": SupplyChainRelationship,
    "ThesisDocument": ThesisDocument,
    "ThesisSection": ThesisSection,
    "InvestmentIdea": InvestmentIdea,
    "OpportunityCandidate": OpportunityCandidate,
    "IdeaEvaluation": IdeaEvaluation,
    "IdeaLifecycleEvent": IdeaLifecycleEvent,
    "IdeaComparisonRun": IdeaComparisonRun,
    "IdeaComparisonRanking": IdeaComparisonRanking,
    "FactorScore": FactorScore,
    "MissingInformationRequirement": MissingInformationRequirement,
    "OptimizationMission": OptimizationMission,
    "OptimizationRun": OptimizationRun,
    "OptimizationActionSnapshot": OptimizationActionSnapshot,
    "OptimizationAlert": OptimizationAlert,
    "SourceFreshness": SourceFreshness,
    "ManagementQualityAssessment": ManagementQualityAssessment,
    "ManagementQualityScorecardRow": ManagementQualityScorecardRow,
    "ManagementQualityAccomplishment": ManagementQualityAccomplishment,
    "ManagementQualitySetback": ManagementQualitySetback,
}
OPTIONAL_NODE_TYPES = {
    "Thesis",
    "Evaluation",
    "Catalyst",
    "KillCondition",
    "ThesisClaim",
    "ActionItem",
    "MonitorDefinition",
    "MissionDefinition",
    "MonitorHit",
    "WatchTrigger",
    "Approval",
    "ActionRun",
    "ActionEvent",
    "WorkflowRun",
    "WorkflowArtifact",
    "Recommendation",
    "CourseOfAction",
    "CourseOfActionComparison",
    "ScenarioAssumption",
    "SimulatedOutcome",
    "DecisionOutcome",
    "CourseOfActionRationale",
    "CourseOfActionDissent",
    "ReportRun",
    "SourceManifest",
    "DocumentArtifact",
    "MediaArtifact",
    "ExtractionRun",
    "Observation",
    "Classification",
    "PatternDetection",
    "AnalystFeedback",
    "ProvenanceEvent",
    "RelationVersionRef",
    "SchemaDefinitionRef",
    "OntologyRunRef",
    "AgentSessionRef",
    "ModelCallRef",
    "ToolCallRef",
    "ComputedSnapshotRef",
    "MarketRegimeSnapshot",
    "SignalFactorScore",
    "ForwardOutlook",
    "RegimeEpisode",
    "PositionRiskSnapshot",
    "PortfolioRiskSnapshot",
    "EquityOverview",
    "CompanyFinancialProfile",
    "ExtrinsicSensitivity",
    "IndustryForceAssessment",
    "SupplyDemandOutlook",
    "SupplyChainRelationship",
    "ThesisDocument",
    "ThesisSection",
    "InvestmentIdea",
    "OpportunityCandidate",
    "IdeaEvaluation",
    "IdeaLifecycleEvent",
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


class OntologySchemaValidationError(ValueError):
    pass


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


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
    allow_current: bool = True,
) -> OntologyNode:
    try:
        schema_cls = NODE_SCHEMAS[node.type]
    except KeyError as exc:
        raise OntologySchemaValidationError(f"Unsupported node type: {node.type}") from exc

    node_id = node.id
    label = node.label
    payload = dict(node.properties or {})
    payload_version = int(payload.get("schema_version") or node.schema_version or 0)

    current_version = _schema_version_for(schema_cls)
    try:
        payload = _upgrade_node_payload(node.type, payload, from_version=payload_version, to_version=current_version)
    except Exception as exc:
        raise OntologySchemaValidationError(
            f"Missing compatible upgrade for {node.type} node {node.id}: {exc}"
        ) from exc

    try:
        model = cast(OntologyObject, schema_cls.model_validate(payload))
    except ValidationError as exc:
        raise OntologySchemaValidationError(f"Invalid {node.type} node {node.id}: {exc}") from exc

    expected_id = expected_node_id(node.type, model)
    if node_id != expected_id:
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
    allow_current: bool = True,
    source_id: str | None = None,
    target_id: str | None = None,
) -> OntologyEdge:
    payload = dict(edge.properties or {})
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
    allow_current: bool = True,
    source_id: str | None = None,
    target_id: str | None = None,
) -> OntologyEdge:
    relation_source_id = source_id or edge.source_id
    relation_target_id = target_id or edge.target_id
    _validate_relation(edge.relation_type, relation_source_id, relation_target_id, node_types)
    normalized = normalize_edge(
        edge,
        run_id=run_id,
        allow_current=allow_current,
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
            valid_edges.append(validate_edge_relation(edge, node_types, allow_current=True))
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
    allow_current: bool = True,
    skip_optional_invalid: bool = False,
    require_core_edges: bool = True,
) -> NormalizedGraph:
    normalized_nodes: dict[str, OntologyNode] = {}
    id_map: dict[str, str] = {}
    skipped_old_ids: set[str] = set()
    warnings: list[str] = []

    for node in nodes:
        try:
            normalized_node = normalize_node(node, run_id=run_id, allow_current=allow_current)
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
                allow_current=allow_current,
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


def expected_node_id(node_type: str, model: OntologyObject) -> str:
    if isinstance(model, Position):
        return portfolio_position_uid(model.model_dump())
    if isinstance(model, HedgePosition):
        return hedge_position_uid(model.model_dump())
    if isinstance(model, Asset):
        return asset_id(model.ticker)
    if isinstance(model, Instrument):
        return instrument_id(model.instrument_id)
    if isinstance(model, Issuer):
        return issuer_id(model.issuer_id)
    if isinstance(model, Investor):
        return investor_id(model.investor_id)
    if isinstance(model, Account):
        return account_id(model.account_id)
    if isinstance(model, Portfolio):
        return portfolio_id(model.portfolio_id)
    if isinstance(model, InvestmentPolicy):
        return investment_policy_id(model.policy_id)
    if isinstance(model, RiskLimit):
        return risk_limit_id(model.limit_id)
    if isinstance(model, RiskMetric):
        return risk_metric_id(model.metric_id)
    if isinstance(model, Scenario):
        return scenario_id(model.scenario_id)
    if isinstance(model, PolicyGateResult):
        return policy_gate_result_id(model.gate_result_id)
    if isinstance(model, TradeProposal):
        return trade_proposal_id(model.proposal_id)
    if isinstance(model, SourceRecord):
        return source_record_object_id(model.source_record_id)
    if isinstance(model, ObjectVersionRef):
        return object_version_ref_id(model.ref_id)
    if isinstance(model, ExecutedAction):
        return executed_action_id(model.executed_action_id)
    if isinstance(model, ExecutedDecisionRecord):
        return executed_decision_record_id(model.decision_record_id)
    if isinstance(model, AuditEvent):
        return audit_event_id(model.event_id)
    if isinstance(model, Sector):
        return sector_id(model.name)
    if isinstance(model, MacroIndicator):
        return macro_indicator_id(model.indicator_key)
    if isinstance(model, Signal):
        return signal_id(model.source, model.name)
    if isinstance(model, Thesis):
        return thesis_id(model.ticker)
    if isinstance(model, Evaluation):
        return evaluation_id(model.ticker, model.evaluated_at)
    if isinstance(model, Catalyst):
        return catalyst_id(model.ticker, model.name, model.description)
    if isinstance(model, KillCondition):
        return kill_condition_id(model.ticker, model.condition)
    if isinstance(model, ThesisClaim):
        return thesis_claim_id(model.ticker, model.claim)
    if isinstance(model, Evidence):
        return evidence_id(model.evidence_id)
    if isinstance(model, Citation):
        return citation_id(model.citation_id)
    if isinstance(model, ActionItem):
        return action_item_id(model.description)
    if isinstance(model, MonitorHit):
        return monitor_hit_id(model.hit_id or model.fingerprint or f"{model.entity_id}:{model.hit_type}")
    if isinstance(model, WatchTrigger):
        return watch_trigger_id(model.trigger_id or model.condition)
    if isinstance(model, Approval):
        if model.supersedes_approval_id:
            replacement_identity = _stable_hash(
                {
                    "action_id": model.action_id,
                    "payload": model.proposed_change,
                    "supersedes_approval_id": model.supersedes_approval_id,
                }
            )
            return approval_id(f"{model.entity_type}:{replacement_identity}")
        return approval_id(f"{model.entity_type}:{model.action_input_hash or model.created_at}")
    if isinstance(model, ActionRun):
        return action_run_id(f"{model.action_id}:{model.started_at}")
    if isinstance(model, ActionEvent):
        return action_event_id(f"{model.action_run_id}:{model.event_type}:{model.created_at}")
    if isinstance(model, ProvenanceEvent):
        return provenance_event_id(model.event_id)
    if isinstance(model, RelationVersionRef):
        return relation_version_ref_id(model.ref_id)
    if isinstance(model, SchemaDefinitionRef):
        return schema_definition_ref_id(model.ref_id)
    if isinstance(model, OntologyRunRef):
        return ontology_run_ref_id(model.run_id)
    if isinstance(model, AgentSessionRef):
        return agent_session_ref_id(model.session_id)
    if isinstance(model, ModelCallRef):
        return model_call_ref_id(model.call_id)
    if isinstance(model, ToolCallRef):
        return tool_call_ref_id(model.call_id)
    if isinstance(model, ComputedSnapshotRef):
        return computed_snapshot_ref_id(model.snapshot_key)
    if isinstance(model, MarketRegimeSnapshot):
        return market_regime_snapshot_id(model.snapshot_id)
    if isinstance(model, SignalFactorScore):
        return signal_factor_score_id(model.factor_score_id)
    if isinstance(model, ForwardOutlook):
        return forward_outlook_id(model.outlook_id)
    if isinstance(model, RegimeEpisode):
        return regime_episode_id(model.episode_id)
    if isinstance(model, PositionRiskSnapshot):
        return position_risk_snapshot_id(model.snapshot_id)
    if isinstance(model, PortfolioRiskSnapshot):
        return portfolio_risk_snapshot_id(model.snapshot_id)
    if isinstance(model, WorkflowRun):
        return workflow_run_id(model.run_id)
    if isinstance(model, WorkflowArtifact):
        return workflow_artifact_id(model.artifact_id)
    if isinstance(model, Recommendation):
        return recommendation_id(
            model.recommendation_id
            or model.idempotency_key
            or f"{model.report_type}:{model.as_of}:{model.action}:{model.ticker}"
        )
    if isinstance(model, CourseOfAction):
        return course_of_action_id(model.course_of_action_id or model.idempotency_key)
    if isinstance(model, CourseOfActionComparison):
        return course_of_action_comparison_id(model.comparison_id)
    if isinstance(model, ScenarioAssumption):
        return scenario_assumption_id(model.assumption_id)
    if isinstance(model, SimulatedOutcome):
        return simulated_outcome_id(model.outcome_id)
    if isinstance(model, DecisionOutcome):
        return decision_outcome_id(model.decision_outcome_id)
    if isinstance(model, CourseOfActionRationale):
        return course_of_action_rationale_id(model.rationale_id)
    if isinstance(model, CourseOfActionDissent):
        return course_of_action_dissent_id(model.dissent_id)
    if isinstance(model, ReportRun):
        return report_run_id(model.report_id)
    if isinstance(model, SourceManifest):
        return source_manifest_id(model.manifest_id)
    if isinstance(model, DocumentArtifact):
        return document_artifact_id(model.document_type, model.document_id)
    if isinstance(model, MediaArtifact):
        return media_artifact_id(model.media_id)
    if isinstance(model, ExtractionRun):
        return extraction_run_id(model.extraction_run_id)
    if isinstance(model, Observation):
        return observation_id(model.observation_id)
    if isinstance(model, Classification):
        return classification_id(model.classification_id)
    if isinstance(model, PatternDetection):
        return pattern_detection_id(model.pattern_id)
    if isinstance(model, AnalystFeedback):
        return analyst_feedback_id(model.feedback_id)
    if isinstance(model, EquityOverview):
        return equity_overview_id(model.overview_id)
    if isinstance(model, CompanyFinancialProfile):
        return company_financial_profile_id(model.profile_id)
    if isinstance(model, ExtrinsicSensitivity):
        return extrinsic_sensitivity_id(model.sensitivity_id)
    if isinstance(model, IndustryForceAssessment):
        return industry_force_assessment_id(model.force_id)
    if isinstance(model, SupplyDemandOutlook):
        return supply_demand_outlook_id(model.outlook_id)
    if isinstance(model, SupplyChainRelationship):
        return supply_chain_relationship_id(model.relationship_id)
    if isinstance(model, ThesisDocument):
        return thesis_document_id(model.thesis_document_id)
    if isinstance(model, ThesisSection):
        return thesis_section_id(model.section_id)
    if isinstance(model, InvestmentIdea):
        return investment_idea_id(model.idea_id)
    if isinstance(model, OpportunityCandidate):
        return opportunity_candidate_id(model.candidate_id or model.idempotency_key)
    if isinstance(model, IdeaEvaluation):
        return idea_evaluation_id(model.evaluation_id)
    if isinstance(model, IdeaLifecycleEvent):
        return idea_lifecycle_event_id(model.event_id)
    if isinstance(model, IdeaComparisonRun):
        return idea_comparison_run_id(model.comparison_run_id)
    if isinstance(model, IdeaComparisonRanking):
        return idea_comparison_ranking_id(model.ranking_id)
    if isinstance(model, FactorScore):
        return factor_score_id(model.factor_score_id)
    if isinstance(model, MissingInformationRequirement):
        return missing_information_requirement_id(model.requirement_id)
    if isinstance(model, MonitorDefinition):
        return monitor_definition_id(model.monitor_id or model.name)
    if isinstance(model, MissionDefinition):
        return mission_definition_id(model.mission_id or model.name)
    if isinstance(model, OptimizationMission):
        return optimization_mission_id(model.mission_id)
    if isinstance(model, OptimizationRun):
        return optimization_run_id(model.run_id)
    if isinstance(model, OptimizationActionSnapshot):
        return optimization_action_snapshot_id(model.snapshot_id)
    if isinstance(model, OptimizationAlert):
        return optimization_alert_id(model.alert_id)
    if isinstance(model, SourceFreshness):
        return source_freshness_id(model.freshness_id)
    if isinstance(model, ManagementQualityAssessment):
        return management_quality_assessment_id(model.assessment_id)
    if isinstance(model, ManagementQualityScorecardRow):
        return management_quality_scorecard_row_id(model.row_id)
    if isinstance(model, ManagementQualityAccomplishment):
        return management_quality_accomplishment_id(model.accomplishment_id)
    if isinstance(model, ManagementQualitySetback):
        return management_quality_setback_id(model.setback_id)
    raise OntologySchemaValidationError(f"Unsupported node schema for type {node_type}")


def node_from_schema(
    *,
    node_id: str,
    node_type: EntityType,
    label: str,
    model: OntologyObject,
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
        allow_current=False,
    )


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
    if current_version == to_version:
        return dict(payload)
    if current_version > to_version:
        raise ValueError(f"future schema version {current_version} cannot be read as v{to_version}")
    raise ValueError(f"unsupported schema version {current_version} for {node_type}; expected {to_version}")


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


def _label_for(node_type: str, label: str, model: OntologyObject) -> str:
    if isinstance(model, (Position, Asset)):
        return model.ticker
    if isinstance(model, Sector):
        return model.name
    if isinstance(model, MacroIndicator):
        return model.name
    if isinstance(model, Signal):
        return model.name
    if isinstance(model, Thesis):
        return f"Thesis: {model.ticker}"
    if isinstance(model, Evaluation):
        return f"Eval: {model.ticker}"
    if isinstance(model, Catalyst):
        return model.name
    if isinstance(model, InvestmentIdea):
        return model.ticker
    if isinstance(model, OpportunityCandidate):
        label = model.summary or model.trigger
        ticker = model.ticker or "unknown"
        return f"{ticker}: {label[:80]}"
    if isinstance(model, IdeaComparisonRanking):
        return f"{model.ticker} rank {model.rank}"
    if isinstance(model, CourseOfAction):
        return f"{model.action}: {model.ticker or model.instrument_id or model.course_of_action_id}"
    if isinstance(model, CourseOfActionComparison):
        return model.objective
    if isinstance(model, ScenarioAssumption):
        return model.name
    if isinstance(model, SimulatedOutcome):
        return f"Simulated outcome: {model.outcome_id}"
    if isinstance(model, DecisionOutcome):
        process = model.process_label or model.outcome_status
        return f"Decision outcome: {process}"
    if isinstance(model, CourseOfActionRationale):
        return model.summary[:80]
    if isinstance(model, CourseOfActionDissent):
        return model.claim[:80]
    if isinstance(model, FactorScore):
        return model.factor_name
    if isinstance(model, MissingInformationRequirement):
        return model.field
    if isinstance(model, OptimizationMission):
        return model.name
    if isinstance(model, OptimizationAlert):
        return model.change_summary
    if isinstance(model, SourceFreshness):
        return f"{model.source_name}: {model.status}"
    if isinstance(model, MarketRegimeSnapshot):
        return f"Market regime: {model.regime_label}"
    if isinstance(model, SignalFactorScore):
        return f"{model.factor_name}: {model.status}"
    if isinstance(model, ForwardOutlook):
        return f"Forward outlook: {model.label}"
    if isinstance(model, RegimeEpisode):
        return f"Regime episode: {model.regime}"
    if isinstance(model, PositionRiskSnapshot):
        return f"Position risk: {model.ticker or model.snapshot_id}"
    if isinstance(model, PortfolioRiskSnapshot):
        return f"Portfolio risk: {model.snapshot_id}"
    if isinstance(model, EquityOverview):
        return f"Equity overview: {model.ticker or model.issuer_id}"
    if isinstance(model, CompanyFinancialProfile):
        return f"Financial profile: {model.ticker or model.issuer_id}"
    if isinstance(model, ExtrinsicSensitivity):
        return model.factor
    if isinstance(model, IndustryForceAssessment):
        return model.force
    if isinstance(model, SupplyDemandOutlook):
        return f"{model.outlook_type} outlook"
    if isinstance(model, SupplyChainRelationship):
        return f"{model.counterparty_role}: {model.counterparty_name}"
    if isinstance(model, ThesisDocument):
        return f"Thesis document: {model.ticker}"
    if isinstance(model, ThesisSection):
        return model.heading
    if isinstance(model, ManagementQualityAssessment):
        return f"Management quality: {model.ticker or model.issuer_id}"
    if isinstance(model, ManagementQualityScorecardRow):
        return model.question
    if isinstance(model, ManagementQualityAccomplishment):
        return model.title or model.text[:80]
    if isinstance(model, ManagementQualitySetback):
        return model.title or model.text[:80]
    return label
