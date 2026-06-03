"""Authoritative ontology object and relation write service."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

from ontology.models import OntologyEdge, OntologyNode
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
    canonical_ticker,
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
    hedge_position_uid,
    idea_comparison_ranking_id,
    idea_comparison_run_id,
    idea_evaluation_id,
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
    model_call_ref_id,
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
from ontology.schemas.registry import NODE_SCHEMAS, normalize_edge, normalize_node
from ontology.schemas.relations import PROVENANCE_RELATION_TYPES, RELATION_REGISTRY, get_relation_definition
from ontology.temporal_repository import (
    ObjectVersionWrite,
    RelationVersionWrite,
    TemporalActor,
    TemporalOntologyRepository,
)

logger = logging.getLogger(__name__)

_GOVERNED_OBJECT_TYPES = {
    "ActionRun",
    "AnalystFeedback",
    "Approval",
    "AuditEvent",
    "Catalyst",
    "Citation",
    "Classification",
    "DocumentArtifact",
    "Evidence",
    "Evaluation",
    "ExecutedAction",
    "ExecutedDecisionRecord",
    "ExtractionRun",
    "FactorScore",
    "HedgePosition",
    "IdeaComparisonRanking",
    "KillCondition",
    "ManagementQualityAccomplishment",
    "ManagementQualityAssessment",
    "ManagementQualityScorecardRow",
    "ManagementQualitySetback",
    "MarketRegimeSnapshot",
    "MediaArtifact",
    "MissingInformationRequirement",
    "ObjectVersionRef",
    "Observation",
    "PatternDetection",
    "PolicyGateResult",
    "PortfolioRiskSnapshot",
    "Position",
    "PositionRiskSnapshot",
    "CourseOfAction",
    "CourseOfActionComparison",
    "ScenarioAssumption",
    "SimulatedOutcome",
    "DecisionOutcome",
    "CourseOfActionRationale",
    "CourseOfActionDissent",
    "OpportunityCandidate",
    "Recommendation",
    "RegimeEpisode",
    "SignalFactorScore",
    "SourceManifest",
    "SourceRecord",
    "ForwardOutlook",
    "Thesis",
    "ThesisClaim",
    "ThesisDocument",
    "ThesisSection",
    "TradeProposal",
    "WorkflowArtifact",
    "WatchTrigger",
    "EquityOverview",
    "CompanyFinancialProfile",
    "ExtrinsicSensitivity",
    "IndustryForceAssessment",
    "SupplyDemandOutlook",
    "SupplyChainRelationship",
}
_GOVERNED_RELATION_TYPES = {
    "action_run_mutates_object_version",
    "analyst_feedback_targets_object",
    "approval_applies_action_run",
    "action_run_produces_executed_action",
    "approval_targets_recommendation",
    "approval_targets_research_object",
    "approval_targets_trade_proposal",
    "approval_targets_workflow_artifact",
    "claim_disconfirmed_by_evidence",
    "claim_supported_by_evidence",
    "evidence_has_citation",
    "evidence_cites_citation",
    "executed_action_mutates_object_version",
    "executed_decision_applies_approval",
    "executed_decision_records_action_run",
    "recommendation_supported_by_evidence",
    "recommendation_contradicted_by_evidence",
    "recommendation_has_policy_gate_result",
    "recommendation_has_trade_proposal",
    "recommendation_uses_position_risk_snapshot",
    "recommendation_uses_portfolio_risk_snapshot",
    "trade_proposal_requires_approval",
    "document_artifact_materializes_research_object",
    "equity_overview_covers_issuer",
    "equity_overview_covers_instrument",
    "equity_overview_has_financial_profile",
    "equity_overview_has_extrinsic_sensitivity",
    "equity_overview_has_industry_force",
    "equity_overview_has_supply_demand_outlook",
    "equity_overview_has_supply_chain_relationship",
    "thesis_document_covers_issuer",
    "thesis_document_covers_instrument",
    "thesis_document_has_section",
    "computed_snapshot_materializes_object_version",
    "course_of_action_targets_account",
    "course_of_action_targets_portfolio",
    "course_of_action_targets_position",
    "course_of_action_targets_instrument",
    "course_of_action_targets_thesis",
    "course_of_action_targets_catalyst",
    "course_of_action_uses_position_risk_snapshot",
    "course_of_action_uses_portfolio_risk_snapshot",
    "course_of_action_uses_scenario",
    "course_of_action_links_recommendation",
    "scenario_has_assumption",
    "course_of_action_has_simulated_outcome",
    "recommendation_has_decision_outcome",
    "course_of_action_has_decision_outcome",
    "decision_outcome_contrasts_simulated_outcome",
    "course_of_action_has_rationale",
    "course_of_action_supported_by_evidence",
    "course_of_action_contradicted_by_evidence",
    "course_of_action_has_dissent",
    "course_of_action_requires_approval",
    "approval_targets_course_of_action",
    "action_run_applies_course_of_action",
    "comparison_includes_course_of_action",
    "comparison_selects_course_of_action",
    "market_regime_has_factor_score",
    "market_regime_has_forward_outlook",
    "market_regime_has_episode",
    "factor_score_uses_source_record",
    "factor_score_uses_computed_snapshot",
    "source_manifest_governs_source_record",
    "source_record_produces_document_artifact",
    "source_record_produces_media_artifact",
    "artifact_has_extraction_run",
    "extraction_run_produces_observation",
    "extraction_run_produces_classification",
    "extraction_run_produces_pattern_detection",
}


def source_record_object_uid_for(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("source_record:"):
        suffix = text.removeprefix("source_record:")
        if suffix and ":" not in suffix and source_record_object_id(suffix) == text:
            return text
        return source_record_object_id(text)
    return source_record_object_id(text)


class OntologyWriteContractError(ValueError):
    """Raised when an ontology write violates the authoritative write contract."""


class OntologyObjectService:
    """Typed write boundary for temporal ontology objects and relations."""

    def __init__(self, repository: TemporalOntologyRepository | None = None):
        self.repo = repository or TemporalOntologyRepository()

    def get_object(
        self,
        object_uid: str,
        *,
        as_of: datetime | str | None = None,
        tx_as_of: datetime | str | None = None,
    ) -> dict[str, Any] | None:
        row = self.repo.get_object(object_uid, as_of=as_of, tx_as_of=tx_as_of)
        return with_temporal_meta(row) if row else None

    def query_objects(
        self,
        object_type: str | None = None,
        filters: Mapping[str, Any] | None = None,
        *,
        as_of: datetime | str | None = None,
        tx_as_of: datetime | str | None = None,
        include_history: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = self.repo.query_objects(
            object_type,
            filters=filters,
            as_of=as_of,
            tx_as_of=tx_as_of,
            include_history=include_history,
            limit=limit,
            offset=offset,
        )
        return [with_temporal_meta(row) for row in rows]

    def write_object(
        self,
        object_type: str,
        business_key: str,
        properties: Mapping[str, Any],
        valid_from: datetime | str,
        valid_to: datetime | str | None = None,
        *,
        actor: Any = None,
        provenance: Mapping[str, Any] | str | None = None,
        action_run_id: int | None = None,
        approval_id: str | int | None = None,
        source_record_id: str | None = None,
        input_hash: str | None = None,
        temporal_confidence: str = "native",
    ) -> dict[str, Any]:
        _check_registered_object_type(object_type)
        actor_fields = _actor_fields(actor)
        provenance_event_id = _provenance_event_id(provenance)
        _require_write_provenance("object", object_type, provenance_event_id)
        _require_governed_lineage(
            "object",
            object_type,
            provenance_event_id=provenance_event_id,
            action_run_id=action_run_id,
            approval_id=approval_id,
            source_record_id=source_record_id,
        )
        object_uid = object_uid_for(object_type, business_key, properties)
        normalized = normalize_object_payload(object_uid, object_type, business_key, properties)
        row = self.repo.write_object_version(
            ObjectVersionWrite(
                object_uid=normalized["object_uid"],
                object_type=normalized["object_type"],
                business_key=normalized["business_key"],
                schema_name=normalized["schema_name"],
                schema_version=normalized["schema_version"],
                properties=normalized["properties"],
                valid_from=valid_from,
                valid_to=valid_to,
                source_record_id=source_record_id,
                provenance_event_id=provenance_event_id,
                action_run_id=action_run_id,
                approval_id=approval_id,
                actor_type=actor_fields.actor_type,
                actor_id=actor_fields.actor_id,
                input_hash=input_hash,
                temporal_confidence=temporal_confidence or "native",
            )
        )
        return with_temporal_meta(row)

    def write_relation(
        self,
        source_uid: str,
        target_uid: str,
        relation_type: str,
        properties: Mapping[str, Any] | None,
        valid_from: datetime | str,
        valid_to: datetime | str | None = None,
        *,
        actor: Any = None,
        provenance: Mapping[str, Any] | str | None = None,
        action_run_id: int | None = None,
        approval_id: str | int | None = None,
        source_record_id: str | None = None,
        input_hash: str | None = None,
        temporal_confidence: str = "native",
    ) -> dict[str, Any]:
        _check_registered_relation_type(relation_type)
        actor_fields = _actor_fields(actor)
        provenance_event_id = _provenance_event_id(provenance)
        _require_write_provenance("relation", relation_type, provenance_event_id)
        _require_governed_lineage(
            "relation",
            relation_type,
            provenance_event_id=provenance_event_id,
            action_run_id=action_run_id,
            approval_id=approval_id,
            source_record_id=source_record_id,
        )
        normalized = normalize_relation_payload(source_uid, target_uid, relation_type, properties or {})
        _require_registered_relation_properties(relation_type, normalized["properties"], source_uid, target_uid)
        row = self.repo.write_relation_version(
            RelationVersionWrite(
                relation_uid=relation_uid_for(source_uid, target_uid, relation_type, normalized["properties"]),
                source_object_uid=source_uid,
                target_object_uid=target_uid,
                relation_type=relation_type,
                relation_schema_name=normalized["relation_schema_name"],
                relation_schema_version=normalized["relation_schema_version"],
                properties=normalized["properties"],
                valid_from=valid_from,
                valid_to=valid_to,
                source_record_id=source_record_id,
                provenance_event_id=provenance_event_id,
                action_run_id=action_run_id,
                approval_id=approval_id,
                actor_type=actor_fields.actor_type,
                actor_id=actor_fields.actor_id,
                input_hash=input_hash,
                temporal_confidence=temporal_confidence or "native",
            )
        )
        return with_temporal_meta(row)

    def query_relations(
        self,
        relation_type: str | None = None,
        *,
        source_object_uid: str | None = None,
        target_object_uid: str | None = None,
        as_of: datetime | str | None = None,
        tx_as_of: datetime | str | None = None,
        include_history: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = self.repo.query_relations(
            relation_type,
            source_object_uid=source_object_uid,
            target_object_uid=target_object_uid,
            as_of=as_of,
            tx_as_of=tx_as_of,
            include_history=include_history,
            limit=limit,
            offset=offset,
        )
        return [with_temporal_meta(row) for row in rows]

    def correct_object_version(
        self,
        version_id: str,
        *,
        properties: Mapping[str, Any],
        actor: Any = None,
        provenance: Mapping[str, Any] | str | None = None,
        input_hash: str | None = None,
    ) -> dict[str, Any]:
        provenance_event_id = _provenance_event_id(provenance)
        _require_write_provenance("object correction", version_id, provenance_event_id)
        row = self.repo.correct_object_version(
            version_id,
            properties=dict(properties),
            actor=_actor_fields(actor),
            provenance_event_id=provenance_event_id,
            input_hash=input_hash,
        )
        return with_temporal_meta(row)

    def expire_object(
        self,
        object_uid: str,
        *,
        valid_from: datetime | str | None = None,
        valid_to: datetime | str | None = None,
        tx_to: datetime | str | None = None,
    ) -> int:
        return self.repo.expire_object_versions(object_uid, valid_from=valid_from, valid_to=valid_to, tx_to=tx_to)

    def expire_relation(
        self,
        relation_uid: str,
        *,
        valid_from: datetime | str | None = None,
        valid_to: datetime | str | None = None,
        tx_to: datetime | str | None = None,
    ) -> int:
        return self.repo.expire_relation_versions(relation_uid, valid_from=valid_from, valid_to=valid_to, tx_to=tx_to)


def normalize_object_payload(
    object_uid: str,
    object_type: str,
    business_key: str,
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    props = dict(properties or {})
    if object_type in NODE_SCHEMAS:
        props = _with_object_identity_fields(object_type, business_key, props)
        node = normalize_node(
            OntologyNode(
                id=object_uid,
                type=cast(Any, object_type),
                label=str(props.get("label") or business_key or object_uid),
                properties=props,
                schema_name=object_type,
                schema_version=1,
            ),
        )
        return {
            "object_uid": node.id,
            "object_type": node.type,
            "business_key": business_key,
            "schema_name": node.schema_name,
            "schema_version": node.schema_version,
            "properties": node.properties,
        }
    return {
        "object_uid": object_uid,
        "object_type": object_type,
        "business_key": business_key,
        "schema_name": object_type,
        "schema_version": int(props.pop("schema_version", 1) or 1),
        "properties": props,
    }


def normalize_relation_payload(
    source_uid: str,
    target_uid: str,
    relation_type: str,
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    props = dict(properties or {})
    if relation_type in RELATION_REGISTRY:
        edge = normalize_edge(
            OntologyEdge(
                source_id=source_uid,
                target_id=target_uid,
                relation_type=cast(Any, relation_type),
                properties=props,
                schema_version=1,
                relation_schema_name=relation_type,
                relation_schema_version=1,
            ),
        )
        return {
            "relation_schema_name": edge.relation_schema_name,
            "relation_schema_version": edge.relation_schema_version,
            "properties": edge.properties,
        }
    return {
        "relation_schema_name": relation_type,
        "relation_schema_version": int(props.pop("relation_schema_version", 1) or 1),
        "properties": props,
    }


def object_uid_for(object_type: str, business_key: str, properties: Mapping[str, Any] | None = None) -> str:
    props = dict(properties or {})
    key = str(business_key or "").strip()
    if object_type == "Position":
        if key.startswith("position:"):
            return key
        return position_id(canonical_ticker(props.get("ticker") or key))
    if object_type == "HedgePosition":
        if key.lower().startswith(("hedge_position:", "position:")):
            return hedge_position_uid(key)
        row = dict(props)
        if not row.get("ticker") and not row.get("position_id"):
            row["ticker"] = canonical_ticker(key)
        return hedge_position_uid(row)
    if object_type == "Asset":
        if key.startswith("asset:"):
            return key
        return asset_id(canonical_ticker(props.get("ticker") or key))
    if object_type == "Instrument":
        if key.startswith("instrument:"):
            return key
        return instrument_id(props.get("instrument_id") or props.get("ticker") or key)
    if object_type == "Issuer":
        if key.startswith("issuer:"):
            return key
        return issuer_id(props.get("issuer_id") or props.get("name") or props.get("ticker") or key)
    if object_type == "Investor":
        if key.startswith("investor:"):
            return key
        return investor_id(props.get("investor_id") or key)
    if object_type == "Account":
        if key.startswith("account:"):
            return key
        return account_id(props.get("account_id") or key)
    if object_type == "Portfolio":
        if key.startswith("portfolio:"):
            return key
        return portfolio_id(props.get("portfolio_id") or key)
    if object_type == "InvestmentPolicy":
        if key.startswith("investment_policy:"):
            return key
        return investment_policy_id(props.get("policy_id") or key)
    if object_type == "RiskLimit":
        if key.startswith("risk_limit:"):
            return key
        return risk_limit_id(props.get("limit_id") or key)
    if object_type == "RiskMetric":
        if key.startswith("risk_metric:"):
            return key
        return risk_metric_id(props.get("metric_id") or key)
    if object_type == "Scenario":
        if key.startswith("scenario:"):
            return key
        return scenario_id(props.get("scenario_id") or key)
    if object_type == "PolicyGateResult":
        if key.startswith("policy_gate_result:"):
            return key
        return policy_gate_result_id(props.get("gate_result_id") or key)
    if object_type == "TradeProposal":
        if key.startswith("trade_proposal:"):
            return key
        return trade_proposal_id(props.get("proposal_id") or key)
    if object_type == "SourceRecord":
        return source_record_object_uid_for(props.get("source_record_id") or key)
    if object_type == "ObjectVersionRef":
        if key.startswith("object_version_ref:"):
            return key
        return object_version_ref_id(props.get("ref_id") or key)
    if object_type == "RelationVersionRef":
        if key.startswith("relation_version_ref:"):
            return key
        return relation_version_ref_id(props.get("ref_id") or props.get("version_id") or key)
    if object_type == "SchemaDefinitionRef":
        if key.startswith("schema_definition_ref:"):
            return key
        return schema_definition_ref_id(props.get("ref_id") or key)
    if object_type == "OntologyRunRef":
        if key.startswith("ontology_run_ref:"):
            return key
        return ontology_run_ref_id(props.get("run_id") or key)
    if object_type == "AgentSessionRef":
        if key.startswith("agent_session_ref:"):
            return key
        return agent_session_ref_id(props.get("session_id") or key)
    if object_type == "ModelCallRef":
        if key.startswith("model_call_ref:"):
            return key
        return model_call_ref_id(props.get("call_id") or key)
    if object_type == "ToolCallRef":
        if key.startswith("tool_call_ref:"):
            return key
        return tool_call_ref_id(props.get("call_id") or key)
    if object_type == "ComputedSnapshotRef":
        if key.startswith("computed_snapshot_ref:"):
            return key
        return computed_snapshot_ref_id(props.get("snapshot_key") or props.get("snapshot_id") or key)
    if object_type == "MarketRegimeSnapshot":
        if key.startswith("market_regime_snapshot:"):
            return key
        return market_regime_snapshot_id(props.get("snapshot_id") or key)
    if object_type == "SignalFactorScore":
        if key.startswith("signal_factor_score:"):
            return key
        return signal_factor_score_id(props.get("factor_score_id") or key)
    if object_type == "ForwardOutlook":
        if key.startswith("forward_outlook:"):
            return key
        return forward_outlook_id(props.get("outlook_id") or key)
    if object_type == "RegimeEpisode":
        if key.startswith("regime_episode:"):
            return key
        return regime_episode_id(props.get("episode_id") or key)
    if object_type == "PositionRiskSnapshot":
        if key.startswith("position_risk_snapshot:"):
            return key
        return position_risk_snapshot_id(props.get("snapshot_id") or key)
    if object_type == "PortfolioRiskSnapshot":
        if key.startswith("portfolio_risk_snapshot:"):
            return key
        return portfolio_risk_snapshot_id(props.get("snapshot_id") or key)
    if object_type == "ExecutedAction":
        if key.startswith("executed_action:"):
            return key
        return executed_action_id(props.get("executed_action_id") or key)
    if object_type == "ExecutedDecisionRecord":
        if key.startswith("executed_decision_record:"):
            return key
        return executed_decision_record_id(props.get("decision_record_id") or key)
    if object_type == "AuditEvent":
        if key.startswith("audit_event:"):
            return key
        return audit_event_id(props.get("event_id") or key)
    if object_type == "Sector":
        if key.startswith("sector:"):
            return key
        return sector_id(str(props.get("name") or key))
    if object_type == "MacroIndicator":
        if key.startswith("macro_indicator:"):
            return key
        return macro_indicator_id(str(props.get("indicator_key") or key))
    if object_type == "Signal":
        if key.startswith("signal:"):
            return key
        source = props.get("source") or props.get("module") or props.get("adapter") or "unknown"
        name = props.get("name") or props.get("signal_key") or key
        return signal_id(source, name)
    if object_type == "Thesis":
        if key.startswith("thesis:"):
            return key
        return thesis_id(canonical_ticker(props.get("ticker") or key))
    if object_type == "Evaluation":
        if key.startswith("evaluation:"):
            return key
        ticker = canonical_ticker(props.get("ticker") or key.split(":", 1)[0])
        evaluated_at = str(props.get("evaluated_at") or key)
        return evaluation_id(ticker, evaluated_at)
    if object_type == "Catalyst":
        if key.startswith("catalyst:"):
            return key
        ticker = canonical_ticker(props.get("ticker") or key.split(":", 1)[0])
        name = str(props.get("name") or props.get("description") or key)
        description = str(props.get("description") or name)
        return catalyst_id(ticker, name, description)
    if object_type == "KillCondition":
        if key.startswith("kill_condition:"):
            return key
        ticker = canonical_ticker(props.get("ticker") or key.split(":", 1)[0])
        return kill_condition_id(ticker, props.get("condition") or key)
    if object_type == "ThesisClaim":
        if key.startswith("thesis_claim:"):
            return key
        ticker = canonical_ticker(props.get("ticker") or key.split(":", 1)[0])
        return thesis_claim_id(ticker, props.get("claim") or key)
    if object_type == "Evidence":
        if key.startswith("evidence:"):
            return key
        return evidence_id(props.get("evidence_id") or props.get("source_record_id") or key)
    if object_type == "Citation":
        if key.startswith("citation:"):
            return key
        return citation_id(props.get("citation_id") or props.get("url") or props.get("source_path") or key)
    if object_type == "ActionItem":
        if key.startswith("action_item:"):
            return key
        return action_item_id(key)
    if object_type == "WatchTrigger":
        if key.startswith("watch_trigger:"):
            return key
        return watch_trigger_id(key)
    if object_type == "Approval":
        if key.startswith("approval:"):
            return key
        return approval_id(key)
    if object_type == "ActionRun":
        if key.startswith("action_run:"):
            return key
        return action_run_id(key)
    if object_type == "ActionEvent":
        if key.startswith("action_event:"):
            return key
        return action_event_id(key)
    if object_type == "WorkflowRun":
        if key.startswith("workflow_run:"):
            return key
        return workflow_run_id(props.get("run_id") or key)
    if object_type == "WorkflowArtifact":
        if key.startswith("workflow_artifact:"):
            return key
        return workflow_artifact_id(props.get("artifact_id") or key)
    if object_type == "Recommendation":
        if key.startswith("recommendation:"):
            return key
        return recommendation_id(props.get("recommendation_id") or props.get("idempotency_key") or key)
    if object_type == "CourseOfAction":
        if key.startswith("course_of_action:"):
            return key
        return course_of_action_id(props.get("course_of_action_id") or props.get("idempotency_key") or key)
    if object_type == "CourseOfActionComparison":
        if key.startswith("course_of_action_comparison:"):
            return key
        return course_of_action_comparison_id(props.get("comparison_id") or key)
    if object_type == "ScenarioAssumption":
        if key.startswith("scenario_assumption:"):
            return key
        return scenario_assumption_id(props.get("assumption_id") or key)
    if object_type == "SimulatedOutcome":
        if key.startswith("simulated_outcome:"):
            return key
        return simulated_outcome_id(props.get("outcome_id") or key)
    if object_type == "DecisionOutcome":
        if key.startswith("decision_outcome:"):
            return key
        return decision_outcome_id(props.get("decision_outcome_id") or key)
    if object_type == "CourseOfActionRationale":
        if key.startswith("course_of_action_rationale:"):
            return key
        return course_of_action_rationale_id(props.get("rationale_id") or key)
    if object_type == "CourseOfActionDissent":
        if key.startswith("course_of_action_dissent:"):
            return key
        return course_of_action_dissent_id(props.get("dissent_id") or key)
    if object_type == "ReportRun":
        if key.startswith("report_run:"):
            return key
        return report_run_id(props.get("report_id") or key)
    if object_type == "SourceManifest":
        if key.startswith("source_manifest:"):
            return key
        return source_manifest_id(props.get("manifest_id") or key)
    if object_type == "DocumentArtifact":
        if key.startswith("document_artifact:"):
            return key
        return document_artifact_id(props.get("document_type") or "document", props.get("document_id") or key)
    if object_type == "MediaArtifact":
        if key.startswith("media_artifact:"):
            return key
        return media_artifact_id(props.get("media_id") or key)
    if object_type == "ExtractionRun":
        if key.startswith("extraction_run:"):
            return key
        return extraction_run_id(props.get("extraction_run_id") or key)
    if object_type == "Observation":
        if key.startswith("observation:"):
            return key
        return observation_id(props.get("observation_id") or key)
    if object_type == "Classification":
        if key.startswith("classification:"):
            return key
        return classification_id(props.get("classification_id") or key)
    if object_type == "PatternDetection":
        if key.startswith("pattern_detection:"):
            return key
        return pattern_detection_id(props.get("pattern_id") or key)
    if object_type == "AnalystFeedback":
        if key.startswith("analyst_feedback:"):
            return key
        return analyst_feedback_id(props.get("feedback_id") or key)
    if object_type == "EquityOverview":
        if key.startswith("equity_overview:"):
            return key
        return equity_overview_id(props.get("overview_id") or props.get("ticker") or key)
    if object_type == "CompanyFinancialProfile":
        if key.startswith("company_financial_profile:"):
            return key
        return company_financial_profile_id(props.get("profile_id") or key)
    if object_type == "ExtrinsicSensitivity":
        if key.startswith("extrinsic_sensitivity:"):
            return key
        return extrinsic_sensitivity_id(props.get("sensitivity_id") or key)
    if object_type == "IndustryForceAssessment":
        if key.startswith("industry_force_assessment:"):
            return key
        return industry_force_assessment_id(props.get("force_id") or key)
    if object_type == "SupplyDemandOutlook":
        if key.startswith("supply_demand_outlook:"):
            return key
        return supply_demand_outlook_id(props.get("outlook_id") or key)
    if object_type == "SupplyChainRelationship":
        if key.startswith("supply_chain_relationship:"):
            return key
        return supply_chain_relationship_id(props.get("relationship_id") or key)
    if object_type == "ThesisDocument":
        if key.startswith("thesis_document:"):
            return key
        return thesis_document_id(props.get("thesis_document_id") or props.get("ticker") or key)
    if object_type == "ThesisSection":
        if key.startswith("thesis_section:"):
            return key
        return thesis_section_id(props.get("section_id") or key)
    if object_type == "ProvenanceEvent":
        return provenance_event_id(props.get("event_id") or props.get("id") or key)
    if object_type == "InvestmentIdea":
        return investment_idea_id(props.get("idea_id") or props.get("id") or key)
    if object_type == "OpportunityCandidate":
        if key.startswith("opportunity_candidate:"):
            return key
        return opportunity_candidate_id(props.get("candidate_id") or props.get("idempotency_key") or key)
    if object_type == "IdeaEvaluation":
        return idea_evaluation_id(props.get("evaluation_id") or props.get("id") or key)
    if object_type == "IdeaComparisonRun":
        return idea_comparison_run_id(props.get("comparison_run_id") or props.get("run_id") or props.get("id") or key)
    if object_type == "IdeaComparisonRanking":
        return idea_comparison_ranking_id(props.get("ranking_id") or props.get("id") or key)
    if object_type == "FactorScore":
        return factor_score_id(props.get("factor_score_id") or props.get("id") or key)
    if object_type == "MissingInformationRequirement":
        return missing_information_requirement_id(props.get("requirement_id") or props.get("id") or key)
    if object_type == "OptimizationMission":
        return optimization_mission_id(props.get("mission_id") or props.get("id") or key)
    if object_type == "OptimizationRun":
        return optimization_run_id(props.get("run_id") or props.get("id") or key)
    if object_type == "OptimizationActionSnapshot":
        return optimization_action_snapshot_id(props.get("snapshot_id") or props.get("id") or key)
    if object_type == "OptimizationAlert":
        return optimization_alert_id(props.get("alert_id") or props.get("id") or key)
    if object_type == "SourceFreshness":
        return source_freshness_id(props.get("freshness_id") or props.get("id") or key)
    if object_type == "ManagementQualityAssessment":
        return management_quality_assessment_id(props.get("assessment_id") or props.get("id") or key)
    if object_type == "ManagementQualityScorecardRow":
        return management_quality_scorecard_row_id(props.get("row_id") or props.get("id") or key)
    if object_type == "ManagementQualityAccomplishment":
        return management_quality_accomplishment_id(props.get("accomplishment_id") or props.get("id") or key)
    if object_type == "ManagementQualitySetback":
        return management_quality_setback_id(props.get("setback_id") or props.get("id") or key)
    if ":" in key and key.split(":", 1)[0]:
        return key
    return f"{_slug(object_type)}:{_slug(key)}"


def relation_uid_for(
    source_uid: str,
    target_uid: str,
    relation_type: str,
    properties: Mapping[str, Any] | None = None,
) -> str:
    props = dict(properties or {})
    if relation_type in PROVENANCE_RELATION_TYPES:
        raw = ":".join(
            [
                relation_type,
                str(props.get("event_id") or "unknown_event"),
                str(props.get("source_ref_type") or "unknown_source"),
                str(props.get("source_ref_id") or source_uid),
                str(props.get("source_ref_version") or "latest"),
                str(props.get("target_ref_type") or "unknown_target"),
                str(props.get("target_ref_id") or target_uid),
                str(props.get("target_ref_version") or "latest"),
            ]
        )
    else:
        raw = f"{relation_type}:{source_uid}->{target_uid}"
    return raw if len(raw) <= 180 else f"{relation_type}:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:32]}"


def with_temporal_meta(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    properties = payload.get("properties_json")
    if isinstance(properties, dict):
        payload["properties"] = properties
    temporal = {
        "object_uid": payload.get("object_uid"),
        "relation_uid": payload.get("relation_uid"),
        "version_id": str(payload.get("version_id")) if payload.get("version_id") is not None else None,
        "valid_from": _iso(payload.get("valid_from")),
        "valid_to": _iso(payload.get("valid_to")),
        "tx_from": _iso(payload.get("tx_from")),
        "tx_to": _iso(payload.get("tx_to")),
        "temporal_confidence": payload.get("temporal_confidence"),
    }
    meta_raw = payload.get("_meta")
    meta = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}
    meta["temporal"] = {key: value for key, value in temporal.items() if value is not None}
    payload["_meta"] = meta
    return payload


def _actor_fields(actor: Any) -> TemporalActor:
    if actor is None:
        return TemporalActor()
    if isinstance(actor, Mapping):
        return TemporalActor(
            actor_type=str(actor.get("actor_type")) if actor.get("actor_type") is not None else None,
            actor_id=str(actor.get("actor_id")) if actor.get("actor_id") is not None else None,
        )
    return TemporalActor(
        actor_type=str(getattr(actor, "actor_type", "")) or None,
        actor_id=str(getattr(actor, "actor_id", "")) or None,
    )


def _provenance_event_id(provenance: Mapping[str, Any] | str | None) -> str | None:
    if provenance is None:
        return None
    if isinstance(provenance, str):
        return provenance or None
    for key in ("provenance_event_id", "event_id", "id"):
        value = provenance.get(key)
        if value:
            return str(value)
    return None


def _require_governed_lineage(
    surface: str,
    name: str,
    *,
    provenance_event_id: str | None,
    action_run_id: int | None,
    approval_id: str | int | None,
    source_record_id: str | None,
) -> None:
    governed = name in _GOVERNED_OBJECT_TYPES if surface == "object" else name in _GOVERNED_RELATION_TYPES
    if not governed:
        return
    if provenance_event_id or action_run_id is not None or approval_id is not None or source_record_id:
        return
    raise ValueError(f"Governed ontology {surface} write '{name}' requires provenance or action lineage")


def _strict_write_contract_enabled() -> bool:
    return True


def _check_registered_object_type(object_type: str) -> None:
    if object_type in NODE_SCHEMAS:
        return
    message = f"Unregistered ontology object type: {object_type}"
    if _strict_write_contract_enabled():
        raise OntologyWriteContractError(message)
    raise OntologyWriteContractError(message)


def _check_registered_relation_type(relation_type: str) -> None:
    if relation_type in RELATION_REGISTRY:
        return
    message = f"Unregistered ontology relation type: {relation_type}"
    if _strict_write_contract_enabled():
        raise OntologyWriteContractError(message)
    raise OntologyWriteContractError(message)


def _require_write_provenance(surface: str, name: str, provenance_event_id: str | None) -> None:
    if provenance_event_id:
        return
    raise OntologyWriteContractError(f"Ontology {surface} write '{name}' requires provenance")


def _require_registered_relation_properties(
    relation_type: str,
    properties: Mapping[str, Any],
    source_uid: str,
    target_uid: str,
) -> None:
    definition = get_relation_definition(relation_type)
    missing = [name for name in sorted(definition.required_properties) if _missing_property(properties.get(name))]
    if missing:
        fields = ", ".join(missing)
        raise OntologyWriteContractError(
            f"Ontology relation write '{relation_type}' {source_uid}->{target_uid} missing required properties: {fields}"
        )


def _missing_property(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _with_object_identity_fields(object_type: str, business_key: str, props: dict[str, Any]) -> dict[str, Any]:
    out = dict(props)
    key = str(business_key or "").strip()
    if object_type == "ProvenanceEvent":
        out.setdefault("event_id", out.get("id") or key)
    elif object_type == "RelationVersionRef":
        out.setdefault("ref_id", out.get("version_id") or out.get("relation_uid") or out.get("id") or key)
        out.setdefault("version_id", out.get("ref_id"))
    elif object_type == "SchemaDefinitionRef":
        out.setdefault("ref_id", out.get("id") or key)
    elif object_type == "OntologyRunRef":
        out.setdefault("run_id", out.get("id") or key)
    elif object_type == "AgentSessionRef":
        out.setdefault("session_id", out.get("id") or key)
    elif object_type == "ModelCallRef":
        out.setdefault("call_id", out.get("id") or key)
    elif object_type == "ToolCallRef":
        out.setdefault("call_id", out.get("id") or key)
    elif object_type == "ComputedSnapshotRef":
        out.setdefault("snapshot_key", out.get("snapshot_id") or out.get("id") or key)
    elif object_type == "SourceManifest":
        out.setdefault("manifest_id", out.get("id") or key)
    elif object_type == "MediaArtifact":
        out.setdefault("media_id", out.get("id") or key)
    elif object_type == "ExtractionRun":
        out.setdefault("extraction_run_id", out.get("run_id") or out.get("id") or key)
    elif object_type == "Observation":
        out.setdefault("observation_id", out.get("id") or key)
    elif object_type == "Classification":
        out.setdefault("classification_id", out.get("id") or key)
    elif object_type == "PatternDetection":
        out.setdefault("pattern_id", out.get("id") or key)
    elif object_type == "AnalystFeedback":
        out.setdefault("feedback_id", out.get("id") or key)
    elif object_type == "MarketRegimeSnapshot":
        out.setdefault("snapshot_id", out.get("id") or key)
    elif object_type == "SignalFactorScore":
        out.setdefault("factor_score_id", out.get("id") or key)
    elif object_type == "ForwardOutlook":
        out.setdefault("outlook_id", out.get("id") or key)
    elif object_type == "RegimeEpisode":
        out.setdefault("episode_id", out.get("id") or key)
    elif object_type == "PositionRiskSnapshot":
        out.setdefault("snapshot_id", out.get("id") or key)
    elif object_type == "PortfolioRiskSnapshot":
        out.setdefault("snapshot_id", out.get("id") or key)
    elif object_type == "EquityOverview":
        out.setdefault("overview_id", out.get("id") or key)
    elif object_type == "CompanyFinancialProfile":
        out.setdefault("profile_id", out.get("id") or key)
    elif object_type == "ExtrinsicSensitivity":
        out.setdefault("sensitivity_id", out.get("id") or key)
    elif object_type == "IndustryForceAssessment":
        out.setdefault("force_id", out.get("id") or key)
    elif object_type == "SupplyDemandOutlook":
        out.setdefault("outlook_id", out.get("id") or key)
    elif object_type == "SupplyChainRelationship":
        out.setdefault("relationship_id", out.get("id") or key)
    elif object_type == "ThesisDocument":
        out.setdefault("thesis_document_id", out.get("id") or key)
    elif object_type == "ThesisSection":
        out.setdefault("section_id", out.get("id") or key)
    elif object_type == "InvestmentIdea":
        out.setdefault("idea_id", out.get("id") or key)
    elif object_type == "OpportunityCandidate":
        out.setdefault("candidate_id", out.get("id") or key)
    elif object_type == "IdeaEvaluation":
        out.setdefault("evaluation_id", out.get("id") or key)
    elif object_type == "IdeaComparisonRun":
        out.setdefault("comparison_run_id", out.get("run_id") or out.get("id") or key)
        out.setdefault("run_id", out.get("comparison_run_id"))
    elif object_type == "IdeaComparisonRanking":
        out.setdefault("ranking_id", out.get("id") or key)
    elif object_type == "FactorScore":
        out.setdefault("factor_score_id", out.get("id") or key)
    elif object_type == "MissingInformationRequirement":
        out.setdefault("requirement_id", out.get("id") or key)
    elif object_type == "OptimizationMission":
        out.setdefault("mission_id", out.get("id") or key)
    elif object_type == "OptimizationRun":
        out.setdefault("run_id", out.get("id") or key)
    elif object_type == "OptimizationActionSnapshot":
        out.setdefault("snapshot_id", out.get("id") or key)
    elif object_type == "OptimizationAlert":
        out.setdefault("alert_id", out.get("id") or key)
    elif object_type == "SourceFreshness":
        out.setdefault("freshness_id", out.get("id") or key)
    elif object_type == "ManagementQualityAssessment":
        out.setdefault("assessment_id", out.get("id") or key)
    elif object_type == "ManagementQualityScorecardRow":
        out.setdefault("row_id", out.get("id") or key)
    elif object_type == "ManagementQualityAccomplishment":
        out.setdefault("accomplishment_id", out.get("id") or key)
    elif object_type == "ManagementQualitySetback":
        out.setdefault("setback_id", out.get("id") or key)
    return out


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(value or "").strip().lower()).strip("_")
    return text or "unknown"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)
