from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, get_args

from pydantic import Field, field_validator

from ontology.models import EntityType, RelationType
from ontology.schemas.base import NonBlankStr, OntologySchemaBase, clean_optional_text, clean_text
from ontology.schemas.objects import SignalDirection

REFERENCES_ASSET: RelationType = "references_asset"
PORTFOLIO_HOLDS_POSITION: RelationType = "portfolio_holds_position"
POSITION_REFERENCES_INSTRUMENT: RelationType = "position_references_instrument"
INSTRUMENT_ISSUED_BY_ISSUER: RelationType = "instrument_issued_by_issuer"
THESIS_COVERS_INSTRUMENT: RelationType = "thesis_covers_instrument"
CLAIM_SUPPORTED_BY_EVIDENCE: RelationType = "claim_supported_by_evidence"
CLAIM_DISCONFIRMED_BY_EVIDENCE: RelationType = "claim_disconfirmed_by_evidence"
EVIDENCE_HAS_CITATION: RelationType = "evidence_has_citation"
BELONGS_TO_SECTOR: RelationType = "belongs_to_sector"
HAS_THESIS: RelationType = "has_thesis"
EVALUATED_BY: RelationType = "evaluated_by"
HAS_CATALYST: RelationType = "has_catalyst"
EMITS_SIGNAL: RelationType = "emits_signal"
AFFECTED_BY: RelationType = "affected_by"
EXPOSED_TO_SIGNAL: RelationType = "exposed_to_signal"
POSITION_HAS_HEDGE: RelationType = "position_has_hedge"
THESIS_HAS_KILL_CONDITION: RelationType = "thesis_has_kill_condition"
THESIS_HAS_CLAIM: RelationType = "thesis_has_claim"
CLAIM_LINKS_CATALYST: RelationType = "claim_links_catalyst"
CLAIM_LINKS_KILL_CONDITION: RelationType = "claim_links_kill_condition"
ACTION_ITEM_TARGETS_OBJECT: RelationType = "action_item_targets_object"
WATCH_TRIGGER_TARGETS_OBJECT: RelationType = "watch_trigger_targets_object"
APPROVAL_PROPOSES_ACTION: RelationType = "approval_proposes_action"
APPROVAL_APPLIES_ACTION_RUN: RelationType = "approval_applies_action_run"
ACTION_RUN_MUTATES_OBJECT_VERSION: RelationType = "action_run_mutates_object_version"
WORKFLOW_RUN_PRODUCES_ARTIFACT: RelationType = "workflow_run_produces_artifact"
WORKFLOW_ARTIFACT_PROPOSES_APPROVAL: RelationType = "workflow_artifact_proposes_approval"
REPORT_RUN_PRODUCES_RECOMMENDATION: RelationType = "report_run_produces_recommendation"
SOURCE_RECORD_MATERIALIZES_OBJECT: RelationType = "source_record_materializes_object"
RECOMMENDATION_SUPPORTED_BY_SOURCE_RECORD: RelationType = "recommendation_supported_by_source_record"
RECOMMENDATION_USES_RISK_METRIC: RelationType = "recommendation_uses_risk_metric"
RECOMMENDATION_USES_SCENARIO: RelationType = "recommendation_uses_scenario"
INVESTOR_OWNS_ACCOUNT: RelationType = "investor_owns_account"
ACCOUNT_HAS_PORTFOLIO: RelationType = "account_has_portfolio"
ACCOUNT_GOVERNED_BY_POLICY: RelationType = "account_governed_by_policy"
POLICY_HAS_RISK_LIMIT: RelationType = "policy_has_risk_limit"
RECOMMENDATION_TARGETS_ACCOUNT: RelationType = "recommendation_targets_account"
RECOMMENDATION_TARGETS_PORTFOLIO: RelationType = "recommendation_targets_portfolio"
RECOMMENDATION_TARGETS_INSTRUMENT: RelationType = "recommendation_targets_instrument"
TRADE_PROPOSAL_DERIVES_FROM_RECOMMENDATION: RelationType = "trade_proposal_derives_from_recommendation"
TRADE_PROPOSAL_TARGETS_ASSET: RelationType = "trade_proposal_targets_asset"
TRADE_PROPOSAL_REQUIRES_APPROVAL: RelationType = "trade_proposal_requires_approval"
APPROVAL_TARGETS_RECOMMENDATION: RelationType = "approval_targets_recommendation"
APPROVAL_TARGETS_TRADE_PROPOSAL: RelationType = "approval_targets_trade_proposal"
APPROVAL_TARGETS_WORKFLOW_ARTIFACT: RelationType = "approval_targets_workflow_artifact"
APPROVAL_TARGETS_RESEARCH_OBJECT: RelationType = "approval_targets_research_object"
ACTION_RUN_PRODUCES_EXECUTED_ACTION: RelationType = "action_run_produces_executed_action"
EXECUTED_ACTION_MUTATES_OBJECT_VERSION: RelationType = "executed_action_mutates_object_version"
EXECUTED_DECISION_APPLIES_APPROVAL: RelationType = "executed_decision_applies_approval"
EXECUTED_DECISION_RECORDS_ACTION_RUN: RelationType = "executed_decision_records_action_run"
SOURCE_RECORD_MATERIALIZES_OBJECT_VERSION: RelationType = "source_record_materializes_object_version"
AUDIT_EVENT_OBSERVES_ACTION_RUN: RelationType = "audit_event_observes_action_run"
POLICY_GATE_EVALUATES_RECOMMENDATION: RelationType = "policy_gate_evaluates_recommendation"
POLICY_GATE_EVALUATES_TRADE_PROPOSAL: RelationType = "policy_gate_evaluates_trade_proposal"
POLICY_GATE_USES_RISK_METRIC: RelationType = "policy_gate_uses_risk_metric"
POLICY_GATE_USES_SCENARIO: RelationType = "policy_gate_uses_scenario"
PROVENANCE_USED: RelationType = "provenance_used"
PROVENANCE_PRODUCED: RelationType = "provenance_produced"
PROVENANCE_SCHEMA_BOUND: RelationType = "provenance_schema_bound"
PROVENANCE_EXECUTED: RelationType = "provenance_executed"
PROVENANCE_EXECUTED_AS: RelationType = "provenance_executed_as"
PROVENANCE_TRIGGERED: RelationType = "provenance_triggered"
PROVENANCE_PROPOSED: RelationType = "provenance_proposed"
PROVENANCE_RESOLVED_BY: RelationType = "provenance_resolved_by"
PROVENANCE_APPROVED_EXECUTION: RelationType = "provenance_approved_execution"
PROVENANCE_AUDITED_BY: RelationType = "provenance_audited_by"
PROVENANCE_UPDATED: RelationType = "provenance_updated"
IDEA_HAS_EVALUATION: RelationType = "idea_has_evaluation"
COMPARISON_RUN_HAS_RANKING: RelationType = "comparison_run_has_ranking"
RANKING_TARGETS_IDEA: RelationType = "ranking_targets_idea"
RANKING_USES_EVALUATION: RelationType = "ranking_uses_evaluation"
RESEARCH_OBJECT_HAS_FACTOR_SCORE: RelationType = "research_object_has_factor_score"
RESEARCH_OBJECT_HAS_MISSING_INFORMATION: RelationType = "research_object_has_missing_information"
RESEARCH_OBJECT_SUPPORTED_BY_EVIDENCE: RelationType = "research_object_supported_by_evidence"
RESEARCH_OBJECT_DISCONFIRMED_BY_EVIDENCE: RelationType = "research_object_disconfirmed_by_evidence"
RESEARCH_OBJECT_USES_DOCUMENT: RelationType = "research_object_uses_document"
RESEARCH_OBJECT_LINKS_RECOMMENDATION: RelationType = "research_object_links_recommendation"
RESEARCH_OBJECT_LINKS_APPROVAL: RelationType = "research_object_links_approval"
RESEARCH_OBJECT_LINKS_ACTION_ITEM: RelationType = "research_object_links_action_item"
MANAGEMENT_QUALITY_ASSESSES_ISSUER: RelationType = "management_quality_assesses_issuer"
MANAGEMENT_QUALITY_HAS_SCORECARD_ROW: RelationType = "management_quality_has_scorecard_row"
MANAGEMENT_QUALITY_HAS_ACCOMPLISHMENT: RelationType = "management_quality_has_accomplishment"
MANAGEMENT_QUALITY_HAS_SETBACK: RelationType = "management_quality_has_setback"
OPTIMIZATION_MISSION_HAS_RUN: RelationType = "optimization_mission_has_run"
OPTIMIZATION_RUN_HAS_SNAPSHOT: RelationType = "optimization_run_has_snapshot"
OPTIMIZATION_ALERT_CURRENT_SNAPSHOT: RelationType = "optimization_alert_current_snapshot"
OPTIMIZATION_ALERT_PREVIOUS_SNAPSHOT: RelationType = "optimization_alert_previous_snapshot"
OPTIMIZATION_SNAPSHOT_TARGETS_POSITION: RelationType = "optimization_snapshot_targets_position"
OPTIMIZATION_SNAPSHOT_TARGETS_INSTRUMENT: RelationType = "optimization_snapshot_targets_instrument"
OPTIMIZATION_ALERT_LINKS_APPROVAL: RelationType = "optimization_alert_links_approval"
OPTIMIZATION_ALERT_LINKS_ACTION_ITEM: RelationType = "optimization_alert_links_action_item"
OPTIMIZATION_OBJECT_HAS_SOURCE_FRESHNESS: RelationType = "optimization_object_has_source_freshness"
COMPUTED_SNAPSHOT_MATERIALIZES_OBJECT_VERSION: RelationType = "computed_snapshot_materializes_object_version"
MARKET_REGIME_HAS_FACTOR_SCORE: RelationType = "market_regime_has_factor_score"
MARKET_REGIME_HAS_FORWARD_OUTLOOK: RelationType = "market_regime_has_forward_outlook"
MARKET_REGIME_HAS_EPISODE: RelationType = "market_regime_has_episode"
FACTOR_SCORE_USES_SOURCE_RECORD: RelationType = "factor_score_uses_source_record"
FACTOR_SCORE_USES_COMPUTED_SNAPSHOT: RelationType = "factor_score_uses_computed_snapshot"
MARKET_REGIME_USES_RISK_SNAPSHOT: RelationType = "market_regime_uses_risk_snapshot"
MARKET_REGIME_REFERENCES_MACRO_INDICATOR: RelationType = "market_regime_references_macro_indicator"
FACTOR_SCORE_REFERENCES_SECTOR: RelationType = "factor_score_references_sector"
RECOMMENDATION_SUPPORTED_BY_EVIDENCE: RelationType = "recommendation_supported_by_evidence"
RECOMMENDATION_CONTRADICTED_BY_EVIDENCE: RelationType = "recommendation_contradicted_by_evidence"
EVIDENCE_CITES_CITATION: RelationType = "evidence_cites_citation"
RECOMMENDATION_HAS_POLICY_GATE_RESULT: RelationType = "recommendation_has_policy_gate_result"
RECOMMENDATION_HAS_TRADE_PROPOSAL: RelationType = "recommendation_has_trade_proposal"
RECOMMENDATION_USES_POSITION_RISK_SNAPSHOT: RelationType = "recommendation_uses_position_risk_snapshot"
RECOMMENDATION_USES_PORTFOLIO_RISK_SNAPSHOT: RelationType = "recommendation_uses_portfolio_risk_snapshot"
DOCUMENT_ARTIFACT_MATERIALIZES_RESEARCH_OBJECT: RelationType = "document_artifact_materializes_research_object"
EQUITY_OVERVIEW_COVERS_ISSUER: RelationType = "equity_overview_covers_issuer"
EQUITY_OVERVIEW_COVERS_INSTRUMENT: RelationType = "equity_overview_covers_instrument"
EQUITY_OVERVIEW_HAS_FINANCIAL_PROFILE: RelationType = "equity_overview_has_financial_profile"
EQUITY_OVERVIEW_HAS_EXTRINSIC_SENSITIVITY: RelationType = "equity_overview_has_extrinsic_sensitivity"
EQUITY_OVERVIEW_HAS_INDUSTRY_FORCE: RelationType = "equity_overview_has_industry_force"
EQUITY_OVERVIEW_HAS_SUPPLY_DEMAND_OUTLOOK: RelationType = "equity_overview_has_supply_demand_outlook"
THESIS_DOCUMENT_COVERS_ISSUER: RelationType = "thesis_document_covers_issuer"
THESIS_DOCUMENT_COVERS_INSTRUMENT: RelationType = "thesis_document_covers_instrument"
THESIS_DOCUMENT_HAS_SECTION: RelationType = "thesis_document_has_section"

PROVENANCE_RELATION_TYPES: frozenset[RelationType] = frozenset(
    {
        PROVENANCE_USED,
        PROVENANCE_PRODUCED,
        PROVENANCE_SCHEMA_BOUND,
        PROVENANCE_EXECUTED,
        PROVENANCE_EXECUTED_AS,
        PROVENANCE_TRIGGERED,
        PROVENANCE_PROPOSED,
        PROVENANCE_RESOLVED_BY,
        PROVENANCE_APPROVED_EXECUTION,
        PROVENANCE_AUDITED_BY,
        PROVENANCE_UPDATED,
    }
)
PROVENANCE_ENDPOINT_TYPES: frozenset[EntityType] = frozenset(get_args(EntityType))
PROVENANCE_REQUIRED_PROPERTIES = frozenset(
    {
        "event_id",
        "source_ref_type",
        "source_ref_id",
        "target_ref_type",
        "target_ref_id",
        "redaction_policy",
        "retention_class",
    }
)


class RelationCardinality(StrEnum):
    MANY_TO_MANY = "many_to_many"
    SOURCE_UNIQUE = "source_unique"
    TARGET_UNIQUE = "target_unique"
    SOURCE_AND_TARGET_UNIQUE = "source_and_target_unique"


@dataclass(frozen=True, slots=True)
class RelationDefinition:
    name: RelationType
    source_type: EntityType
    target_type: EntityType
    cardinality: RelationCardinality
    required_properties: frozenset[str]
    optional: bool = False
    allowed_source_types: frozenset[EntityType] | None = None
    allowed_target_types: frozenset[EntityType] | None = None


RELATION_REGISTRY: dict[str, RelationDefinition] = {
    REFERENCES_ASSET: RelationDefinition(
        name=REFERENCES_ASSET,
        source_type="Position",
        target_type="Asset",
        cardinality=RelationCardinality.SOURCE_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
    ),
    PORTFOLIO_HOLDS_POSITION: RelationDefinition(
        name=PORTFOLIO_HOLDS_POSITION,
        source_type="Portfolio",
        target_type="Position",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    POSITION_REFERENCES_INSTRUMENT: RelationDefinition(
        name=POSITION_REFERENCES_INSTRUMENT,
        source_type="Position",
        target_type="Instrument",
        cardinality=RelationCardinality.SOURCE_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    INSTRUMENT_ISSUED_BY_ISSUER: RelationDefinition(
        name=INSTRUMENT_ISSUED_BY_ISSUER,
        source_type="Instrument",
        target_type="Issuer",
        cardinality=RelationCardinality.SOURCE_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    THESIS_COVERS_INSTRUMENT: RelationDefinition(
        name=THESIS_COVERS_INSTRUMENT,
        source_type="Thesis",
        target_type="Instrument",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    CLAIM_SUPPORTED_BY_EVIDENCE: RelationDefinition(
        name=CLAIM_SUPPORTED_BY_EVIDENCE,
        source_type="ThesisClaim",
        target_type="Evidence",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    CLAIM_DISCONFIRMED_BY_EVIDENCE: RelationDefinition(
        name=CLAIM_DISCONFIRMED_BY_EVIDENCE,
        source_type="ThesisClaim",
        target_type="Evidence",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EVIDENCE_HAS_CITATION: RelationDefinition(
        name=EVIDENCE_HAS_CITATION,
        source_type="Evidence",
        target_type="Citation",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    BELONGS_TO_SECTOR: RelationDefinition(
        name=BELONGS_TO_SECTOR,
        source_type="Asset",
        target_type="Sector",
        cardinality=RelationCardinality.SOURCE_UNIQUE,
        required_properties=frozenset({"ontology_run_id", "source"}),
    ),
    HAS_THESIS: RelationDefinition(
        name=HAS_THESIS,
        source_type="Position",
        target_type="Thesis",
        cardinality=RelationCardinality.SOURCE_AND_TARGET_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EVALUATED_BY: RelationDefinition(
        name=EVALUATED_BY,
        source_type="Thesis",
        target_type="Evaluation",
        cardinality=RelationCardinality.TARGET_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    HAS_CATALYST: RelationDefinition(
        name=HAS_CATALYST,
        source_type="Thesis",
        target_type="Catalyst",
        cardinality=RelationCardinality.TARGET_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EMITS_SIGNAL: RelationDefinition(
        name=EMITS_SIGNAL,
        source_type="MacroIndicator",
        target_type="Signal",
        cardinality=RelationCardinality.TARGET_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
    ),
    AFFECTED_BY: RelationDefinition(
        name=AFFECTED_BY,
        source_type="Sector",
        target_type="MacroIndicator",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
    ),
    EXPOSED_TO_SIGNAL: RelationDefinition(
        name=EXPOSED_TO_SIGNAL,
        source_type="Position",
        target_type="Signal",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset(
            {"component", "source", "name", "threshold", "direction", "contribution", "ontology_run_id"}
        ),
    ),
    POSITION_HAS_HEDGE: RelationDefinition(
        name=POSITION_HAS_HEDGE,
        source_type="Position",
        target_type="HedgePosition",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    THESIS_HAS_KILL_CONDITION: RelationDefinition(
        name=THESIS_HAS_KILL_CONDITION,
        source_type="Thesis",
        target_type="KillCondition",
        cardinality=RelationCardinality.TARGET_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    THESIS_HAS_CLAIM: RelationDefinition(
        name=THESIS_HAS_CLAIM,
        source_type="Thesis",
        target_type="ThesisClaim",
        cardinality=RelationCardinality.TARGET_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    CLAIM_LINKS_CATALYST: RelationDefinition(
        name=CLAIM_LINKS_CATALYST,
        source_type="ThesisClaim",
        target_type="Catalyst",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    CLAIM_LINKS_KILL_CONDITION: RelationDefinition(
        name=CLAIM_LINKS_KILL_CONDITION,
        source_type="ThesisClaim",
        target_type="KillCondition",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    ACTION_ITEM_TARGETS_OBJECT: RelationDefinition(
        name=ACTION_ITEM_TARGETS_OBJECT,
        source_type="ActionItem",
        target_type="Thesis",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "target_object_type"}),
        optional=True,
    ),
    WATCH_TRIGGER_TARGETS_OBJECT: RelationDefinition(
        name=WATCH_TRIGGER_TARGETS_OBJECT,
        source_type="WatchTrigger",
        target_type="Thesis",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "target_object_type"}),
        optional=True,
    ),
    APPROVAL_PROPOSES_ACTION: RelationDefinition(
        name=APPROVAL_PROPOSES_ACTION,
        source_type="Approval",
        target_type="ActionRun",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "action_id"}),
        optional=True,
    ),
    APPROVAL_APPLIES_ACTION_RUN: RelationDefinition(
        name=APPROVAL_APPLIES_ACTION_RUN,
        source_type="Approval",
        target_type="ActionRun",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    ACTION_RUN_MUTATES_OBJECT_VERSION: RelationDefinition(
        name=ACTION_RUN_MUTATES_OBJECT_VERSION,
        source_type="ActionRun",
        target_type="ObjectVersionRef",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "object_uid", "version_id"}),
        optional=True,
    ),
    WORKFLOW_RUN_PRODUCES_ARTIFACT: RelationDefinition(
        name=WORKFLOW_RUN_PRODUCES_ARTIFACT,
        source_type="WorkflowRun",
        target_type="WorkflowArtifact",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    WORKFLOW_ARTIFACT_PROPOSES_APPROVAL: RelationDefinition(
        name=WORKFLOW_ARTIFACT_PROPOSES_APPROVAL,
        source_type="WorkflowArtifact",
        target_type="Approval",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    REPORT_RUN_PRODUCES_RECOMMENDATION: RelationDefinition(
        name=REPORT_RUN_PRODUCES_RECOMMENDATION,
        source_type="ReportRun",
        target_type="Recommendation",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    SOURCE_RECORD_MATERIALIZES_OBJECT: RelationDefinition(
        name=SOURCE_RECORD_MATERIALIZES_OBJECT,
        source_type="SourceRecord",
        target_type="ObjectVersionRef",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "source_record_id", "object_uid"}),
        optional=True,
    ),
    RECOMMENDATION_SUPPORTED_BY_SOURCE_RECORD: RelationDefinition(
        name=RECOMMENDATION_SUPPORTED_BY_SOURCE_RECORD,
        source_type="Recommendation",
        target_type="SourceRecord",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_USES_RISK_METRIC: RelationDefinition(
        name=RECOMMENDATION_USES_RISK_METRIC,
        source_type="Recommendation",
        target_type="RiskMetric",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_USES_SCENARIO: RelationDefinition(
        name=RECOMMENDATION_USES_SCENARIO,
        source_type="Recommendation",
        target_type="Scenario",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    INVESTOR_OWNS_ACCOUNT: RelationDefinition(
        name=INVESTOR_OWNS_ACCOUNT,
        source_type="Investor",
        target_type="Account",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    ACCOUNT_HAS_PORTFOLIO: RelationDefinition(
        name=ACCOUNT_HAS_PORTFOLIO,
        source_type="Account",
        target_type="Portfolio",
        cardinality=RelationCardinality.SOURCE_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    ACCOUNT_GOVERNED_BY_POLICY: RelationDefinition(
        name=ACCOUNT_GOVERNED_BY_POLICY,
        source_type="Account",
        target_type="InvestmentPolicy",
        cardinality=RelationCardinality.SOURCE_UNIQUE,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    POLICY_HAS_RISK_LIMIT: RelationDefinition(
        name=POLICY_HAS_RISK_LIMIT,
        source_type="InvestmentPolicy",
        target_type="RiskLimit",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_TARGETS_ACCOUNT: RelationDefinition(
        name=RECOMMENDATION_TARGETS_ACCOUNT,
        source_type="Recommendation",
        target_type="Account",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_TARGETS_PORTFOLIO: RelationDefinition(
        name=RECOMMENDATION_TARGETS_PORTFOLIO,
        source_type="Recommendation",
        target_type="Portfolio",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_TARGETS_INSTRUMENT: RelationDefinition(
        name=RECOMMENDATION_TARGETS_INSTRUMENT,
        source_type="Recommendation",
        target_type="Instrument",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    TRADE_PROPOSAL_DERIVES_FROM_RECOMMENDATION: RelationDefinition(
        name=TRADE_PROPOSAL_DERIVES_FROM_RECOMMENDATION,
        source_type="TradeProposal",
        target_type="Recommendation",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    TRADE_PROPOSAL_TARGETS_ASSET: RelationDefinition(
        name=TRADE_PROPOSAL_TARGETS_ASSET,
        source_type="TradeProposal",
        target_type="Asset",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    TRADE_PROPOSAL_REQUIRES_APPROVAL: RelationDefinition(
        name=TRADE_PROPOSAL_REQUIRES_APPROVAL,
        source_type="TradeProposal",
        target_type="Approval",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    APPROVAL_TARGETS_RECOMMENDATION: RelationDefinition(
        name=APPROVAL_TARGETS_RECOMMENDATION,
        source_type="Approval",
        target_type="Recommendation",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    APPROVAL_TARGETS_TRADE_PROPOSAL: RelationDefinition(
        name=APPROVAL_TARGETS_TRADE_PROPOSAL,
        source_type="Approval",
        target_type="TradeProposal",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    APPROVAL_TARGETS_WORKFLOW_ARTIFACT: RelationDefinition(
        name=APPROVAL_TARGETS_WORKFLOW_ARTIFACT,
        source_type="Approval",
        target_type="WorkflowArtifact",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    APPROVAL_TARGETS_RESEARCH_OBJECT: RelationDefinition(
        name=APPROVAL_TARGETS_RESEARCH_OBJECT,
        source_type="Approval",
        target_type="Thesis",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "target_object_type"}),
        optional=True,
    ),
    ACTION_RUN_PRODUCES_EXECUTED_ACTION: RelationDefinition(
        name=ACTION_RUN_PRODUCES_EXECUTED_ACTION,
        source_type="ActionRun",
        target_type="ExecutedAction",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EXECUTED_ACTION_MUTATES_OBJECT_VERSION: RelationDefinition(
        name=EXECUTED_ACTION_MUTATES_OBJECT_VERSION,
        source_type="ExecutedAction",
        target_type="ObjectVersionRef",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "object_uid", "version_id"}),
        optional=True,
    ),
    EXECUTED_DECISION_APPLIES_APPROVAL: RelationDefinition(
        name=EXECUTED_DECISION_APPLIES_APPROVAL,
        source_type="ExecutedDecisionRecord",
        target_type="Approval",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EXECUTED_DECISION_RECORDS_ACTION_RUN: RelationDefinition(
        name=EXECUTED_DECISION_RECORDS_ACTION_RUN,
        source_type="ExecutedDecisionRecord",
        target_type="ActionRun",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    SOURCE_RECORD_MATERIALIZES_OBJECT_VERSION: RelationDefinition(
        name=SOURCE_RECORD_MATERIALIZES_OBJECT_VERSION,
        source_type="SourceRecord",
        target_type="ObjectVersionRef",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "source_record_id", "object_uid", "version_id"}),
        optional=True,
    ),
    AUDIT_EVENT_OBSERVES_ACTION_RUN: RelationDefinition(
        name=AUDIT_EVENT_OBSERVES_ACTION_RUN,
        source_type="AuditEvent",
        target_type="ActionRun",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    POLICY_GATE_EVALUATES_RECOMMENDATION: RelationDefinition(
        name=POLICY_GATE_EVALUATES_RECOMMENDATION,
        source_type="PolicyGateResult",
        target_type="Recommendation",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    POLICY_GATE_EVALUATES_TRADE_PROPOSAL: RelationDefinition(
        name=POLICY_GATE_EVALUATES_TRADE_PROPOSAL,
        source_type="PolicyGateResult",
        target_type="TradeProposal",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    POLICY_GATE_USES_RISK_METRIC: RelationDefinition(
        name=POLICY_GATE_USES_RISK_METRIC,
        source_type="PolicyGateResult",
        target_type="RiskMetric",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    POLICY_GATE_USES_SCENARIO: RelationDefinition(
        name=POLICY_GATE_USES_SCENARIO,
        source_type="PolicyGateResult",
        target_type="Scenario",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    IDEA_HAS_EVALUATION: RelationDefinition(
        name=IDEA_HAS_EVALUATION,
        source_type="InvestmentIdea",
        target_type="IdeaEvaluation",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    COMPARISON_RUN_HAS_RANKING: RelationDefinition(
        name=COMPARISON_RUN_HAS_RANKING,
        source_type="IdeaComparisonRun",
        target_type="IdeaComparisonRanking",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RANKING_TARGETS_IDEA: RelationDefinition(
        name=RANKING_TARGETS_IDEA,
        source_type="IdeaComparisonRanking",
        target_type="InvestmentIdea",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RANKING_USES_EVALUATION: RelationDefinition(
        name=RANKING_USES_EVALUATION,
        source_type="IdeaComparisonRanking",
        target_type="IdeaEvaluation",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RESEARCH_OBJECT_HAS_FACTOR_SCORE: RelationDefinition(
        name=RESEARCH_OBJECT_HAS_FACTOR_SCORE,
        source_type="IdeaEvaluation",
        target_type="FactorScore",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
        allowed_source_types=frozenset({"IdeaEvaluation", "ManagementQualityAssessment"}),
    ),
    RESEARCH_OBJECT_HAS_MISSING_INFORMATION: RelationDefinition(
        name=RESEARCH_OBJECT_HAS_MISSING_INFORMATION,
        source_type="IdeaEvaluation",
        target_type="MissingInformationRequirement",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
        allowed_source_types=frozenset({"IdeaEvaluation", "ManagementQualityAssessment"}),
    ),
    RESEARCH_OBJECT_SUPPORTED_BY_EVIDENCE: RelationDefinition(
        name=RESEARCH_OBJECT_SUPPORTED_BY_EVIDENCE,
        source_type="IdeaEvaluation",
        target_type="Evidence",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
        allowed_source_types=frozenset(
            {
                "IdeaEvaluation",
                "ManagementQualityAssessment",
                "ManagementQualityScorecardRow",
                "ManagementQualityAccomplishment",
                "ManagementQualitySetback",
                "EquityOverview",
                "CompanyFinancialProfile",
                "ExtrinsicSensitivity",
                "IndustryForceAssessment",
                "SupplyDemandOutlook",
                "ThesisDocument",
                "ThesisSection",
            }
        ),
    ),
    RESEARCH_OBJECT_DISCONFIRMED_BY_EVIDENCE: RelationDefinition(
        name=RESEARCH_OBJECT_DISCONFIRMED_BY_EVIDENCE,
        source_type="IdeaEvaluation",
        target_type="Evidence",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
        allowed_source_types=frozenset(
            {
                "IdeaEvaluation",
                "ManagementQualityAssessment",
                "ManagementQualityScorecardRow",
                "ManagementQualityAccomplishment",
                "ManagementQualitySetback",
                "EquityOverview",
                "CompanyFinancialProfile",
                "ExtrinsicSensitivity",
                "IndustryForceAssessment",
                "SupplyDemandOutlook",
                "ThesisDocument",
                "ThesisSection",
            }
        ),
    ),
    RESEARCH_OBJECT_USES_DOCUMENT: RelationDefinition(
        name=RESEARCH_OBJECT_USES_DOCUMENT,
        source_type="InvestmentIdea",
        target_type="DocumentArtifact",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "document_role"}),
        optional=True,
        allowed_source_types=frozenset(
            {
                "InvestmentIdea",
                "IdeaEvaluation",
                "ManagementQualityAssessment",
                "EquityOverview",
                "ThesisDocument",
            }
        ),
    ),
    RESEARCH_OBJECT_LINKS_RECOMMENDATION: RelationDefinition(
        name=RESEARCH_OBJECT_LINKS_RECOMMENDATION,
        source_type="IdeaEvaluation",
        target_type="Recommendation",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
        allowed_source_types=frozenset({"IdeaEvaluation"}),
    ),
    RESEARCH_OBJECT_LINKS_APPROVAL: RelationDefinition(
        name=RESEARCH_OBJECT_LINKS_APPROVAL,
        source_type="IdeaEvaluation",
        target_type="Approval",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
        allowed_source_types=frozenset({"IdeaEvaluation", "ManagementQualityAssessment"}),
    ),
    RESEARCH_OBJECT_LINKS_ACTION_ITEM: RelationDefinition(
        name=RESEARCH_OBJECT_LINKS_ACTION_ITEM,
        source_type="IdeaEvaluation",
        target_type="ActionItem",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
        allowed_source_types=frozenset({"IdeaEvaluation", "OptimizationAlert"}),
    ),
    MANAGEMENT_QUALITY_ASSESSES_ISSUER: RelationDefinition(
        name=MANAGEMENT_QUALITY_ASSESSES_ISSUER,
        source_type="ManagementQualityAssessment",
        target_type="Issuer",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    MANAGEMENT_QUALITY_HAS_SCORECARD_ROW: RelationDefinition(
        name=MANAGEMENT_QUALITY_HAS_SCORECARD_ROW,
        source_type="ManagementQualityAssessment",
        target_type="ManagementQualityScorecardRow",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    MANAGEMENT_QUALITY_HAS_ACCOMPLISHMENT: RelationDefinition(
        name=MANAGEMENT_QUALITY_HAS_ACCOMPLISHMENT,
        source_type="ManagementQualityAssessment",
        target_type="ManagementQualityAccomplishment",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    MANAGEMENT_QUALITY_HAS_SETBACK: RelationDefinition(
        name=MANAGEMENT_QUALITY_HAS_SETBACK,
        source_type="ManagementQualityAssessment",
        target_type="ManagementQualitySetback",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    OPTIMIZATION_MISSION_HAS_RUN: RelationDefinition(
        name=OPTIMIZATION_MISSION_HAS_RUN,
        source_type="OptimizationMission",
        target_type="OptimizationRun",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    OPTIMIZATION_RUN_HAS_SNAPSHOT: RelationDefinition(
        name=OPTIMIZATION_RUN_HAS_SNAPSHOT,
        source_type="OptimizationRun",
        target_type="OptimizationActionSnapshot",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    OPTIMIZATION_ALERT_CURRENT_SNAPSHOT: RelationDefinition(
        name=OPTIMIZATION_ALERT_CURRENT_SNAPSHOT,
        source_type="OptimizationAlert",
        target_type="OptimizationActionSnapshot",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    OPTIMIZATION_ALERT_PREVIOUS_SNAPSHOT: RelationDefinition(
        name=OPTIMIZATION_ALERT_PREVIOUS_SNAPSHOT,
        source_type="OptimizationAlert",
        target_type="OptimizationActionSnapshot",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    OPTIMIZATION_SNAPSHOT_TARGETS_POSITION: RelationDefinition(
        name=OPTIMIZATION_SNAPSHOT_TARGETS_POSITION,
        source_type="OptimizationActionSnapshot",
        target_type="Position",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    OPTIMIZATION_SNAPSHOT_TARGETS_INSTRUMENT: RelationDefinition(
        name=OPTIMIZATION_SNAPSHOT_TARGETS_INSTRUMENT,
        source_type="OptimizationActionSnapshot",
        target_type="Instrument",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    OPTIMIZATION_ALERT_LINKS_APPROVAL: RelationDefinition(
        name=OPTIMIZATION_ALERT_LINKS_APPROVAL,
        source_type="OptimizationAlert",
        target_type="Approval",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    OPTIMIZATION_ALERT_LINKS_ACTION_ITEM: RelationDefinition(
        name=OPTIMIZATION_ALERT_LINKS_ACTION_ITEM,
        source_type="OptimizationAlert",
        target_type="ActionItem",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    OPTIMIZATION_OBJECT_HAS_SOURCE_FRESHNESS: RelationDefinition(
        name=OPTIMIZATION_OBJECT_HAS_SOURCE_FRESHNESS,
        source_type="OptimizationRun",
        target_type="SourceFreshness",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
        allowed_source_types=frozenset({"OptimizationRun", "OptimizationActionSnapshot", "OptimizationAlert"}),
    ),
    COMPUTED_SNAPSHOT_MATERIALIZES_OBJECT_VERSION: RelationDefinition(
        name=COMPUTED_SNAPSHOT_MATERIALIZES_OBJECT_VERSION,
        source_type="ComputedSnapshotRef",
        target_type="ObjectVersionRef",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "object_uid", "version_id"}),
        optional=True,
    ),
    MARKET_REGIME_HAS_FACTOR_SCORE: RelationDefinition(
        name=MARKET_REGIME_HAS_FACTOR_SCORE,
        source_type="MarketRegimeSnapshot",
        target_type="SignalFactorScore",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    MARKET_REGIME_HAS_FORWARD_OUTLOOK: RelationDefinition(
        name=MARKET_REGIME_HAS_FORWARD_OUTLOOK,
        source_type="MarketRegimeSnapshot",
        target_type="ForwardOutlook",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    MARKET_REGIME_HAS_EPISODE: RelationDefinition(
        name=MARKET_REGIME_HAS_EPISODE,
        source_type="MarketRegimeSnapshot",
        target_type="RegimeEpisode",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    FACTOR_SCORE_USES_SOURCE_RECORD: RelationDefinition(
        name=FACTOR_SCORE_USES_SOURCE_RECORD,
        source_type="SignalFactorScore",
        target_type="SourceRecord",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    FACTOR_SCORE_USES_COMPUTED_SNAPSHOT: RelationDefinition(
        name=FACTOR_SCORE_USES_COMPUTED_SNAPSHOT,
        source_type="SignalFactorScore",
        target_type="ComputedSnapshotRef",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    MARKET_REGIME_USES_RISK_SNAPSHOT: RelationDefinition(
        name=MARKET_REGIME_USES_RISK_SNAPSHOT,
        source_type="MarketRegimeSnapshot",
        target_type="PositionRiskSnapshot",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
        allowed_target_types=frozenset({"PositionRiskSnapshot", "PortfolioRiskSnapshot"}),
    ),
    MARKET_REGIME_REFERENCES_MACRO_INDICATOR: RelationDefinition(
        name=MARKET_REGIME_REFERENCES_MACRO_INDICATOR,
        source_type="MarketRegimeSnapshot",
        target_type="MacroIndicator",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    FACTOR_SCORE_REFERENCES_SECTOR: RelationDefinition(
        name=FACTOR_SCORE_REFERENCES_SECTOR,
        source_type="SignalFactorScore",
        target_type="Sector",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_SUPPORTED_BY_EVIDENCE: RelationDefinition(
        name=RECOMMENDATION_SUPPORTED_BY_EVIDENCE,
        source_type="Recommendation",
        target_type="Evidence",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_CONTRADICTED_BY_EVIDENCE: RelationDefinition(
        name=RECOMMENDATION_CONTRADICTED_BY_EVIDENCE,
        source_type="Recommendation",
        target_type="Evidence",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EVIDENCE_CITES_CITATION: RelationDefinition(
        name=EVIDENCE_CITES_CITATION,
        source_type="Evidence",
        target_type="Citation",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_HAS_POLICY_GATE_RESULT: RelationDefinition(
        name=RECOMMENDATION_HAS_POLICY_GATE_RESULT,
        source_type="Recommendation",
        target_type="PolicyGateResult",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_HAS_TRADE_PROPOSAL: RelationDefinition(
        name=RECOMMENDATION_HAS_TRADE_PROPOSAL,
        source_type="Recommendation",
        target_type="TradeProposal",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_USES_POSITION_RISK_SNAPSHOT: RelationDefinition(
        name=RECOMMENDATION_USES_POSITION_RISK_SNAPSHOT,
        source_type="Recommendation",
        target_type="PositionRiskSnapshot",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    RECOMMENDATION_USES_PORTFOLIO_RISK_SNAPSHOT: RelationDefinition(
        name=RECOMMENDATION_USES_PORTFOLIO_RISK_SNAPSHOT,
        source_type="Recommendation",
        target_type="PortfolioRiskSnapshot",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    DOCUMENT_ARTIFACT_MATERIALIZES_RESEARCH_OBJECT: RelationDefinition(
        name=DOCUMENT_ARTIFACT_MATERIALIZES_RESEARCH_OBJECT,
        source_type="DocumentArtifact",
        target_type="EquityOverview",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id", "document_role"}),
        optional=True,
        allowed_target_types=frozenset({"EquityOverview", "ThesisDocument", "ManagementQualityAssessment"}),
    ),
    EQUITY_OVERVIEW_COVERS_ISSUER: RelationDefinition(
        name=EQUITY_OVERVIEW_COVERS_ISSUER,
        source_type="EquityOverview",
        target_type="Issuer",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EQUITY_OVERVIEW_COVERS_INSTRUMENT: RelationDefinition(
        name=EQUITY_OVERVIEW_COVERS_INSTRUMENT,
        source_type="EquityOverview",
        target_type="Instrument",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EQUITY_OVERVIEW_HAS_FINANCIAL_PROFILE: RelationDefinition(
        name=EQUITY_OVERVIEW_HAS_FINANCIAL_PROFILE,
        source_type="EquityOverview",
        target_type="CompanyFinancialProfile",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EQUITY_OVERVIEW_HAS_EXTRINSIC_SENSITIVITY: RelationDefinition(
        name=EQUITY_OVERVIEW_HAS_EXTRINSIC_SENSITIVITY,
        source_type="EquityOverview",
        target_type="ExtrinsicSensitivity",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EQUITY_OVERVIEW_HAS_INDUSTRY_FORCE: RelationDefinition(
        name=EQUITY_OVERVIEW_HAS_INDUSTRY_FORCE,
        source_type="EquityOverview",
        target_type="IndustryForceAssessment",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    EQUITY_OVERVIEW_HAS_SUPPLY_DEMAND_OUTLOOK: RelationDefinition(
        name=EQUITY_OVERVIEW_HAS_SUPPLY_DEMAND_OUTLOOK,
        source_type="EquityOverview",
        target_type="SupplyDemandOutlook",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    THESIS_DOCUMENT_COVERS_ISSUER: RelationDefinition(
        name=THESIS_DOCUMENT_COVERS_ISSUER,
        source_type="ThesisDocument",
        target_type="Issuer",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    THESIS_DOCUMENT_COVERS_INSTRUMENT: RelationDefinition(
        name=THESIS_DOCUMENT_COVERS_INSTRUMENT,
        source_type="ThesisDocument",
        target_type="Instrument",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
    THESIS_DOCUMENT_HAS_SECTION: RelationDefinition(
        name=THESIS_DOCUMENT_HAS_SECTION,
        source_type="ThesisDocument",
        target_type="ThesisSection",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=frozenset({"ontology_run_id"}),
        optional=True,
    ),
}
for _provenance_relation_type in PROVENANCE_RELATION_TYPES:
    RELATION_REGISTRY[_provenance_relation_type] = RelationDefinition(
        name=_provenance_relation_type,
        source_type="ProvenanceEvent",
        target_type="ObjectVersionRef",
        cardinality=RelationCardinality.MANY_TO_MANY,
        required_properties=PROVENANCE_REQUIRED_PROPERTIES,
        optional=True,
        allowed_source_types=PROVENANCE_ENDPOINT_TYPES,
        allowed_target_types=PROVENANCE_ENDPOINT_TYPES,
    )

ALLOWED_RELATIONS: dict[str, tuple[EntityType, EntityType]] = {
    name: (definition.source_type, definition.target_type) for name, definition in RELATION_REGISTRY.items()
}
OPTIONAL_RELATIONS = {name for name, definition in RELATION_REGISTRY.items() if definition.optional}
RELATION_TYPE_SQL_VALUES = ", ".join(f"'{relation_type}'" for relation_type in RELATION_REGISTRY)


class RelationPropertiesV1(OntologySchemaBase):
    ontology_run_id: NonBlankStr
    event_id: str | None = None
    source: str | None = None
    action_id: str | None = None
    object_uid: str | None = None
    object_type: str | None = None
    version_id: str | None = None
    source_record_id: str | None = None
    target_object_type: str | None = None
    target_object_uid: str | None = None
    approval_id: str | None = None
    artifact_key: str | None = None
    document_role: str | None = None
    relation_role: str | None = None
    link_type: str | None = None
    source_ref_type: str | None = None
    source_ref_id: str | None = None
    source_ref_version: str | None = None
    target_ref_type: str | None = None
    target_ref_id: str | None = None
    target_ref_version: str | None = None
    redaction_policy: str | None = None
    retention_class: str | None = None
    lineage_root_id: str | None = None
    metadata: dict[str, Any] | list[Any] | str | int | float | bool | None = None

    @field_validator("ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator(
        "source",
        "action_id",
        "object_uid",
        "object_type",
        "version_id",
        "source_record_id",
        "target_object_type",
        "target_object_uid",
        "approval_id",
        "artifact_key",
        "document_role",
        "relation_role",
        "link_type",
        "source_ref_type",
        "source_ref_id",
        "source_ref_version",
        "target_ref_type",
        "target_ref_id",
        "target_ref_version",
        "event_id",
        "redaction_policy",
        "retention_class",
        "lineage_root_id",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        return clean_optional_text(value)


class PositionSignalExposureV1(OntologySchemaBase):
    component: NonBlankStr
    source: NonBlankStr
    name: NonBlankStr
    value: float | int | str | bool | None = None
    threshold: NonBlankStr
    direction: SignalDirection
    contribution: float = Field(ge=0.0, le=1.0)
    ontology_run_id: NonBlankStr

    @field_validator("component", "source", "name", "threshold", "ontology_run_id", mode="before")
    @classmethod
    def _required_text(cls, value: object) -> str:
        return clean_text(value)

    @field_validator("direction", mode="before")
    @classmethod
    def _direction(cls, value: object) -> str:
        text = str(value or "").strip().lower()
        if text in {"deteriorating", "stable", "improving", "neutral", "unknown"}:
            return text
        return "unknown"


EdgePropertiesV1 = RelationPropertiesV1 | PositionSignalExposureV1


def get_relation_definition(relation_type: str) -> RelationDefinition:
    try:
        return RELATION_REGISTRY[relation_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported relation type: {relation_type}") from exc


def edge_schema_name(relation_type: str) -> str:
    get_relation_definition(relation_type)
    if relation_type == EXPOSED_TO_SIGNAL:
        return "PositionSignalExposure"
    return "Relation"


def edge_schema_for_relation(relation_type: str):
    get_relation_definition(relation_type)
    if relation_type == EXPOSED_TO_SIGNAL:
        return PositionSignalExposureV1
    return RelationPropertiesV1


def dump_edge_properties(model: EdgePropertiesV1) -> dict[str, Any]:
    return model.model_dump(mode="json")
