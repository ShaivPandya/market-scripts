from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EntityType = Literal[
    "Asset",
    "Instrument",
    "Issuer",
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
    "ExecutedDecisionRecord",
    "AuditEvent",
    "Sector",
    "MacroIndicator",
    "Signal",
    "Position",
    "HedgePosition",
    "Thesis",
    "Evaluation",
    "Catalyst",
    "KillCondition",
    "ThesisClaim",
    "Evidence",
    "Citation",
    "ActionItem",
    "WatchTrigger",
    "ResearchNote",
    "Approval",
    "ActionRun",
    "ActionEvent",
    "ProvenanceEvent",
    "ProvenanceLink",
    "WorkflowRun",
    "WorkflowArtifact",
    "Recommendation",
    "ReportRun",
    "DocumentArtifact",
    "InvestmentIdea",
    "IdeaEvaluation",
    "IdeaComparisonRun",
    "OptimizationMission",
    "OptimizationRun",
    "OptimizationActionSnapshot",
    "OptimizationAlert",
]
RelationType = Literal[
    "references_asset",
    "portfolio_holds_position",
    "position_references_instrument",
    "instrument_issued_by_issuer",
    "thesis_covers_instrument",
    "claim_supported_by_evidence",
    "claim_disconfirmed_by_evidence",
    "evidence_has_citation",
    "belongs_to_sector",
    "has_thesis",
    "evaluated_by",
    "has_catalyst",
    "emits_signal",
    "affected_by",
    "exposed_to_signal",
    "position_has_hedge",
    "thesis_has_kill_condition",
    "thesis_has_claim",
    "claim_links_catalyst",
    "claim_links_kill_condition",
    "action_item_targets_object",
    "watch_trigger_targets_object",
    "approval_proposes_action",
    "approval_applies_action_run",
    "action_run_mutates_object_version",
    "workflow_run_produces_artifact",
    "report_run_produces_recommendation",
    "workflow_artifact_proposes_approval",
    "source_record_materializes_object",
    "recommendation_supported_by_source_record",
    "recommendation_uses_risk_metric",
    "recommendation_uses_scenario",
    "trade_proposal_targets_asset",
    "trade_proposal_requires_approval",
    "approval_targets_recommendation",
    "approval_targets_trade_proposal",
    "approval_targets_workflow_artifact",
    "action_run_produces_executed_action",
    "executed_action_mutates_object_version",
    "source_record_materializes_object_version",
    "audit_event_observes_action_run",
    "investor_owns_account",
    "account_has_portfolio",
    "account_governed_by_policy",
    "policy_has_mandate",
    "policy_has_risk_limit",
    "recommendation_targets_account",
    "recommendation_targets_portfolio",
    "recommendation_targets_instrument",
    "trade_proposal_derives_from_recommendation",
    "approval_targets_research_object",
    "executed_decision_applies_approval",
    "executed_decision_records_action_run",
    "policy_gate_evaluates_recommendation",
    "policy_gate_evaluates_trade_proposal",
    "policy_gate_uses_risk_metric",
    "policy_gate_uses_scenario",
    "provenance_event_records_link",
]
ParserSource = Literal["structured", "llm", "deterministic_fallback"]


@dataclass(slots=True)
class OntologyNode:
    id: str
    type: EntityType
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    schema_name: str = "legacy"
    schema_version: int = 0


@dataclass(slots=True)
class OntologyEdge:
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    schema_name: str = "legacy"
    schema_version: int = 0
    relation_schema_name: str = "legacy"
    relation_schema_version: int = 0


@dataclass(slots=True)
class InterpretedQuery:
    intent: str
    source: ParserSource
    filters: dict[str, Any] = field(default_factory=dict)
    entity: str | None = None
    original_query: str | None = None
