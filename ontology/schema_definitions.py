from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

SCHEMA_KIND_ONTOLOGY_OBJECT = "ontology_object"
SCHEMA_KIND_ONTOLOGY_RELATION = "ontology_relation"
SCHEMA_KIND_ONTOLOGY_EDGE_PROPERTIES = "ontology_edge_properties"
SCHEMA_KIND_DOMAIN_ACTION = "domain_action"
SCHEMA_KIND_API_REQUEST = "api_request"

SCHEMA_KINDS = {
    SCHEMA_KIND_ONTOLOGY_OBJECT,
    SCHEMA_KIND_ONTOLOGY_RELATION,
    SCHEMA_KIND_ONTOLOGY_EDGE_PROPERTIES,
    SCHEMA_KIND_DOMAIN_ACTION,
    SCHEMA_KIND_API_REQUEST,
}


@dataclass(frozen=True, slots=True)
class SchemaDefinition:
    schema_kind: str
    schema_name: str
    schema_version: int
    definition: dict[str, Any]
    compatibility: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    deprecated_at: str | None = None

    @property
    def definition_hash(self) -> str:
        raw = json.dumps(self.definition, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def row(self) -> tuple[str, str, int, str, str, str, str, str | None]:
        return (
            self.schema_kind,
            self.schema_name,
            int(self.schema_version),
            json.dumps(self.definition, sort_keys=True, default=str),
            self.definition_hash,
            json.dumps(self.compatibility, sort_keys=True, default=str),
            self.status,
            self.deprecated_at,
        )


def create_schema_registry_tables(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_definitions (
            schema_kind TEXT NOT NULL,
            schema_name TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            definition_json TEXT NOT NULL,
            definition_hash TEXT NOT NULL,
            compatibility_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            deprecated_at TEXT,
            PRIMARY KEY (schema_kind, schema_name, schema_version)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_schema_definitions_kind_status
        ON schema_definitions(schema_kind, status)
        """
    )


def create_ontology_binding_tables(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ontology_run_schema_bindings (
            run_id TEXT NOT NULL,
            schema_kind TEXT NOT NULL,
            schema_name TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            definition_hash TEXT NOT NULL,
            PRIMARY KEY (run_id, schema_kind, schema_name, schema_version),
            FOREIGN KEY (run_id) REFERENCES ontology_runs(run_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ontology_run_schema_bindings_run
        ON ontology_run_schema_bindings(run_id)
        """
    )


def seed_schema_definitions(conn: Any, definitions: Iterable[SchemaDefinition]) -> None:
    rows = [definition.row() for definition in definitions]
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO schema_definitions(
            schema_kind,
            schema_name,
            schema_version,
            definition_json,
            definition_hash,
            compatibility_json,
            status,
            deprecated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(schema_kind, schema_name, schema_version) DO UPDATE SET
            definition_json = excluded.definition_json,
            definition_hash = excluded.definition_hash,
            compatibility_json = excluded.compatibility_json,
            status = excluded.status,
            deprecated_at = excluded.deprecated_at
        """,
        rows,
    )


def ontology_schema_definitions() -> list[SchemaDefinition]:
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
        ModelCallRef,
        ObjectVersionRef,
        Observation,
        OntologyRunRef,
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
        DecisionOutcome,
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
    from ontology.schemas.relations import RELATION_REGISTRY, PositionSignalExposure, RelationProperties

    object_models: Sequence[tuple[str, type[BaseModel]]] = (
        ("Position", Position),
        ("HedgePosition", HedgePosition),
        ("Asset", Asset),
        ("Instrument", Instrument),
        ("Issuer", Issuer),
        ("Investor", Investor),
        ("Account", Account),
        ("Portfolio", Portfolio),
        ("InvestmentPolicy", InvestmentPolicy),
        ("RiskLimit", RiskLimit),
        ("RiskMetric", RiskMetric),
        ("Scenario", Scenario),
        ("PolicyGateResult", PolicyGateResult),
        ("TradeProposal", TradeProposal),
        ("SourceRecord", SourceRecord),
        ("ObjectVersionRef", ObjectVersionRef),
        ("ExecutedAction", ExecutedAction),
        ("ExecutedDecisionRecord", ExecutedDecisionRecord),
        ("AuditEvent", AuditEvent),
        ("Sector", Sector),
        ("MacroIndicator", MacroIndicator),
        ("Signal", Signal),
        ("Thesis", Thesis),
        ("Evaluation", Evaluation),
        ("Catalyst", Catalyst),
        ("KillCondition", KillCondition),
        ("ThesisClaim", ThesisClaim),
        ("Evidence", Evidence),
        ("Citation", Citation),
        ("ActionItem", ActionItem),
        ("WatchTrigger", WatchTrigger),
        ("Approval", Approval),
        ("ActionRun", ActionRun),
        ("ActionEvent", ActionEvent),
        ("ProvenanceEvent", ProvenanceEvent),
        ("RelationVersionRef", RelationVersionRef),
        ("SchemaDefinitionRef", SchemaDefinitionRef),
        ("OntologyRunRef", OntologyRunRef),
        ("AgentSessionRef", AgentSessionRef),
        ("ModelCallRef", ModelCallRef),
        ("ToolCallRef", ToolCallRef),
        ("ComputedSnapshotRef", ComputedSnapshotRef),
        ("MarketRegimeSnapshot", MarketRegimeSnapshot),
        ("SignalFactorScore", SignalFactorScore),
        ("ForwardOutlook", ForwardOutlook),
        ("RegimeEpisode", RegimeEpisode),
        ("PositionRiskSnapshot", PositionRiskSnapshot),
        ("PortfolioRiskSnapshot", PortfolioRiskSnapshot),
        ("WorkflowRun", WorkflowRun),
        ("WorkflowArtifact", WorkflowArtifact),
        ("Recommendation", Recommendation),
        ("CourseOfAction", CourseOfAction),
        ("CourseOfActionComparison", CourseOfActionComparison),
        ("ScenarioAssumption", ScenarioAssumption),
        ("SimulatedOutcome", SimulatedOutcome),
        ("DecisionOutcome", DecisionOutcome),
        ("CourseOfActionRationale", CourseOfActionRationale),
        ("CourseOfActionDissent", CourseOfActionDissent),
        ("ReportRun", ReportRun),
        ("SourceManifest", SourceManifest),
        ("DocumentArtifact", DocumentArtifact),
        ("MediaArtifact", MediaArtifact),
        ("ExtractionRun", ExtractionRun),
        ("Observation", Observation),
        ("Classification", Classification),
        ("PatternDetection", PatternDetection),
        ("AnalystFeedback", AnalystFeedback),
        ("EquityOverview", EquityOverview),
        ("CompanyFinancialProfile", CompanyFinancialProfile),
        ("ExtrinsicSensitivity", ExtrinsicSensitivity),
        ("IndustryForceAssessment", IndustryForceAssessment),
        ("SupplyDemandOutlook", SupplyDemandOutlook),
        ("SupplyChainRelationship", SupplyChainRelationship),
        ("ThesisDocument", ThesisDocument),
        ("ThesisSection", ThesisSection),
        ("InvestmentIdea", InvestmentIdea),
        ("IdeaEvaluation", IdeaEvaluation),
        ("IdeaComparisonRun", IdeaComparisonRun),
        ("IdeaComparisonRanking", IdeaComparisonRanking),
        ("FactorScore", FactorScore),
        ("MissingInformationRequirement", MissingInformationRequirement),
        ("OptimizationMission", OptimizationMission),
        ("OptimizationRun", OptimizationRun),
        ("OptimizationActionSnapshot", OptimizationActionSnapshot),
        ("OptimizationAlert", OptimizationAlert),
        ("SourceFreshness", SourceFreshness),
        ("ManagementQualityAssessment", ManagementQualityAssessment),
        ("ManagementQualityScorecardRow", ManagementQualityScorecardRow),
        ("ManagementQualityAccomplishment", ManagementQualityAccomplishment),
        ("ManagementQualitySetback", ManagementQualitySetback),
    )
    definitions = [
        SchemaDefinition(
            SCHEMA_KIND_ONTOLOGY_OBJECT,
            name,
            1,
            _pydantic_definition(model),
        )
        for name, model in object_models
    ]
    for relation_type, relation in sorted(RELATION_REGISTRY.items()):
        definitions.append(
            SchemaDefinition(
                SCHEMA_KIND_ONTOLOGY_RELATION,
                relation_type,
                1,
                {
                    "name": relation.name,
                    "source_type": relation.source_type,
                    "target_type": relation.target_type,
                    "cardinality": str(relation.cardinality),
                    "required_properties": sorted(relation.required_properties),
                    "optional": bool(relation.optional),
                },
            )
        )
    definitions.extend(
        [
            SchemaDefinition(
                SCHEMA_KIND_ONTOLOGY_EDGE_PROPERTIES,
                "Relation",
                1,
                _pydantic_definition(RelationProperties),
            ),
            SchemaDefinition(
                SCHEMA_KIND_ONTOLOGY_EDGE_PROPERTIES,
                "PositionSignalExposure",
                1,
                _pydantic_definition(PositionSignalExposure),
            ),
        ]
    )
    return definitions


def domain_action_schema_definitions() -> list[SchemaDefinition]:
    from ontology.action_registry import iter_actions

    return [
        SchemaDefinition(
            SCHEMA_KIND_DOMAIN_ACTION,
            action.action_id,
            int(action.schema_version),
            _pydantic_definition(action.input_model),
            compatibility={"handler": action.action_id},
        )
        for action in iter_actions()
    ]


def definition_hash_map(definitions: Iterable[SchemaDefinition]) -> dict[tuple[str, str, int], str]:
    return {
        (definition.schema_kind, definition.schema_name, int(definition.schema_version)): definition.definition_hash
        for definition in definitions
    }


def current_definition_hash(schema_kind: str, schema_name: str, schema_version: int) -> str:
    definitions = [
        *ontology_schema_definitions(),
        *domain_action_schema_definitions(),
    ]
    by_key = definition_hash_map(definitions)
    key = (schema_kind, schema_name, int(schema_version))
    if key in by_key:
        return by_key[key]
    fallback = {
        "schema_kind": schema_kind,
        "schema_name": schema_name,
        "schema_version": int(schema_version),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    return hashlib.sha256(json.dumps(fallback, sort_keys=True).encode("utf-8")).hexdigest()


def _pydantic_definition(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()
