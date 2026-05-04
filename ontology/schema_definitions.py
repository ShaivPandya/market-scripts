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
        AccountV1,
        ActionEventV1,
        ActionItemV1,
        ActionRunV1,
        ApprovalV1,
        AssetV1,
        AuditEventV1,
        CatalystV1,
        DocumentArtifactV1,
        EvaluationV1,
        ExecutedActionV1,
        HedgePositionV1,
        InvestmentPolicyV1,
        InvestorV1,
        KillConditionV1,
        MacroIndicatorV1,
        MandateV1,
        ObjectVersionRefV1,
        PolicyGateResultV1,
        PortfolioV1,
        PositionV1,
        RecommendationV1,
        ReportRunV1,
        ResearchNoteV1,
        RiskLimitV1,
        RiskMetricV1,
        ScenarioV1,
        SectorV1,
        SignalV1,
        SourceRecordV1,
        ThesisClaimV1,
        ThesisV1,
        TradeProposalV1,
        WatchTriggerV1,
        WorkflowArtifactV1,
        WorkflowRunV1,
    )
    from ontology.schemas.relations import RELATION_REGISTRY, PositionSignalExposureV1, RelationPropertiesV1

    object_models: Sequence[tuple[str, type[BaseModel]]] = (
        ("Position", PositionV1),
        ("HedgePosition", HedgePositionV1),
        ("Asset", AssetV1),
        ("Investor", InvestorV1),
        ("Account", AccountV1),
        ("Portfolio", PortfolioV1),
        ("Mandate", MandateV1),
        ("InvestmentPolicy", InvestmentPolicyV1),
        ("RiskLimit", RiskLimitV1),
        ("RiskMetric", RiskMetricV1),
        ("Scenario", ScenarioV1),
        ("PolicyGateResult", PolicyGateResultV1),
        ("TradeProposal", TradeProposalV1),
        ("SourceRecord", SourceRecordV1),
        ("ObjectVersionRef", ObjectVersionRefV1),
        ("ExecutedAction", ExecutedActionV1),
        ("AuditEvent", AuditEventV1),
        ("Sector", SectorV1),
        ("MacroIndicator", MacroIndicatorV1),
        ("Signal", SignalV1),
        ("Thesis", ThesisV1),
        ("Evaluation", EvaluationV1),
        ("Catalyst", CatalystV1),
        ("KillCondition", KillConditionV1),
        ("ThesisClaim", ThesisClaimV1),
        ("ActionItem", ActionItemV1),
        ("WatchTrigger", WatchTriggerV1),
        ("ResearchNote", ResearchNoteV1),
        ("Approval", ApprovalV1),
        ("ActionRun", ActionRunV1),
        ("ActionEvent", ActionEventV1),
        ("WorkflowRun", WorkflowRunV1),
        ("WorkflowArtifact", WorkflowArtifactV1),
        ("Recommendation", RecommendationV1),
        ("ReportRun", ReportRunV1),
        ("DocumentArtifact", DocumentArtifactV1),
    )
    definitions = [
        SchemaDefinition(
            SCHEMA_KIND_ONTOLOGY_OBJECT,
            name,
            1,
            _pydantic_definition(model),
            compatibility={"upgrades_from": [{"schema_name": "legacy", "schema_version": 0}]},
        )
        for name, model in object_models
    ]
    definitions.append(
        SchemaDefinition(
            SCHEMA_KIND_ONTOLOGY_OBJECT,
            "legacy",
            0,
            {"title": "Legacy ontology object", "type": "object", "additionalProperties": True},
            compatibility={"upgrades_to": [{"schema_version": 1}]},
            status="deprecated",
        )
    )

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
                compatibility={"upgrades_from": [{"schema_name": "legacy", "schema_version": 0}]},
            )
        )
    definitions.append(
        SchemaDefinition(
            SCHEMA_KIND_ONTOLOGY_RELATION,
            "legacy",
            0,
            {"title": "Legacy ontology relation", "type": "object", "additionalProperties": True},
            compatibility={"upgrades_to": [{"schema_version": 1}]},
            status="deprecated",
        )
    )

    definitions.extend(
        [
            SchemaDefinition(
                SCHEMA_KIND_ONTOLOGY_EDGE_PROPERTIES,
                "Relation",
                1,
                _pydantic_definition(RelationPropertiesV1),
                compatibility={"upgrades_from": [{"schema_name": "legacy", "schema_version": 0}]},
            ),
            SchemaDefinition(
                SCHEMA_KIND_ONTOLOGY_EDGE_PROPERTIES,
                "PositionSignalExposure",
                1,
                _pydantic_definition(PositionSignalExposureV1),
                compatibility={"upgrades_from": [{"schema_name": "legacy", "schema_version": 0}]},
            ),
            SchemaDefinition(
                SCHEMA_KIND_ONTOLOGY_EDGE_PROPERTIES,
                "legacy",
                0,
                {"title": "Legacy ontology edge properties", "type": "object", "additionalProperties": True},
                compatibility={"upgrades_to": [{"schema_version": 1}]},
                status="deprecated",
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
