"""Authoritative ontology object and relation write service."""

from __future__ import annotations

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
    approval_id,
    asset_id,
    audit_event_id,
    canonical_ticker,
    catalyst_id,
    citation_id,
    document_artifact_id,
    evaluation_id,
    evidence_id,
    executed_action_id,
    executed_decision_record_id,
    hedge_position_id,
    instrument_id,
    investment_policy_id,
    investor_id,
    issuer_id,
    kill_condition_id,
    macro_indicator_id,
    mandate_id,
    object_version_ref_id,
    policy_gate_result_id,
    portfolio_id,
    position_id,
    recommendation_id,
    report_run_id,
    research_note_id,
    risk_limit_id,
    risk_metric_id,
    scenario_id,
    sector_id,
    signal_id,
    source_record_object_id,
    thesis_claim_id,
    thesis_id,
    trade_proposal_id,
    watch_trigger_id,
    workflow_artifact_id,
    workflow_run_id,
)
from ontology.schemas.registry import NODE_SCHEMAS, normalize_edge, normalize_node
from ontology.schemas.relations import RELATION_REGISTRY
from ontology.temporal_repository import (
    ObjectVersionWrite,
    RelationVersionWrite,
    TemporalActor,
    TemporalOntologyRepository,
)

_GOVERNED_OBJECT_TYPES = {
    "ActionRun",
    "Approval",
    "AuditEvent",
    "Catalyst",
    "Citation",
    "DocumentArtifact",
    "Evidence",
    "Evaluation",
    "ExecutedAction",
    "ExecutedDecisionRecord",
    "HedgePosition",
    "KillCondition",
    "ObjectVersionRef",
    "PolicyGateResult",
    "Position",
    "Recommendation",
    "ResearchNote",
    "Thesis",
    "ThesisClaim",
    "TradeProposal",
    "WorkflowArtifact",
    "WatchTrigger",
}
_GOVERNED_RELATION_TYPES = {
    "action_run_mutates_object_version",
    "approval_applies_action_run",
    "action_run_produces_executed_action",
    "approval_targets_recommendation",
    "approval_targets_research_object",
    "approval_targets_trade_proposal",
    "approval_targets_workflow_artifact",
    "claim_disconfirmed_by_evidence",
    "claim_supported_by_evidence",
    "evidence_has_citation",
    "executed_action_mutates_object_version",
    "executed_decision_applies_approval",
    "executed_decision_records_action_run",
    "trade_proposal_requires_approval",
}


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
        approval_id: int | None = None,
        source_record_id: str | None = None,
        input_hash: str | None = None,
        temporal_confidence: str = "native",
    ) -> dict[str, Any]:
        actor_fields = _actor_fields(actor)
        provenance_event_id = _provenance_event_id(provenance)
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
        approval_id: int | None = None,
        source_record_id: str | None = None,
        input_hash: str | None = None,
        temporal_confidence: str = "native",
    ) -> dict[str, Any]:
        actor_fields = _actor_fields(actor)
        provenance_event_id = _provenance_event_id(provenance)
        _require_governed_lineage(
            "relation",
            relation_type,
            provenance_event_id=provenance_event_id,
            action_run_id=action_run_id,
            approval_id=approval_id,
            source_record_id=source_record_id,
        )
        normalized = normalize_relation_payload(source_uid, target_uid, relation_type, properties or {})
        row = self.repo.write_relation_version(
            RelationVersionWrite(
                relation_uid=relation_uid_for(source_uid, target_uid, relation_type),
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
        row = self.repo.correct_object_version(
            version_id,
            properties=dict(properties),
            actor=_actor_fields(actor),
            provenance_event_id=_provenance_event_id(provenance),
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
        node = normalize_node(
            OntologyNode(
                id=object_uid,
                type=cast(Any, object_type),
                label=str(props.get("label") or business_key or object_uid),
                properties=props,
                schema_name=object_type,
                schema_version=1,
            ),
            allow_legacy=True,
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
                relation_schema_name=relation_type,
                relation_schema_version=1,
            ),
            allow_legacy=True,
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
        if key.startswith("hedge_position:"):
            return key
        return hedge_position_id(canonical_ticker(props.get("ticker") or key))
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
    if object_type == "Mandate":
        if key.startswith("mandate:"):
            return key
        return mandate_id(props.get("mandate_id") or key)
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
        if key.startswith("source_record:"):
            return key
        return source_record_object_id(props.get("source_record_id") or key)
    if object_type == "ObjectVersionRef":
        if key.startswith("object_version_ref:"):
            return key
        return object_version_ref_id(props.get("ref_id") or key)
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
        return kill_condition_id(ticker, props.get("legacy_id") or props.get("condition") or key)
    if object_type == "ThesisClaim":
        if key.startswith("thesis_claim:"):
            return key
        ticker = canonical_ticker(props.get("ticker") or key.split(":", 1)[0])
        return thesis_claim_id(ticker, props.get("legacy_id") or props.get("claim") or key)
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
        return action_item_id(props.get("legacy_id") or key)
    if object_type == "WatchTrigger":
        if key.startswith("watch_trigger:"):
            return key
        return watch_trigger_id(props.get("legacy_id") or key)
    if object_type == "ResearchNote":
        if key.startswith("research_note:"):
            return key
        return research_note_id(props.get("legacy_id") or key)
    if object_type == "Approval":
        if key.startswith("approval:"):
            return key
        return approval_id(props.get("legacy_id") or key)
    if object_type == "ActionRun":
        if key.startswith("action_run:"):
            return key
        return action_run_id(props.get("legacy_id") or key)
    if object_type == "ActionEvent":
        if key.startswith("action_event:"):
            return key
        return action_event_id(props.get("legacy_id") or key)
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
        return recommendation_id(
            props.get("legacy_id") or props.get("recommendation_id") or props.get("idempotency_key") or key
        )
    if object_type == "ReportRun":
        if key.startswith("report_run:"):
            return key
        return report_run_id(props.get("report_id") or key)
    if object_type == "DocumentArtifact":
        if key.startswith("document_artifact:"):
            return key
        return document_artifact_id(props.get("document_type") or "document", props.get("document_id") or key)
    if ":" in key and key.split(":", 1)[0]:
        return key
    return f"{_slug(object_type)}:{_slug(key)}"


def relation_uid_for(source_uid: str, target_uid: str, relation_type: str) -> str:
    raw = f"{relation_type}:{source_uid}->{target_uid}"
    return raw if len(raw) <= 180 else f"{relation_type}:{_slug(source_uid)}:{_slug(target_uid)}"


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
    approval_id: int | None,
    source_record_id: str | None,
) -> None:
    governed = name in _GOVERNED_OBJECT_TYPES if surface == "object" else name in _GOVERNED_RELATION_TYPES
    if not governed:
        return
    if provenance_event_id or action_run_id is not None or approval_id is not None or source_record_id:
        return
    raise ValueError(f"Governed ontology {surface} write '{name}' requires provenance or action lineage")


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(value or "").strip().lower()).strip("_")
    return text or "unknown"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)
