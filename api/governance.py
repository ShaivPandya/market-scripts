"""Mandatory audit/provenance helpers for governed financial lifecycle writes."""

from __future__ import annotations

from typing import Any

from api.audit import summarize_for_audit
from api.provenance import deterministic_id, stable_hash

SCHEMA_VERSION = 1
CRITICAL_FINANCIAL = "financial_critical"
OPERATIONAL = "operational"
REDACTION_POLICY = "audit_summary_v1"
FINANCIAL_RETENTION_CLASS = "financial_lineage_7y"

EVENT_RECOMMENDATION_GENERATED = "recommendation.generated"
EVENT_SOURCE_USED = "source.used"
EVENT_MODEL_CALL_COMPLETED = "model_call.completed"
EVENT_PROMPT_USED = "prompt.used"
EVENT_TOOL_CALL_COMPLETED = "tool_call.completed"
EVENT_POLICY_GATE_EVALUATED = "policy_gate.evaluated"
EVENT_APPROVAL_CREATED = "approval.created"
EVENT_APPROVAL_RESOLVED = "approval.resolved"
EVENT_ACTION_APPLIED = "action.applied"
EVENT_OBJECT_VERSION_CHANGED = "object_version.changed"
EVENT_LINEAGE_MATERIALIZED = "lineage.materialized"

REF_RECOMMENDATION = "recommendation"
REF_REPORT_RUN = "report_run"
REF_SOURCE_RECORD = "source_record"
REF_MODEL_CALL = "model_call"
REF_PROMPT_TEMPLATE = "prompt_template"
REF_TOOL_CALL = "tool_call"
REF_POLICY_GATE_RESULT = "policy_gate_result"
REF_APPROVAL = "approval"
REF_ACTION_RUN = "action_run"
REF_AUDIT_EVENT = "audit_event"
REF_ONTOLOGY_OBJECT_VERSION = "ontology_object_version"
REF_RELATION_VERSION = "relation_version"
REF_WORKFLOW_ARTIFACT = "workflow_artifact"


class GovernanceWriteError(RuntimeError):
    """Raised when mandatory governance state cannot be recorded or queued."""


def lineage_root(ref_type: str, ref_id: Any) -> str:
    return f"{ref_type}:{ref_id}"


def payload_hash(value: Any) -> str | None:
    return stable_hash(value) if value is not None else None


def redacted(value: Any) -> Any:
    return summarize_for_audit(value)


def audit_event(
    *,
    action_name: str,
    status: str,
    lineage_root_id: str,
    object_refs: list[dict[str, Any]] | None = None,
    actor_id: str | None = None,
    actor_type: str = "system",
    before_summary: Any | None = None,
    after_summary: Any | None = None,
    source_lineage: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
    event_id: str | None = None,
    idempotency_key: str | None = None,
    producer_name: str | None = None,
    producer_version: str | None = None,
) -> dict[str, Any]:
    stable_key = idempotency_key or deterministic_id("audit", lineage_root_id, action_name, status)
    return {
        "event_id": event_id or stable_key,
        "action_name": action_name,
        "action_category": "governance",
        "status": status,
        "actor_id": actor_id,
        "actor_type": actor_type,
        "object_refs": object_refs or [],
        "before_summary": redacted(before_summary),
        "after_summary": redacted(after_summary),
        "source_lineage": redacted(source_lineage),
        "metadata": redacted(metadata),
        "error": error,
        "schema_version": SCHEMA_VERSION,
        "criticality": CRITICAL_FINANCIAL,
        "lineage_root_id": lineage_root_id,
        "idempotency_key": stable_key,
        "producer_name": producer_name,
        "producer_version": producer_version,
        "redaction_policy": REDACTION_POLICY,
        "retention_class": FINANCIAL_RETENTION_CLASS,
    }


def provenance_event(
    *,
    event_type: str,
    event_name: str,
    lineage_root_id: str,
    event_id: str | None = None,
    status: str = "succeeded",
    actor_type: str | None = None,
    actor_id: str | None = None,
    parent_event_id: str | None = None,
    workflow_run_id: str | None = None,
    ontology_run_id: str | None = None,
    agent_session_id: str | None = None,
    action_run_id: int | None = None,
    approval_id: int | None = None,
    audit_event_id: str | None = None,
    input_value: Any | None = None,
    output_value: Any | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
    idempotency_key: str | None = None,
    producer_name: str | None = None,
    producer_version: str | None = None,
) -> dict[str, Any]:
    stable_event_id = event_id or deterministic_id("pv", lineage_root_id, event_type, event_name)
    stable_key = idempotency_key or stable_event_id
    return {
        "id": stable_event_id,
        "event_type": event_type,
        "event_name": event_name,
        "status": status,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "parent_event_id": parent_event_id,
        "workflow_run_id": workflow_run_id,
        "ontology_run_id": ontology_run_id,
        "agent_session_id": agent_session_id,
        "action_run_id": action_run_id,
        "approval_id": approval_id,
        "audit_event_id": audit_event_id,
        "input_hash": payload_hash(input_value),
        "output_hash": payload_hash(output_value),
        "summary": redacted(summary),
        "metadata": redacted(metadata),
        "schema_version": SCHEMA_VERSION,
        "criticality": CRITICAL_FINANCIAL,
        "lineage_root_id": lineage_root_id,
        "idempotency_key": stable_key,
        "producer_name": producer_name,
        "producer_version": producer_version,
        "redaction_policy": REDACTION_POLICY,
        "retention_class": FINANCIAL_RETENTION_CLASS,
        "error": error,
    }


def provenance_link(
    *,
    event_id: str,
    source_ref_type: str,
    source_ref_id: Any,
    target_ref_type: str,
    target_ref_id: Any,
    link_type: str,
    lineage_root_id: str,
    source_ref_version: str | None = None,
    target_ref_version: str | None = None,
    metadata: Any | None = None,
    link_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": link_id,
        "event_id": event_id,
        "source_ref_type": source_ref_type,
        "source_ref_id": str(source_ref_id),
        "source_ref_version": source_ref_version,
        "target_ref_type": target_ref_type,
        "target_ref_id": str(target_ref_id),
        "target_ref_version": target_ref_version,
        "link_type": link_type,
        "metadata": redacted(metadata),
        "lineage_root_id": lineage_root_id,
    }


def record_now_tx(conn: Any, event_bundle: dict[str, Any]) -> dict[str, int]:
    try:
        from portfolio import core_db

        return core_db._materialize_governance_bundle_tx(conn, event_bundle)
    except Exception as exc:
        raise GovernanceWriteError("Failed to materialize mandatory governance bundle") from exc


def enqueue_outbox_tx(
    conn: Any,
    event_bundle: dict[str, Any],
    *,
    idempotency_key: str | None = None,
    lineage_root_id: str | None = None,
    retention_class: str = FINANCIAL_RETENTION_CLASS,
) -> dict[str, Any]:
    try:
        from portfolio import core_db

        return core_db._enqueue_governance_outbox_tx(
            conn,
            event_bundle,
            idempotency_key=idempotency_key,
            lineage_root_id=lineage_root_id,
            retention_class=retention_class,
        )
    except Exception as exc:
        raise GovernanceWriteError("Failed to enqueue mandatory governance bundle") from exc
