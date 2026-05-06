"""Mandatory audit/provenance helpers for governed financial lifecycle writes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from api.audit import summarize_for_audit
from api.provenance import deterministic_id, stable_hash
from ontology.object_service import OntologyObjectService

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
REF_SOURCE = "source"
REF_DOMAIN_ACTION = "domain_action"
REF_TOOL = "tool"
REF_MODEL = "model"

LINK_USED = "used"
LINK_PRODUCED = "produced"
LINK_EXECUTED_AS = "executed_as"
LINK_TRIGGERED = "triggered"
LINK_PROPOSED = "proposed"
LINK_GATED = "gated"
LINK_APPROVED_EXECUTION = "approved_execution"
LINK_RESOLVED_BY = "resolved_by"
LINK_APPLIED_BY = "applied_by"
LINK_UPDATED = "updated"
LINK_AUDITED_BY = "audited_by"
LINK_SCHEMA_BOUND = "schema_bound"


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


def event_bundle(
    *,
    lineage_root_id: str,
    provenance_events: list[dict[str, Any]] | None = None,
    audit_events: list[dict[str, Any]] | None = None,
    provenance_links: list[dict[str, Any]] | None = None,
    idempotency_key: str | None = None,
    **updates: Any,
) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "lineage_root_id": lineage_root_id,
        "idempotency_key": idempotency_key or deterministic_id("governance", lineage_root_id),
    }
    if provenance_events:
        bundle["provenance_events"] = provenance_events
    if audit_events:
        bundle["audit_events"] = audit_events
    if provenance_links:
        bundle["provenance_links"] = provenance_links
    for key, value in updates.items():
        if value:
            bundle[key] = value
    return bundle


def lifecycle_event_bundle(
    *,
    event_name: str,
    ref_type: str,
    ref_id: Any,
    status: str = "succeeded",
    event_type: str | None = None,
    actor_type: str | None = None,
    actor_id: str | None = None,
    parent_event_id: str | None = None,
    action_run_id: int | None = None,
    approval_id: int | None = None,
    workflow_run_id: str | None = None,
    input_value: Any | None = None,
    output_value: Any | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    object_refs: list[dict[str, Any]] | None = None,
    source_lineage: Any | None = None,
    error: str | None = None,
    producer_name: str | None = None,
    producer_version: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    root = lineage_root(ref_type, ref_id)
    pv_event_id = deterministic_id("pv", root, event_name, status)
    audit_id = deterministic_id("audit", root, event_name, status)
    refs = object_refs or [{"type": ref_type, "id": ref_id}]
    bundle = event_bundle(
        lineage_root_id=root,
        idempotency_key=idempotency_key or deterministic_id("governance", root, event_name, status),
        provenance_events=[
            provenance_event(
                event_id=pv_event_id,
                event_type=event_type or ref_type,
                event_name=event_name,
                status="failed" if status == "failed" else status,
                actor_type=actor_type,
                actor_id=actor_id,
                parent_event_id=parent_event_id,
                workflow_run_id=workflow_run_id,
                action_run_id=action_run_id,
                approval_id=approval_id,
                input_value=input_value,
                output_value=output_value,
                summary=summary,
                metadata=metadata,
                error=error,
                lineage_root_id=root,
                idempotency_key=deterministic_id("pvkey", root, event_name, status),
                producer_name=producer_name,
                producer_version=producer_version,
            )
        ],
        audit_events=[
            audit_event(
                event_id=audit_id,
                action_name=event_name,
                status=status,
                lineage_root_id=root,
                actor_type=actor_type or "system",
                actor_id=actor_id,
                object_refs=refs,
                after_summary=summary,
                source_lineage=source_lineage,
                metadata=metadata,
                error=error,
                idempotency_key=deterministic_id("auditkey", root, event_name, status),
                producer_name=producer_name,
                producer_version=producer_version,
            )
        ],
    )
    return bundle


def source_used_bundle(
    *,
    target_ref_type: str,
    target_ref_id: Any,
    source_ref_id: Any,
    source_name: str,
    source_version: str | None = None,
    source_record_hash: str | None = None,
    lineage_root_id: str | None = None,
) -> dict[str, Any]:
    root = lineage_root_id or lineage_root(target_ref_type, target_ref_id)
    event_id = deterministic_id("pv:source", root, source_ref_id)
    return event_bundle(
        lineage_root_id=root,
        idempotency_key=deterministic_id("governance:source", root, source_ref_id),
        provenance_events=[
            provenance_event(
                event_id=event_id,
                event_type="source_record",
                event_name=EVENT_SOURCE_USED,
                lineage_root_id=root,
                summary={"source_name": source_name, "source_ref_id": str(source_ref_id)},
                metadata={"source_version": source_version, "source_record_hash": source_record_hash},
            )
        ],
        audit_events=[
            audit_event(
                action_name=EVENT_SOURCE_USED,
                status="succeeded",
                lineage_root_id=root,
                object_refs=[
                    {"type": REF_SOURCE_RECORD, "id": source_ref_id},
                    {"type": target_ref_type, "id": target_ref_id},
                ],
                after_summary={"source_name": source_name, "source_ref_id": str(source_ref_id)},
                metadata={"source_version": source_version, "source_record_hash": source_record_hash},
            )
        ],
        provenance_links=[
            provenance_link(
                event_id=event_id,
                source_ref_type=REF_SOURCE_RECORD,
                source_ref_id=source_ref_id,
                source_ref_version=source_version,
                target_ref_type=target_ref_type,
                target_ref_id=target_ref_id,
                link_type=LINK_USED,
                lineage_root_id=root,
            )
        ],
    )


def model_prompt_tool_bundle(
    *,
    target_ref_type: str,
    target_ref_id: Any,
    model: str | None = None,
    prompt_hash: str | None = None,
    tool_name: str | None = None,
    tool_version: str | None = None,
    run_id: Any | None = None,
    lineage_root_id: str | None = None,
) -> dict[str, Any]:
    root = lineage_root_id or lineage_root(target_ref_type, target_ref_id)
    events: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    refs = [
        (REF_MODEL_CALL, model, EVENT_MODEL_CALL_COMPLETED, model),
        (REF_PROMPT_TEMPLATE, prompt_hash, EVENT_PROMPT_USED, prompt_hash),
        (REF_TOOL_CALL, run_id or tool_name, EVENT_TOOL_CALL_COMPLETED, tool_name),
    ]
    for ref_type, ref_id, event_name, label in refs:
        if not ref_id:
            continue
        event_id = deterministic_id("pv", root, event_name, ref_id)
        events.append(
            provenance_event(
                event_id=event_id,
                event_type=ref_type,
                event_name=event_name,
                lineage_root_id=root,
                summary={"ref_type": ref_type, "ref_id": str(ref_id), "label": label},
                metadata={"tool_version": tool_version},
            )
        )
        audits.append(
            audit_event(
                action_name=event_name,
                status="succeeded",
                lineage_root_id=root,
                object_refs=[{"type": ref_type, "id": ref_id}, {"type": target_ref_type, "id": target_ref_id}],
                after_summary={"ref_type": ref_type, "ref_id": str(ref_id), "label": label},
                metadata={"tool_version": tool_version},
            )
        )
        links.append(
            provenance_link(
                event_id=event_id,
                source_ref_type=ref_type,
                source_ref_id=ref_id,
                source_ref_version=tool_version if ref_type == REF_TOOL_CALL else None,
                target_ref_type=target_ref_type,
                target_ref_id=target_ref_id,
                link_type=LINK_USED,
                lineage_root_id=root,
            )
        )
    return event_bundle(
        lineage_root_id=root,
        idempotency_key=deterministic_id("governance:producer", root, model, prompt_hash, tool_name, run_id),
        provenance_events=events,
        audit_events=audits,
        provenance_links=links,
    )


def object_version_changed_bundle(
    *,
    version_ref_type: str,
    version_ref_id: Any,
    producer_event_id: str | None,
    action_run_id: int | None = None,
    approval_id: int | None = None,
    object_uid: str | None = None,
    metadata: Any | None = None,
) -> dict[str, Any]:
    root = lineage_root(version_ref_type, version_ref_id)
    event_id = producer_event_id or deterministic_id("pv:object_version", version_ref_type, version_ref_id)
    return event_bundle(
        lineage_root_id=root,
        idempotency_key=deterministic_id("governance:object_version", version_ref_type, version_ref_id, event_id),
        provenance_events=[
            provenance_event(
                event_id=event_id,
                event_type="object_version",
                event_name=EVENT_OBJECT_VERSION_CHANGED,
                lineage_root_id=root,
                action_run_id=action_run_id,
                approval_id=approval_id,
                summary={
                    "version_ref_type": version_ref_type,
                    "version_ref_id": str(version_ref_id),
                    "object_uid": object_uid,
                },
                metadata=metadata,
            )
        ],
        audit_events=[
            audit_event(
                action_name=EVENT_OBJECT_VERSION_CHANGED,
                status="succeeded",
                lineage_root_id=root,
                object_refs=[{"type": version_ref_type, "id": version_ref_id}],
                after_summary={"version_ref_id": str(version_ref_id), "object_uid": object_uid},
                metadata=metadata,
            )
        ],
        provenance_links=[
            provenance_link(
                event_id=event_id,
                source_ref_type=REF_ACTION_RUN if action_run_id is not None else "producer_event",
                source_ref_id=action_run_id if action_run_id is not None else event_id,
                target_ref_type=version_ref_type,
                target_ref_id=version_ref_id,
                link_type=LINK_UPDATED if action_run_id is not None else LINK_PRODUCED,
                lineage_root_id=root,
            )
        ],
    )


def record_now_tx(conn: Any, event_bundle: dict[str, Any]) -> dict[str, int]:
    try:
        objects = OntologyObjectService()
        now = datetime.now(UTC).isoformat()
        provenance_count = 0
        audit_count = 0
        link_count = 0
        for event in event_bundle.get("provenance_events") or []:
            event_id = str(event.get("event_id") or deterministic_id("provenance_event", event))
            objects.write_object(
                "ProvenanceEvent",
                event_id,
                {**event, "event_id": event_id},
                now,
                provenance=event.get("lineage_root_id") or event_id,
            )
            provenance_count += 1
        for event in event_bundle.get("audit_events") or []:
            event_id = str(event.get("event_id") or deterministic_id("audit_event", event))
            objects.write_object(
                "AuditEvent",
                event_id,
                {
                    "event_id": event_id,
                    "occurred_at": now,
                    "actor_type": event.get("actor_type") or "system",
                    "actor_id": event.get("actor_id"),
                    "action_name": event.get("action_name") or "governance.event",
                    "action_category": "governance",
                    "status": event.get("status") or "succeeded",
                    "object_refs": event.get("object_refs") or [],
                    "before_summary": event.get("before_summary"),
                    "after_summary": event.get("after_summary"),
                    "source_lineage": event.get("source_lineage"),
                    "metadata": event.get("metadata"),
                    "lineage_root_id": event.get("lineage_root_id"),
                    "retention_class": event.get("retention_class") or FINANCIAL_RETENTION_CLASS,
                    "ontology_run_id": "operational",
                },
                now,
                provenance=event.get("lineage_root_id") or event_id,
            )
            audit_count += 1
        for link in event_bundle.get("provenance_links") or []:
            link_id = str(link.get("link_id") or deterministic_id("provenance_link", link))
            objects.write_object(
                "ProvenanceLink",
                link_id,
                {**link, "link_id": link_id},
                now,
                provenance=link.get("lineage_root_id") or link.get("event_id") or link_id,
            )
            link_count += 1
        return {"provenance_events": provenance_count, "audit_events": audit_count, "provenance_links": link_count}
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
    return {
        "status": "not_queued",
        "lineage_state": "ontology",
        "idempotency_key": idempotency_key or event_bundle.get("idempotency_key"),
        "lineage_root_id": lineage_root_id or event_bundle.get("lineage_root_id"),
        "retention_class": retention_class,
    }
