"""Provenance writer and redaction helpers.

Ontology-primary runtimes treat provenance as part of the write contract and
fail closed when required events, refs, relations, redaction, or retention data
are invalid.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from api.audit import summarize_for_audit
from api.logging_config import request_id_var
from api.optional_ontology_writes import should_attempt_optional_ontology_write
from ontology.object_service import OntologyObjectService, source_record_object_uid_for
from ontology.schemas.identity import (
    action_item_id,
    action_run_id,
    agent_session_ref_id,
    approval_id,
    audit_event_id,
    catalyst_id,
    computed_snapshot_ref_id,
    document_artifact_id,
    evaluation_id,
    kill_condition_id,
    model_call_ref_id,
    object_version_ref_id,
    ontology_run_ref_id,
    policy_gate_result_id,
    portfolio_id,
    position_id,
    provenance_event_id,
    recommendation_id,
    relation_version_ref_id,
    report_run_id,
    schema_definition_ref_id,
    slug,
    source_record_object_id,
    thesis_claim_id,
    thesis_id,
    tool_call_ref_id,
    watch_trigger_id,
    workflow_artifact_id,
    workflow_run_id,
)
from ontology.schemas.relations import (
    PROVENANCE_APPROVED_EXECUTION,
    PROVENANCE_AUDITED_BY,
    PROVENANCE_EXECUTED,
    PROVENANCE_EXECUTED_AS,
    PROVENANCE_PRODUCED,
    PROVENANCE_PROPOSED,
    PROVENANCE_RESOLVED_BY,
    PROVENANCE_SCHEMA_BOUND,
    PROVENANCE_TRIGGERED,
    PROVENANCE_UPDATED,
    PROVENANCE_USED,
)

logger = logging.getLogger("api.provenance")

EVENT_SOURCE_ADAPTER_RUN = "source_adapter_run"
EVENT_ONTOLOGY_RUN = "ontology_run"
EVENT_AGENT_TURN = "agent_turn"
EVENT_MODEL_CALL = "model_call"
EVENT_TOOL_CALL = "tool_call"
EVENT_WORKFLOW_RUN = "workflow_run"
EVENT_WORKFLOW_ARTIFACT = "workflow_artifact"
EVENT_APPROVAL = "approval"
EVENT_ACTION_RUN = "action_run"

REF_SOURCE_RECORD = "source_record"
REF_SOURCE_ADAPTER_RUN = "source_adapter_run"
REF_ONTOLOGY_RUN = "ontology_run"
REF_ONTOLOGY_OBJECT_VERSION = "ontology_object_version"
REF_RELATION_VERSION = "relation_version"
REF_SCHEMA_DEFINITION = "schema_definition"
REF_AGENT_SESSION = "agent_session"
REF_MODEL_CALL = "model_call"
REF_TOOL_CALL = "tool_call"
REF_WORKFLOW_RUN = "workflow_run"
REF_WORKFLOW_ARTIFACT = "workflow_artifact"
REF_APPROVAL = "approval"
REF_ACTION_RUN = "action_run"
REF_AUDIT_EVENT = "audit_event"
REF_COMPUTED_SNAPSHOT_VERSION = "computed_snapshot_version"
REF_PRODUCER_EVENT = "producer_event"

LINK_USED = "used"
LINK_PRODUCED = "produced"
LINK_SCHEMA_BOUND = "schema_bound"
LINK_EXECUTED = "executed"
LINK_EXECUTED_AS = "executed_as"
LINK_TRIGGERED = "triggered"
LINK_PROPOSED = "proposed"
LINK_RESOLVED_BY = "resolved_by"
LINK_APPROVED_EXECUTION = "approved_execution"
LINK_AUDITED_BY = "audited_by"
LINK_UPDATED = "updated"

LINK_RELATION_TYPES = {
    LINK_USED: PROVENANCE_USED,
    LINK_PRODUCED: PROVENANCE_PRODUCED,
    LINK_SCHEMA_BOUND: PROVENANCE_SCHEMA_BOUND,
    LINK_EXECUTED: PROVENANCE_EXECUTED,
    LINK_EXECUTED_AS: PROVENANCE_EXECUTED_AS,
    LINK_TRIGGERED: PROVENANCE_TRIGGERED,
    LINK_PROPOSED: PROVENANCE_PROPOSED,
    LINK_RESOLVED_BY: PROVENANCE_RESOLVED_BY,
    LINK_APPROVED_EXECUTION: PROVENANCE_APPROVED_EXECUTION,
    LINK_AUDITED_BY: PROVENANCE_AUDITED_BY,
    LINK_UPDATED: PROVENANCE_UPDATED,
}

DEFAULT_REDACTION_POLICY = "audit_summary_v1"
DEFAULT_RETENTION_CLASS = "provenance_365d"
FINANCIAL_RETENTION_CLASS = "financial_lineage_7y"
SOURCE_REF_RETENTION_CLASS = "source_ref_90d"
WORKFLOW_ARTIFACT_RETENTION_CLASS = "workflow_artifact_365d"

_HASH_ONLY_KEY_PARTS = (
    "args",
    "arguments",
    "content",
    "conversation",
    "document",
    "input",
    "instructions",
    "messages",
    "output",
    "prompt",
    "raw",
    "response",
    "result",
    "secret",
    "synthesis",
    "token",
    "transcript",
)
_MAX_SUMMARY_DEPTH = 3
_MAX_SUMMARY_KEYS = 24
_MAX_SUMMARY_ITEMS = 5
_MAX_SUMMARY_STRING = 160


class ProvenanceWriteError(RuntimeError):
    """Raised when a caller requires provenance before proceeding."""


def stable_hash(value: Any, *, length: int = 16) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:
        raw = str(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def deterministic_id(prefix: str, *parts: Any) -> str:
    safe_prefix = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(prefix or "pv")).strip("_") or "pv"
    safe_parts = [
        re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(part)).strip("_")
        for part in parts
        if part is not None and str(part).strip()
    ]
    raw = ":".join([safe_prefix, *safe_parts])
    if len(raw) <= 180:
        return raw
    return f"{safe_prefix}:{stable_hash(parts, length=24)}"


def input_hash(value: Any) -> str | None:
    return stable_hash(value) if value is not None else None


def output_hash(value: Any) -> str | None:
    return stable_hash(value) if value is not None else None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _flatten_object(row: Mapping[str, Any]) -> dict[str, Any]:
    props = dict(row.get("properties") or row.get("properties_json") or {})
    props["id"] = str(row.get("object_uid") or props.get("id") or "")
    props["object_uid"] = props["id"]
    return props


def _flatten_relation(row: Mapping[str, Any]) -> dict[str, Any]:
    props = dict(row.get("properties") or row.get("properties_json") or {})
    props["id"] = str(row.get("relation_uid") or props.get("id") or "")
    props["relation_uid"] = props["id"]
    return {**dict(row), "properties": props}


def _raise_if_required(exc: Exception, message: str, fail_closed: bool) -> None:
    raise ProvenanceWriteError(message) from exc


def _prefixed_or(prefix: str, helper, value: Any) -> str:
    text = str(value or "").strip()
    return text if text.startswith(f"{prefix}:") else helper(text)


def ref_object_uid_for(ref_type: str, ref_id: Any) -> str:
    text = str(ref_id or "").strip()
    if not text:
        raise ProvenanceWriteError(f"Provenance ref {ref_type} requires a non-empty id")
    if ref_type == REF_SOURCE_RECORD:
        return source_record_object_uid_for(text)
    if ref_type == "recommendation":
        return _prefixed_or("recommendation", recommendation_id, text)
    if ref_type == "report_run":
        return _prefixed_or("report_run", report_run_id, text)
    if ref_type == "policy_gate_result":
        return _prefixed_or("policy_gate_result", policy_gate_result_id, text)
    if ref_type == "portfolio_positions":
        return portfolio_id(text)
    if ref_type == "portfolio_position":
        return position_id(text)
    if ref_type == "hedge_positions":
        return portfolio_id(text)
    if ref_type == "hedge_position":
        return position_id(text)
    if ref_type == "thesis":
        return thesis_id(text)
    if ref_type == "thesis_evaluation":
        ticker, _, evaluated_at = text.partition(":")
        return evaluation_id(ticker, evaluated_at or "latest")
    if ref_type == "catalyst":
        return _prefixed_or("catalyst", lambda value: f"catalyst:{slug(value)}", text)
    if ref_type == "kill_condition":
        return _prefixed_or("kill_condition", lambda value: f"kill_condition:{slug(value)}", text)
    if ref_type == "thesis_claim":
        return _prefixed_or("thesis_claim", lambda value: f"thesis_claim:{slug(value)}", text)
    if ref_type == "action_item":
        return _prefixed_or("action_item", action_item_id, text)
    if ref_type == "watch_trigger":
        return _prefixed_or("watch_trigger", watch_trigger_id, text)
    if ref_type in {"news_digest", "management_quality"}:
        return document_artifact_id(ref_type, text)
    if ref_type in {"domain_action", "tool", "prompt_template"}:
        return tool_call_ref_id(f"{ref_type}:{text}")
    if ref_type in {"model"}:
        return model_call_ref_id(f"{ref_type}:{text}")
    if ref_type in {"source", "user", "agent", "workflow"}:
        return source_record_object_id(f"{ref_type}:{text}")
    if ref_type in {REF_SOURCE_ADAPTER_RUN, REF_PRODUCER_EVENT}:
        return provenance_event_id(text)
    if ref_type == REF_ONTOLOGY_RUN:
        return ontology_run_ref_id(text)
    if ref_type == REF_ONTOLOGY_OBJECT_VERSION:
        return object_version_ref_id(text)
    if ref_type == REF_RELATION_VERSION:
        return relation_version_ref_id(text)
    if ref_type == REF_SCHEMA_DEFINITION:
        return schema_definition_ref_id(text)
    if ref_type == REF_AGENT_SESSION:
        return agent_session_ref_id(text)
    if ref_type == REF_MODEL_CALL:
        return model_call_ref_id(text)
    if ref_type == REF_TOOL_CALL:
        return tool_call_ref_id(text)
    if ref_type == REF_WORKFLOW_RUN:
        return _prefixed_or("workflow_run", workflow_run_id, text)
    if ref_type == REF_WORKFLOW_ARTIFACT:
        return _prefixed_or("workflow_artifact", workflow_artifact_id, text)
    if ref_type == REF_APPROVAL:
        return _prefixed_or("approval", approval_id, text)
    if ref_type == REF_ACTION_RUN:
        return _prefixed_or("action_run", action_run_id, text)
    if ref_type == REF_AUDIT_EVENT:
        return _prefixed_or("audit_event", audit_event_id, text)
    if ref_type == REF_COMPUTED_SNAPSHOT_VERSION:
        return computed_snapshot_ref_id(text)
    raise ProvenanceWriteError(f"Unsupported provenance ref type: {ref_type}")


def _ref_object_payload(ref_type: str, ref_id: str, ref_version: str | None) -> tuple[str, str, dict[str, Any]] | None:
    uid = ref_object_uid_for(ref_type, ref_id)
    if ref_type == REF_ONTOLOGY_OBJECT_VERSION:
        return (
            "ObjectVersionRef",
            uid,
            {
                "ref_id": ref_id,
                "object_uid": ref_id,
                "version_id": ref_version or ref_id,
                "temporal_confidence": "referenced",
            },
        )
    if ref_type == REF_RELATION_VERSION:
        return (
            "RelationVersionRef",
            uid,
            {
                "ref_id": ref_id,
                "relation_uid": ref_id,
                "version_id": ref_version or ref_id,
            },
        )
    if ref_type == REF_SCHEMA_DEFINITION:
        try:
            schema_version_value = int(ref_version or 1)
        except (TypeError, ValueError):
            schema_version_value = 1
        return (
            "SchemaDefinitionRef",
            uid,
            {
                "ref_id": ref_id,
                "schema_kind": "unknown",
                "schema_name": ref_id,
                "schema_version_value": schema_version_value,
            },
        )
    if ref_type == REF_ONTOLOGY_RUN:
        return ("OntologyRunRef", uid, {"run_id": ref_id})
    if ref_type == REF_AGENT_SESSION:
        return ("AgentSessionRef", uid, {"session_id": ref_id})
    if ref_type == REF_MODEL_CALL:
        return ("ModelCallRef", uid, {"call_id": ref_id})
    if ref_type == REF_TOOL_CALL:
        return ("ToolCallRef", uid, {"call_id": ref_id})
    if ref_type in {"domain_action", "tool", "prompt_template"}:
        return ("ToolCallRef", uid, {"call_id": f"{ref_type}:{ref_id}", "tool_name": ref_type})
    if ref_type == "model":
        return ("ModelCallRef", uid, {"call_id": f"{ref_type}:{ref_id}", "model": ref_id})
    if ref_type == REF_COMPUTED_SNAPSHOT_VERSION:
        return ("ComputedSnapshotRef", uid, {"snapshot_key": ref_id, "snapshot_id": ref_version})
    return None


def _materialize_ref_object(
    objects: OntologyObjectService,
    *,
    ref_type: str,
    ref_id: str,
    ref_version: str | None,
    valid_from: str,
    provenance_event_id_value: str,
) -> str:
    payload = _ref_object_payload(ref_type, ref_id, ref_version)
    if payload is None:
        return ref_object_uid_for(ref_type, ref_id)
    object_type, uid, props = payload
    objects.write_object(
        object_type,
        uid,
        {**props, "ontology_run_id": "operational"},
        valid_from,
        actor={"actor_type": "system", "actor_id": "provenance"},
        provenance=provenance_event_id_value,
    )
    return uid


def redacted_summary(value: Any) -> Any:
    return _provenance_summary(value)


def _hash_only(value: Any) -> dict[str, Any]:
    if value is None:
        return {"redacted": True, "type": "none"}
    if isinstance(value, str):
        return {"redacted": True, "type": "text", "length": len(value), "sha256": stable_hash(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"redacted": True, "type": "list", "count": len(value), "sha256": stable_hash(value)}
    if isinstance(value, Mapping):
        return {
            "redacted": True,
            "type": "dict",
            "field_names": sorted(str(key) for key in value.keys())[:_MAX_SUMMARY_KEYS],
            "sha256": stable_hash(value),
        }
    return {"redacted": True, "type": type(value).__name__, "sha256": stable_hash(value)}


def _is_hash_only_key(key: Any) -> bool:
    lowered = str(key or "").strip().lower()
    if lowered.endswith("_hash") or lowered.endswith("_fingerprint") or lowered in {"hash", "sha256"}:
        return False
    return any(part in lowered for part in _HASH_ONLY_KEY_PARTS)


def _provenance_summary(value: Any, *, _depth: int = 0, _key: str | None = None) -> Any:
    if _key is not None and _is_hash_only_key(_key):
        return _hash_only(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_SUMMARY_STRING:
            return value
        return {"type": "text", "length": len(value), "sha256": stable_hash(value)}
    if _depth >= _MAX_SUMMARY_DEPTH:
        return _hash_only(value)
    if isinstance(value, Mapping):
        keys = [str(key) for key in value.keys()]
        out: dict[str, Any] = {"field_names": sorted(keys)[:_MAX_SUMMARY_KEYS]}
        for key, item in list(value.items())[:_MAX_SUMMARY_KEYS]:
            key_str = str(key)
            out[key_str] = _provenance_summary(item, _depth=_depth + 1, _key=key_str)
        if len(keys) > _MAX_SUMMARY_KEYS:
            out["truncated_key_count"] = len(keys) - _MAX_SUMMARY_KEYS
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        return {
            "type": "list",
            "count": len(items),
            "items": [_provenance_summary(item, _depth=_depth + 1) for item in items[:_MAX_SUMMARY_ITEMS]],
            "sha256": stable_hash(value),
        }
    return summarize_for_audit(value)


def _shape_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        keys = sorted(str(key) for key in value.keys())
        summary: dict[str, Any] = {"type": "dict", "field_names": keys[:24]}
        for key in ("ticker", "status", "action", "entity_type", "entity_id", "id"):
            item = value.get(key)
            if item is not None and not isinstance(item, (dict, list, tuple)):
                summary[key] = item
        if len(keys) > 24:
            summary["truncated_key_count"] = len(keys) - 24
        return summary
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"type": "list", "count": len(value)}
    return {"type": type(value).__name__}


def _actor_fields(actor: Any) -> tuple[str | None, str | None, str | None]:
    if actor is None:
        return None, None, None
    if isinstance(actor, Mapping):
        return (
            str(actor["actor_type"]) if actor.get("actor_type") is not None else None,
            str(actor["actor_id"]) if actor.get("actor_id") is not None else None,
            str(actor["parent_actor_id"]) if actor.get("parent_actor_id") is not None else None,
        )
    actor_type = getattr(actor, "actor_type", None)
    actor_id = getattr(actor, "actor_id", None)
    parent_actor_id = getattr(actor, "parent_actor_id", None)
    return (
        str(actor_type) if actor_type is not None else None,
        str(actor_id) if actor_id is not None else None,
        str(parent_actor_id) if parent_actor_id is not None else None,
    )


def start_event(
    *,
    event_type: str,
    event_name: str,
    event_id: str | None = None,
    actor: Any = None,
    parent_event_id: str | None = None,
    workflow_run_id: str | None = None,
    ontology_run_id: str | None = None,
    agent_session_id: str | None = None,
    action_run_id: str | int | None = None,
    approval_id: str | int | None = None,
    audit_event_id: str | None = None,
    input_value: Any | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    started_at: str | None = None,
    request_id: str | None = None,
    retention_class: str = DEFAULT_RETENTION_CLASS,
    fail_closed: bool = False,
    schema_version: int = 1,
    criticality: str = "operational",
    lineage_root_id: str | None = None,
    idempotency_key: str | None = None,
    producer_name: str | None = None,
    producer_version: str | None = None,
) -> dict | None:
    if not should_attempt_optional_ontology_write(fail_closed=fail_closed):
        return None

    try:
        actor_type, actor_id, parent_actor_id = _actor_fields(actor)
        started = started_at or _now()
        uid = event_id or deterministic_id("provenance_event", event_type, event_name, idempotency_key, started)
        objects = OntologyObjectService()
        existing = objects.get_object(uid) or objects.get_object(provenance_event_id(uid))
        if existing:
            return _flatten_object(existing)
        row = objects.write_object(
            "ProvenanceEvent",
            uid,
            {
                "event_id": uid,
                "event_type": event_type,
                "event_name": event_name,
                "status": "started",
                "started_at": started,
                "actor_type": actor_type,
                "actor_id": actor_id,
                "parent_actor_id": parent_actor_id,
                "request_id": request_id if request_id is not None else (request_id_var.get("") or None),
                "parent_event_id": parent_event_id,
                "workflow_run_id": workflow_run_id,
                "ontology_run_id": ontology_run_id,
                "agent_session_id": agent_session_id,
                "action_run_id": str(action_run_id) if action_run_id is not None else None,
                "approval_id": str(approval_id) if approval_id is not None else None,
                "audit_event_id": audit_event_id,
                "input_hash": input_hash(input_value),
                "summary": redacted_summary(summary),
                "metadata": redacted_summary(metadata),
                "schema_version": schema_version,
                "criticality": criticality,
                "lineage_root_id": lineage_root_id,
                "idempotency_key": idempotency_key,
                "producer_name": producer_name,
                "producer_version": producer_version,
                "redaction_policy": DEFAULT_REDACTION_POLICY,
                "retention_class": retention_class,
            },
            started,
            actor={"actor_type": actor_type, "actor_id": actor_id},
            provenance=lineage_root_id or uid,
            input_hash=idempotency_key or input_hash(input_value),
        )
        return _flatten_object(row)
    except Exception as exc:
        logger.debug("Failed to start provenance event type=%s name=%s", event_type, event_name, exc_info=True)
        _raise_if_required(exc, f"Failed to write mandatory provenance event {event_type}:{event_name}", fail_closed)
        return None


def finish_event(
    event_id: str | None,
    *,
    status: str,
    output_value: Any | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
    fail_closed: bool = False,
) -> dict | None:
    if not should_attempt_optional_ontology_write(fail_closed=fail_closed):
        return None
    if not event_id:
        raise ProvenanceWriteError("Provenance link requires event_id")
    try:
        completed = _now()
        objects = OntologyObjectService()
        existing = objects.get_object(event_id) or objects.get_object(provenance_event_id(event_id))
        if not existing:
            raise ProvenanceWriteError(f"Cannot finish unknown provenance event {event_id}")
        existing_props = dict(existing.get("properties") or existing.get("properties_json") or {})
        current_status = str(existing_props.get("status") or "")
        if current_status in {"succeeded", "failed"}:
            if current_status == status:
                return _flatten_object(existing)
            raise ProvenanceWriteError(
                f"Provenance event {event_id} is already terminal with status {current_status}; cannot transition to {status}"
            )
        if current_status != "started" or status not in {"succeeded", "failed"}:
            raise ProvenanceWriteError(f"Invalid provenance event transition {event_id}: {current_status}->{status}")
        row = objects.write_object(
            "ProvenanceEvent",
            event_id,
            {
                **existing_props,
                "event_id": event_id,
                "status": status,
                "completed_at": completed,
                "output_hash": output_hash(output_value),
                "summary": redacted_summary(summary),
                "metadata": redacted_summary(metadata),
                "error": error,
            },
            completed,
            provenance=event_id,
            input_hash=output_hash(output_value),
        )
        return _flatten_object(row)
    except ProvenanceWriteError:
        raise
    except Exception as exc:
        logger.debug("Failed to finish provenance event id=%s", event_id, exc_info=True)
        _raise_if_required(exc, f"Failed to finish mandatory provenance event {event_id}", fail_closed)
        return None


@contextmanager
def event_scope(**kwargs: Any) -> Iterator[dict | None]:
    event = start_event(**kwargs)
    started = time.perf_counter()
    try:
        yield event
    except Exception as exc:
        finish_event(
            event.get("id") if event else None,
            status="failed",
            metadata={"duration_ms": round((time.perf_counter() - started) * 1000, 1)},
            error=str(exc) or exc.__class__.__name__,
        )
        raise
    else:
        finish_event(
            event.get("id") if event else None,
            status="succeeded",
            metadata={"duration_ms": round((time.perf_counter() - started) * 1000, 1)},
        )


def link_refs(
    *,
    event_id: str | None,
    source_ref_type: str,
    source_ref_id: str,
    target_ref_type: str,
    target_ref_id: str,
    link_type: str,
    source_ref_version: str | None = None,
    target_ref_version: str | None = None,
    metadata: Any | None = None,
    link_id: str | None = None,
    lineage_root_id: str | None = None,
    fail_closed: bool = False,
) -> dict | None:
    if not should_attempt_optional_ontology_write(fail_closed=fail_closed):
        return None
    if not event_id:
        raise ProvenanceWriteError("Provenance relation requires event_id")
    try:
        relation_type = LINK_RELATION_TYPES[link_type]
        now = _now()
        objects = OntologyObjectService()
        provenance_id = lineage_root_id or event_id
        source_uid = _materialize_ref_object(
            objects,
            ref_type=source_ref_type,
            ref_id=str(source_ref_id),
            ref_version=source_ref_version,
            valid_from=now,
            provenance_event_id_value=provenance_id,
        )
        target_uid = _materialize_ref_object(
            objects,
            ref_type=target_ref_type,
            ref_id=str(target_ref_id),
            ref_version=target_ref_version,
            valid_from=now,
            provenance_event_id_value=provenance_id,
        )
        row = objects.write_relation(
            source_uid,
            target_uid,
            relation_type,
            {
                "event_id": event_id,
                "ontology_run_id": "operational",
                "source_ref_type": source_ref_type,
                "source_ref_id": str(source_ref_id),
                "source_ref_version": source_ref_version,
                "target_ref_type": target_ref_type,
                "target_ref_id": str(target_ref_id),
                "target_ref_version": target_ref_version,
                "redaction_policy": DEFAULT_REDACTION_POLICY,
                "retention_class": FINANCIAL_RETENTION_CLASS if lineage_root_id else DEFAULT_RETENTION_CLASS,
                "lineage_root_id": lineage_root_id,
                "metadata": redacted_summary(metadata),
            },
            now,
            provenance=provenance_id,
        )
        return _flatten_relation(row)
    except KeyError as exc:
        logger.debug("Unsupported provenance link type=%s", link_type, exc_info=True)
        _raise_if_required(exc, f"Unsupported provenance link type: {link_type}", fail_closed)
        return None
    except ProvenanceWriteError:
        raise
    except Exception as exc:
        logger.debug(
            "Failed to link provenance refs event=%s source=%s:%s target=%s:%s",
            event_id,
            source_ref_type,
            source_ref_id,
            target_ref_type,
            target_ref_id,
            exc_info=True,
        )
        _raise_if_required(exc, f"Failed to write mandatory provenance link for event {event_id}", fail_closed)
        return None


def record_source_ref(
    *,
    adapter_run_event_id: str | None,
    source_name: str,
    record_kind: str,
    record_key: Any,
    record_value: Any,
    as_of: str | None = None,
    summary: Any | None = None,
    retention_class: str = SOURCE_REF_RETENTION_CLASS,
    fail_closed: bool = False,
) -> dict | None:
    if not should_attempt_optional_ontology_write(fail_closed=fail_closed):
        return None
    if not adapter_run_event_id:
        raise ProvenanceWriteError("Source record provenance requires adapter_run_event_id")
    record_key_hash = stable_hash(record_key)
    record_hash = stable_hash(record_value)
    record_ref_id = deterministic_id("source_record", source_name, record_kind, record_key_hash)
    try:
        now = _now()
        row = OntologyObjectService().write_object(
            "SourceRecord",
            record_ref_id,
            {
                "source_record_id": record_ref_id,
                "vendor": source_name,
                "source_name": source_name,
                "source_version": "unknown",
                "dataset": source_name,
                "record_kind": record_kind,
                "record_key_hash": record_key_hash,
                "payload_hash": record_hash,
                "status": "ok",
                "quality": "ok",
                "as_of": as_of,
                "load_time": now,
                "provenance_event_id": adapter_run_event_id,
                "metadata": {
                    "summary": redacted_summary(summary),
                    "redaction_policy": DEFAULT_REDACTION_POLICY,
                    "retention_class": retention_class,
                },
            },
            as_of or now,
            provenance=adapter_run_event_id,
            input_hash=record_hash,
        )
        return _flatten_object(row)
    except Exception as exc:
        logger.debug("Failed to record source ref source=%s kind=%s", source_name, record_kind, exc_info=True)
        _raise_if_required(exc, f"Failed to write mandatory source ref {source_name}:{record_kind}", fail_closed)
        return None


def record_workflow_artifact(
    *,
    workflow_run_id: str,
    artifact_key: str,
    artifact_index: int,
    artifact_value: Any,
    approval_id: str | int | None = None,
    provenance_event_id: str | None = None,
    retention_class: str = WORKFLOW_ARTIFACT_RETENTION_CLASS,
    fail_closed: bool = False,
) -> dict | None:
    if not should_attempt_optional_ontology_write(fail_closed=fail_closed):
        return None
    if not provenance_event_id:
        raise ProvenanceWriteError("Workflow artifact provenance requires provenance_event_id")
    artifact_hash = stable_hash(artifact_value)
    artifact_id = deterministic_id("workflow_artifact", workflow_run_id, artifact_key, artifact_index, artifact_hash)
    try:
        now = _now()
        row = OntologyObjectService().write_object(
            "WorkflowArtifact",
            artifact_id,
            {
                "artifact_id": artifact_id,
                "workflow_run_id": workflow_run_id,
                "artifact_key": artifact_key,
                "artifact_index": artifact_index,
                "artifact_value": redacted_summary(_shape_summary(artifact_value)),
                "artifact_hash": artifact_hash,
                "state": "extracted",
                "approval_id": str(approval_id) if approval_id is not None else None,
                "provenance_event_id": provenance_event_id,
                "metadata": {
                    "redaction_policy": DEFAULT_REDACTION_POLICY,
                    "retention_class": retention_class,
                },
            },
            now,
            provenance=provenance_event_id or artifact_id,
            input_hash=artifact_hash,
        )
        return _flatten_object(row)
    except Exception as exc:
        logger.debug("Failed to record workflow artifact run=%s key=%s", workflow_run_id, artifact_key, exc_info=True)
        _raise_if_required(
            exc, f"Failed to write mandatory workflow artifact {workflow_run_id}:{artifact_key}", fail_closed
        )
        return None
