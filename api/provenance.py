"""Best-effort provenance writer and redaction helpers.

The provenance subsystem intentionally mirrors audit semantics: callers should
not fail operational work because provenance storage is temporarily unavailable.
Only hashes and compact redacted summaries are stored here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from api.audit import summarize_for_audit
from api.logging_config import request_id_var

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

DEFAULT_REDACTION_POLICY = "audit_summary_v1"
DEFAULT_RETENTION_CLASS = "provenance_365d"
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
    action_run_id: int | None = None,
    approval_id: int | None = None,
    audit_event_id: str | None = None,
    input_value: Any | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    started_at: str | None = None,
    request_id: str | None = None,
    retention_class: str = DEFAULT_RETENTION_CLASS,
) -> dict | None:
    try:
        from portfolio import core_db

        actor_type, actor_id, parent_actor_id = _actor_fields(actor)
        return core_db.upsert_provenance_event(
            event_id=event_id,
            event_type=event_type,
            event_name=event_name,
            status="started",
            started_at=started_at,
            actor_type=actor_type,
            actor_id=actor_id,
            parent_actor_id=parent_actor_id,
            request_id=request_id if request_id is not None else (request_id_var.get("") or None),
            parent_event_id=parent_event_id,
            workflow_run_id=workflow_run_id,
            ontology_run_id=ontology_run_id,
            agent_session_id=agent_session_id,
            action_run_id=action_run_id,
            approval_id=approval_id,
            audit_event_id=audit_event_id,
            input_hash=input_hash(input_value),
            summary=redacted_summary(summary),
            metadata=redacted_summary(metadata),
            redaction_policy=DEFAULT_REDACTION_POLICY,
            retention_class=retention_class,
        )
    except Exception:
        logger.debug("Failed to start provenance event type=%s name=%s", event_type, event_name, exc_info=True)
        return None


def finish_event(
    event_id: str | None,
    *,
    status: str,
    output_value: Any | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
) -> dict | None:
    if not event_id:
        return None
    try:
        from portfolio import core_db

        return core_db.finish_provenance_event(
            event_id,
            status=status,
            output_hash=output_hash(output_value),
            summary=redacted_summary(summary),
            metadata=redacted_summary(metadata),
            error=error,
        )
    except Exception:
        logger.debug("Failed to finish provenance event id=%s", event_id, exc_info=True)
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
) -> dict | None:
    if not event_id:
        return None
    try:
        from portfolio import core_db

        return core_db.upsert_provenance_link(
            link_id=link_id,
            event_id=event_id,
            source_ref_type=source_ref_type,
            source_ref_id=str(source_ref_id),
            source_ref_version=source_ref_version,
            target_ref_type=target_ref_type,
            target_ref_id=str(target_ref_id),
            target_ref_version=target_ref_version,
            link_type=link_type,
            metadata=redacted_summary(metadata),
        )
    except Exception:
        logger.debug(
            "Failed to link provenance refs event=%s source=%s:%s target=%s:%s",
            event_id,
            source_ref_type,
            source_ref_id,
            target_ref_type,
            target_ref_id,
            exc_info=True,
        )
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
) -> dict | None:
    if not adapter_run_event_id:
        return None
    record_key_hash = stable_hash(record_key)
    record_hash = stable_hash(record_value)
    record_ref_id = deterministic_id("source_record", source_name, record_kind, record_key_hash)
    try:
        from portfolio import core_db

        return core_db.upsert_source_record_ref(
            record_ref_id=record_ref_id,
            adapter_run_event_id=adapter_run_event_id,
            source_name=source_name,
            record_kind=record_kind,
            record_key_hash=record_key_hash,
            record_hash=record_hash,
            as_of=as_of,
            summary=redacted_summary(summary),
            redaction_policy=DEFAULT_REDACTION_POLICY,
            retention_class=retention_class,
        )
    except Exception:
        logger.debug("Failed to record source ref source=%s kind=%s", source_name, record_kind, exc_info=True)
        return None


def record_workflow_artifact(
    *,
    workflow_run_id: str,
    artifact_key: str,
    artifact_index: int,
    artifact_value: Any,
    approval_id: int | None = None,
    provenance_event_id: str | None = None,
    retention_class: str = WORKFLOW_ARTIFACT_RETENTION_CLASS,
) -> dict | None:
    artifact_hash = stable_hash(artifact_value)
    artifact_id = deterministic_id("workflow_artifact", workflow_run_id, artifact_key, artifact_index, artifact_hash)
    try:
        from portfolio import core_db

        return core_db.upsert_workflow_artifact_record(
            artifact_id=artifact_id,
            workflow_run_id=workflow_run_id,
            artifact_key=artifact_key,
            artifact_index=artifact_index,
            artifact_hash=artifact_hash,
            summary=redacted_summary(_shape_summary(artifact_value)),
            approval_id=approval_id,
            provenance_event_id=provenance_event_id,
            redaction_policy=DEFAULT_REDACTION_POLICY,
            retention_class=retention_class,
        )
    except Exception:
        logger.debug("Failed to record workflow artifact run=%s key=%s", workflow_run_id, artifact_key, exc_info=True)
        return None
