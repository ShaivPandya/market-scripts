"""Append-only audit event writer and redaction helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from api.logging_config import request_id_var
from ontology.object_service import OntologyObjectService
from ontology.schemas.identity import audit_event_id

logger = logging.getLogger("api.audit")

_SENSITIVE_KEY_PARTS = (
    "args",
    "arguments",
    "authorization",
    "api_key",
    "apikey",
    "cookie",
    "content",
    "conversation",
    "document",
    "input",
    "instructions",
    "jwt",
    "messages",
    "output",
    "password",
    "prompt",
    "raw",
    "response",
    "result",
    "secret",
    "session",
    "synthesis",
    "token",
    "transcript",
)
_MAX_STRING = 160
_MAX_ITEMS = 5
_MAX_KEYS = 24
_MAX_DEPTH = 3


class AuditWriteError(RuntimeError):
    """Raised when a caller requires an audit row before proceeding."""


def _stable_hash(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:
        raw = str(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _ontology_primary_writes_enabled() -> bool:
    try:
        from ontology.domain_write_service import ontology_primary_writes_enabled

        return ontology_primary_writes_enabled()
    except Exception:
        return False


def _is_sensitive_key(key: Any) -> bool:
    lowered = str(key or "").strip().lower()
    if lowered.endswith("_hash") or lowered.endswith("_fingerprint") or lowered in {"hash", "sha256"}:
        return False
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _redacted_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"redacted": True, "type": "none"}
    if isinstance(value, (str, bytes, list, tuple, dict)):
        return {"redacted": True, "type": type(value).__name__, "sha256": _stable_hash(value)}
    return {"redacted": True, "type": type(value).__name__}


def summarize_for_audit(value: Any, *, _depth: int = 0, _key: str | None = None) -> Any:
    """Return a compact, JSON-safe, redacted summary for audit storage."""

    if _key is not None and _is_sensitive_key(_key):
        return _redacted_summary(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_STRING:
            return value
        return {"type": "text", "length": len(value), "sha256": _stable_hash(value)}
    if _depth >= _MAX_DEPTH:
        return {"type": type(value).__name__, "sha256": _stable_hash(value)}
    if isinstance(value, Mapping):
        keys = [str(k) for k in value.keys()]
        out: dict[str, Any] = {"field_names": sorted(keys)[:_MAX_KEYS]}
        for key, item in list(value.items())[:_MAX_KEYS]:
            key_str = str(key)
            out[key_str] = summarize_for_audit(item, _depth=_depth + 1, _key=key_str)
        if len(keys) > _MAX_KEYS:
            out["truncated_key_count"] = len(keys) - _MAX_KEYS
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        return {
            "type": "list",
            "count": len(items),
            "items": [summarize_for_audit(item, _depth=_depth + 1) for item in items[:_MAX_ITEMS]],
            "sha256": _stable_hash(value),
        }
    return str(value)


def _summary_object(value: Any) -> dict[str, Any] | None:
    summarized = summarize_for_audit(value)
    if summarized is None:
        return None
    if isinstance(summarized, dict):
        return summarized
    return {"value": summarized}


def _actor_fields(actor: Any) -> tuple[str | None, str, str | None]:
    if actor is None:
        return None, "system", None
    if isinstance(actor, Mapping):
        return (
            str(actor["actor_id"]) if actor.get("actor_id") is not None else None,
            str(actor.get("actor_type") or "system"),
            str(actor["parent_actor_id"]) if actor.get("parent_actor_id") is not None else None,
        )
    actor_id = getattr(actor, "actor_id", None)
    actor_type = getattr(actor, "actor_type", None)
    parent_actor_id = getattr(actor, "parent_actor_id", None)
    return (
        str(actor_id) if actor_id is not None else None,
        str(actor_type or "system"),
        str(parent_actor_id) if parent_actor_id is not None else None,
    )


def _normalize_object_refs(object_refs: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for ref in object_refs or ():
        if not isinstance(ref, Mapping):
            continue
        ref_type = ref.get("type") or ref.get("object_type")
        ref_id = ref.get("id") or ref.get("object_id")
        item: dict[str, Any] = {}
        if ref_type is not None:
            item["type"] = str(ref_type)
        if ref_id is not None:
            item["id"] = str(ref_id)
        for key in ("role", "name"):
            if ref.get(key) is not None:
                item[key] = str(ref[key])
        if item:
            refs.append(item)
    return refs


def emit_audit_event(
    action_name: str,
    action_category: str,
    status: str,
    *,
    actor: Any = None,
    object_refs: Sequence[Mapping[str, Any]] | None = None,
    before_summary: Any | None = None,
    after_summary: Any | None = None,
    source_lineage: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
    request_id: str | None = None,
    fail_closed: bool = False,
    schema_version: int = 1,
    criticality: str = "operational",
    lineage_root_id: str | None = None,
    idempotency_key: str | None = None,
    producer_name: str | None = None,
    producer_version: str | None = None,
    redaction_policy: str = "audit_summary_v1",
    retention_class: str = "audit_365d",
) -> dict[str, Any] | None:
    """Append one audit event.

    Operational callers keep best-effort behavior. Critical financial paths
    pass ``fail_closed=True`` so missing audit rows stop the business write.
    """

    actor_id, actor_type, parent_actor_id = _actor_fields(actor)
    refs = _normalize_object_refs(object_refs)
    rid = request_id if request_id is not None else request_id_var.get("")
    if not _ontology_primary_writes_enabled():
        try:
            from portfolio import core_db

            return core_db.record_audit_event(
                action_name=action_name,
                action_category=action_category,
                status=status,
                request_id=rid or None,
                actor_id=actor_id,
                actor_type=actor_type,
                parent_actor_id=parent_actor_id,
                object_refs=refs,
                before_summary=summarize_for_audit(before_summary),
                after_summary=summarize_for_audit(after_summary),
                source_lineage=summarize_for_audit(source_lineage),
                metadata=summarize_for_audit(metadata),
                error=error,
                schema_version=schema_version,
                criticality=criticality,
                lineage_root_id=lineage_root_id,
                idempotency_key=idempotency_key,
                producer_name=producer_name,
                producer_version=producer_version,
                redaction_policy=redaction_policy,
                retention_class=retention_class,
            )
        except Exception as exc:
            logger.warning("Failed to write audit event action=%s status=%s", action_name, status, exc_info=True)
            if fail_closed:
                raise AuditWriteError(f"Failed to write mandatory audit event {action_name}:{status}") from exc
            return None

    try:
        now = datetime.now(UTC).isoformat()
        event_key = idempotency_key or f"{action_name}:{status}:{rid}:{_stable_hash(object_refs)}:{now}"
        event_id = audit_event_id(event_key)
        payload: dict[str, Any] = {
            "event_id": event_id,
            "occurred_at": now,
            "actor_id": actor_id,
            "actor_type": actor_type,
            "action_name": action_name,
            "action_category": action_category,
            "status": status,
            "object_refs": refs,
            "before_summary": _summary_object(before_summary),
            "after_summary": _summary_object(after_summary),
            "source_lineage": _summary_object(source_lineage),
            "metadata": {
                "request_id": rid or None,
                "parent_actor_id": parent_actor_id,
                "error": error,
                "schema_version": schema_version,
                "criticality": criticality,
                "idempotency_key": idempotency_key,
                "producer_name": producer_name,
                "producer_version": producer_version,
                "redaction_policy": redaction_policy,
                "summary": _summary_object(metadata),
            },
            "lineage_root_id": lineage_root_id,
            "retention_class": retention_class,
            "ontology_run_id": "operational",
        }
        row = OntologyObjectService().write_object(
            "AuditEvent",
            event_id,
            payload,
            now,
            actor={"actor_type": actor_type, "actor_id": actor_id},
            provenance=lineage_root_id or event_id,
            input_hash=idempotency_key or _stable_hash(payload),
        )
        props = dict(row.get("properties") or row.get("properties_json") or {})
        return {**props, "id": row.get("object_uid") or props.get("event_id")}
    except Exception as exc:
        logger.warning("Failed to write audit event action=%s status=%s", action_name, status, exc_info=True)
        if fail_closed:
            raise AuditWriteError(f"Failed to write mandatory audit event {action_name}:{status}") from exc
        return None
