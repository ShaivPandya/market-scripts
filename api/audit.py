"""Append-only audit event writer and redaction helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from api.logging_config import request_id_var

logger = logging.getLogger("api.audit")

_SENSITIVE_KEY_PARTS = (
    "authorization",
    "api_key",
    "apikey",
    "cookie",
    "content",
    "document",
    "jwt",
    "password",
    "prompt",
    "raw",
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


def _stable_hash(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:
        raw = str(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _is_sensitive_key(key: Any) -> bool:
    lowered = str(key or "").strip().lower()
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
) -> dict[str, Any] | None:
    """Best-effort append-only audit writer.

    Audit write failures are logged and swallowed so operational paths do not
    fail because the audit subsystem is temporarily unavailable.
    """

    actor_id, actor_type, parent_actor_id = _actor_fields(actor)
    refs = _normalize_object_refs(object_refs)
    rid = request_id if request_id is not None else request_id_var.get("")
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
        )
    except Exception:
        logger.warning("Failed to write audit event action=%s status=%s", action_name, status, exc_info=True)
        return None
