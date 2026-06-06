"""First-class conviction history materialization and retrieval."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from ontology.schemas.identity import conviction_history_entry_id

logger = logging.getLogger(__name__)

OPERATIONAL_ONTOLOGY_RUN_ID = "operational"
CONVICTION_FIELDS = ("conviction", "group_conviction")
PORTFOLIO_ACTION_IDS = ("update_portfolio_positions", "update_hedge_positions")


def _object_props(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    props = dict(row.get("properties") or row.get("properties_json") or {})
    object_uid = str(row.get("object_uid") or props.get("id") or "")
    if object_uid:
        props["id"] = object_uid
        props["object_uid"] = object_uid
    meta = row.get("_meta")
    if isinstance(meta, dict):
        props["_meta"] = meta
    return props


def _stable_hash(payload: object) -> str:
    import json

    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize_conviction(value: object) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        level = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= level <= 5:
        return level
    return None


def _version_timestamp(row: dict[str, Any]) -> str:
    for key in ("updated_at", "created_at"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    meta = row.get("_meta")
    temporal = meta.get("temporal") if isinstance(meta, dict) else None
    if isinstance(temporal, dict):
        for key in ("tx_from", "valid_from"):
            value = str(temporal.get(key) or "").strip()
            if value:
                return value
    return ""


def _version_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return (_version_timestamp(row), str(row.get("id") or row.get("object_uid") or ""))


def _timestamp_sort_value(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        from datetime import datetime

        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def conviction_transitions(
    version_rows: list[dict[str, Any]],
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    sorted_versions = sorted(version_rows, key=_version_sort_key)
    transitions: list[dict[str, Any]] = []
    previous: int | None = None
    for version in sorted_versions:
        current = _normalize_conviction(version.get(field_name))
        changed_at = _version_timestamp(version)
        if previous is not None and current != previous:
            transitions.append(
                {
                    "previous_conviction": previous,
                    "new_conviction": current,
                    "changed_at": changed_at,
                }
            )
        if current is not None:
            previous = current
    return transitions


def deterministic_entry_key(
    *,
    entity_type: str,
    entity_id: str,
    conviction_field: str,
    changed_at: str,
    previous_conviction: int | None,
    new_conviction: int | None,
) -> str:
    payload = {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "conviction_field": conviction_field,
        "changed_at": changed_at,
        "previous_conviction": previous_conviction,
        "new_conviction": new_conviction,
    }
    return _stable_hash(payload)


def build_conviction_history_props(
    *,
    entity_type: str,
    entity_id: str,
    ticker: str,
    conviction_field: str,
    previous_conviction: int | None,
    new_conviction: int | None,
    changed_at: str,
    conviction_source_kind: str = "human",
    reason: str | None = None,
    note: str | None = None,
    actor_type: str | None = None,
    actor_id: str | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
    raw_target_weight: float | None = None,
    upgrade_condition: str | None = None,
    downgrade_condition: str | None = None,
    ai_confidence: float | None = None,
    ai_confidence_reason: str | None = None,
    approval_id: str | None = None,
    action_run_id: str | None = None,
    provenance_event_id: str | None = None,
    evaluation_id: str | None = None,
    recommendation_id: str | None = None,
    source_refs: list[str] | None = None,
    linked_refs: list[str] | None = None,
) -> dict[str, Any]:
    normalized_ticker = str(ticker or "").strip().upper()
    entry_key = deterministic_entry_key(
        entity_type=entity_type,
        entity_id=entity_id,
        conviction_field=conviction_field,
        changed_at=changed_at,
        previous_conviction=previous_conviction,
        new_conviction=new_conviction,
    )
    return {
        "entry_id": entry_key,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "ticker": normalized_ticker,
        "conviction_field": conviction_field,
        "previous_conviction": previous_conviction,
        "new_conviction": new_conviction,
        "conviction_scale": 5,
        "conviction_source_kind": conviction_source_kind,
        "reason": reason,
        "note": note,
        "changed_at": changed_at,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "source_type": source_type,
        "source_id": source_id,
        "raw_target_weight": raw_target_weight,
        "upgrade_condition": upgrade_condition,
        "downgrade_condition": downgrade_condition,
        "ai_confidence": ai_confidence,
        "ai_confidence_reason": ai_confidence_reason,
        "approval_id": approval_id,
        "action_run_id": action_run_id,
        "provenance_event_id": provenance_event_id,
        "evaluation_id": evaluation_id,
        "recommendation_id": recommendation_id,
        "source_refs": list(source_refs or []),
        "linked_refs": list(linked_refs or []),
        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
    }


def compact_conviction_history_entry(row: dict[str, Any], *, entry_id: int | str | None = None) -> dict[str, Any]:
    approval_uid = str(row.get("approval_id") or "").strip() or None
    numeric_id = entry_id
    if numeric_id is None and approval_uid and approval_uid.startswith("approval:"):
        suffix = approval_uid.split(":", 1)[1]
        if suffix.isdigit():
            numeric_id = int(suffix)
    if numeric_id is None:
        numeric_id = row.get("entry_id") or row.get("id") or row.get("object_uid")
    return {
        "id": numeric_id,
        "entry_id": row.get("entry_id"),
        "ticker": row.get("ticker"),
        "entity_type": row.get("entity_type"),
        "entity_id": row.get("entity_id"),
        "conviction_field": row.get("conviction_field") or "conviction",
        "previous_conviction": row.get("previous_conviction"),
        "new_conviction": row.get("new_conviction"),
        "conviction_scale": row.get("conviction_scale") or 5,
        "conviction_source_kind": row.get("conviction_source_kind"),
        "reason": row.get("reason"),
        "note": row.get("note"),
        "changed_at": row.get("changed_at"),
        "actor": row.get("actor_id"),
        "actor_type": row.get("actor_type"),
        "source": (
            f"{row.get('source_type')}:{row.get('source_id')}"
            if row.get("source_type") and row.get("source_id")
            else row.get("source_type") or row.get("source_id")
        ),
        "raw_target_weight": row.get("raw_target_weight"),
        "upgrade_condition": row.get("upgrade_condition"),
        "downgrade_condition": row.get("downgrade_condition"),
        "ai_confidence": row.get("ai_confidence"),
        "ai_confidence_reason": row.get("ai_confidence_reason"),
        "approval_id": approval_uid,
        "action_run_id": row.get("action_run_id"),
        "provenance_event_id": row.get("provenance_event_id"),
        "evaluation_id": row.get("evaluation_id"),
        "recommendation_id": row.get("recommendation_id"),
        "source_refs": row.get("source_refs") or [],
        "linked_refs": row.get("linked_refs") or [],
    }


def _portfolio_approvals(object_service: Any, ticker: str) -> list[dict[str, Any]]:
    rows = [
        _object_props(row)
        for row in object_service.query_objects(
            "Approval",
            filters={"ticker": ticker, "application_status": "applied"},
            limit=500,
        )
        if str(_object_props(row).get("action_id") or "") in PORTFOLIO_ACTION_IDS
    ]
    return sorted(rows, key=lambda row: str(row.get("application_completed_at") or row.get("resolved_at") or ""))


def _action_runs_by_approval_id(object_service: Any, approvals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for approval in approvals:
        approval_uid = str(approval.get("id") or approval.get("object_uid") or "").strip()
        if not approval_uid:
            continue
        for row in object_service.query_objects("ActionRun", limit=500):
            props = _object_props(row)
            if str(props.get("approval_id") or "") == approval_uid:
                indexed[approval_uid] = props
                break
    return indexed


def _approval_timestamp(approval: dict[str, Any]) -> str:
    return str(
        approval.get("application_completed_at") or approval.get("resolved_at") or approval.get("created_at") or ""
    ).strip()


def _approval_source(approval: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not approval:
        return None, None
    return (
        str(approval.get("source_type") or "").strip() or None,
        str(approval.get("source_id") or "").strip() or None,
    )


def _position_change_for_ticker(approval: dict[str, Any], ticker: str, field_name: str) -> dict[str, Any] | None:
    proposed = approval.get("proposed_change")
    if not isinstance(proposed, dict):
        return None
    for change in proposed.get("position_changes") or []:
        if not isinstance(change, dict):
            continue
        if str(change.get("ticker") or "").strip().upper() != ticker:
            continue
        for field_change in change.get("field_changes") or []:
            if not isinstance(field_change, dict):
                continue
            if str(field_change.get("field") or "") == field_name:
                return field_change
    return None


def _group_change_for_name(approval: dict[str, Any], group_name: str, field_name: str) -> dict[str, Any] | None:
    proposed = approval.get("proposed_change")
    if not isinstance(proposed, dict):
        return None
    for change in proposed.get("group_changes") or []:
        if not isinstance(change, dict):
            continue
        if str(change.get("group_name") or "").strip() != group_name:
            continue
        for field_change in change.get("field_changes") or []:
            if not isinstance(field_change, dict):
                continue
            if str(field_change.get("field") or "") == field_name:
                return field_change
    return None


def _match_portfolio_approval(
    transition: dict[str, Any],
    approvals: list[dict[str, Any]],
    *,
    ticker: str,
    field_name: str,
    group_name: str | None = None,
    used_approval_ids: set[str],
) -> dict[str, Any] | None:
    target_new = _normalize_conviction(transition.get("new_conviction"))
    transition_at = str(transition.get("changed_at") or "").strip()
    candidates: list[dict[str, Any]] = []
    for approval in approvals:
        approval_uid = str(approval.get("id") or approval.get("object_uid") or "")
        if approval_uid in used_approval_ids:
            continue
        if group_name:
            field_change = _group_change_for_name(approval, group_name, field_name)
        else:
            field_change = _position_change_for_ticker(approval, ticker, field_name)
        if not field_change:
            continue
        if _normalize_conviction(field_change.get("after")) == target_new:
            candidates.append(approval)
    if not candidates:
        return None
    if not transition_at:
        return candidates[-1]
    return min(
        candidates,
        key=lambda approval: abs(
            _timestamp_sort_value(_approval_timestamp(approval)) - _timestamp_sort_value(transition_at)
        ),
    )


def _derive_entity_transitions(
    version_rows: list[dict[str, Any]],
    *,
    entity_type: str,
    entity_id: str,
    ticker: str,
    conviction_source_kind: str,
    group_name: str | None = None,
    approvals: list[dict[str, Any]] | None = None,
    action_runs_by_approval: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    used_approval_ids: set[str] = set()
    for field_name in CONVICTION_FIELDS:
        if field_name == "group_conviction" and not group_name:
            continue
        transitions = conviction_transitions(version_rows, field_name=field_name)
        for transition in transitions:
            approval = None
            action_run = None
            if approvals:
                approval = _match_portfolio_approval(
                    transition,
                    approvals,
                    ticker=ticker,
                    field_name=field_name,
                    group_name=group_name if field_name == "group_conviction" else None,
                    used_approval_ids=used_approval_ids,
                )
                if approval:
                    approval_uid = str(approval.get("id") or approval.get("object_uid") or "")
                    if approval_uid:
                        used_approval_ids.add(approval_uid)
                        action_run = (action_runs_by_approval or {}).get(approval_uid)
            source_type, source_id = _approval_source(approval)
            changed_at = (
                _approval_timestamp(approval)
                if approval and _approval_timestamp(approval)
                else str(transition.get("changed_at") or "")
            )
            resolved_entity_type = entity_type
            resolved_entity_id = entity_id
            if field_name == "group_conviction" and group_name:
                resolved_entity_type = "position_group"
                resolved_entity_id = f"group:{group_name}"
            entries.append(
                build_conviction_history_props(
                    entity_type=resolved_entity_type,
                    entity_id=resolved_entity_id,
                    ticker=ticker,
                    conviction_field=field_name,
                    previous_conviction=transition.get("previous_conviction"),
                    new_conviction=transition.get("new_conviction"),
                    changed_at=changed_at,
                    conviction_source_kind=conviction_source_kind,
                    reason=(str(approval.get("reason") or "").strip() or None if approval else None),
                    actor_type=(str(approval.get("source_type") or "").strip() or None if approval else None),
                    actor_id=(
                        str(approval.get("resolved_by_actor_id") or approval.get("requested_by_actor_id") or "").strip()
                        or None
                        if approval
                        else None
                    ),
                    source_type=source_type,
                    source_id=source_id,
                    approval_id=(
                        str(approval.get("id") or approval.get("object_uid") or "") or None if approval else None
                    ),
                    action_run_id=(
                        str(action_run.get("id") or action_run.get("object_uid") or "") or None if action_run else None
                    ),
                    provenance_event_id=(
                        str(
                            (action_run or {}).get("provenance_event_id")
                            or (approval or {}).get("provenance_event_id")
                            or ""
                        ).strip()
                        or None
                        if approval or action_run
                        else None
                    ),
                )
            )
    return entries


def derive_conviction_history_entries(
    object_service: Any,
    *,
    object_type: str,
    entity_type: str,
    conviction_source_kind: str,
    ticker: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    filters = {"ticker": str(ticker or "").strip().upper()} if ticker else None
    version_rows = [
        _object_props(row)
        for row in object_service.query_objects(object_type, filters=filters, include_history=True, limit=limit)
    ]
    if not version_rows:
        return []

    by_entity: dict[str, list[dict[str, Any]]] = {}
    for row in version_rows:
        entity_id = str(row.get("id") or row.get("object_uid") or row.get("position_id") or row.get("idea_id") or "")
        if not entity_id:
            continue
        by_entity.setdefault(entity_id, []).append(row)

    entries: list[dict[str, Any]] = []
    for entity_id, rows in by_entity.items():
        sample = rows[-1]
        row_ticker = str(sample.get("ticker") or ticker or "").strip().upper()
        if not row_ticker:
            continue
        approvals = (
            _portfolio_approvals(object_service, row_ticker) if entity_type in {"position", "hedge_position"} else []
        )
        action_runs = _action_runs_by_approval_id(object_service, approvals)
        entries.extend(
            _derive_entity_transitions(
                rows,
                entity_type=entity_type,
                entity_id=entity_id,
                ticker=row_ticker,
                conviction_source_kind=conviction_source_kind,
                group_name=str(sample.get("group_name") or "").strip() or None,
                approvals=approvals,
                action_runs_by_approval=action_runs,
            )
        )
    return entries


def materialize_conviction_history_entry(
    object_service: Any,
    props: dict[str, Any],
    *,
    now: str,
    actor: dict[str, Any] | None = None,
    provenance: str | None = None,
    input_hash: str | None = None,
) -> dict[str, Any]:
    entry_key = str(props.get("entry_id") or "")
    if not entry_key:
        raise ValueError("conviction history entry requires entry_id")
    uid = conviction_history_entry_id(entry_key)
    row = object_service.write_object(
        "ConvictionHistoryEntry",
        entry_key,
        props,
        now,
        actor=actor,
        provenance=provenance,
        input_hash=input_hash,
    )
    flattened = _object_props(row)
    flattened["id"] = flattened.get("id") or uid
    return flattened


def record_conviction_change(
    object_service: Any,
    *,
    entity_type: str,
    entity_id: str,
    ticker: str,
    conviction_field: str,
    previous_conviction: int | None,
    new_conviction: int | None,
    changed_at: str,
    conviction_source_kind: str,
    reason: str | None = None,
    note: str | None = None,
    actor: dict[str, Any] | None = None,
    actor_type: str | None = None,
    actor_id: str | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
    raw_target_weight: float | None = None,
    upgrade_condition: str | None = None,
    downgrade_condition: str | None = None,
    ai_confidence: float | None = None,
    ai_confidence_reason: str | None = None,
    approval_id: str | None = None,
    action_run_id: str | None = None,
    provenance_event_id: str | None = None,
    evaluation_id: str | None = None,
    recommendation_id: str | None = None,
    provenance: str | None = None,
    input_hash: str | None = None,
) -> dict[str, Any] | None:
    if previous_conviction == new_conviction:
        return None
    props = build_conviction_history_props(
        entity_type=entity_type,
        entity_id=entity_id,
        ticker=ticker,
        conviction_field=conviction_field,
        previous_conviction=previous_conviction,
        new_conviction=new_conviction,
        changed_at=changed_at,
        conviction_source_kind=conviction_source_kind,
        reason=reason,
        note=note,
        actor_type=actor_type,
        actor_id=actor_id,
        source_type=source_type,
        source_id=source_id,
        raw_target_weight=raw_target_weight,
        upgrade_condition=upgrade_condition,
        downgrade_condition=downgrade_condition,
        ai_confidence=ai_confidence,
        ai_confidence_reason=ai_confidence_reason,
        approval_id=approval_id,
        action_run_id=action_run_id,
        provenance_event_id=provenance_event_id,
        evaluation_id=evaluation_id,
        recommendation_id=recommendation_id,
    )
    return materialize_conviction_history_entry(
        object_service,
        props,
        now=changed_at,
        actor=actor,
        provenance=provenance,
        input_hash=input_hash,
    )


def backfill_conviction_history(
    object_service: Any,
    *,
    ticker: str | None = None,
    now: str,
    actor: dict[str, Any] | None = None,
) -> int:
    """Materialize conviction history entries from temporal versions. Returns count written."""
    specs = (
        ("Position", "position", "backfill"),
        ("HedgePosition", "hedge_position", "backfill"),
        ("InvestmentIdea", "investment_idea", "backfill"),
    )
    derived: list[dict[str, Any]] = []
    for object_type, entity_type, source_kind in specs:
        derived.extend(
            derive_conviction_history_entries(
                object_service,
                object_type=object_type,
                entity_type=entity_type,
                conviction_source_kind=source_kind,
                ticker=ticker,
            )
        )

    written = 0
    for props in derived:
        entry_key = str(props.get("entry_id") or "")
        if not entry_key:
            continue
        existing = object_service.get_object(conviction_history_entry_id(entry_key))
        if existing:
            continue
        materialize_conviction_history_entry(object_service, props, now=now, actor=actor)
        written += 1
    return written


def record_position_conviction_changes(
    object_service: Any,
    *,
    before_row: dict[str, Any] | None,
    after_row: dict[str, Any],
    entity_type: str,
    entity_id: str,
    changed_at: str,
    conviction_source_kind: str,
    actor: dict[str, Any] | None = None,
    actor_type: str | None = None,
    actor_id: str | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
    approval_id: str | None = None,
    action_run_id: str | None = None,
    provenance_event_id: str | None = None,
    provenance: str | None = None,
    input_hash: str | None = None,
) -> list[dict[str, Any]]:
    ticker = str(after_row.get("ticker") or before_row.get("ticker") if before_row else "").strip().upper()
    if not ticker:
        return []
    refs: list[dict[str, Any]] = []
    group_name = str(after_row.get("group_name") or (before_row or {}).get("group_name") or "").strip() or None
    for field_name in CONVICTION_FIELDS:
        if field_name == "group_conviction" and not group_name:
            continue
        previous = _normalize_conviction((before_row or {}).get(field_name))
        current = _normalize_conviction(after_row.get(field_name))
        if previous == current:
            continue
        resolved_entity_type = entity_type
        resolved_entity_id = entity_id
        if field_name == "group_conviction" and group_name:
            resolved_entity_type = "position_group"
            resolved_entity_id = f"group:{group_name}"
        row = record_conviction_change(
            object_service,
            entity_type=resolved_entity_type,
            entity_id=resolved_entity_id,
            ticker=ticker,
            conviction_field=field_name,
            previous_conviction=previous,
            new_conviction=current,
            changed_at=changed_at,
            conviction_source_kind=conviction_source_kind,
            actor=actor,
            actor_type=actor_type,
            actor_id=actor_id,
            source_type=source_type,
            source_id=source_id,
            approval_id=approval_id,
            action_run_id=action_run_id,
            provenance_event_id=provenance_event_id,
            provenance=provenance,
            input_hash=input_hash,
        )
        if row:
            refs.append(row)
    return refs
