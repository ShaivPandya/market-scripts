"""Consolidated record evolution timeline for position, thesis, and idea context."""

from __future__ import annotations

from typing import Any, Literal

RecordTimelineContext = Literal["position", "thesis", "idea"]

_KIND_LABELS: dict[str, str] = {
    "conviction_change": "Conviction",
    "thesis_status_change": "Thesis status",
    "lifecycle_event": "Idea lifecycle",
    "evaluation": "Evaluation",
    "idea_evaluation": "Idea evaluation",
    "recommendation_accepted": "Recommendation accepted",
    "approval_applied": "Approval applied",
}


def _clean_text(value: Any, *, max_len: int = 240) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _format_label(kind: str) -> str:
    return _KIND_LABELS.get(kind, kind.replace("_", " ").title())


def _timestamp_sort_value(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        from datetime import datetime

        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _compact_refs(
    *,
    approval_id: Any = None,
    action_run_id: Any = None,
    provenance_event_id: Any = None,
    evaluation_id: Any = None,
    recommendation_id: Any = None,
    source_refs: list[str] | None = None,
    linked_refs: list[str] | None = None,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    if approval_id:
        refs["approval_id"] = str(approval_id)
    if action_run_id:
        refs["action_run_id"] = str(action_run_id)
    if provenance_event_id:
        refs["provenance_event_id"] = str(provenance_event_id)
    if evaluation_id:
        refs["evaluation_id"] = str(evaluation_id)
    if recommendation_id:
        refs["recommendation_id"] = str(recommendation_id)
    if source_refs:
        refs["source_refs"] = list(source_refs)
    if linked_refs:
        refs["linked_refs"] = list(linked_refs)
    if evidence_refs:
        refs["evidence_refs"] = list(evidence_refs)
    return refs


def _timeline_entry(
    *,
    entry_id: str | int,
    kind: str,
    summary: str,
    changed_at: str,
    ticker: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    refs: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "kind": kind,
        "label": _format_label(kind),
        "summary": summary,
        "changed_at": changed_at,
        "ticker": ticker,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "refs": refs or {},
        "payload": payload or {},
    }


def _conviction_summary_text(entry: dict[str, Any]) -> str:
    field = str(entry.get("conviction_field") or "conviction")
    field_label = "group conviction" if field == "group_conviction" else "conviction"
    previous = entry.get("previous_conviction")
    new_value = entry.get("new_conviction")
    if previous is None and new_value is not None:
        return f"{field_label} set to {new_value}"
    if previous is not None and new_value is not None:
        return f"{field_label} {previous} → {new_value}"
    reason = _clean_text(entry.get("reason"))
    return reason or f"{field_label} updated"


def _thesis_status_summary_text(entry: dict[str, Any]) -> str:
    old_status = entry.get("old_status")
    new_status = entry.get("new_status")
    if old_status and new_status:
        return f"{old_status} → {new_status}"
    if new_status:
        return f"Status {new_status}"
    return _clean_text(entry.get("reason")) or "Thesis status updated"


def _lifecycle_summary_text(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type") or "updated").replace("_", " ")
    if event.get("event_type") == "evaluation_accepted":
        action = event.get("after", {}).get("action") if isinstance(event.get("after"), dict) else None
        if action:
            return f"Accepted {action} evaluation"
        return "Accepted evaluation"
    changed_fields = event.get("changed_fields") if isinstance(event.get("changed_fields"), list) else []
    if changed_fields:
        return f"{event_type}: {', '.join(str(field) for field in changed_fields)}"
    reason = _clean_text(event.get("reason"))
    return reason or event_type.title()


def _evaluation_summary_text(entry: dict[str, Any]) -> str:
    action = _clean_text(entry.get("action"))
    confidence = _clean_text(entry.get("confidence"))
    thesis_status = _clean_text(entry.get("thesis_status"))
    parts = [part for part in (action, thesis_status, confidence) if part]
    if parts:
        return " · ".join(parts)
    return "Evaluation recorded"


def _idea_evaluation_summary_text(entry: dict[str, Any]) -> str:
    action = _clean_text(entry.get("action"))
    score = entry.get("score")
    accepted = entry.get("accepted")
    parts: list[str] = []
    if action:
        parts.append(action)
    if score is not None:
        parts.append(f"score {score}")
    if accepted:
        parts.append("accepted")
    return " · ".join(parts) if parts else "Idea evaluation recorded"


def _recommendation_summary_text(entry: dict[str, Any]) -> str:
    action = _clean_text(entry.get("action"))
    report_type = _clean_text(entry.get("report_type"))
    approval_status = _clean_text(entry.get("approval_status"))
    parts = [part for part in (action, report_type, approval_status) if part]
    return " · ".join(parts) if parts else "Recommendation accepted"


def _approval_summary_text(entry: dict[str, Any]) -> str:
    action_id = _clean_text(entry.get("action_id"))
    reason = _clean_text(entry.get("reason"))
    if action_id and reason:
        return f"{action_id}: {reason}"
    return action_id or reason or "Approval applied"


def _entry_from_conviction(entry: dict[str, Any]) -> dict[str, Any]:
    return _timeline_entry(
        entry_id=entry.get("id") or entry.get("entry_id") or f"conviction:{entry.get('changed_at')}",
        kind="conviction_change",
        summary=_conviction_summary_text(entry),
        changed_at=str(entry.get("changed_at") or ""),
        ticker=entry.get("ticker"),
        entity_type=entry.get("entity_type"),
        entity_id=entry.get("entity_id"),
        refs=_compact_refs(
            approval_id=entry.get("approval_id"),
            action_run_id=entry.get("action_run_id"),
            provenance_event_id=entry.get("provenance_event_id"),
            evaluation_id=entry.get("evaluation_id"),
            recommendation_id=entry.get("recommendation_id"),
            source_refs=entry.get("source_refs") or [],
            linked_refs=entry.get("linked_refs") or [],
        ),
        payload={
            "conviction_field": entry.get("conviction_field"),
            "previous_conviction": entry.get("previous_conviction"),
            "new_conviction": entry.get("new_conviction"),
            "reason": entry.get("reason"),
            "ai_confidence": entry.get("ai_confidence"),
        },
    )


def _entry_from_thesis_status(entry: dict[str, Any]) -> dict[str, Any]:
    return _timeline_entry(
        entry_id=entry.get("id") or f"thesis_status:{entry.get('changed_at')}",
        kind="thesis_status_change",
        summary=_thesis_status_summary_text(entry),
        changed_at=str(entry.get("changed_at") or ""),
        ticker=entry.get("ticker"),
        entity_type="thesis",
        entity_id=entry.get("ticker"),
        refs=_compact_refs(
            approval_id=entry.get("approval_id"),
            action_run_id=entry.get("action_run_id"),
            provenance_event_id=entry.get("provenance_event_id"),
        ),
        payload={
            "old_status": entry.get("old_status"),
            "new_status": entry.get("new_status"),
            "reason": entry.get("reason"),
            "actor": entry.get("actor"),
        },
    )


def _entry_from_lifecycle(event: dict[str, Any]) -> dict[str, Any]:
    return _timeline_entry(
        entry_id=event.get("id") or event.get("event_id") or f"lifecycle:{event.get('changed_at')}",
        kind="lifecycle_event",
        summary=_lifecycle_summary_text(event),
        changed_at=str(event.get("changed_at") or ""),
        ticker=event.get("ticker"),
        entity_type="investment_idea",
        entity_id=event.get("idea_id"),
        refs=_compact_refs(
            approval_id=event.get("approval_id") or event.get("action_approval_id"),
            evaluation_id=event.get("evaluation_id"),
            recommendation_id=event.get("recommendation_id"),
        ),
        payload={
            "event_type": event.get("event_type"),
            "changed_fields": event.get("changed_fields") or [],
            "reason": event.get("reason"),
        },
    )


def _entry_from_evaluation(entry: dict[str, Any]) -> dict[str, Any]:
    return _timeline_entry(
        entry_id=entry.get("id") or entry.get("object_uid") or f"evaluation:{entry.get('evaluated_at')}",
        kind="evaluation",
        summary=_evaluation_summary_text(entry),
        changed_at=str(entry.get("evaluated_at") or entry.get("created_at") or ""),
        ticker=entry.get("ticker"),
        entity_type="position",
        entity_id=entry.get("ticker"),
        refs=_compact_refs(),
        payload={
            "action": entry.get("action"),
            "confidence": entry.get("confidence"),
            "thesis_status": entry.get("thesis_status"),
            "risk_flag": entry.get("risk_flag"),
        },
    )


def _entry_from_idea_evaluation(entry: dict[str, Any]) -> dict[str, Any]:
    evidence_refs = [
        str(item.get("source_ref") or item.get("ref") or item.get("id") or "")
        for item in (entry.get("evidence") or [])
        if isinstance(item, dict)
    ]
    evidence_refs = [ref for ref in evidence_refs if ref]
    return _timeline_entry(
        entry_id=entry.get("id") or entry.get("evaluation_id") or f"idea_evaluation:{entry.get('evaluated_at')}",
        kind="idea_evaluation",
        summary=_idea_evaluation_summary_text(entry),
        changed_at=str(entry.get("evaluated_at") or entry.get("accepted_at") or entry.get("created_at") or ""),
        ticker=entry.get("ticker"),
        entity_type="investment_idea",
        entity_id=entry.get("idea_id"),
        refs=_compact_refs(
            approval_id=entry.get("approval_id") or entry.get("action_approval_id"),
            evaluation_id=entry.get("evaluation_id") or entry.get("id"),
            recommendation_id=entry.get("recommendation_id"),
            evidence_refs=evidence_refs or None,
        ),
        payload={
            "action": entry.get("action"),
            "score": entry.get("score"),
            "accepted": entry.get("accepted"),
            "recommendation_status": entry.get("recommendation_status"),
        },
    )


def _entry_from_recommendation(entry: dict[str, Any]) -> dict[str, Any]:
    return _timeline_entry(
        entry_id=entry.get("id") or entry.get("object_uid") or f"recommendation:{entry.get('as_of')}",
        kind="recommendation_accepted",
        summary=_recommendation_summary_text(entry),
        changed_at=str(entry.get("as_of") or entry.get("updated_at") or entry.get("created_at") or ""),
        ticker=entry.get("ticker"),
        entity_type="recommendation",
        entity_id=str(entry.get("id") or entry.get("object_uid") or ""),
        refs=_compact_refs(
            approval_id=entry.get("approval_id"),
            recommendation_id=entry.get("id") or entry.get("object_uid"),
        ),
        payload={
            "action": entry.get("action"),
            "report_type": entry.get("report_type"),
            "approval_status": entry.get("approval_status"),
            "outcome_status": entry.get("outcome_status"),
        },
    )


def _entry_from_approval(entry: dict[str, Any]) -> dict[str, Any]:
    changed_at = str(entry.get("application_completed_at") or entry.get("resolved_at") or entry.get("created_at") or "")
    return _timeline_entry(
        entry_id=entry.get("id") or entry.get("object_uid") or f"approval:{changed_at}",
        kind="approval_applied",
        summary=_approval_summary_text(entry),
        changed_at=changed_at,
        ticker=entry.get("ticker"),
        entity_type=entry.get("entity_type"),
        entity_id=entry.get("entity_id") or entry.get("target_object_uid"),
        refs=_compact_refs(
            approval_id=entry.get("id") or entry.get("object_uid"),
            action_run_id=entry.get("action_run_id"),
            provenance_event_id=entry.get("provenance_event_id"),
        ),
        payload={
            "action_id": entry.get("action_id"),
            "reason": entry.get("reason"),
            "application_status": entry.get("application_status"),
        },
    )


def _dedup_key(entry: dict[str, Any]) -> str:
    refs = entry.get("refs") if isinstance(entry.get("refs"), dict) else {}
    for ref_name in ("approval_id", "action_run_id", "evaluation_id", "recommendation_id"):
        ref_value = refs.get(ref_name)
        if ref_value:
            return f"{ref_name}:{ref_value}:{entry.get('kind')}"
    return f"{entry.get('kind')}:{entry.get('id')}:{entry.get('changed_at')}"


def _deduplicate_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for entry in entries:
        key = _dedup_key(entry)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _sort_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda row: _timestamp_sort_value(str(row.get("changed_at") or "")), reverse=True)


def build_position_record_timeline(
    reads: Any,
    ticker: str,
    *,
    context: RecordTimelineContext = "position",
    limit: int = 30,
    include_approvals: bool = True,
) -> list[dict[str, Any]]:
    normalized = str(ticker or "").strip().upper()
    if not normalized:
        return []

    entries: list[dict[str, Any]] = []
    for row in reads.conviction_history(normalized, entity_type="position", limit=limit):
        entries.append(_entry_from_conviction(row))
    if context in {"position", "thesis"}:
        for row in reads.thesis_status_history(normalized, limit=limit):
            entries.append(_entry_from_thesis_status(row))
        for row in reads.evaluations(normalized, limit=limit):
            entries.append(_entry_from_evaluation(row))
        for row in reads.recommendations(ticker=normalized, approval_status="accepted", limit=limit):
            entries.append(_entry_from_recommendation(row))
        if include_approvals:
            for row in reads.approvals(ticker=normalized, application_status="applied", limit=limit):
                approval_id = str(row.get("id") or row.get("object_uid") or "")
                if any(
                    (entry.get("refs") or {}).get("approval_id") == approval_id
                    for entry in entries
                    if isinstance(entry.get("refs"), dict)
                ):
                    continue
                entries.append(_entry_from_approval(row))

    entries = _deduplicate_entries(entries)
    entries = _sort_entries(entries)
    return entries[: max(1, int(limit))]


def build_idea_record_timeline(
    reads: Any,
    *,
    idea_id: str,
    ticker: str | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    idea_uid = reads.idea_uid(idea_id)
    idea = reads.get(idea_uid) or reads.idea_by_id(idea_id)
    normalized_ticker = str(ticker or (idea or {}).get("ticker") or "").strip().upper()

    entries: list[dict[str, Any]] = []
    for row in reads.idea_lifecycle_events(idea_id, limit=limit):
        entries.append(_entry_from_lifecycle(row))
    if normalized_ticker:
        for row in reads.conviction_history(
            normalized_ticker,
            entity_type="investment_idea",
            entity_id=idea_uid,
            limit=limit,
        ):
            entries.append(_entry_from_conviction(row))
    for row in reads.idea_evaluations(idea_id, limit=limit):
        if row.get("accepted"):
            lifecycle_dup = any(
                entry.get("kind") == "lifecycle_event"
                and (entry.get("refs") or {}).get("evaluation_id")
                == str(row.get("evaluation_id") or row.get("id") or "")
                for entry in entries
            )
            if lifecycle_dup:
                continue
        entries.append(_entry_from_idea_evaluation(row))

    accepted_recommendation_id = (idea or {}).get("accepted_recommendation_id")
    if accepted_recommendation_id:
        recommendation = reads.get(str(accepted_recommendation_id))
        if recommendation:
            entries.append(_entry_from_recommendation(recommendation))

    entries = _deduplicate_entries(entries)
    entries = _sort_entries(entries)
    return entries[: max(1, int(limit))]


def build_record_timeline(
    reads: Any,
    *,
    context: RecordTimelineContext,
    ticker: str | None = None,
    idea_id: str | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    if context == "idea":
        if not idea_id:
            return []
        return build_idea_record_timeline(reads, idea_id=idea_id, ticker=ticker, limit=limit)
    if not ticker:
        return []
    return build_position_record_timeline(reads, ticker, context=context, limit=limit)
