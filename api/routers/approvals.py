"""Approval queue API endpoints -- the gatekeeper for all agent/workflow writes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from api.action_execution import execute_api_action
from api.decision_state import normalize_approval
from api.exceptions import AppError, ConflictError, NotFoundError, ValidationError
from api.routers.auth import ActorDep
from ontology.command_service import (
    OntologyCommandConflict,
    OntologyCommandContext,
    OntologyCommandNotFound,
    OntologyCommandService,
    OntologyCommandValidationError,
)

router = APIRouter()


class ResolveRequest(BaseModel):
    note: str | None = None
    requirement_id: str | None = None


class BulkResolveRequest(BaseModel):
    ids: list[str]
    note: str | None = None


class ReplaceWatchTriggerApprovalRequest(BaseModel):
    condition: str
    trigger_type: str = "custom"
    ticker: str | None = None
    expires_at: str | None = None
    definition: dict[str, Any] | None = None
    reason: str | None = None
    note: str | None = None


def _approval_uid(approval_id: str) -> str:
    text = str(approval_id).strip()
    return text if text.startswith("approval:") else f"approval:{text}"


def _list_approval_records(
    *,
    actor,
    status: str | None,
    ticker: str | None,
    application_status: str | None,
) -> list[dict[str, Any]]:
    return OntologyCommandService().list_approvals(
        status=status,
        ticker=ticker,
        application_status=application_status,
        actor=actor,
    )


def _get_approval_record(approval_id: str, *, actor) -> dict[str, Any]:
    try:
        return OntologyCommandService().get_approval(approval_id, actor=actor)
    except OntologyCommandNotFound as exc:
        raise NotFoundError("Approval", str(approval_id)) from exc


@router.get("/approvals")
def list_approvals(
    actor: ActorDep,
    status: str | None = "pending",
    ticker: str | None = None,
    application_status: str | None = None,
):
    if status == "all":
        status = None
    if application_status == "all":
        application_status = None
    try:
        approvals = _list_approval_records(
            actor=actor,
            status=status,
            ticker=ticker,
            application_status=application_status,
        )
    except ValueError as e:
        raise ValidationError(str(e)) from e
    return {"approvals": [normalize_approval(a) for a in approvals], "count": len(approvals)}


@router.get("/approvals/summary")
def approval_summary(
    actor: ActorDep,
    status: str | None = "pending",
    ticker: str | None = None,
    application_status: str | None = None,
    limit: int = Query(default=5, ge=1, le=50),
):
    normalized_status = None if status == "all" else status
    normalized_application_status = None if application_status == "all" else application_status
    normalized_ticker = ticker.strip().upper() if ticker and ticker.strip() else None
    try:
        approvals = _list_approval_records(
            actor=actor,
            status=normalized_status,
            ticker=normalized_ticker,
            application_status=normalized_application_status,
        )
    except ValueError as e:
        raise ValidationError(str(e)) from e
    normalized: list[dict[str, Any]] = []
    for approval in approvals:
        normalized_approval = normalize_approval(approval)
        if normalized_approval is not None:
            normalized.append(normalized_approval)

    recommendation_approval_count = 0
    for approval in normalized:
        proposed_change = approval.get("proposed_change")
        if isinstance(proposed_change, dict) and proposed_change.get("recommendation_id") is not None:
            recommendation_approval_count += 1
    items = normalized[:limit]
    return {
        "count": len(normalized),
        "items": items,
        "recommendation_approval_count": recommendation_approval_count,
        "has_more": len(normalized) > len(items),
        "status": normalized_status,
        "ticker": normalized_ticker,
        "application_status": normalized_application_status,
        "limit": limit,
    }


@router.get("/approvals/{approval_id}")
def get_approval(approval_id: str, actor: ActorDep):
    approval = _get_approval_record(approval_id, actor=actor)
    approval["provenance_summary"] = {"selector": {"approval_id": approval.get("id")}, "lineage_state": "ontology"}
    return normalize_approval(approval)


@router.post("/approvals/{approval_id}/approve")
def approve_item(approval_id: str, actor: ActorDep, body: ResolveRequest | None = None):
    note = body.note if body else None
    if not str(note or "").strip():
        raise ValidationError("Approval note is required.")
    return execute_api_action(
        "resolve_approval",
        {
            "approval_id": approval_id,
            "status": "approved",
            "note": note,
            "requirement_id": body.requirement_id if body else None,
        },
        source_id="approvals.approve_item",
        actor=actor,
    )


@router.post("/approvals/{approval_id}/reject")
def reject_item(approval_id: str, actor: ActorDep, body: ResolveRequest | None = None):
    return execute_api_action(
        "resolve_approval",
        {
            "approval_id": approval_id,
            "status": "rejected",
            "note": body.note if body else None,
            "requirement_id": body.requirement_id if body else None,
        },
        source_id="approvals.reject_item",
        actor=actor,
    )


@router.post("/approvals/{approval_id}/reject-and-restage")
def reject_and_restage_item(approval_id: str, actor: ActorDep, body: ResolveRequest | None = None):
    service = OntologyCommandService()
    context = OntologyCommandContext(
        actor=actor, source_type="user", source_id=f"approvals.reject_and_restage:{approval_id}"
    )
    try:
        approval = service.get_approval(approval_id, actor=actor)
    except OntologyCommandNotFound as exc:
        raise NotFoundError("Approval", str(approval_id)) from exc
    normalized = normalize_approval(approval)
    if normalized is None:
        raise NotFoundError("Approval", str(approval_id))
    if str(normalized.get("status") or "pending") != "pending":
        raise ConflictError(f"Approval {approval_id} is already {normalized.get('status')}")
    if normalized.get("base_state_status") != "stale":
        raise ConflictError("Approval is not stale and cannot be restaged.")

    action_id = str(approval.get("action_id") or "").strip()
    proposed_change = approval.get("proposed_change")
    if not action_id or not isinstance(proposed_change, dict):
        raise ValidationError("Approval cannot be restaged because it is not an action-backed proposal.")

    try:
        replacement = service.propose_action(
            action_id,
            proposed_change,
            context,
            reason=approval.get("reason") or f"Replacement for stale approval #{approval_id}",
            entity_id=approval.get("entity_id"),
            supersedes_approval_id=str(approval_id),
        )
    except OntologyCommandValidationError as exc:
        raise ValidationError(exc.message) from exc
    except OntologyCommandNotFound as exc:
        raise NotFoundError(exc.resource, exc.identifier) from exc
    except OntologyCommandConflict as exc:
        raise ConflictError(exc.message) from exc

    note = str((body.note if body else "") or "").strip()
    if not note:
        note = f"Superseded by approval {replacement['id']} after underlying state changed."
    original = execute_api_action(
        "resolve_approval",
        {
            "approval_id": approval_id,
            "status": "rejected",
            "note": note,
            "requirement_id": body.requirement_id if body else None,
        },
        source_id="approvals.reject_and_restage",
        actor=actor,
    )
    return {
        "status": "replacement_created",
        "original": normalize_approval(original),
        "replacement": normalize_approval(replacement),
    }


@router.post("/approvals/{approval_id}/replace")
def replace_item(approval_id: str, body: ReplaceWatchTriggerApprovalRequest, actor: ActorDep):
    service = OntologyCommandService()
    context = OntologyCommandContext(actor=actor, source_type="user", source_id=f"approvals.replace:{approval_id}")
    try:
        approval = service.get_approval(approval_id, actor=actor)
    except OntologyCommandNotFound as exc:
        raise NotFoundError("Approval", str(approval_id)) from exc
    normalized = normalize_approval(approval)
    if normalized is None:
        raise NotFoundError("Approval", str(approval_id))
    if str(normalized.get("status") or "pending") != "pending":
        raise ConflictError(f"Approval {approval_id} is already {normalized.get('status')}")

    action_id = str(approval.get("action_id") or "").strip()
    if action_id != "create_watch_trigger":
        raise ValidationError("Only pending watch trigger creation approvals can be edited.")

    replacement_payload = body.model_dump(exclude={"reason", "note"}, exclude_none=True)
    try:
        replacement = service.propose_action(
            "create_watch_trigger",
            replacement_payload,
            context,
            reason=body.reason or approval.get("reason") or f"Edited replacement for approval {approval_id}",
            supersedes_approval_id=str(approval_id),
        )
    except OntologyCommandValidationError as exc:
        raise ValidationError(exc.message) from exc
    except OntologyCommandNotFound as exc:
        raise NotFoundError(exc.resource, exc.identifier) from exc
    except OntologyCommandConflict as exc:
        raise ConflictError(exc.message) from exc

    note = str(body.note or "").strip() or f"Superseded by edited approval {replacement['id']}."
    original = execute_api_action(
        "resolve_approval",
        {"approval_id": approval_id, "status": "rejected", "note": note},
        source_id="approvals.replace",
        actor=actor,
    )
    return {
        "status": "replacement_created",
        "original": normalize_approval(original),
        "replacement": normalize_approval(replacement),
    }


@router.post("/approvals/bulk-approve")
def bulk_approve(body: BulkResolveRequest, actor: ActorDep):
    if not str(body.note or "").strip():
        raise ValidationError("Bulk approval note is required.")
    results = []
    for aid in body.ids:
        try:
            execute_api_action(
                "resolve_approval",
                {"approval_id": aid, "status": "approved", "note": body.note},
                source_id="approvals.bulk_approve",
                actor=actor,
            )
            results.append({"id": aid, "status": "approved"})
        except ConflictError as e:
            results.append({"id": aid, "status": "failed", "message": str(e)})
        except AppError as e:
            results.append({"id": aid, "status": "error", "message": e.message or "Not found or already resolved"})
    return {"results": results}


@router.post("/approvals/bulk-reject")
def bulk_reject(body: BulkResolveRequest, actor: ActorDep):
    results = []
    for aid in body.ids:
        try:
            execute_api_action(
                "resolve_approval",
                {"approval_id": aid, "status": "rejected", "note": body.note},
                source_id="approvals.bulk_reject",
                actor=actor,
            )
            results.append({"id": aid, "status": "rejected"})
        except AppError as e:
            results.append({"id": aid, "status": "error", "message": e.message or "Not found or already resolved"})
    return {"results": results}
