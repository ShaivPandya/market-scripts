"""Approval queue API endpoints -- the gatekeeper for all agent/workflow writes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.action_execution import execute_api_action
from api.exceptions import AppError, ConflictError, NotFoundError, ValidationError

router = APIRouter()


class ResolveRequest(BaseModel):
    note: str | None = None


class BulkResolveRequest(BaseModel):
    ids: list[int]
    note: str | None = None


@router.get("/approvals")
def list_approvals(
    status: str | None = "pending",
    ticker: str | None = None,
    application_status: str | None = None,
):
    from portfolio.core_db import get_pending_approvals

    if status == "all":
        status = None
    if application_status == "all":
        application_status = None
    try:
        approvals = get_pending_approvals(status=status, ticker=ticker, application_status=application_status)
    except ValueError as e:
        raise ValidationError(str(e)) from e
    return {"approvals": approvals, "count": len(approvals)}


@router.get("/approvals/{approval_id}")
def get_approval(approval_id: int):
    from portfolio.core_db import get_pending_approval

    approval = get_pending_approval(approval_id)
    if not approval:
        raise NotFoundError("Approval", str(approval_id))
    return approval


@router.post("/approvals/{approval_id}/approve")
def approve_item(approval_id: int, body: ResolveRequest | None = None):
    return execute_api_action(
        "resolve_approval",
        {"approval_id": approval_id, "status": "approved", "note": body.note if body else None},
        source_id="approvals.approve_item",
    )


@router.post("/approvals/{approval_id}/reject")
def reject_item(approval_id: int, body: ResolveRequest | None = None):
    return execute_api_action(
        "resolve_approval",
        {"approval_id": approval_id, "status": "rejected", "note": body.note if body else None},
        source_id="approvals.reject_item",
    )


@router.post("/approvals/bulk-approve")
def bulk_approve(body: BulkResolveRequest):
    results = []
    for aid in body.ids:
        try:
            execute_api_action(
                "resolve_approval",
                {"approval_id": aid, "status": "approved", "note": body.note},
                source_id="approvals.bulk_approve",
            )
            results.append({"id": aid, "status": "approved"})
        except ConflictError as e:
            results.append({"id": aid, "status": "failed", "message": str(e)})
        except AppError as e:
            results.append({"id": aid, "status": "error", "message": e.message or "Not found or already resolved"})
    return {"results": results}


@router.post("/approvals/bulk-reject")
def bulk_reject(body: BulkResolveRequest):
    results = []
    for aid in body.ids:
        try:
            execute_api_action(
                "resolve_approval",
                {"approval_id": aid, "status": "rejected", "note": body.note},
                source_id="approvals.bulk_reject",
            )
            results.append({"id": aid, "status": "rejected"})
        except AppError as e:
            results.append({"id": aid, "status": "error", "message": e.message or "Not found or already resolved"})
    return {"results": results}
