"""Approval queue API endpoints -- the gatekeeper for all agent/workflow writes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.exceptions import ConflictError, NotFoundError, ValidationError

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
    from portfolio.core_db import ApprovalApplicationError, resolve_approval

    try:
        return resolve_approval(approval_id, "approved", body.note if body else None)
    except ApprovalApplicationError as e:
        raise ConflictError(str(e)) from e
    except ValueError as e:
        if "not found" in str(e).lower() or "No pending" in str(e):
            raise NotFoundError("Approval", str(approval_id)) from e
        raise ValidationError(str(e)) from e


@router.post("/approvals/{approval_id}/reject")
def reject_item(approval_id: int, body: ResolveRequest | None = None):
    from portfolio.core_db import resolve_approval

    try:
        return resolve_approval(approval_id, "rejected", body.note if body else None)
    except ValueError as e:
        if "not found" in str(e).lower() or "No pending" in str(e):
            raise NotFoundError("Approval", str(approval_id)) from e
        raise ValidationError(str(e)) from e


@router.post("/approvals/bulk-approve")
def bulk_approve(body: BulkResolveRequest):
    from portfolio.core_db import ApprovalApplicationError, resolve_approval

    results = []
    for aid in body.ids:
        try:
            resolve_approval(aid, "approved", body.note)
            results.append({"id": aid, "status": "approved"})
        except ApprovalApplicationError as e:
            results.append({"id": aid, "status": "failed", "message": str(e)})
        except ValueError as e:
            results.append({"id": aid, "status": "error", "message": str(e) or "Not found or already resolved"})
    return {"results": results}


@router.post("/approvals/bulk-reject")
def bulk_reject(body: BulkResolveRequest):
    from portfolio.core_db import resolve_approval

    results = []
    for aid in body.ids:
        try:
            resolve_approval(aid, "rejected", body.note)
            results.append({"id": aid, "status": "rejected"})
        except ValueError as e:
            results.append({"id": aid, "status": "error", "message": str(e) or "Not found or already resolved"})
    return {"results": results}
