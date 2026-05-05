"""Approval queue API endpoints -- the gatekeeper for all agent/workflow writes."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from api.action_execution import execute_api_action
from api.decision_state import normalize_approval
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
    return {"approvals": [normalize_approval(a) for a in approvals], "count": len(approvals)}


@router.get("/approvals/summary")
def approval_summary(
    status: str | None = "pending",
    ticker: str | None = None,
    application_status: str | None = None,
    limit: int = Query(default=5, ge=1, le=50),
):
    from portfolio.core_db import get_pending_approvals

    normalized_status = None if status == "all" else status
    normalized_application_status = None if application_status == "all" else application_status
    normalized_ticker = ticker.strip().upper() if ticker and ticker.strip() else None
    try:
        approvals = get_pending_approvals(
            status=normalized_status,
            ticker=normalized_ticker,
            application_status=normalized_application_status,
        )
    except ValueError as e:
        raise ValidationError(str(e)) from e
    normalized = [normalize_approval(a) for a in approvals]
    recommendation_approval_count = len(
        [
            approval
            for approval in normalized
            if isinstance(approval.get("proposed_change"), dict)
            and approval["proposed_change"].get("recommendation_id") is not None
        ]
    )
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
def get_approval(approval_id: int):
    from portfolio.core_db import get_pending_approval, provenance_summary

    approval = get_pending_approval(approval_id)
    if not approval:
        raise NotFoundError("Approval", str(approval_id))
    approval["provenance_summary"] = provenance_summary(approval_id=approval_id)
    return normalize_approval(approval)


@router.post("/approvals/{approval_id}/approve")
def approve_item(approval_id: int, body: ResolveRequest | None = None):
    note = body.note if body else None
    if not str(note or "").strip():
        raise ValidationError("Approval note is required.")
    return execute_api_action(
        "resolve_approval",
        {"approval_id": approval_id, "status": "approved", "note": note},
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
    if not str(body.note or "").strip():
        raise ValidationError("Bulk approval note is required.")
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
