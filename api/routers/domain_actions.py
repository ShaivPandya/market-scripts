"""Generic staged domain action endpoints."""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.action_execution import execute_api_action, stage_api_action
from api.exceptions import ValidationError

router = APIRouter()


class DomainActionRequest(BaseModel):
    payload: dict[str, Any]
    reason: str | None = None
    approval_note: str | None = None


class BreakGlassRequest(BaseModel):
    payload: dict[str, Any]
    reason_code: Literal["incident_recovery", "data_correction", "migration_repair", "market_hours_exception"]
    reason: str = Field(..., min_length=8)


@router.post("/domain-actions/{action_id}/proposals")
def create_domain_action_proposal(action_id: str, body: DomainActionRequest):
    return stage_api_action(
        action_id,
        body.payload,
        source_id=f"domain_actions.{action_id}.proposal",
        reason=body.reason,
    )


@router.post("/domain-actions/{action_id}/self-apply")
def self_apply_domain_action(action_id: str, body: DomainActionRequest):
    if not str(body.approval_note or "").strip():
        raise ValidationError("approval_note is required for self-apply.")
    return stage_api_action(
        action_id,
        body.payload,
        source_id=f"domain_actions.{action_id}.self_apply",
        reason=body.reason,
        apply=True,
        approval_note=body.approval_note,
    )


@router.post("/domain-actions/{action_id}/break-glass")
def break_glass_domain_action(action_id: str, body: BreakGlassRequest):
    if (os.getenv("BREAK_GLASS_ENABLED") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise HTTPException(status_code=403, detail="Break-glass execution is disabled.")
    result = execute_api_action(
        action_id,
        body.payload,
        source_id=f"break_glass.{body.reason_code}",
        request_mode="break_glass",
    )
    return {
        "status": "break_glass_applied",
        "action_id": action_id,
        "reason_code": body.reason_code,
        "result": result,
    }
