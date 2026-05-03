"""Watch trigger CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.action_execution import execute_api_action

router = APIRouter()


class CreateTriggerRequest(BaseModel):
    condition: str
    trigger_type: str = "custom"
    ticker: str | None = None
    expires_at: str | None = None
    definition: dict | None = None


@router.get("/triggers")
def list_triggers(
    status: str | None = None,
    ticker: str | None = None,
):
    from portfolio.core_db import get_watch_triggers

    triggers = get_watch_triggers(status=status, ticker=ticker)
    return {"triggers": triggers, "count": len(triggers)}


@router.post("/triggers")
def create_trigger(body: CreateTriggerRequest):
    return execute_api_action(
        "create_watch_trigger",
        body.model_dump(),
        source_id="triggers.create_trigger",
    )


@router.put("/triggers/{trigger_id}/fire")
def fire_trigger(trigger_id: int):
    return execute_api_action(
        "fire_watch_trigger",
        {"trigger_id": trigger_id},
        source_id="triggers.fire_trigger",
    )


@router.put("/triggers/{trigger_id}/cancel")
def cancel_trigger(trigger_id: int):
    return execute_api_action(
        "cancel_watch_trigger",
        {"trigger_id": trigger_id},
        source_id="triggers.cancel_trigger",
    )
