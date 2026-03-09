"""Watch trigger CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.exceptions import NotFoundError

router = APIRouter()


class CreateTriggerRequest(BaseModel):
    condition: str
    trigger_type: str = "custom"
    ticker: str | None = None
    expires_at: str | None = None


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
    from portfolio.core_db import create_watch_trigger

    return create_watch_trigger(
        condition=body.condition,
        trigger_type=body.trigger_type,
        ticker=body.ticker,
        source_type="user",
        expires_at=body.expires_at,
    )


@router.put("/triggers/{trigger_id}/fire")
def fire_trigger(trigger_id: int):
    from portfolio.core_db import fire_watch_trigger

    try:
        return fire_watch_trigger(trigger_id)
    except ValueError as e:
        raise NotFoundError("Watch trigger", str(trigger_id)) from e


@router.put("/triggers/{trigger_id}/cancel")
def cancel_trigger(trigger_id: int):
    from portfolio.core_db import cancel_watch_trigger

    try:
        return cancel_watch_trigger(trigger_id)
    except ValueError as e:
        raise NotFoundError("Watch trigger", str(trigger_id)) from e
