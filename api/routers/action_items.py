"""Action item CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.exceptions import NotFoundError

router = APIRouter()


class CreateActionRequest(BaseModel):
    description: str
    action_type: str = "review"
    ticker: str | None = None
    urgency: str = "normal"


class CompleteActionRequest(BaseModel):
    resolution_note: str = ""


@router.get("/actions")
def list_actions(
    status: str | None = None,
    ticker: str | None = None,
):
    from portfolio.core_db import get_action_items

    items = get_action_items(status=status, ticker=ticker)
    return {"actions": items, "count": len(items)}


@router.get("/actions/{item_id}")
def get_action(item_id: int):
    from portfolio.core_db import get_action_items

    items = get_action_items()
    for item in items:
        if item["id"] == item_id:
            return item
    raise NotFoundError("Action item", str(item_id))


@router.post("/actions")
def create_action(body: CreateActionRequest):
    from portfolio.core_db import create_action_item

    return create_action_item(
        description=body.description,
        action_type=body.action_type,
        ticker=body.ticker,
        urgency=body.urgency,
        source_type="user",
    )


@router.put("/actions/{item_id}/complete")
def complete_action(item_id: int, body: CompleteActionRequest | None = None):
    from portfolio.core_db import complete_action_item

    try:
        return complete_action_item(item_id, body.resolution_note if body else "")
    except ValueError as e:
        raise NotFoundError("Action item", str(item_id)) from e


@router.put("/actions/{item_id}/dismiss")
def dismiss_action(item_id: int):
    from portfolio.core_db import dismiss_action_item

    try:
        return dismiss_action_item(item_id)
    except ValueError as e:
        raise NotFoundError("Action item", str(item_id)) from e
