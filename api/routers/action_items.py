"""Action item CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.action_execution import execute_api_action
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
    return execute_api_action(
        "create_action_item",
        body.model_dump(),
        source_id="action_items.create_action",
    )


@router.put("/actions/{item_id}/complete")
def complete_action(item_id: int, body: CompleteActionRequest | None = None):
    return execute_api_action(
        "complete_action_item",
        {"item_id": item_id, "resolution_note": body.resolution_note if body else ""},
        source_id="action_items.complete_action",
    )


@router.put("/actions/{item_id}/dismiss")
def dismiss_action(item_id: int):
    return execute_api_action(
        "dismiss_action_item",
        {"item_id": item_id},
        source_id="action_items.dismiss_action",
    )
