"""Action item CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.action_execution import stage_api_action
from api.exceptions import NotFoundError
from ontology.object_service import OntologyObjectService

router = APIRouter()


class CreateActionRequest(BaseModel):
    description: str
    action_type: str = "review"
    ticker: str | None = None
    urgency: str = "normal"
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


class CompleteActionRequest(BaseModel):
    resolution_note: str = ""
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


class DismissActionRequest(BaseModel):
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.get("/actions")
def list_actions(
    status: str | None = None,
    ticker: str | None = None,
):
    filters = {}
    if status:
        filters["status"] = status
    if ticker:
        filters["ticker"] = ticker.strip().upper()
    items = [_flatten_object(row) for row in OntologyObjectService().query_objects("ActionItem", filters=filters)]
    return {"actions": items, "count": len(items)}


@router.get("/actions/{item_id}")
def get_action(item_id: str):
    item = OntologyObjectService().get_object(
        item_id if item_id.startswith("action_item:") else f"action_item:{item_id}"
    )
    if item:
        return _flatten_object(item)
    raise NotFoundError("Action item", str(item_id))


@router.post("/actions")
def create_action(body: CreateActionRequest):
    return stage_api_action(
        "create_action_item",
        body.model_dump(exclude={"reason", "apply", "approval_note"}),
        source_id="action_items.create_action",
        reason=body.reason or "Create action item",
        apply=body.apply,
        approval_note=body.approval_note,
    )


@router.put("/actions/{item_id}/complete")
def complete_action(item_id: str, body: CompleteActionRequest | None = None):
    return stage_api_action(
        "complete_action_item",
        {"item_id": item_id, "resolution_note": body.resolution_note if body else ""},
        source_id="action_items.complete_action",
        reason=(body.reason if body else None) or f"Complete action item {item_id}",
        apply=body.apply if body else False,
        approval_note=body.approval_note if body else None,
        entity_id=item_id,
    )


@router.put("/actions/{item_id}/dismiss")
def dismiss_action(item_id: str, body: DismissActionRequest | None = None):
    return stage_api_action(
        "dismiss_action_item",
        {"item_id": item_id},
        source_id="action_items.dismiss_action",
        reason=(body.reason if body else None) or f"Dismiss action item {item_id}",
        apply=body.apply if body else False,
        approval_note=body.approval_note if body else None,
        entity_id=item_id,
    )


def _flatten_object(row: dict) -> dict:
    props = dict(row.get("properties") or row.get("properties_json") or {})
    props["id"] = str(row.get("object_uid") or props.get("id") or "")
    props["object_uid"] = props["id"]
    return props
