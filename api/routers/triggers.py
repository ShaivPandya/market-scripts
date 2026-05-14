"""Watch trigger CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.action_execution import stage_api_action
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()


class CreateTriggerRequest(BaseModel):
    condition: str
    trigger_type: str = "custom"
    ticker: str | None = None
    expires_at: str | None = None
    definition: dict | None = None
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


class ReplaceTriggerRequest(CreateTriggerRequest):
    pass


class TriggerMutationRequest(BaseModel):
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.get("/triggers")
def list_triggers(
    status: str | None = None,
    ticker: str | None = None,
):
    triggers = OntologyRuntimeReadService().watch_triggers(status=status, ticker=ticker)
    return {"triggers": triggers, "count": len(triggers)}


@router.post("/triggers")
def create_trigger(body: CreateTriggerRequest):
    return stage_api_action(
        "create_watch_trigger",
        body.model_dump(exclude={"reason", "apply", "approval_note"}),
        source_id="triggers.create_trigger",
        reason=body.reason or "Create watch trigger",
        apply=body.apply,
        approval_note=body.approval_note,
    )


@router.put("/triggers/{trigger_id}/fire")
def fire_trigger(trigger_id: str, body: TriggerMutationRequest | None = None):
    return stage_api_action(
        "fire_watch_trigger",
        {"trigger_id": trigger_id},
        source_id="triggers.fire_trigger",
        reason=(body.reason if body else None) or f"Fire watch trigger {trigger_id}",
        apply=body.apply if body else False,
        approval_note=body.approval_note if body else None,
        entity_id=trigger_id,
    )


@router.put("/triggers/{trigger_id}/cancel")
def cancel_trigger(trigger_id: str, body: TriggerMutationRequest | None = None):
    return stage_api_action(
        "cancel_watch_trigger",
        {"trigger_id": trigger_id},
        source_id="triggers.cancel_trigger",
        reason=(body.reason if body else None) or f"Cancel watch trigger {trigger_id}",
        apply=body.apply if body else False,
        approval_note=body.approval_note if body else None,
        entity_id=trigger_id,
    )


@router.put("/triggers/{trigger_id}/replace")
def replace_trigger(trigger_id: str, body: ReplaceTriggerRequest):
    return stage_api_action(
        "replace_watch_trigger",
        {"trigger_id": trigger_id, **body.model_dump(exclude={"reason", "apply", "approval_note"})},
        source_id="triggers.replace_trigger",
        reason=body.reason or f"Replace watch trigger {trigger_id}",
        apply=body.apply,
        approval_note=body.approval_note,
        entity_id=trigger_id,
    )
