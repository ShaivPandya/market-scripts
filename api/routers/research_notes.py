"""Research notes CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.action_execution import stage_api_action

router = APIRouter()


class CreateResearchNoteRequest(BaseModel):
    title: str
    content: str
    ticker: str | None = None
    note_type: str = "general"
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.get("/research-notes")
def list_research_notes(
    ticker: str | None = None,
    limit: int = 20,
):
    from portfolio.core_db import get_research_notes

    safe_limit = max(1, min(int(limit), 100))
    notes = get_research_notes(ticker=ticker, limit=safe_limit)
    return {"notes": notes, "count": len(notes)}


@router.post("/research-notes")
def create_research_note(body: CreateResearchNoteRequest):
    return stage_api_action(
        "create_research_note",
        body.model_dump(exclude={"reason", "apply", "approval_note"}),
        source_id="research_notes.create_research_note",
        reason=body.reason or "Create research note",
        apply=body.apply,
        approval_note=body.approval_note,
    )
