"""Research notes CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class CreateResearchNoteRequest(BaseModel):
    title: str
    content: str
    ticker: str | None = None
    note_type: str = "general"


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
    from portfolio.core_db import create_research_note

    return create_research_note(
        title=body.title,
        content=body.content,
        ticker=body.ticker,
        note_type=body.note_type,
        source_type="user",
    )
