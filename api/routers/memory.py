"""
Conversation memory endpoints for the AI agent.

Provides persistence, summarization, and retrieval of past agent
conversations so the agent can maintain continuity across sessions.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api import memory_db
from llm_utils import MODEL_LOW, call_llm_text, has_llm_api_key

router = APIRouter()
logger = logging.getLogger("api.memory")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SaveSessionRequest(BaseModel):
    messages: list[dict[str, Any]]
    session_id: str | None = None


class SaveSessionResponse(BaseModel):
    session_id: str
    started_at: str
    ended_at: str
    message_count: int


class SummarizeResponse(BaseModel):
    session_id: str
    summary: str
    key_tickers: list[str]
    key_topics: list[str]


class SessionListItem(BaseModel):
    session_id: str
    started_at: str | None
    ended_at: str | None
    message_count: int
    key_tickers: list[str] | None
    key_topics: list[str] | None
    summary: str | None
    title: str | None = None
    title_source: str | None = None
    title_updated_at: str | None = None
    has_active_job: bool = False


class UpdateSessionRequest(BaseModel):
    title: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/memory/sessions", response_model=SaveSessionResponse)
def save_session(req: SaveSessionRequest):
    """Save a conversation transcript."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages to save")
    result = memory_db.save_session(
        messages=req.messages,
        session_id=req.session_id,
    )
    return SaveSessionResponse(**result)


@router.post("/memory/sessions/{session_id}/summarize", response_model=SummarizeResponse)
def summarize_session(session_id: str):
    """Generate an AI summary for a saved session."""
    session = memory_db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    transcript = session.get("transcript", [])
    if not transcript:
        raise HTTPException(status_code=400, detail="Session has no transcript")

    summary, tickers, topics = _summarize_transcript(transcript)

    memory_db.update_summary(
        session_id=session_id,
        summary=summary,
        key_tickers=tickers,
        key_topics=topics,
    )

    # Index summary for semantic search (best-effort)
    try:
        from api.retrieval import index_document

        index_content = summary
        if tickers:
            index_content = f"Tickers: {', '.join(tickers)}\n\n{index_content}"
        if topics:
            index_content = f"Topics: {', '.join(topics)}\n\n{index_content}"
        index_document(
            doc_type="conversation_summary",
            content=index_content,
            ticker=tickers[0] if tickers else None,
            doc_id=f"conversation-{session_id}",
        )
    except Exception:
        logger.debug("Failed to index conversation summary for retrieval", exc_info=True)

    return SummarizeResponse(
        session_id=session_id,
        summary=summary,
        key_tickers=tickers,
        key_topics=topics,
    )


def _agent_job_session_id(row: dict[str, Any]) -> str | None:
    payload = row.get("payload_json") if isinstance(row, dict) else None
    if isinstance(payload, dict):
        session_id = payload.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None


def _active_agent_chat_session_ids() -> set[str]:
    from api.job_queue import list_active_jobs

    active: set[str] = set()
    for row in list_active_jobs():
        if str(row.get("job_type") or "") != "agent_chat_turn":
            continue
        sid = _agent_job_session_id(row)
        if sid:
            active.add(sid)
    return active


@router.get("/memory/sessions", response_model=list[SessionListItem])
def list_sessions(limit: int = 20):
    """List recent sessions (without full transcripts)."""
    rows = memory_db.list_sessions(limit=min(limit, 100))
    active_sessions = _active_agent_chat_session_ids()
    return [SessionListItem(**{**r, "has_active_job": r["session_id"] in active_sessions}) for r in rows]


@router.get("/memory/sessions/{session_id}")
def get_session(session_id: str):
    """Load a full session including transcript and any active agent job metadata."""
    session = memory_db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    from api.job_queue import list_active_jobs

    active_jobs: list[dict[str, Any]] = []
    for row in list_active_jobs():
        if str(row.get("job_type") or "") != "agent_chat_turn":
            continue
        if _agent_job_session_id(row) != session_id:
            continue
        payload = row.get("payload_json") if isinstance(row.get("payload_json"), dict) else {}
        active_jobs.append(
            {
                "job_id": row.get("job_id"),
                "status": row.get("status"),
                "client_turn_id": payload.get("client_turn_id") if isinstance(payload, dict) else None,
            }
        )
    return {**session, "active_jobs": active_jobs}


@router.patch("/memory/sessions/{session_id}", response_model=SessionListItem)
def update_session(session_id: str, req: UpdateSessionRequest):
    """Update user-editable session metadata."""
    try:
        title = memory_db.normalize_session_title(req.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session = memory_db.rename_session(session_id, title)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionListItem(**session)


@router.delete("/memory/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a session."""
    deleted = memory_db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ---------------------------------------------------------------------------
# Summarization (Haiku)
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = """\
Summarize this investment research conversation in 150-200 words.

Extract:
- Key tickers discussed (as uppercase symbols, e.g. CRWD, AAPL)
- Key topics (2-5 short phrases, e.g. "thesis review", "macro risk", "earnings prep")
- Decisions made or conclusions reached
- Open questions or action items

Be specific about numbers, dates, and conclusions.
Return strict JSON with keys: summary (string), key_tickers (array of strings), key_topics (array of strings).
No markdown, no extra text.

Conversation transcript:
"""


def _summarize_transcript(
    transcript: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    """Use Haiku to summarize a conversation. Falls back to deterministic extraction."""
    # Build a compact text representation
    lines: list[str] = []
    for msg in transcript:
        role = msg.get("role", "?")
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"[{role}] {content[:2000]}")

    text = "\n".join(lines[-40:])  # Last 40 messages max

    if has_llm_api_key():
        try:
            return _summarize_with_haiku(text)
        except Exception:
            logger.warning("LLM summarization failed, using fallback", exc_info=True)

    return _summarize_deterministic(transcript)


def _summarize_with_haiku(text: str) -> tuple[str, list[str], list[str]]:
    import json as json_mod

    response_text, _citations, _resp = call_llm_text(
        prompt=_SUMMARIZE_PROMPT + text,
        model=MODEL_LOW,
        api_key=None,
        max_tokens=1024,
    )

    if not response_text:
        raise ValueError("Empty LLM response")

    # Parse JSON from response
    parsed = _parse_json(response_text)
    if not isinstance(parsed, dict):
        raise ValueError("Non-dict Haiku response")

    summary = str(parsed.get("summary", "")).strip()
    tickers = [str(t).upper() for t in parsed.get("key_tickers", []) if isinstance(t, str)]
    topics = [str(t) for t in parsed.get("key_topics", []) if isinstance(t, str)]

    if not summary:
        raise ValueError("Empty summary from Haiku")

    return summary, tickers, topics


def _summarize_deterministic(
    transcript: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    """Fallback: extract tickers and build a basic summary without LLM."""
    import re

    ticker_re = re.compile(r"\b[A-Z][A-Z0-9.]{1,9}\b")
    stop_words = {
        "AND",
        "THE",
        "WITH",
        "FROM",
        "THAT",
        "WHAT",
        "WHICH",
        "SHOW",
        "PORTFOLIO",
        "RISK",
        "EXPOSURE",
        "SIGNAL",
        "SIGNALS",
    }

    all_text = " ".join(str(m.get("content", "")) for m in transcript)
    tickers = sorted(set(ticker_re.findall(all_text)) - stop_words)[:10]

    user_msgs = [str(m.get("content", "")).strip() for m in transcript if m.get("role") == "user"]
    topics = [msg[:80] for msg in user_msgs[:5] if msg]

    msg_count = len(transcript)
    summary = f"Conversation with {msg_count} messages."
    if tickers:
        summary += f" Tickers discussed: {', '.join(tickers[:5])}."
    if user_msgs:
        summary += f" First question: {user_msgs[0][:120]}"

    return summary, tickers, [t[:60] for t in topics]


def _parse_json(text: str) -> Any:
    import json as json_mod

    try:
        return json_mod.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json_mod.loads(text[start : end + 1])
        except Exception:
            pass
    return None
