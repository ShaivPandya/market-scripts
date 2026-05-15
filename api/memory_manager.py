"""
Rolling memory manager for agent chat.

Replaces the "full transcript every turn" pattern with:
  - A verbatim window of recent messages
  - A rolling summary of older turns (incrementally updated via the configured low-tier LLM)
  - Retrieval-augmented context from past sessions / theses

The frontend sends only the new message + session_id; this module
assembles the optimal context window for the configured LLM.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

from api import memory_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

VERBATIM_WINDOW = 8  # keep last 8 messages (~4 exchanges) verbatim
SUMMARIZE_THRESHOLD = 16  # trigger rolling summarization when total exceeds this
RETRIEVAL_TOP_K = 3  # semantic retrieval hits to include

# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def build_conversation_context(
    session_id: str | None,
    new_user_message: str,
    *,
    enable_retrieval: bool = True,
) -> tuple[list[dict[str, object]], str]:
    """Build the ``messages`` list for a Claude API call.

    Returns ``(conversation, session_id)`` where *conversation* contains:
      1. An optional context preamble (rolling summary + retrieval hits)
      2. The last ``VERBATIM_WINDOW`` messages from the session
      3. The new user message

    The caller should pass *conversation* directly to ``client.messages.stream()``.
    """
    session = memory_db.get_or_create_session(session_id)
    sid = session["session_id"]
    server_msgs: list[dict[str, Any]] = session["server_messages"]
    rolling_summary: str | None = session.get("rolling_summary")

    # Split into old (summarised) and recent (verbatim)
    if len(server_msgs) > VERBATIM_WINDOW:
        recent = server_msgs[-VERBATIM_WINDOW:]
    else:
        recent = server_msgs

    # --- Build optional context preamble ---
    preamble_parts: list[str] = []

    if rolling_summary:
        preamble_parts.append("## Conversation History (summarised)\n" + rolling_summary)

    if enable_retrieval:
        retrieval_context = _retrieve_relevant(new_user_message)
        if retrieval_context:
            preamble_parts.append(retrieval_context)

    # --- Assemble conversation ---
    conversation: list[dict[str, object]] = []

    if preamble_parts:
        context_text = "\n\n".join(preamble_parts)
        # Inject as a user/assistant pair so it doesn't confuse role alternation
        conversation.append(
            {
                "role": "user",
                "content": f"[Context from earlier in this conversation and past research]\n\n{context_text}",
            }
        )
        conversation.append(
            {"role": "assistant", "content": "Understood — I have this context available and will use it as needed."}
        )

    # Verbatim recent messages
    for msg in recent:
        conversation.append({"role": msg["role"], "content": msg["content"]})

    # The new user message
    conversation.append({"role": "user", "content": new_user_message})

    return conversation, sid


# ---------------------------------------------------------------------------
# Post-turn finalization
# ---------------------------------------------------------------------------


def finalize_turn(
    session_id: str,
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
) -> None:
    """Persist the turn and trigger rolling summarization if needed.

    Designed to be called in a background thread so it doesn't block SSE.
    """
    try:
        total = memory_db.append_messages(session_id, [user_message, assistant_message])
    except Exception:
        logger.exception("Failed to append messages to session %s", session_id)
        return

    _ensure_session_title(session_id, user_message, assistant_message, total)

    if total >= SUMMARIZE_THRESHOLD:
        _maybe_summarize(session_id)


def finalize_turn_async(
    session_id: str,
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
) -> None:
    """Fire-and-forget wrapper that runs finalize_turn in a daemon thread."""
    t = threading.Thread(
        target=finalize_turn,
        args=(session_id, user_message, assistant_message),
        daemon=True,
    )
    t.start()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _retrieve_relevant(query: str) -> str | None:
    """Run semantic retrieval for past sessions / theses. Non-blocking."""
    try:
        from api.retrieval import search as retrieval_search

        hits = retrieval_search(
            query=query,
            doc_types=["conversation_summary", "thesis", "news_digest"],
            top_k=RETRIEVAL_TOP_K,
        )
        if not hits:
            return None

        lines = ["## Relevant Past Research"]
        for h in hits:
            ticker = h.get("ticker") or ""
            score = h.get("relevance_score", 0.0)
            snippet = h.get("content_snippet", "")
            lines.append(f"- [{h['doc_type']}] {ticker} (score={score:.2f}): {snippet}")
        return "\n".join(lines)
    except Exception:
        logger.debug("Retrieval failed, skipping", exc_info=True)
        return None


def _ensure_session_title(
    session_id: str,
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
    total_messages: int,
) -> None:
    """Set a quick deterministic title, then schedule first-turn LLM refinement."""
    user_text = str(user_message.get("content") or "")
    memory_db.set_deterministic_title_if_missing(session_id, user_text)

    if total_messages > 2:
        return
    meta = memory_db.get_session_title_metadata(session_id)
    if not meta or meta.get("title_source") != "deterministic":
        return
    _refine_session_title_async(session_id, user_message, assistant_message)


def _refine_session_title_async(
    session_id: str,
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
) -> None:
    t = threading.Thread(
        target=_refine_session_title,
        args=(session_id, user_message, assistant_message),
        daemon=True,
    )
    t.start()


def _refine_session_title(
    session_id: str,
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
) -> None:
    try:
        meta = memory_db.get_session_title_metadata(session_id)
        if not meta or meta.get("title_source") != "deterministic":
            return

        from llm_utils import MODEL_LOW, call_llm_text, has_llm_api_key

        if not has_llm_api_key():
            return

        user_text = str(user_message.get("content") or "").strip()
        assistant_text = str(assistant_message.get("content") or "").strip()
        if not user_text:
            return

        prompt = f"""\
Generate a concise title for this investment research chat.

Rules:
- 2 to 6 words.
- 80 characters maximum.
- Preserve important tickers.
- Return only the title, with no quotes, labels, markdown, or punctuation wrapper.

User:
{user_text[:1500]}

Assistant:
{assistant_text[:1500]}
"""
        response_text, _citations, _resp = call_llm_text(
            prompt=prompt,
            model=MODEL_LOW,
            api_key=None,
            max_tokens=64,
        )
        title = _clean_generated_title(response_text)
        if not title:
            return
        memory_db.update_generated_title(session_id, title)
    except Exception:
        logger.debug("Failed to refine conversation title for session %s", session_id, exc_info=True)


def _clean_generated_title(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.splitlines()[0].strip()
    text = re.sub(r"^title\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" \"'`*_-.")
    if not text:
        return None
    try:
        return memory_db.normalize_session_title(text)
    except ValueError:
        fallback = memory_db.deterministic_title_from_text(text)
        if not fallback:
            return None
        try:
            return memory_db.normalize_session_title(fallback)
        except ValueError:
            return None


def _maybe_summarize(session_id: str) -> None:
    """Summarize older messages if they exceed the verbatim window."""
    try:
        session = memory_db.get_or_create_session(session_id)
        server_msgs: list[dict[str, Any]] = session["server_messages"]

        if len(server_msgs) <= VERBATIM_WINDOW:
            return

        old_msgs = server_msgs[:-VERBATIM_WINDOW]
        existing_summary = session.get("rolling_summary") or ""

        # Build text for Haiku
        lines: list[str] = []
        if existing_summary:
            lines.append(f"[Previous summary]\n{existing_summary}\n")
        lines.append("[New messages to incorporate]")
        for msg in old_msgs:
            role = msg.get("role", "?")
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"[{role}] {content[:2000]}")

        text = "\n".join(lines[-60:])  # cap input

        from llm_utils import has_llm_api_key

        if not has_llm_api_key():
            logger.debug("No valid API key for rolling summarization")
            return

        from api.routers.memory import _summarize_with_haiku

        summary, tickers, topics = _summarize_with_haiku(text)
        memory_db.update_rolling_summary(session_id, summary)

        # Also index the summary for future retrieval
        try:
            from api.retrieval import index_document

            ticker_str = ", ".join(tickers) if tickers else ""
            index_document(
                doc_type="conversation_summary",
                content=f"Tickers: {ticker_str}\n\n{summary}" if ticker_str else summary,
                ticker=tickers[0] if tickers else None,
                doc_id=f"rolling-{session_id}",
            )
        except Exception:
            logger.debug("Failed to index rolling summary", exc_info=True)

    except Exception:
        logger.exception("Rolling summarization failed for session %s", session_id)
