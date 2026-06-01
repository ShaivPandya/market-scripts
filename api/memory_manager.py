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


def _include_in_verbatim(msg: dict[str, Any]) -> bool:
    """Exclude in-progress assistant placeholders from LLM context."""
    if msg.get("role") == "assistant" and msg.get("is_streaming"):
        return False
    return True


def build_conversation_context(
    session_id: str | None,
    new_user_message: str,
    *,
    enable_retrieval: bool = True,
    client_turn_id: str | None = None,
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
    verbatim_msgs = [m for m in server_msgs if _include_in_verbatim(m)]
    if len(verbatim_msgs) > VERBATIM_WINDOW:
        recent = verbatim_msgs[-VERBATIM_WINDOW:]
    else:
        recent = verbatim_msgs

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

    # The new user message (skip if begin_turn already persisted it for this turn)
    skip_duplicate_user = False
    if client_turn_id and recent:
        last = recent[-1]
        if (
            last.get("role") == "user"
            and str(last.get("client_turn_id")) == client_turn_id
            and str(last.get("content") or "") == new_user_message
        ):
            skip_duplicate_user = True
    if not skip_duplicate_user:
        conversation.append({"role": "user", "content": new_user_message})

    return conversation, sid


# ---------------------------------------------------------------------------
# Post-turn finalization
# ---------------------------------------------------------------------------


def begin_turn(
    session_id: str,
    user_message: dict[str, Any],
    assistant_placeholder: dict[str, Any],
) -> None:
    """Persist user message and streaming assistant placeholder at turn start."""
    client_turn_id = str(user_message.get("client_turn_id") or "")
    if not client_turn_id:
        return
    try:
        placeholder = {
            **assistant_placeholder,
            "is_streaming": assistant_placeholder.get("is_streaming", True),
        }
        memory_db.begin_turn(session_id, user_message, placeholder)
    except Exception:
        logger.exception("Failed to begin turn for session %s", session_id)


def begin_turn_async(
    session_id: str,
    user_message: dict[str, Any],
    assistant_placeholder: dict[str, Any],
) -> None:
    t = threading.Thread(
        target=begin_turn,
        args=(session_id, user_message, assistant_placeholder),
        daemon=True,
    )
    t.start()


def update_assistant_message(
    session_id: str,
    client_turn_id: str | None,
    patch: dict[str, Any],
) -> None:
    if not client_turn_id:
        return
    try:
        memory_db.update_assistant_message(session_id, client_turn_id, patch)
    except Exception:
        logger.debug("Failed to update assistant message for session %s", session_id, exc_info=True)


def update_assistant_message_async(
    session_id: str,
    client_turn_id: str | None,
    patch: dict[str, Any],
) -> None:
    t = threading.Thread(
        target=update_assistant_message,
        args=(session_id, client_turn_id, patch),
        daemon=True,
    )
    t.start()


def complete_turn(
    session_id: str,
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
) -> None:
    """Finalize a turn, using in-place update when incremental persistence exists."""
    client_turn_id = str(user_message.get("client_turn_id") or assistant_message.get("client_turn_id") or "")
    final_assistant = {**assistant_message, "is_streaming": False}
    try:
        if client_turn_id and memory_db.turn_exists(session_id, client_turn_id):
            total = memory_db.complete_turn_messages(
                session_id,
                client_turn_id,
                user_message,
                final_assistant,
            )
        else:
            total = memory_db.append_messages(session_id, [user_message, final_assistant])
    except Exception:
        logger.exception("Failed to complete turn for session %s", session_id)
        return

    _ensure_session_title(session_id, user_message, final_assistant, total)

    if total >= SUMMARIZE_THRESHOLD:
        _maybe_summarize(session_id)


def complete_turn_async(
    session_id: str,
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
) -> None:
    t = threading.Thread(
        target=complete_turn,
        args=(session_id, user_message, assistant_message),
        daemon=True,
    )
    t.start()


def fail_turn(
    session_id: str,
    client_turn_id: str | None,
    *,
    status: str = "cancelled",
    content: str | None = None,
) -> None:
    if not client_turn_id:
        return
    try:
        memory_db.fail_turn(session_id, client_turn_id, status=status, content=content)
    except Exception:
        logger.debug("Failed to mark turn failed for session %s", session_id, exc_info=True)


def fail_turn_async(
    session_id: str,
    client_turn_id: str | None,
    *,
    status: str = "cancelled",
    content: str | None = None,
) -> None:
    t = threading.Thread(
        target=fail_turn,
        args=(session_id, client_turn_id),
        kwargs={"status": status, "content": content},
        daemon=True,
    )
    t.start()


def finalize_turn(
    session_id: str,
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
) -> None:
    """Persist the turn and trigger rolling summarization if needed.

    Designed to be called in a background thread so it doesn't block SSE.
    """
    complete_turn(session_id, user_message, assistant_message)


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
