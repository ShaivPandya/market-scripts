"""Lightweight durable worker for replayable agent chat turns."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from api.agent_models import AgentChatJobRequest, AgentChatRequest
from api.job_events import append_job_event
from api.job_queue import get_job
from llm_utils import api_key_env, selected_provider


def _env_int(name: str, *, default: int, minimum: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _delta_flush_interval_s() -> float:
    return _env_int("AGENT_DELTA_FLUSH_INTERVAL_MS", default=500, minimum=0) / 1000.0


def _delta_flush_bytes() -> int:
    return _env_int("AGENT_DELTA_FLUSH_BYTES", default=1024, minimum=1)


def _parse_sse_frame(raw: str) -> tuple[str, dict[str, Any]] | None:
    event_type: str | None = None
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if not event_type or not data_lines:
        return None
    try:
        payload = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return event_type, payload


def _append_agent_delta(
    job_id: str,
    buffer: list[str],
    *,
    force: bool = False,
    state: dict[str, float],
) -> None:
    if not buffer:
        return
    now = time.monotonic()
    text = "".join(buffer)
    last = state.get("last_delta_flush", 0.0)
    if not force and len(text.encode("utf-8")) < _delta_flush_bytes() and now - last < _delta_flush_interval_s():
        return
    buffer.clear()
    state["last_delta_flush"] = now
    append_job_event(job_id, "delta", {"text": text})


def _job_cancelled(job_id: str) -> bool:
    row = get_job(job_id)
    return bool(row and str(row.get("status") or "") == "cancelled")


def _format_stream_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    raw = str(exc)
    lowered = raw.lower()

    if status_code == 401 or "invalid x-api-key" in lowered or "authentication_error" in lowered:
        try:
            provider = selected_provider()
            key_env = api_key_env(provider)
        except Exception:
            provider = "configured provider"
            key_env = "the selected provider API key"
        return f"Agent authentication failed. Set a valid {provider} API key in {key_env} and restart the backend."

    if status_code == 529 or "overloaded" in lowered:
        return "The AI model is temporarily overloaded. Please try again in a few seconds."

    if status_code == 429 or "rate_limit" in lowered:
        return "Rate limit reached. Please wait a moment before sending another message."

    return raw


def _run_agent_chat_turn_job(req: AgentChatJobRequest, *, job_id: str) -> dict[str, Any]:
    """Execute one agent turn and persist replayable chat events."""
    worker_req = AgentChatRequest.model_validate(
        {
            **req.model_dump(exclude={"actor"}),
            "finalize_synchronously": True,
            "allow_workflow_handoff": False,
        }
    )
    append_job_event(job_id, "status", {"status": "running", "session_id": worker_req.session_id})
    delta_buffer: list[str] = []
    flush_state = {"last_delta_flush": time.monotonic()}
    terminal_payload: dict[str, Any] | None = None
    error_message: str | None = None

    async def _consume() -> None:
        nonlocal terminal_payload, error_message
        from api.routers.agent import agent_chat
        from ontology.policy import actor_from_dict

        actor = actor_from_dict(req.actor)
        response = agent_chat(worker_req, actor)
        buffer = ""
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                text = chunk.decode("utf-8", errors="replace")
            else:
                text = str(chunk)
            buffer += text
            frames = buffer.split("\n\n")
            buffer = frames.pop() or ""
            for frame in frames:
                parsed = _parse_sse_frame(frame)
                if parsed is None:
                    continue
                event_type, payload = parsed
                if event_type == "ping":
                    if _job_cancelled(job_id):
                        return
                    continue
                if event_type == "delta":
                    delta_text = payload.get("text")
                    if isinstance(delta_text, str) and delta_text:
                        delta_buffer.append(delta_text)
                        _append_agent_delta(job_id, delta_buffer, state=flush_state)
                    if _job_cancelled(job_id):
                        return
                    continue

                _append_agent_delta(job_id, delta_buffer, force=True, state=flush_state)
                append_job_event(job_id, event_type, payload)
                if event_type == "error":
                    error_message = str(payload.get("message") or "Agent chat turn failed")
                elif event_type == "done":
                    terminal_payload = payload
                if _job_cancelled(job_id):
                    return

            if _job_cancelled(job_id):
                return

        _append_agent_delta(job_id, delta_buffer, force=True, state=flush_state)

    try:
        asyncio.run(_consume())
    except Exception as exc:
        message = _format_stream_error(exc)
        append_job_event(job_id, "error", {"message": message})
        raise RuntimeError(message) from exc

    if _job_cancelled(job_id):
        return {"status": "cancelled", "session_id": worker_req.session_id}

    if error_message:
        raise RuntimeError(error_message)

    if terminal_payload is None:
        terminal_payload = {"usage": {}, "session_id": worker_req.session_id}
        append_job_event(job_id, "done", terminal_payload)

    return {"status": "done", **terminal_payload}
