"""
AI Agent chat endpoint with streaming (SSE) and function calling.

Uses the OpenAI Responses API with GPT-5.4 and the tool definitions from
:mod:`api.agent_tools` to answer cross-cutting investment questions by
fetching live data from the platform's analysis modules.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.agent_tools import TOOL_DEFINITIONS, execute_tool
from api.exceptions import ConfigurationError

router = APIRouter()
logger = logging.getLogger("api.agent")

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "auto_report" / "prompts"


def _load_required_prompt_file(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise ConfigurationError(f"Missing required prompt file: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ConfigurationError(f"Prompt file is empty: {path}")
    return content


def _build_agent_instructions() -> str:
    core_md = _load_required_prompt_file("system.md")
    agent_md = _load_required_prompt_file("agent_system.md")
    return "\n\n---\n\n".join([core_md, agent_md])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


MAX_TOOL_CONTINUATION_ROUNDS = 8


def _tool_error_message(result_str: str) -> str | None:
    try:
        payload = json.loads(result_str)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    err = payload.get("error")
    if isinstance(err, str) and err.strip():
        return err.strip()
    return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/agent/chat")
def agent_chat(req: AgentChatRequest):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ConfigurationError("OPENAI_API_KEY")
    instructions = _build_agent_instructions()

    def generate():  # noqa: C901 — complex but linear control flow
        from openai import OpenAI

        client = OpenAI()
        input_messages = [{"role": m.role, "content": m.content} for m in req.messages]

        try:
            stream = client.responses.create(
                model="gpt-5.4",
                instructions=instructions,
                input=input_messages,
                tools=TOOL_DEFINITIONS,
                stream=True,
            )

            response_id: str | None = None
            pending_tool_calls: list[dict[str, str]] = []
            arg_buffers: dict[str, str] = {}
            item_to_call: dict[str, str] = {}
            active_call_id: str | None = None

            for event in stream:
                if event.type == "response.created":
                    response_id = event.response.id

                elif event.type == "response.output_text.delta":
                    yield _sse("delta", {"text": event.delta})

                elif event.type == "response.function_call_arguments.delta":
                    delta = str(getattr(event, "delta", ""))
                    call_id = getattr(event, "call_id", None)
                    if not call_id:
                        item_id = getattr(event, "item_id", None)
                        if isinstance(item_id, str):
                            call_id = item_to_call.get(item_id)
                    if not call_id:
                        call_id = active_call_id
                    if call_id:
                        arg_buffers[call_id] = arg_buffers.get(call_id, "") + delta

                elif event.type == "response.output_item.added":
                    if event.item.type == "function_call":
                        active_call_id = event.item.call_id
                        arg_buffers.setdefault(event.item.call_id, "")
                        item_id = getattr(event.item, "id", None)
                        if isinstance(item_id, str):
                            item_to_call[item_id] = event.item.call_id
                        yield _sse(
                            "tool_call",
                            {
                                "name": event.item.name,
                                "id": event.item.call_id,
                            },
                        )

                elif event.type == "response.output_item.done":
                    if event.item.type == "function_call":
                        call_id = event.item.call_id
                        raw_args = arg_buffers.pop(call_id, "")
                        try:
                            args = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            args = {}
                        result_str = execute_tool(event.item.name, args)
                        pending_tool_calls.append({"call_id": call_id, "output": result_str})
                        err_msg = _tool_error_message(result_str)
                        payload = {
                            "name": event.item.name,
                            "id": call_id,
                            "status": "error" if err_msg else "ok",
                        }
                        if err_msg:
                            payload["message"] = err_msg
                        yield _sse("tool_result", payload)

                elif event.type == "response.completed":
                    if not pending_tool_calls:
                        usage = {}
                        if hasattr(event.response, "usage") and event.response.usage:
                            usage = {
                                "input_tokens": event.response.usage.input_tokens,
                                "output_tokens": event.response.usage.output_tokens,
                            }
                        yield _sse("done", {"usage": usage})

            if pending_tool_calls and not response_id:
                yield _sse("error", {"message": "Missing response ID for tool-call continuation."})
                yield _sse("done", {"usage": {}})
                return

            continuation_round = 0
            while pending_tool_calls and response_id:
                continuation_round += 1
                if continuation_round > MAX_TOOL_CONTINUATION_ROUNDS:
                    yield _sse(
                        "error",
                        {"message": (f"Tool-call loop limit reached ({MAX_TOOL_CONTINUATION_ROUNDS} rounds).")},
                    )
                    yield _sse("done", {"usage": {}})
                    return

                tool_outputs = [
                    {
                        "type": "function_call_output",
                        "call_id": tc["call_id"],
                        "output": tc["output"],
                    }
                    for tc in pending_tool_calls
                ]
                pending_tool_calls = []
                arg_buffers = {}
                item_to_call = {}
                active_call_id = None

                stream = client.responses.create(
                    model="gpt-5.4",
                    previous_response_id=response_id,
                    input=tool_outputs,
                    tools=TOOL_DEFINITIONS,
                    stream=True,
                )

                for event in stream:
                    if event.type == "response.created":
                        response_id = event.response.id

                    elif event.type == "response.output_text.delta":
                        yield _sse("delta", {"text": event.delta})

                    elif event.type == "response.function_call_arguments.delta":
                        delta = str(getattr(event, "delta", ""))
                        call_id = getattr(event, "call_id", None)
                        if not call_id:
                            item_id = getattr(event, "item_id", None)
                            if isinstance(item_id, str):
                                call_id = item_to_call.get(item_id)
                        if not call_id:
                            call_id = active_call_id
                        if call_id:
                            arg_buffers[call_id] = arg_buffers.get(call_id, "") + delta

                    elif event.type == "response.output_item.added":
                        if event.item.type == "function_call":
                            active_call_id = event.item.call_id
                            arg_buffers.setdefault(event.item.call_id, "")
                            item_id = getattr(event.item, "id", None)
                            if isinstance(item_id, str):
                                item_to_call[item_id] = event.item.call_id
                            yield _sse(
                                "tool_call",
                                {
                                    "name": event.item.name,
                                    "id": event.item.call_id,
                                },
                            )

                    elif event.type == "response.output_item.done":
                        if event.item.type == "function_call":
                            call_id = event.item.call_id
                            raw_args = arg_buffers.pop(call_id, "")
                            try:
                                args = json.loads(raw_args) if raw_args else {}
                            except json.JSONDecodeError:
                                args = {}
                            result_str = execute_tool(event.item.name, args)
                            pending_tool_calls.append({"call_id": call_id, "output": result_str})
                            err_msg = _tool_error_message(result_str)
                            payload = {
                                "name": event.item.name,
                                "id": call_id,
                                "status": "error" if err_msg else "ok",
                            }
                            if err_msg:
                                payload["message"] = err_msg
                            yield _sse("tool_result", payload)

                    elif event.type == "response.completed":
                        if not pending_tool_calls:
                            usage = {}
                            if hasattr(event.response, "usage") and event.response.usage:
                                usage = {
                                    "input_tokens": event.response.usage.input_tokens,
                                    "output_tokens": event.response.usage.output_tokens,
                                }
                            yield _sse("done", {"usage": usage})

        except Exception as exc:
            logger.exception("Agent stream error")
            yield _sse("error", {"message": str(exc)})
            yield _sse("done", {"usage": {}})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
