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

        # Build messages for the first call
        input_messages = [{"role": m.role, "content": m.content} for m in req.messages]

        try:
            # Initial streaming call
            stream = client.responses.create(
                model="gpt-5.4",
                instructions=instructions,
                input=input_messages,
                tools=TOOL_DEFINITIONS,
                stream=True,
            )

            # Collect output items from the stream
            response_id: str | None = None
            pending_tool_calls: list[dict] = []
            current_fn_name: str | None = None
            current_fn_call_id: str | None = None
            current_fn_args: str = ""

            for event in stream:
                # Capture the response ID for continuations
                if event.type == "response.created":
                    response_id = event.response.id

                # Text deltas → stream to client
                elif event.type == "response.output_text.delta":
                    yield _sse("delta", {"text": event.delta})

                # Function call started
                elif event.type == "response.function_call_arguments.delta":
                    current_fn_args += event.delta

                elif event.type == "response.output_item.added":
                    if event.item.type == "function_call":
                        current_fn_name = event.item.name
                        current_fn_call_id = event.item.call_id
                        current_fn_args = ""
                        yield _sse(
                            "tool_call",
                            {
                                "name": current_fn_name,
                                "id": current_fn_call_id,
                            },
                        )

                elif event.type == "response.output_item.done":
                    if event.item.type == "function_call":
                        # Execute the tool
                        try:
                            args = json.loads(current_fn_args) if current_fn_args else {}
                        except json.JSONDecodeError:
                            args = {}
                        result_str = execute_tool(event.item.name, args)
                        pending_tool_calls.append(
                            {
                                "call_id": event.item.call_id,
                                "output": result_str,
                            }
                        )
                        yield _sse(
                            "tool_result",
                            {
                                "name": event.item.name,
                                "id": event.item.call_id,
                                "status": "ok",
                            },
                        )

                elif event.type == "response.completed":
                    # If there were tool calls, we need to continue
                    if not pending_tool_calls:
                        usage = {}
                        if hasattr(event.response, "usage") and event.response.usage:
                            usage = {
                                "input_tokens": event.response.usage.input_tokens,
                                "output_tokens": event.response.usage.output_tokens,
                            }
                        yield _sse("done", {"usage": usage})

            # Tool-call continuation loop
            while pending_tool_calls and response_id:
                tool_outputs = []
                for tc in pending_tool_calls:
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": tc["call_id"],
                            "output": tc["output"],
                        }
                    )
                pending_tool_calls = []

                stream = client.responses.create(
                    model="gpt-5.4",
                    previous_response_id=response_id,
                    input=tool_outputs,
                    tools=TOOL_DEFINITIONS,
                    stream=True,
                )

                current_fn_name = None
                current_fn_call_id = None
                current_fn_args = ""

                for event in stream:
                    if event.type == "response.created":
                        response_id = event.response.id

                    elif event.type == "response.output_text.delta":
                        yield _sse("delta", {"text": event.delta})

                    elif event.type == "response.function_call_arguments.delta":
                        current_fn_args += event.delta

                    elif event.type == "response.output_item.added":
                        if event.item.type == "function_call":
                            current_fn_name = event.item.name
                            current_fn_call_id = event.item.call_id
                            current_fn_args = ""
                            yield _sse(
                                "tool_call",
                                {
                                    "name": current_fn_name,
                                    "id": current_fn_call_id,
                                },
                            )

                    elif event.type == "response.output_item.done":
                        if event.item.type == "function_call":
                            try:
                                args = json.loads(current_fn_args) if current_fn_args else {}
                            except json.JSONDecodeError:
                                args = {}
                            result_str = execute_tool(event.item.name, args)
                            pending_tool_calls.append(
                                {
                                    "call_id": event.item.call_id,
                                    "output": result_str,
                                }
                            )
                            yield _sse(
                                "tool_result",
                                {
                                    "name": event.item.name,
                                    "id": event.item.call_id,
                                    "status": "ok",
                                },
                            )

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
