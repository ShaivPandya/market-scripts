"""
AI Agent chat endpoint with streaming (SSE) and function calling.

Uses Anthropic's Messages API with Claude Sonnet 4.6 and the tool definitions from
:mod:`api.agent_tools` to answer cross-cutting investment questions by
fetching live data from the platform's analysis modules.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
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
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 8_192
ANTHROPIC_TOOL_DEFINITIONS: list[dict] = [
    {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
    }
    for tool in TOOL_DEFINITIONS
    if isinstance(tool.get("name"), str)
]


def _read_anthropic_api_key() -> str:
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip().strip("\"'")
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY")

    # A common misconfiguration is placing an OpenAI key into ANTHROPIC_API_KEY.
    if api_key.startswith("sk-proj-") or (api_key.startswith("sk-") and not api_key.startswith("sk-ant-")):
        raise ConfigurationError("ANTHROPIC_API_KEY (must be an Anthropic key beginning with sk-ant-)")

    return api_key


def _format_stream_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    raw = str(exc)
    lowered = raw.lower()

    if status_code == 401 or "invalid x-api-key" in lowered or "authentication_error" in lowered:
        return (
            "Agent authentication failed. Set a valid Anthropic API key in ANTHROPIC_API_KEY "
            "and restart the backend."
        )

    return raw


def _execute_tools_parallel(
    calls: list[dict],
) -> list[tuple[dict, str]]:
    """Execute tool calls in parallel using threads."""
    if len(calls) == 1:
        c = calls[0]
        return [(c, execute_tool(c["name"], c["args"]))]
    with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
        futures = [(c, pool.submit(execute_tool, c["name"], c["args"])) for c in calls]
        return [(c, f.result()) for c, f in futures]


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


def _serialize_content_blocks(blocks: list[object]) -> list[dict]:
    serialized: list[dict] = []
    for block in blocks:
        if isinstance(block, dict):
            serialized.append(block)
            continue

        model_dump = getattr(block, "model_dump", None)
        if callable(model_dump):
            serialized.append(model_dump(exclude_none=True))
            continue

        to_dict = getattr(block, "to_dict", None)
        if callable(to_dict):
            serialized.append(to_dict())
    return serialized


def _extract_tool_calls(content_blocks: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for block in content_blocks:
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        call_id = block.get("id")
        args = block.get("input", {})
        if not isinstance(name, str) or not isinstance(call_id, str):
            continue
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "call_id": call_id, "args": args})
    return calls


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/agent/chat")
def agent_chat(req: AgentChatRequest):
    api_key = _read_anthropic_api_key()
    instructions = _build_agent_instructions()

    def generate():  # noqa: C901 — complex but linear control flow
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        conversation: list[dict[str, object]] = [{"role": m.role, "content": m.content} for m in req.messages]
        continuation_round = 0
        force_tool_use = True

        try:
            while True:
                if continuation_round >= MAX_TOOL_CONTINUATION_ROUNDS:
                    yield _sse(
                        "error",
                        {"message": (f"Tool-call loop limit reached ({MAX_TOOL_CONTINUATION_ROUNDS} rounds).")},
                    )
                    yield _sse("done", {"usage": {}})
                    return

                stream_kwargs: dict[str, object] = dict(
                    model=CLAUDE_MODEL,
                    max_tokens=CLAUDE_MAX_TOKENS,
                    system=instructions,
                    messages=conversation,
                    tools=ANTHROPIC_TOOL_DEFINITIONS,
                )
                if force_tool_use:
                    stream_kwargs["tool_choice"] = {"type": "any"}

                with client.messages.stream(**stream_kwargs) as stream:
                    for event in stream:
                        if event.type == "content_block_delta" and event.delta.type == "text_delta":
                            yield _sse("delta", {"text": event.delta.text})
                        elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                            yield _sse(
                                "tool_call",
                                {
                                    "name": event.content_block.name,
                                    "id": event.content_block.id,
                                },
                            )
                    final_message = stream.get_final_message()

                assistant_content = _serialize_content_blocks(list(final_message.content))
                deferred_calls = _extract_tool_calls(assistant_content)

                if deferred_calls:
                    tool_results: list[dict] = []
                    for call_info, result_str in _execute_tools_parallel(deferred_calls):
                        err_msg = _tool_error_message(result_str)
                        payload = {
                            "name": call_info["name"],
                            "id": call_info["call_id"],
                            "status": "error" if err_msg else "ok",
                        }
                        if err_msg:
                            payload["message"] = err_msg
                        yield _sse("tool_result", payload)

                        result_block: dict[str, object] = {
                            "type": "tool_result",
                            "tool_use_id": call_info["call_id"],
                            "content": result_str,
                        }
                        if err_msg:
                            result_block["is_error"] = True
                        tool_results.append(result_block)

                    conversation.append({"role": "assistant", "content": assistant_content})
                    conversation.append({"role": "user", "content": tool_results})
                    force_tool_use = False
                    continuation_round += 1
                    continue

                if final_message.stop_reason == "pause_turn":
                    conversation.append({"role": "assistant", "content": assistant_content})
                    conversation.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
                    force_tool_use = False
                    continuation_round += 1
                    continue

                usage = {}
                if hasattr(final_message, "usage") and final_message.usage:
                    usage = {
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                    }
                yield _sse("done", {"usage": usage})
                return

        except Exception as exc:
            logger.exception("Agent stream error")
            yield _sse("error", {"message": _format_stream_error(exc)})
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
