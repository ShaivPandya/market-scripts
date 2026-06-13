"""Production OpenAI-compatible adapter for the first-party Talisman provider (TL-86)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TalismanEndpointConfig:
    base_url: str
    api_key: str
    timeout_s: float = 120.0

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
    ) -> TalismanEndpointConfig:
        resolved_base_url = (base_url or os.environ.get("TALISMAN_BASE_URL") or "").strip().rstrip("/")
        resolved_api_key = (api_key if api_key is not None else os.environ.get("TALISMAN_API_KEY") or "").strip()
        resolved_timeout = timeout_s
        if resolved_timeout is None:
            raw_timeout = (os.environ.get("TALISMAN_TIMEOUT_S") or "120").strip()
            try:
                resolved_timeout = float(raw_timeout)
            except ValueError:
                resolved_timeout = 120.0
        if not resolved_base_url:
            raise ValueError("TALISMAN_BASE_URL is required for the talisman provider")
        return cls(
            base_url=resolved_base_url,
            api_key=resolved_api_key or "talisman-placeholder",
            timeout_s=resolved_timeout,
        )


def talisman_base_url_from_env() -> str | None:
    value = (os.environ.get("TALISMAN_BASE_URL") or "").strip().rstrip("/")
    return value or None


def talisman_timeout_s_from_env() -> float:
    raw = (os.environ.get("TALISMAN_TIMEOUT_S") or "120").strip()
    try:
        return float(raw)
    except ValueError:
        return 120.0


def openai_compatible_client(*, base_url: str, api_key: str, timeout_s: float) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=api_key or "talisman-placeholder",
        base_url=base_url.rstrip("/"),
        timeout=timeout_s,
    )


def usage_from_chat_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        }
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0),
    }


def openai_function_tools(tool_defs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in tool_defs or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            tools.append(tool)
            continue
        name = tool.get("name")
        parameters = tool.get("parameters") or tool.get("input_schema") or {"type": "object", "properties": {}}
        if isinstance(name, str):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description") or "",
                        "parameters": parameters,
                    },
                }
            )
    return tools


def conversation_to_chat_messages(
    conversation: list[dict[str, Any]],
    *,
    system: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    for item in conversation:
        role = str(item.get("role") or "user")
        content = item.get("content")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"text", "input_text", "output_text"}:
                    text_parts.append(str(block.get("text") or ""))
            content = "".join(text_parts)
        if role == "assistant" and item.get("tool_calls"):
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": item.get("tool_calls"),
                }
            )
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("tool_call_id") or item.get("id"),
                    "content": content if isinstance(content, str) else json.dumps(content),
                }
            )
            continue
        messages.append({"role": role, "content": content if isinstance(content, str) else json.dumps(content)})
    return messages


def call_chat_completions_text(
    *,
    client: Any,
    model: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 4096,
    json_schema: dict[str, Any] | None = None,
    json_schema_name: str | None = None,
    strict_json_schema: bool = True,
) -> tuple[str, list[tuple[str, str]], Any]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if json_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": json_schema_name or "structured_output",
                "schema": json_schema,
                "strict": strict_json_schema,
            },
        }

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message
    text = (message.content or "").strip()
    if not text and getattr(message, "parsed", None) is not None:
        text = json.dumps(message.parsed)
    return text, [], response


def call_chat_completions_tools(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int = 4096,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    converted_tools = openai_function_tools(tools)
    if converted_tools:
        kwargs["tools"] = converted_tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


def serialize_chat_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    tool_calls = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        tool_calls.append(
            {
                "id": getattr(call, "id", None),
                "type": getattr(call, "type", "function"),
                "function": {
                    "name": getattr(function, "name", None) if function is not None else None,
                    "arguments": getattr(function, "arguments", None) if function is not None else None,
                },
            }
        )
    return {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
        "tool_calls": tool_calls or None,
    }


def extract_chat_tool_calls(message: Any) -> list[dict[str, Any]]:
    serialized = serialize_chat_message(message)
    calls: list[dict[str, Any]] = []
    for call in serialized.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        raw_function = call.get("function")
        function: dict[str, Any] = raw_function if isinstance(raw_function, dict) else {}
        name = function.get("name")
        call_id = call.get("id")
        raw_args = function.get("arguments", {})
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {}
            args = parsed_args if isinstance(parsed_args, dict) else {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        if isinstance(name, str) and isinstance(call_id, str):
            calls.append({"name": name, "call_id": call_id, "args": args})
    return calls


def chat_finish_reason(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("finish_reason") or message.get("stop_reason") or "")
    return str(getattr(message, "finish_reason", "") or getattr(message, "stop_reason", "") or "")


def stream_chat_completions_events(
    *,
    client: Any,
    stream_kwargs: dict[str, object],
    text_parts: list[str] | None = None,
):
    """Yield SSE events and return the aggregated assistant message."""
    from api.routers.agent import _sse

    instructions = stream_kwargs.get("instructions") or stream_kwargs.get("system")
    conversation = stream_kwargs.get("input") or stream_kwargs.get("messages") or []
    if not isinstance(conversation, list):
        conversation = []
    messages = conversation_to_chat_messages(
        conversation, system=instructions if isinstance(instructions, str) else None
    )
    raw_tools = stream_kwargs.get("tools")
    tools = openai_function_tools(raw_tools if isinstance(raw_tools, list) else None)
    tool_choice = stream_kwargs.get("tool_choice")
    max_tokens = int(str(stream_kwargs.get("max_output_tokens") or stream_kwargs.get("max_tokens") or 4096))
    model = str(stream_kwargs.get("model") or "")

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    emitted_call_ids: set[str] = set()
    aggregate_tool_calls: dict[int, dict[str, Any]] = {}
    final_message: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": []}
    last_chunk: Any = None

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        last_chunk = chunk
        if not getattr(chunk, "choices", None):
            continue
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            if text_parts is not None:
                text_parts.append(content)
            final_message["content"] = str(final_message.get("content") or "") + content
            yield _sse("delta", {"text": content})
        for tool_delta in getattr(delta, "tool_calls", None) or []:
            index = int(getattr(tool_delta, "index", 0) or 0)
            entry = aggregate_tool_calls.setdefault(
                index,
                {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if getattr(tool_delta, "id", None):
                entry["id"] = tool_delta.id
            function = getattr(tool_delta, "function", None)
            if function is not None:
                if getattr(function, "name", None):
                    entry["function"]["name"] = function.name
                if getattr(function, "arguments", None):
                    entry["function"]["arguments"] = str(entry["function"]["arguments"]) + str(function.arguments)
            call_id = entry.get("id") or f"talisman:{index}"
            name = entry["function"].get("name")
            if isinstance(name, str) and name and call_id not in emitted_call_ids:
                emitted_call_ids.add(call_id)
                yield _sse("tool_call", {"name": name, "id": call_id})

    if aggregate_tool_calls:
        final_message["tool_calls"] = [aggregate_tool_calls[index] for index in sorted(aggregate_tool_calls)]
    usage = usage_from_chat_response(last_chunk) if last_chunk is not None else {"input_tokens": 0, "output_tokens": 0}
    final_message["usage"] = usage
    finish_reason = ""
    if last_chunk is not None and getattr(last_chunk, "choices", None):
        finish_reason = str(getattr(last_chunk.choices[0], "finish_reason", "") or "")
    final_message["finish_reason"] = finish_reason
    return final_message
