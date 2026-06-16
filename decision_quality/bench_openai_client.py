"""Benchmark-only OpenAI-compatible client for TalismanBench release checks.

This module is intentionally separate from production provider wiring (TL-86).
It patches ``llm_utils`` call sites during a scoped context so owned-model
candidates can be evaluated against an OpenAI-compatible inference endpoint.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import llm_utils

BENCH_AGENT_MODE_ENV = "TALISMAN_BENCH_AGENT_MODE"


@dataclass(frozen=True)
class BenchOpenAIConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 120.0
    combination_id: str | None = None
    host_id: str | None = None
    model_id: str | None = None
    cost_per_1k_input_tokens_usd: float | None = None
    cost_per_1k_output_tokens_usd: float | None = None

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        api_key_env: str = "TALISMAN_BENCH_CANDIDATE_API_KEY",
        model: str | None = None,
        model_env: str = "TALISMAN_BENCH_CANDIDATE_MODEL",
        timeout_s: float = 120.0,
        combination_id: str | None = None,
        host_id: str | None = None,
        model_id: str | None = None,
        cost_per_1k_input_tokens_usd: float | None = None,
        cost_per_1k_output_tokens_usd: float | None = None,
    ) -> BenchOpenAIConfig:
        resolved_base_url = (base_url or os.environ.get("TALISMAN_BENCH_CANDIDATE_BASE_URL") or "").strip()
        resolved_model = (model or os.environ.get(model_env) or "").strip()
        api_key = os.environ.get(api_key_env, "").strip()
        if not resolved_base_url:
            raise ValueError("candidate OpenAI-compatible base_url is required")
        if not resolved_model:
            raise ValueError("candidate model id is required")
        return cls(
            base_url=resolved_base_url.rstrip("/"),
            api_key=api_key,
            model=resolved_model,
            timeout_s=timeout_s,
            combination_id=combination_id,
            host_id=host_id,
            model_id=model_id,
            cost_per_1k_input_tokens_usd=cost_per_1k_input_tokens_usd,
            cost_per_1k_output_tokens_usd=cost_per_1k_output_tokens_usd,
        )


def _openai_client(config: BenchOpenAIConfig) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=config.api_key or "bench-placeholder",
        base_url=config.base_url,
        timeout=config.timeout_s,
    )


def _usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def estimate_cost_usd(
    usage: dict[str, int],
    *,
    cost_per_1k_input_tokens_usd: float | None,
    cost_per_1k_output_tokens_usd: float | None,
) -> float | None:
    if cost_per_1k_input_tokens_usd is None and cost_per_1k_output_tokens_usd is None:
        return None
    input_cost = (usage.get("input_tokens", 0) / 1000.0) * float(cost_per_1k_input_tokens_usd or 0.0)
    output_cost = (usage.get("output_tokens", 0) / 1000.0) * float(cost_per_1k_output_tokens_usd or 0.0)
    return round(input_cost + output_cost, 6)


def call_openai_compatible_text(
    *,
    config: BenchOpenAIConfig,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 4096,
    json_schema: dict[str, Any] | None = None,
    json_schema_name: str | None = None,
    strict_json_schema: bool = True,
) -> tuple[str, list[tuple[str, str]], Any]:
    client = _openai_client(config)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if json_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": json_schema_name or "bench_structured_output",
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


def call_openai_compatible_json(
    *,
    config: BenchOpenAIConfig,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 4096,
    json_schema: dict[str, Any] | None = None,
    json_schema_name: str | None = None,
    require_object: bool = True,
    strict_json_schema: bool = True,
) -> tuple[Any, list[tuple[str, str]], Any, dict[str, Any]]:
    text, citations, response = call_openai_compatible_text(
        config=config,
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        json_schema=json_schema,
        json_schema_name=json_schema_name,
        strict_json_schema=strict_json_schema,
    )
    parsed = llm_utils.parse_json_text(text)
    usage = _usage_from_response(response)
    diagnostics: dict[str, Any] = {
        "status": "ok",
        "provider": "bench_openai_compatible",
        "model": config.model,
        "attempts": 1,
        "usage": usage,
        "estimated_cost_usd": estimate_cost_usd(
            usage,
            cost_per_1k_input_tokens_usd=config.cost_per_1k_input_tokens_usd,
            cost_per_1k_output_tokens_usd=config.cost_per_1k_output_tokens_usd,
        ),
    }
    if require_object and not isinstance(parsed, dict):
        diagnostics["status"] = "invalid_json_object"
    return parsed, citations, response, diagnostics


def call_openai_compatible_tools(
    *,
    config: BenchOpenAIConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int = 4096,
) -> tuple[Any, dict[str, Any]]:
    client = _openai_client(config)
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message
    usage = _usage_from_response(response)
    diagnostics = {
        "status": "ok",
        "provider": "bench_openai_compatible",
        "model": config.model,
        "usage": usage,
        "estimated_cost_usd": estimate_cost_usd(
            usage,
            cost_per_1k_input_tokens_usd=config.cost_per_1k_input_tokens_usd,
            cost_per_1k_output_tokens_usd=config.cost_per_1k_output_tokens_usd,
        ),
        "tool_calls": [
            {
                "id": getattr(call, "id", None),
                "name": getattr(call.function, "name", None) if getattr(call, "function", None) else None,
                "arguments": getattr(call.function, "arguments", None) if getattr(call, "function", None) else None,
            }
            for call in (getattr(message, "tool_calls", None) or [])
        ],
    }
    return message, diagnostics


def _responses_to_chat_messages(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in conversation:
        role = str(item.get("role") or "user")
        content = item.get("content")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
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


def _openai_tools_from_stream_kwargs(stream_kwargs: dict[str, object]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    raw_tools = stream_kwargs.get("tools")
    if not isinstance(raw_tools, list):
        return tools
    for tool in raw_tools:
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


def stream_bench_openai_compatible(
    *,
    config: BenchOpenAIConfig,
    stream_kwargs: dict[str, object],
    text_parts: list[str] | None = None,
):
    """Yield SSE-compatible events from an OpenAI-compatible chat-completions stream."""
    from api.routers.agent import _sse

    client = _openai_client(config)
    instructions = stream_kwargs.get("instructions") or stream_kwargs.get("system")
    conversation = stream_kwargs.get("input") or stream_kwargs.get("messages") or stream_kwargs.get("contents") or []
    if not isinstance(conversation, list):
        conversation = []
    messages: list[dict[str, Any]] = []
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    messages.extend(_responses_to_chat_messages(conversation))

    tools = _openai_tools_from_stream_kwargs(stream_kwargs)
    tool_choice = stream_kwargs.get("tool_choice")
    max_tokens = int(
        str(
            stream_kwargs.get("max_output_tokens")
            or stream_kwargs.get("max_tokens")
            or stream_kwargs.get("max_output_tokens", 4096)
            or 4096
        )
    )

    kwargs: dict[str, Any] = {
        "model": config.model,
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

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
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
            call_id = entry.get("id") or f"bench:{index}"
            name = entry["function"].get("name")
            if isinstance(name, str) and name and call_id not in emitted_call_ids:
                emitted_call_ids.add(call_id)
                yield _sse("tool_call", {"name": name, "id": call_id})

    if aggregate_tool_calls:
        final_message["tool_calls"] = [aggregate_tool_calls[index] for index in sorted(aggregate_tool_calls)]
    usage = _usage_from_response(chunk) if "chunk" in locals() else {"input_tokens": 0, "output_tokens": 0}
    final_message["usage"] = usage
    return final_message


_ORIGINAL_CALL_LLM_TEXT = llm_utils.call_llm_text
_ORIGINAL_CALL_LLM_JSON = llm_utils.call_llm_json
_ORIGINAL_GET_LLM_CLIENT = llm_utils.get_llm_client
_ORIGINAL_MODEL_FOR_TIER = llm_utils.model_for_tier
_ORIGINAL_RESOLVE_MODEL = llm_utils.resolve_model
_ORIGINAL_SELECTED_PROVIDER = llm_utils.selected_provider
_ORIGINAL_SELECTED_PROVIDER_FOR_TIER = llm_utils.selected_provider_for_tier
_ACTIVE_CONFIG: BenchOpenAIConfig | None = None
_AGENT_PATCHES: dict[str, Any] = {}


def _patched_call_llm_text(**kwargs: Any) -> tuple[str, list[tuple[str, str]], Any]:
    if _ACTIVE_CONFIG is None:
        return _ORIGINAL_CALL_LLM_TEXT(**kwargs)
    return call_openai_compatible_text(
        config=_ACTIVE_CONFIG,
        prompt=str(kwargs.get("prompt") or ""),
        system=kwargs.get("system"),
        max_tokens=int(kwargs.get("max_tokens") or 4096),
        json_schema=kwargs.get("json_schema"),
        json_schema_name=kwargs.get("json_schema_name"),
    )


def _patched_call_llm_json(**kwargs: Any) -> tuple[Any, list[tuple[str, str]], Any, dict[str, Any]]:
    if _ACTIVE_CONFIG is None:
        return _ORIGINAL_CALL_LLM_JSON(**kwargs)
    return call_openai_compatible_json(
        config=_ACTIVE_CONFIG,
        prompt=str(kwargs.get("prompt") or ""),
        system=kwargs.get("system"),
        max_tokens=int(kwargs.get("max_tokens") or 4096),
        json_schema=kwargs.get("json_schema"),
        json_schema_name=kwargs.get("json_schema_name"),
        require_object=bool(kwargs.get("require_object", True)),
    )


def _patched_get_llm_client(provider: str | None = None, api_key: str | None = None) -> Any:
    if _ACTIVE_CONFIG is None or not _agent_mode_enabled():
        return _ORIGINAL_GET_LLM_CLIENT(provider, api_key)
    return _openai_client(_ACTIVE_CONFIG)


def _patched_model_for_tier(tier: str, provider: str | None = None) -> str:
    if _ACTIVE_CONFIG is None or not _agent_mode_enabled():
        return _ORIGINAL_MODEL_FOR_TIER(tier, provider)
    return _ACTIVE_CONFIG.model


def _patched_resolve_model(model: str, provider: str | None = None) -> str:
    if _ACTIVE_CONFIG is None or not _agent_mode_enabled():
        return _ORIGINAL_RESOLVE_MODEL(model, provider)
    return _ACTIVE_CONFIG.model


def _patched_selected_provider() -> str:
    if _ACTIVE_CONFIG is None or not _agent_mode_enabled():
        return _ORIGINAL_SELECTED_PROVIDER()
    return llm_utils.PROVIDER_OPENAI


def _patched_selected_provider_for_tier(tier: str) -> str:
    if _ACTIVE_CONFIG is None or not _agent_mode_enabled():
        return _ORIGINAL_SELECTED_PROVIDER_FOR_TIER(tier)
    return llm_utils.PROVIDER_OPENAI


def _agent_mode_enabled() -> bool:
    return os.environ.get(BENCH_AGENT_MODE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _patch_agent_stream() -> None:
    if _AGENT_PATCHES:
        return
    import api.routers.agent as agent_router

    original_stream = agent_router._stream_llm_response

    def patched_stream(
        client: Any, provider: str, stream_kwargs: dict[str, object], text_parts: list[str] | None = None, **kwargs: Any
    ):
        if _ACTIVE_CONFIG is not None and _agent_mode_enabled():
            return (
                yield from stream_bench_openai_compatible(
                    config=_ACTIVE_CONFIG, stream_kwargs=stream_kwargs, text_parts=text_parts
                )
            )
        return (yield from original_stream(client, provider, stream_kwargs, text_parts, **kwargs))

    agent_router._stream_llm_response = patched_stream
    _AGENT_PATCHES["agent_router"] = agent_router
    _AGENT_PATCHES["original_stream"] = original_stream


def _restore_agent_stream() -> None:
    agent_router = _AGENT_PATCHES.get("agent_router")
    original_stream = _AGENT_PATCHES.get("original_stream")
    if agent_router is not None and original_stream is not None:
        agent_router._stream_llm_response = original_stream
    _AGENT_PATCHES.clear()


@contextmanager
def activate_bench_openai(
    config: BenchOpenAIConfig,
    *,
    agent_mode: bool | None = None,
) -> Iterator[BenchOpenAIConfig]:
    global _ACTIVE_CONFIG
    previous_text = llm_utils.call_llm_text
    previous_json = llm_utils.call_llm_json
    previous_client = llm_utils.get_llm_client
    previous_model_for_tier = llm_utils.model_for_tier
    previous_resolve_model = llm_utils.resolve_model
    previous_selected_provider = llm_utils.selected_provider
    previous_selected_provider_for_tier = llm_utils.selected_provider_for_tier
    previous_agent_mode = os.environ.get(BENCH_AGENT_MODE_ENV)

    resolved_agent_mode = _agent_mode_enabled() if agent_mode is None else agent_mode
    if resolved_agent_mode:
        os.environ[BENCH_AGENT_MODE_ENV] = "1"
        _patch_agent_stream()
    else:
        os.environ.pop(BENCH_AGENT_MODE_ENV, None)

    _ACTIVE_CONFIG = config
    llm_utils.call_llm_text = _patched_call_llm_text
    llm_utils.call_llm_json = _patched_call_llm_json
    llm_utils.get_llm_client = _patched_get_llm_client
    llm_utils.model_for_tier = _patched_model_for_tier
    llm_utils.resolve_model = _patched_resolve_model
    llm_utils.selected_provider = _patched_selected_provider
    llm_utils.selected_provider_for_tier = _patched_selected_provider_for_tier
    try:
        yield config
    finally:
        _ACTIVE_CONFIG = None
        llm_utils.call_llm_text = previous_text
        llm_utils.call_llm_json = previous_json
        llm_utils.get_llm_client = previous_client
        llm_utils.model_for_tier = previous_model_for_tier
        llm_utils.resolve_model = previous_resolve_model
        llm_utils.selected_provider = previous_selected_provider
        llm_utils.selected_provider_for_tier = previous_selected_provider_for_tier
        if previous_agent_mode is None:
            os.environ.pop(BENCH_AGENT_MODE_ENV, None)
        else:
            os.environ[BENCH_AGENT_MODE_ENV] = previous_agent_mode
        _restore_agent_stream()
