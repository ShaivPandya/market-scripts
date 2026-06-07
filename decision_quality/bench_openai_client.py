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


@dataclass(frozen=True)
class BenchOpenAIConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 120.0

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        api_key_env: str = "TALISMAN_BENCH_CANDIDATE_API_KEY",
        model: str | None = None,
        model_env: str = "TALISMAN_BENCH_CANDIDATE_MODEL",
        timeout_s: float = 120.0,
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


def call_openai_compatible_text(
    *,
    config: BenchOpenAIConfig,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 4096,
    json_schema: dict[str, Any] | None = None,
    json_schema_name: str | None = None,
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
                "strict": True,
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
) -> tuple[Any, list[tuple[str, str]], Any, dict[str, Any]]:
    text, citations, response = call_openai_compatible_text(
        config=config,
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        json_schema=json_schema,
        json_schema_name=json_schema_name,
    )
    parsed = llm_utils.parse_json_text(text)
    diagnostics: dict[str, Any] = {
        "status": "ok",
        "provider": "bench_openai_compatible",
        "model": config.model,
        "attempts": 1,
        "usage": _usage_from_response(response),
    }
    if require_object and not isinstance(parsed, dict):
        diagnostics["status"] = "invalid_json_object"
    return parsed, citations, response, diagnostics


_ORIGINAL_CALL_LLM_TEXT = llm_utils.call_llm_text
_ORIGINAL_CALL_LLM_JSON = llm_utils.call_llm_json
_ACTIVE_CONFIG: BenchOpenAIConfig | None = None


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


@contextmanager
def activate_bench_openai(config: BenchOpenAIConfig) -> Iterator[BenchOpenAIConfig]:
    global _ACTIVE_CONFIG
    previous_text = llm_utils.call_llm_text
    previous_json = llm_utils.call_llm_json
    _ACTIVE_CONFIG = config
    llm_utils.call_llm_text = _patched_call_llm_text
    llm_utils.call_llm_json = _patched_call_llm_json
    try:
        yield config
    finally:
        _ACTIVE_CONFIG = None
        llm_utils.call_llm_text = previous_text
        llm_utils.call_llm_json = previous_json
