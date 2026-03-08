from __future__ import annotations

from collections.abc import Sequence
from typing import Any

MODEL_HAIKU = "claude-haiku-4-5"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-6"


def _obj_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def extract_text(response: Any) -> str:
    parts: list[str] = []
    for block in _obj_get(response, "content", []) or []:
        if _obj_get(block, "type") == "text":
            text = _obj_get(block, "text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts).strip()


def extract_citations(response: Any) -> list[tuple[str, str]]:
    seen_urls: set[str] = set()
    citations: list[tuple[str, str]] = []

    for block in _obj_get(response, "content", []) or []:
        if _obj_get(block, "type") != "text":
            continue
        for citation in _obj_get(block, "citations", []) or []:
            url = _obj_get(citation, "url")
            if not isinstance(url, str) or not url or url in seen_urls:
                continue
            title = _obj_get(citation, "title")
            label = title if isinstance(title, str) and title else url
            seen_urls.add(url)
            citations.append((label, url))

    return citations


def _extract_json_object(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def call_claude_text(
    *,
    prompt: str,
    model: str,
    api_key: str | None,
    max_tokens: int = 4096,
    system: str | None = None,
    allowed_domains: Sequence[str] | None = None,
    max_web_search_uses: int = 5,
) -> tuple[str, list[tuple[str, str]], Any]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    if allowed_domains is not None:
        kwargs["tools"] = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": max_web_search_uses,
                "allowed_domains": list(allowed_domains),
            }
        ]

    response = client.messages.create(**kwargs)
    while _obj_get(response, "stop_reason") == "pause_turn":
        messages.append({"role": "assistant", "content": _obj_get(response, "content", [])})
        messages.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
        kwargs["messages"] = messages
        response = client.messages.create(**kwargs)

    return extract_text(response), extract_citations(response), response


def parse_json_text(text: str) -> Any:
    import json

    cleaned = _extract_json_object(text)
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(cleaned[start : end + 1])
        except Exception:
            return None
