from __future__ import annotations

import base64
import os
import threading
from collections.abc import Sequence
from typing import Any

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDERS = {PROVIDER_ANTHROPIC, PROVIDER_OPENAI}

MODEL_LOW = "low"
MODEL_MID = "mid"
MODEL_HIGH = "high"
MODEL_TIERS = {MODEL_LOW, MODEL_MID, MODEL_HIGH}

REASONING_MEDIUM = "medium"
REASONING_HIGH = "high"
REASONING_NONE = "none"
REASONING_XHIGH = "xhigh"
REASONING_MAX = "max"
OPENAI_REASONING_EFFORTS = (
    REASONING_NONE,
    REASONING_MEDIUM,
    REASONING_XHIGH,
)
ANTHROPIC_REASONING_EFFORTS = (
    REASONING_NONE,
    REASONING_HIGH,
    REASONING_MAX,
)
REASONING_EFFORTS = set(OPENAI_REASONING_EFFORTS) | set(ANTHROPIC_REASONING_EFFORTS)
DEFAULT_REASONING_EFFORT_BY_PROVIDER = {
    PROVIDER_ANTHROPIC: REASONING_HIGH,
    PROVIDER_OPENAI: REASONING_MEDIUM,
}

ANTHROPIC_DEFAULT_MODELS = {
    MODEL_LOW: "claude-haiku-4-5",
    MODEL_MID: "claude-sonnet-4-6",
    MODEL_HIGH: "claude-opus-4-7",
}
OPENAI_DEFAULT_MODELS = {
    MODEL_LOW: "gpt-5.4-mini",
    MODEL_MID: "gpt-5.4",
    MODEL_HIGH: "gpt-5.5",
}

# Compatibility aliases for older call sites. These now represent tiers.
MODEL_HAIKU = MODEL_LOW
MODEL_SONNET = MODEL_MID
MODEL_OPUS = MODEL_HIGH

_LEGACY_MODEL_TO_TIER = {
    "claude-haiku-4-5": MODEL_LOW,
    "claude-sonnet-4-6": MODEL_MID,
    "claude-opus-4-7": MODEL_HIGH,
    "gpt-5.4-mini": MODEL_LOW,
    "gpt-5.4": MODEL_MID,
    "gpt-5.5": MODEL_HIGH,
}
_MODEL_ENV_BY_PROVIDER = {
    PROVIDER_ANTHROPIC: {
        MODEL_LOW: "ANTHROPIC_MODEL_LOW",
        MODEL_MID: "ANTHROPIC_MODEL_MID",
        MODEL_HIGH: "ANTHROPIC_MODEL_HIGH",
    },
    PROVIDER_OPENAI: {
        MODEL_LOW: "OPENAI_MODEL_LOW",
        MODEL_MID: "OPENAI_MODEL_MID",
        MODEL_HIGH: "OPENAI_MODEL_HIGH",
    },
}
_API_KEY_ENV_BY_PROVIDER = {
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
    PROVIDER_OPENAI: "OPENAI_API_KEY",
}
_CLIENT_CACHE: dict[tuple[str, str | None], Any] = {}
_CLIENT_FACTORY_CACHE: dict[str, Any] = {}
_CLIENT_LOCK = threading.Lock()


def _obj_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _stored_provider() -> str | None:
    try:
        from api.llm_settings import get_llm_provider_setting

        return get_llm_provider_setting()
    except Exception:
        return None


def selected_provider() -> str:
    provider = (_stored_provider() or os.environ.get("LLM_PROVIDER") or PROVIDER_ANTHROPIC).strip().lower()
    if provider not in PROVIDERS:
        raise ValueError("LLM_PROVIDER must be 'anthropic' or 'openai'")
    return provider


def api_key_env(provider: str | None = None) -> str:
    return _API_KEY_ENV_BY_PROVIDER[_normalize_provider(provider)]


def get_api_key(provider: str | None = None) -> str | None:
    value = (os.environ.get(api_key_env(provider)) or "").strip().strip("\"'")
    return value or None


def has_llm_api_key(provider: str | None = None) -> bool:
    return get_api_key(provider) is not None


def require_api_key(provider: str | None = None) -> str:
    resolved_provider = _normalize_provider(provider)
    api_key = get_api_key(resolved_provider)
    if not api_key:
        raise RuntimeError(f"{api_key_env(resolved_provider)} is required for LLM_PROVIDER={resolved_provider}")
    if resolved_provider == PROVIDER_ANTHROPIC and (
        api_key.startswith("sk-proj-") or (api_key.startswith("sk-") and not api_key.startswith("sk-ant-"))
    ):
        raise RuntimeError("ANTHROPIC_API_KEY must be an Anthropic key beginning with sk-ant-")
    return api_key


def model_for_tier(tier: str, provider: str | None = None) -> str:
    resolved_provider = _normalize_provider(provider)
    normalized_tier = _normalize_tier(tier)
    env_name = _MODEL_ENV_BY_PROVIDER[resolved_provider][normalized_tier]
    override = (os.environ.get(env_name) or "").strip()
    if override:
        return override
    defaults = ANTHROPIC_DEFAULT_MODELS if resolved_provider == PROVIDER_ANTHROPIC else OPENAI_DEFAULT_MODELS
    return defaults[normalized_tier]


def reasoning_effort_for_tier(tier: str, provider: str | None = None) -> str:
    resolved_provider = _normalize_provider(provider)
    normalized_tier = _normalize_tier(tier)
    resolved_model = model_for_tier(normalized_tier, resolved_provider)
    fallback = DEFAULT_REASONING_EFFORT_BY_PROVIDER[resolved_provider]
    try:
        from api.llm_settings import get_llm_reasoning_effort_setting

        effort = get_llm_reasoning_effort_setting(resolved_provider, normalized_tier)
    except Exception:
        effort = fallback
    return effort if effort in reasoning_effort_options(resolved_provider, resolved_model) else fallback


def reasoning_effort_options(provider: str, model: str | None = None) -> list[str]:
    resolved_provider = _normalize_provider(provider)
    if resolved_provider == PROVIDER_OPENAI:
        return [REASONING_NONE, REASONING_MEDIUM, REASONING_XHIGH]

    return [REASONING_NONE, REASONING_HIGH, REASONING_MAX]


def resolve_model(model: str, provider: str | None = None) -> str:
    resolved_provider = _normalize_provider(provider)
    tier = _model_to_tier(model)
    if tier is None:
        return model
    return model_for_tier(tier, resolved_provider)


def get_llm_client(provider: str | None = None, api_key: str | None = None) -> Any:
    resolved_provider = _normalize_provider(provider)
    resolved_key = api_key if api_key is not None else get_api_key(resolved_provider)
    cache_key = (resolved_provider, resolved_key)
    factory = _client_factory(resolved_provider)

    if cache_key in _CLIENT_CACHE and _CLIENT_FACTORY_CACHE.get(resolved_provider) is factory:
        return _CLIENT_CACHE[cache_key]

    with _CLIENT_LOCK:
        if cache_key in _CLIENT_CACHE and _CLIENT_FACTORY_CACHE.get(resolved_provider) is factory:
            return _CLIENT_CACHE[cache_key]

        client = factory(api_key=resolved_key) if resolved_key else factory()
        _CLIENT_FACTORY_CACHE[resolved_provider] = factory
        _CLIENT_CACHE[cache_key] = client
        return client


def extract_text(response: Any) -> str:
    output_text = _obj_get(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []

    for block in _obj_get(response, "content", []) or []:
        if _obj_get(block, "type") == "text":
            text = _obj_get(block, "text")
            if isinstance(text, str) and text:
                parts.append(text)

    for item in _obj_get(response, "output", []) or []:
        if _obj_get(item, "type") != "message":
            continue
        for block in _obj_get(item, "content", []) or []:
            text = _obj_get(block, "text")
            if isinstance(text, str) and text:
                parts.append(text)

    return "".join(parts).strip()


def extract_citations(response: Any) -> list[tuple[str, str]]:
    seen_urls: set[str] = set()
    citations: list[tuple[str, str]] = []

    def add(title: Any, url: Any) -> None:
        if not isinstance(url, str) or not url or url in seen_urls:
            return
        label = title if isinstance(title, str) and title else url
        seen_urls.add(url)
        citations.append((label, url))

    for block in _obj_get(response, "content", []) or []:
        if _obj_get(block, "type") != "text":
            continue
        for citation in _obj_get(block, "citations", []) or []:
            add(_obj_get(citation, "title"), _obj_get(citation, "url"))

    for item in _obj_get(response, "output", []) or []:
        if _obj_get(item, "type") == "message":
            for block in _obj_get(item, "content", []) or []:
                for annotation in _obj_get(block, "annotations", []) or []:
                    if _obj_get(annotation, "type") == "url_citation":
                        add(_obj_get(annotation, "title"), _obj_get(annotation, "url"))
        action = _obj_get(item, "action")
        for source in _obj_get(action, "sources", []) or []:
            add(_obj_get(source, "title"), _obj_get(source, "url"))

    for source in _obj_get(response, "sources", []) or []:
        add(_obj_get(source, "title"), _obj_get(source, "url"))

    return citations


def _prepare_text_egress(
    *,
    provider: str,
    purpose: str,
    model: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
) -> tuple[str, str | None]:
    from api.agent_governance import prepare_model_egress
    from ontology.policy import admin_actor

    sanitized, _manifest = prepare_model_egress(
        provider=provider,
        purpose=purpose,
        stream_kwargs={
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        actor=admin_actor(source="llm_utils"),
    )
    messages = sanitized.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        content = messages[0].get("content")
        if isinstance(content, str):
            prompt = content
    sanitized_system = sanitized.get("system")
    return prompt, sanitized_system if isinstance(sanitized_system, str) else None


def call_llm_text(
    *,
    prompt: str,
    model: str,
    api_key: str | None = None,
    max_tokens: int = 4096,
    system: str | None = None,
    allowed_domains: Sequence[str] | None = None,
    max_web_search_uses: int = 5,
    provider: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, list[tuple[str, str]], Any]:
    resolved_provider = _normalize_provider(provider)
    prompt, system = _prepare_text_egress(
        provider=resolved_provider,
        purpose="llm_utils.call_llm_text",
        model=model,
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
    )
    if resolved_provider == PROVIDER_ANTHROPIC:
        response = _call_anthropic_messages(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            system=system,
            allowed_domains=allowed_domains,
            max_web_search_uses=max_web_search_uses,
            reasoning_effort=reasoning_effort,
        )
    else:
        response = _call_openai_response(
            input_items=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            system=system,
            allowed_domains=allowed_domains,
            reasoning_effort=reasoning_effort,
        )
    return extract_text(response), extract_citations(response), response


def call_llm_pdf_text(
    *,
    pdf_bytes: bytes,
    prompt: str,
    model: str,
    api_key: str | None = None,
    max_tokens: int = 4096,
    system: str | None = None,
    filename: str = "document.pdf",
    provider: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, list[tuple[str, str]], Any]:
    resolved_provider = _normalize_provider(provider)
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    if resolved_provider == PROVIDER_ANTHROPIC:
        response = _call_anthropic_messages(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            system=system,
            reasoning_effort=reasoning_effort,
        )
    else:
        response = _call_openai_response(
            input_items=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": filename,
                            "file_data": f"data:application/pdf;base64,{pdf_b64}",
                        },
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            system=system,
            reasoning_effort=reasoning_effort,
        )
    return extract_text(response), extract_citations(response), response


def call_claude_text(
    *,
    prompt: str,
    model: str,
    api_key: str | None,
    max_tokens: int = 4096,
    system: str | None = None,
    allowed_domains: Sequence[str] | None = None,
    max_web_search_uses: int = 5,
    reasoning_effort: str | None = None,
) -> tuple[str, list[tuple[str, str]], Any]:
    return call_llm_text(
        prompt=prompt,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        system=system,
        allowed_domains=allowed_domains,
        max_web_search_uses=max_web_search_uses,
        reasoning_effort=reasoning_effort,
    )


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


def _call_anthropic_messages(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str | None,
    max_tokens: int,
    system: str | None = None,
    allowed_domains: Sequence[str] | None = None,
    max_web_search_uses: int = 5,
    reasoning_effort: str | None = None,
) -> Any:
    client = get_llm_client(PROVIDER_ANTHROPIC, api_key=api_key)
    resolved_model = resolve_model(model, PROVIDER_ANTHROPIC)
    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    apply_reasoning_config(
        kwargs,
        provider=PROVIDER_ANTHROPIC,
        model=resolved_model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
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
    return response


def _call_openai_response(
    *,
    input_items: list[dict[str, Any]],
    model: str,
    api_key: str | None,
    max_tokens: int,
    system: str | None = None,
    allowed_domains: Sequence[str] | None = None,
    reasoning_effort: str | None = None,
) -> Any:
    client = get_llm_client(PROVIDER_OPENAI, api_key=api_key)
    resolved_model = resolve_model(model, PROVIDER_OPENAI)
    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "input": input_items,
        "max_output_tokens": max_tokens,
    }
    apply_reasoning_config(
        kwargs,
        provider=PROVIDER_OPENAI,
        model=resolved_model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    if system:
        kwargs["instructions"] = system
    if allowed_domains is not None:
        kwargs["tools"] = [
            {
                "type": "web_search",
                "filters": {"allowed_domains": list(allowed_domains)},
                "search_context_size": "medium",
            }
        ]
        kwargs["include"] = ["web_search_call.action.sources"]
    return client.responses.create(**kwargs)


def apply_reasoning_config(
    kwargs: dict[str, Any],
    *,
    provider: str,
    model: str,
    max_tokens: int,
    reasoning_effort: str | None,
) -> None:
    resolved_provider = _normalize_provider(provider)
    effort = _normalize_reasoning_effort(reasoning_effort, provider=resolved_provider, model=model)
    if effort is None:
        return

    if resolved_provider == PROVIDER_OPENAI:
        kwargs["reasoning"] = {"effort": effort}
        return

    if _anthropic_supports_adaptive_thinking(model):
        kwargs["thinking"] = {"type": "adaptive", "display": "omitted"}
        output_config = dict(kwargs.get("output_config") or {})
        output_config["effort"] = effort
        kwargs["output_config"] = output_config
        return

    kwargs["thinking"] = {
        "type": "enabled",
        "budget_tokens": _anthropic_manual_thinking_budget(max_tokens=max_tokens, effort=effort),
        "display": "omitted",
    }
    output_config = dict(kwargs.get("output_config") or {})
    output_config["effort"] = effort
    kwargs["output_config"] = output_config


def _normalize_provider(provider: str | None) -> str:
    resolved = selected_provider() if provider is None else provider.strip().lower()
    if resolved not in PROVIDERS:
        raise ValueError("LLM provider must be 'anthropic' or 'openai'")
    return resolved


def _normalize_tier(tier: str) -> str:
    normalized = (tier or "").strip().lower()
    aliases = {
        "haiku": MODEL_LOW,
        "low": MODEL_LOW,
        "mini": MODEL_LOW,
        "sonnet": MODEL_MID,
        "mid": MODEL_MID,
        "medium": MODEL_MID,
        "opus": MODEL_HIGH,
        "high": MODEL_HIGH,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODEL_TIERS:
        raise ValueError("model tier must be 'low', 'mid', or 'high'")
    return normalized


def _normalize_reasoning_effort(
    effort: str | None,
    *,
    provider: str,
    model: str | None = None,
) -> str | None:
    if effort is None:
        return None
    normalized = effort.strip().lower()
    if normalized in {"", "off", "disabled"}:
        return None
    resolved_provider = _normalize_provider(provider)
    if resolved_provider == PROVIDER_ANTHROPIC and normalized == REASONING_NONE:
        return None
    options = OPENAI_REASONING_EFFORTS if resolved_provider == PROVIDER_OPENAI else ANTHROPIC_REASONING_EFFORTS
    if normalized not in options:
        allowed = "', '".join(options)
        raise ValueError(f"reasoning_effort must be one of '{allowed}'")
    return normalized


def _anthropic_supports_adaptive_thinking(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return any(
        marker in normalized
        for marker in (
            "claude-mythos-preview",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        )
    )


def _anthropic_manual_thinking_budget(*, max_tokens: int, effort: str) -> int:
    if max_tokens < 2048:
        raise ValueError("Anthropic manual thinking requires max_tokens >= 2048")
    cap_by_effort = {
        REASONING_HIGH: 8192,
        REASONING_MAX: 32768,
    }
    return min(cap_by_effort[effort], max(max_tokens // 2, 1024))


def _model_to_tier(model: str) -> str | None:
    normalized = (model or "").strip()
    lowered = normalized.lower()
    if lowered in MODEL_TIERS:
        return lowered
    if lowered in {"haiku", "mini"}:
        return MODEL_LOW
    if lowered in {"sonnet", "medium"}:
        return MODEL_MID
    if lowered == "opus":
        return MODEL_HIGH
    return _LEGACY_MODEL_TO_TIER.get(normalized)


def _extract_json_object(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def _client_factory(provider: str) -> Any:
    if provider == PROVIDER_ANTHROPIC:
        import anthropic

        return anthropic.Anthropic
    from openai import OpenAI

    return OpenAI
