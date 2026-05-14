from __future__ import annotations

import base64
import copy
import importlib
import os
import threading
from collections.abc import Sequence
from typing import Any

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDERS = {PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_GEMINI}

MODEL_LOW = "low"
MODEL_MID = "mid"
MODEL_HIGH = "high"
MODEL_TIERS = {MODEL_LOW, MODEL_MID, MODEL_HIGH}

REASONING_MEDIUM = "medium"
REASONING_HIGH = "high"
REASONING_NONE = "none"
REASONING_MINIMAL = "minimal"
REASONING_LOW = "low"
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
GEMINI_FLASH_REASONING_EFFORTS = (
    REASONING_MINIMAL,
    REASONING_LOW,
    REASONING_MEDIUM,
    REASONING_HIGH,
)
GEMINI_PRO_REASONING_EFFORTS = (
    REASONING_LOW,
    REASONING_MEDIUM,
    REASONING_HIGH,
)
REASONING_EFFORTS = (
    set(OPENAI_REASONING_EFFORTS) | set(ANTHROPIC_REASONING_EFFORTS) | set(GEMINI_FLASH_REASONING_EFFORTS)
)
DEFAULT_REASONING_EFFORT_BY_PROVIDER_TIER = {
    PROVIDER_ANTHROPIC: {
        MODEL_LOW: REASONING_HIGH,
        MODEL_MID: REASONING_HIGH,
        MODEL_HIGH: REASONING_HIGH,
    },
    PROVIDER_OPENAI: {
        MODEL_LOW: REASONING_MEDIUM,
        MODEL_MID: REASONING_MEDIUM,
        MODEL_HIGH: REASONING_MEDIUM,
    },
    PROVIDER_GEMINI: {
        MODEL_LOW: REASONING_MINIMAL,
        MODEL_MID: REASONING_HIGH,
        MODEL_HIGH: REASONING_HIGH,
    },
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
GEMINI_DEFAULT_MODELS = {
    MODEL_LOW: "gemini-3.1-flash-lite",
    MODEL_MID: "gemini-3.1-pro-preview-customtools",
    MODEL_HIGH: "gemini-3.1-pro-preview-customtools",
}
DEFAULT_MODELS_BY_PROVIDER = {
    PROVIDER_ANTHROPIC: ANTHROPIC_DEFAULT_MODELS,
    PROVIDER_OPENAI: OPENAI_DEFAULT_MODELS,
    PROVIDER_GEMINI: GEMINI_DEFAULT_MODELS,
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
    "gemini-3.1-flash-lite": MODEL_LOW,
    "gemini-3.1-pro-preview": MODEL_HIGH,
    "gemini-3.1-pro-preview-customtools": MODEL_HIGH,
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
    PROVIDER_GEMINI: {
        MODEL_LOW: "GEMINI_MODEL_LOW",
        MODEL_MID: "GEMINI_MODEL_MID",
        MODEL_HIGH: "GEMINI_MODEL_HIGH",
    },
}
_API_KEY_ENV_BY_PROVIDER = {
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_GEMINI: "GEMINI_API_KEY",
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
        raise ValueError("LLM_PROVIDER must be 'anthropic', 'openai', or 'gemini'")
    return provider


def api_key_env(provider: str | None = None) -> str:
    return _API_KEY_ENV_BY_PROVIDER[_normalize_provider(provider)]


def get_api_key(provider: str | None = None) -> str | None:
    value = (os.environ.get(api_key_env(provider)) or "").strip().strip("\"'")
    return value or None


def has_llm_api_key(provider: str | None = None) -> bool:
    resolved_provider = _normalize_provider(provider)
    return get_api_key(resolved_provider) is not None


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
    return DEFAULT_MODELS_BY_PROVIDER[resolved_provider][normalized_tier]


def reasoning_effort_for_tier(tier: str, provider: str | None = None) -> str:
    resolved_provider = _normalize_provider(provider)
    normalized_tier = _normalize_tier(tier)
    resolved_model = model_for_tier(normalized_tier, resolved_provider)
    fallback = default_reasoning_effort(resolved_provider, normalized_tier)
    try:
        from api.llm_settings import get_llm_reasoning_effort_setting

        effort = get_llm_reasoning_effort_setting(resolved_provider, normalized_tier)
    except Exception:
        effort = fallback
    options = reasoning_effort_options(resolved_provider, resolved_model)
    if fallback not in options:
        fallback = REASONING_HIGH if REASONING_HIGH in options else options[0]
    return effort if effort in options else fallback


def reasoning_effort_options(provider: str, model: str | None = None) -> list[str]:
    resolved_provider = _normalize_provider(provider)
    if resolved_provider == PROVIDER_OPENAI:
        return [REASONING_NONE, REASONING_MEDIUM, REASONING_XHIGH]
    if resolved_provider == PROVIDER_GEMINI:
        return list(_gemini_reasoning_efforts_for_model(model))

    return [REASONING_NONE, REASONING_HIGH, REASONING_MAX]


def default_reasoning_effort(provider: str, tier: str) -> str:
    resolved_provider = _normalize_provider(provider)
    normalized_tier = _normalize_tier(tier)
    return DEFAULT_REASONING_EFFORT_BY_PROVIDER_TIER[resolved_provider][normalized_tier]


def resolve_model(model: str, provider: str | None = None) -> str:
    resolved_provider = _normalize_provider(provider)
    tier = _model_to_tier(model)
    if tier is None:
        return model
    return model_for_tier(tier, resolved_provider)


def get_llm_client(provider: str | None = None, api_key: str | None = None) -> Any:
    resolved_provider = _normalize_provider(provider)
    resolved_key = api_key if api_key is not None else get_api_key(resolved_provider)
    cache_key: tuple[str, str | None] = (resolved_provider, resolved_key)
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

    gemini_text = _obj_get(response, "text")
    if isinstance(gemini_text, str) and gemini_text.strip():
        return gemini_text.strip()

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

    for block in _gemini_response_parts(response):
        if block.get("thought") is True:
            continue
        text = block.get("text")
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

    for candidate in _obj_get(response, "candidates", []) or []:
        metadata = _obj_get(candidate, "grounding_metadata") or _obj_get(candidate, "groundingMetadata")
        for chunk in _obj_get(metadata, "grounding_chunks", _obj_get(metadata, "groundingChunks", [])) or []:
            web = _obj_get(chunk, "web")
            add(_obj_get(web, "title"), _obj_get(web, "uri"))

    return citations


def _gemini_content(role: str, parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": role, "parts": parts}


def _gemini_response_parts(response: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for candidate in _obj_get(response, "candidates", []) or []:
        content = _obj_get(candidate, "content")
        for part in _obj_get(content, "parts", []) or []:
            serialized = _serialize_gemini_part(part)
            if serialized:
                parts.append(serialized)
    return parts


def _serialize_gemini_part(part: Any) -> dict[str, Any]:
    if isinstance(part, dict):
        return dict(part)

    to_json_dict = getattr(part, "to_json_dict", None)
    if callable(to_json_dict):
        try:
            value = to_json_dict()
            if isinstance(value, dict):
                return value
        except Exception:
            pass

    out: dict[str, Any] = {}
    text = _obj_get(part, "text")
    if isinstance(text, str) and text:
        out["text"] = text
    thought = _obj_get(part, "thought")
    if isinstance(thought, bool):
        out["thought"] = thought
    function_call = _obj_get(part, "function_call") or _obj_get(part, "functionCall")
    if function_call:
        out["function_call"] = _serialize_gemini_function_call(function_call)
    function_response = _obj_get(part, "function_response") or _obj_get(part, "functionResponse")
    if function_response:
        out["function_response"] = function_response
    return out


def _serialize_gemini_function_call(function_call: Any) -> dict[str, Any]:
    if isinstance(function_call, dict):
        return dict(function_call)
    to_json_dict = getattr(function_call, "to_json_dict", None)
    if callable(to_json_dict):
        try:
            value = to_json_dict()
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {
        key: value
        for key, value in {
            "id": _obj_get(function_call, "id"),
            "name": _obj_get(function_call, "name"),
            "args": _obj_get(function_call, "args"),
        }.items()
        if value is not None
    }


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
    enable_web_search: bool | None = None,
    max_web_search_uses: int = 5,
    provider: str | None = None,
    reasoning_effort: str | None = None,
    json_schema: dict[str, Any] | None = None,
    json_schema_name: str | None = None,
) -> tuple[str, list[tuple[str, str]], Any]:
    resolved_provider = _normalize_provider(provider)
    web_search_enabled = _web_search_enabled(
        enable_web_search=enable_web_search,
        allowed_domains=allowed_domains,
    )
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
            enable_web_search=web_search_enabled,
            max_web_search_uses=max_web_search_uses,
            reasoning_effort=reasoning_effort,
        )
    elif resolved_provider == PROVIDER_OPENAI:
        response = _call_openai_response(
            input_items=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            system=system,
            enable_web_search=web_search_enabled,
            provider=resolved_provider,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
            json_schema_name=json_schema_name,
        )
    else:
        response = _call_gemini_generate_content(
            contents=[_gemini_content("user", [{"text": prompt}])],
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            system=system,
            enable_web_search=web_search_enabled,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
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
    elif resolved_provider == PROVIDER_OPENAI:
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
            provider=resolved_provider,
            reasoning_effort=reasoning_effort,
        )
    else:
        response = _call_gemini_generate_content(
            contents=[
                _gemini_content(
                    "user",
                    [
                        {"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}},
                        {"text": prompt},
                    ],
                )
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
    enable_web_search: bool | None = None,
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
        enable_web_search=enable_web_search,
        max_web_search_uses=max_web_search_uses,
        provider=PROVIDER_ANTHROPIC,
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


def _gemini_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON Schema into the smaller schema subset accepted by Gemini."""

    root_defs = schema.get("$defs") if isinstance(schema.get("$defs"), dict) else {}

    def resolve_ref(ref: str) -> dict[str, Any] | None:
        prefix = "#/$defs/"
        if not ref.startswith(prefix):
            return None
        value = root_defs.get(ref.removeprefix(prefix))
        return copy.deepcopy(value) if isinstance(value, dict) else None

    def convert(value: Any) -> Any:
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value

        current: dict[str, Any] = value
        ref = current.get("$ref")
        if isinstance(ref, str):
            resolved = resolve_ref(ref)
            if resolved is not None:
                siblings = {key: val for key, val in current.items() if key != "$ref"}
                resolved.update(siblings)
                current = resolved

        converted: dict[str, Any] = {}
        nullable = False
        for key, item in current.items():
            if key in {"$defs", "$schema", "additionalProperties", "default", "examples", "title"}:
                continue
            if key in {"anyOf", "oneOf"} and isinstance(item, list):
                non_null = [candidate for candidate in item if not (isinstance(candidate, dict) and candidate.get("type") == "null")]
                if len(non_null) == 1 and len(non_null) != len(item):
                    nested = convert(non_null[0])
                    if isinstance(nested, dict):
                        converted.update(nested)
                        nullable = True
                    continue
            if key == "type" and isinstance(item, list):
                non_null_types = [type_name for type_name in item if type_name != "null"]
                if len(non_null_types) == 1:
                    converted[key] = non_null_types[0]
                    nullable = True
                    continue
            converted[key] = convert(item)
        if nullable:
            converted["nullable"] = True
        return converted

    converted = convert(schema)
    return converted if isinstance(converted, dict) else {}


def _call_anthropic_messages(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str | None,
    max_tokens: int,
    system: str | None = None,
    enable_web_search: bool = False,
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
    if enable_web_search:
        kwargs["tools"] = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": max_web_search_uses,
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
    enable_web_search: bool = False,
    provider: str = PROVIDER_OPENAI,
    reasoning_effort: str | None = None,
    json_schema: dict[str, Any] | None = None,
    json_schema_name: str | None = None,
) -> Any:
    resolved_provider = _normalize_provider(provider)
    client = get_llm_client(resolved_provider, api_key=api_key)
    resolved_model = resolve_model(model, resolved_provider)
    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "input": input_items,
        "max_output_tokens": max_tokens,
    }
    apply_reasoning_config(
        kwargs,
        provider=resolved_provider,
        model=resolved_model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    if system:
        kwargs["instructions"] = system
    if json_schema:
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": json_schema_name or "structured_output",
                "schema": json_schema,
                "strict": True,
            }
        }
    if enable_web_search and resolved_provider == PROVIDER_OPENAI:
        kwargs["tools"] = [
            {
                "type": "web_search",
                "search_context_size": "medium",
            }
        ]
        kwargs["include"] = ["web_search_call.action.sources"]
    elif enable_web_search:
        raise RuntimeError("This LLM provider does not support managed web search in llm_utils")
    return client.responses.create(**kwargs)


def _call_gemini_generate_content(
    *,
    contents: list[dict[str, Any]],
    model: str,
    api_key: str | None,
    max_tokens: int,
    system: str | None = None,
    enable_web_search: bool = False,
    reasoning_effort: str | None = None,
    json_schema: dict[str, Any] | None = None,
) -> Any:
    client = get_llm_client(PROVIDER_GEMINI, api_key=api_key)
    resolved_model = resolve_model(model, PROVIDER_GEMINI)
    config: dict[str, Any] = {"max_output_tokens": max_tokens}
    if system:
        config["system_instruction"] = system
    if json_schema:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = _gemini_response_schema(json_schema)
    if enable_web_search:
        config["tools"] = [{"google_search": {}}]
    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "contents": contents,
        "config": config,
    }
    apply_reasoning_config(
        kwargs,
        provider=PROVIDER_GEMINI,
        model=resolved_model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    return client.models.generate_content(**kwargs)


def _web_search_enabled(
    *,
    enable_web_search: bool | None,
    allowed_domains: Sequence[str] | None,
) -> bool:
    if enable_web_search is not None:
        return enable_web_search
    return allowed_domains is not None


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

    if resolved_provider == PROVIDER_GEMINI:
        config = dict(kwargs.get("config") or {})
        config["thinking_config"] = {"thinking_level": effort}
        kwargs["config"] = config
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
        raise ValueError("LLM provider must be 'anthropic', 'openai', or 'gemini'")
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
    options: Sequence[str]
    if resolved_provider == PROVIDER_OPENAI:
        options = OPENAI_REASONING_EFFORTS
    elif resolved_provider == PROVIDER_GEMINI:
        if normalized == REASONING_NONE:
            return None
        options = tuple(reasoning_effort_options(resolved_provider, model))
    else:
        options = ANTHROPIC_REASONING_EFFORTS
    if normalized not in options:
        allowed = "', '".join(options)
        raise ValueError(f"reasoning_effort must be one of '{allowed}'")
    return normalized


def _gemini_reasoning_efforts_for_model(model: str | None = None) -> tuple[str, ...]:
    normalized = (model or "").strip().lower()
    if "flash" in normalized:
        return GEMINI_FLASH_REASONING_EFFORTS
    return GEMINI_PRO_REASONING_EFFORTS


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
    if provider == PROVIDER_GEMINI:
        return importlib.import_module("google.genai").Client
    from openai import OpenAI

    return OpenAI
