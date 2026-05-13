"""Live app settings endpoints."""

from __future__ import annotations

import json
import os
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError as PydanticValidationError

from api.audit import emit_audit_event
from api.exceptions import ValidationError
from api.llm_settings import (
    ALLOWED_LLM_PROVIDERS,
    LLM_GATEWAY_POLICY_KEY,
    LLM_PROVIDER_KEY,
    REASONING_EFFORTS,
    _reasoning_key,
    get_gateway_policy_setting,
    get_setting,
    get_settings,
    normalize_gateway_policy,
    set_gateway_policy_setting,
    set_llm_provider_setting,
    set_llm_reasoning_effort_settings,
    set_setting,
)
from api.routers.auth import require_actor
from llm_utils import (
    LOCAL_LLM_BASE_URL_ENV,
    MODEL_HIGH,
    MODEL_LOW,
    MODEL_MID,
    PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI,
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    api_key_env,
    default_reasoning_effort,
    get_api_key,
    get_local_base_url,
    model_for_tier,
    reasoning_effort_options,
    require_api_key,
)
from ontology.policy import Actor

router = APIRouter()
ActorDep = Annotated[Actor, Depends(require_actor)]

Provider = Literal["anthropic", "openai", "gemini", "local"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
PreferenceLevel = Literal["less", "balanced", "more"]
Personality = Literal["friendly", "pragmatic"]
LifecycleState = Literal["draft", "enabled", "deprecated", "disabled"]
DataSensitivity = Literal[
    "public_market",
    "portfolio_private",
    "research_private",
    "account_private",
    "operational_private",
]
CustomInstructionText = Annotated[str, Field(max_length=2000)]
AGENT_RESPONSE_PREFERENCES_KEY = "agent.response_preferences"


class ReasoningEffortSettings(BaseModel):
    low: ReasoningEffort
    mid: ReasoningEffort
    high: ReasoningEffort


class GatewayDeniedRule(BaseModel):
    provider: str = "*"
    model: str = "*"
    data_sensitivity: DataSensitivity

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        normalized = str(value or "*").strip().lower()
        if normalized != "*" and normalized not in ALLOWED_LLM_PROVIDERS:
            raise ValueError("provider must be '*', 'anthropic', 'openai', 'gemini', or 'local'")
        return normalized

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        normalized = str(value or "*").strip()
        if not normalized:
            raise ValueError("model cannot be empty")
        return normalized


class GatewayPolicySettings(BaseModel):
    private_egress_mode: Literal["allow_with_warning"] = "allow_with_warning"
    provider_lifecycle: dict[Provider, LifecycleState]
    model_lifecycle: dict[str, LifecycleState] = Field(default_factory=dict)
    denied_rules: list[GatewayDeniedRule] = Field(default_factory=list)

    @field_validator("model_lifecycle")
    @classmethod
    def _validate_model_lifecycle(cls, value: dict[str, LifecycleState]) -> dict[str, LifecycleState]:
        out: dict[str, LifecycleState] = {}
        for model, state in value.items():
            key = str(model or "").strip()
            if not key:
                raise ValueError("model_lifecycle keys cannot be empty")
            out[key] = state
        return out


class LLMSettingsUpdate(BaseModel):
    provider: Provider
    reasoning_efforts: ReasoningEffortSettings | None = None
    gateway_policy: GatewayPolicySettings | None = None
    gateway_note: str | None = Field(default=None, max_length=2000)


class AgentResponsePreferencesSettings(BaseModel):
    personality: Personality = "pragmatic"
    warmth: PreferenceLevel = "less"
    enthusiasm: PreferenceLevel = "less"
    headers_lists: PreferenceLevel = "less"
    emoji: PreferenceLevel = "less"
    fast_answers: bool = True
    thinking_enabled: bool = False
    custom_instructions: CustomInstructionText | None = ""

    @field_validator("custom_instructions")
    @classmethod
    def _normalize_custom_instructions(cls, value: str | None) -> str:
        return (value or "").strip()


def _provider_label(provider: str) -> str:
    return {
        PROVIDER_ANTHROPIC: "Claude",
        PROVIDER_OPENAI: "OpenAI",
        PROVIDER_GEMINI: "Gemini",
        PROVIDER_LOCAL: "Local",
    }.get(provider, provider.title())


def _provider_status(provider: str) -> dict:
    configured = get_local_base_url() is not None if provider == PROVIDER_LOCAL else get_api_key(provider) is not None
    status = {
        "provider": provider,
        "label": _provider_label(provider),
        "configured": configured,
        "api_key_env": api_key_env(provider),
    }
    if provider == PROVIDER_LOCAL:
        status["base_url_env"] = LOCAL_LLM_BASE_URL_ENV
        status["base_url_configured"] = get_local_base_url() is not None
    return status


def _models_for_provider(provider: str) -> dict:
    return {
        MODEL_LOW: model_for_tier(MODEL_LOW, provider),
        MODEL_MID: model_for_tier(MODEL_MID, provider),
        MODEL_HIGH: model_for_tier(MODEL_HIGH, provider),
    }


def _reasoning_label(effort: str) -> str:
    return {
        "none": "None",
        "minimal": "Minimal",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "xhigh": "XHigh",
        "max": "Max",
    }[effort]


def _reasoning_options_for_provider(provider: str) -> dict:
    models = _models_for_provider(provider)
    return {
        tier: [
            {"effort": effort, "label": _reasoning_label(effort)}
            for effort in reasoning_effort_options(provider, model)
        ]
        for tier, model in models.items()
    }


def _validate_reasoning_efforts(provider: str, efforts: dict[str, str]) -> None:
    models = _models_for_provider(provider)
    for tier, effort in efforts.items():
        allowed = reasoning_effort_options(provider, models[tier])
        if effort not in allowed:
            allowed_list = ", ".join(allowed)
            raise ValidationError(
                f"{effort} is not supported for {models[tier]} reasoning effort. Use one of: {allowed_list}."
            )


def _llm_settings_keys() -> list[str]:
    providers = (PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_LOCAL)
    tiers = (MODEL_LOW, MODEL_MID, MODEL_HIGH)
    return [
        LLM_PROVIDER_KEY,
        LLM_GATEWAY_POLICY_KEY,
        *(_reasoning_key(provider, tier) for provider in providers for tier in tiers),
    ]


def _provider_from_settings(rows: dict[str, dict]) -> str:
    stored = str(rows.get(LLM_PROVIDER_KEY, {}).get("value") or "").strip().lower()
    if stored in ALLOWED_LLM_PROVIDERS:
        return stored

    provider = (os.environ.get("LLM_PROVIDER") or PROVIDER_ANTHROPIC).strip().lower()
    if provider not in ALLOWED_LLM_PROVIDERS:
        raise ValueError("LLM_PROVIDER must be 'anthropic', 'openai', 'gemini', or 'local'")
    return provider


def _reasoning_effort_from_settings(rows: dict[str, dict], provider: str, tier: str, model: str) -> str:
    fallback = default_reasoning_effort(provider, tier)
    key = _reasoning_key(provider, tier)
    effort = str(rows.get(key, {}).get("value") or "").strip().lower()
    if effort not in REASONING_EFFORTS:
        effort = fallback
    options = reasoning_effort_options(provider, model)
    if fallback not in options:
        fallback = "high" if "high" in options else options[0]
    return effort if effort in options else fallback


def _settings_response() -> dict:
    rows = get_settings(_llm_settings_keys())
    provider = _provider_from_settings(rows)
    models_by_provider = {
        PROVIDER_ANTHROPIC: _models_for_provider(PROVIDER_ANTHROPIC),
        PROVIDER_OPENAI: _models_for_provider(PROVIDER_OPENAI),
        PROVIDER_GEMINI: _models_for_provider(PROVIDER_GEMINI),
        PROVIDER_LOCAL: _models_for_provider(PROVIDER_LOCAL),
    }
    gateway_policy = get_gateway_policy_setting(rows)
    return {
        "provider": provider,
        "available_providers": [
            _provider_status(PROVIDER_ANTHROPIC),
            _provider_status(PROVIDER_OPENAI),
            _provider_status(PROVIDER_GEMINI),
            _provider_status(PROVIDER_LOCAL),
        ],
        "models": models_by_provider[provider],
        "models_by_provider": models_by_provider,
        "reasoning_efforts": {
            PROVIDER_ANTHROPIC: {
                tier: _reasoning_effort_from_settings(
                    rows, PROVIDER_ANTHROPIC, tier, models_by_provider[PROVIDER_ANTHROPIC][tier]
                )
                for tier in (MODEL_LOW, MODEL_MID, MODEL_HIGH)
            },
            PROVIDER_OPENAI: {
                tier: _reasoning_effort_from_settings(
                    rows, PROVIDER_OPENAI, tier, models_by_provider[PROVIDER_OPENAI][tier]
                )
                for tier in (MODEL_LOW, MODEL_MID, MODEL_HIGH)
            },
            PROVIDER_GEMINI: {
                tier: _reasoning_effort_from_settings(
                    rows, PROVIDER_GEMINI, tier, models_by_provider[PROVIDER_GEMINI][tier]
                )
                for tier in (MODEL_LOW, MODEL_MID, MODEL_HIGH)
            },
            PROVIDER_LOCAL: {
                tier: _reasoning_effort_from_settings(
                    rows, PROVIDER_LOCAL, tier, models_by_provider[PROVIDER_LOCAL][tier]
                )
                for tier in (MODEL_LOW, MODEL_MID, MODEL_HIGH)
            },
        },
        "reasoning_options": {
            PROVIDER_ANTHROPIC: _reasoning_options_for_provider(PROVIDER_ANTHROPIC),
            PROVIDER_OPENAI: _reasoning_options_for_provider(PROVIDER_OPENAI),
            PROVIDER_GEMINI: _reasoning_options_for_provider(PROVIDER_GEMINI),
            PROVIDER_LOCAL: _reasoning_options_for_provider(PROVIDER_LOCAL),
        },
        "gateway_policy": gateway_policy,
        "local_provider": {
            "configured": get_local_base_url() is not None,
            "base_url_env": LOCAL_LLM_BASE_URL_ENV,
            "api_key_env": api_key_env(PROVIDER_LOCAL),
        },
    }


def _agent_response_preferences_response() -> dict:
    row = get_setting(AGENT_RESPONSE_PREFERENCES_KEY)
    if not row:
        return AgentResponsePreferencesSettings().model_dump()

    try:
        raw = json.loads(str(row.get("value") or "{}"))
        return AgentResponsePreferencesSettings.model_validate(raw).model_dump()
    except (TypeError, json.JSONDecodeError, PydanticValidationError):
        return AgentResponsePreferencesSettings().model_dump()


def _agent_response_preferences_audit_summary(prefs: dict) -> dict:
    custom = str(prefs.get("custom_instructions") or "")
    return {
        "personality": prefs.get("personality"),
        "warmth": prefs.get("warmth"),
        "enthusiasm": prefs.get("enthusiasm"),
        "headers_lists": prefs.get("headers_lists"),
        "emoji": prefs.get("emoji"),
        "fast_answers": prefs.get("fast_answers"),
        "thinking_enabled": prefs.get("thinking_enabled"),
        "custom_instructions_present": bool(custom),
        "custom_instructions_length": len(custom),
    }


@router.get("/settings/llm")
def get_llm_settings():
    return _settings_response()


@router.put("/settings/llm")
def update_llm_settings(body: LLMSettingsUpdate, actor: ActorDep):
    try:
        require_api_key(body.provider)
    except RuntimeError as exc:
        raise ValidationError(str(exc)) from exc

    before = _settings_response()
    if body.reasoning_efforts is not None:
        _validate_reasoning_efforts(body.provider, body.reasoning_efforts.model_dump())
    gateway_policy_changed = False
    normalized_gateway_policy: dict | None = None
    if body.gateway_policy is not None:
        try:
            normalized_gateway_policy = normalize_gateway_policy(body.gateway_policy.model_dump())
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        gateway_policy_changed = normalized_gateway_policy != before.get("gateway_policy")
        if gateway_policy_changed and not str(body.gateway_note or "").strip():
            raise ValidationError("gateway_note is required when changing gateway policy.")
    set_llm_provider_setting(body.provider)
    if body.reasoning_efforts is not None:
        set_llm_reasoning_effort_settings(body.provider, body.reasoning_efforts.model_dump())
    if normalized_gateway_policy is not None:
        set_gateway_policy_setting(normalized_gateway_policy)
    after = _settings_response()
    emit_audit_event(
        "settings.llm_provider.updated",
        "permission",
        "succeeded",
        actor=actor,
        before_summary={
            "provider": before.get("provider"),
            "reasoning_efforts": before.get("reasoning_efforts", {}).get(body.provider),
        },
        after_summary={
            "provider": after.get("provider"),
            "reasoning_efforts": after.get("reasoning_efforts", {}).get(body.provider),
        },
    )
    if gateway_policy_changed:
        emit_audit_event(
            "settings.model_gateway_policy.updated",
            "permission",
            "succeeded",
            actor=actor,
            before_summary=before.get("gateway_policy"),
            after_summary=after.get("gateway_policy"),
            metadata={"note": str(body.gateway_note or "").strip()},
        )
    return after


@router.get("/settings/agent-response-preferences")
def get_agent_response_preferences():
    return _agent_response_preferences_response()


@router.put("/settings/agent-response-preferences")
def update_agent_response_preferences(body: AgentResponsePreferencesSettings, actor: ActorDep):
    before = _agent_response_preferences_response()
    after = AgentResponsePreferencesSettings.model_validate(body.model_dump()).model_dump()
    set_setting(AGENT_RESPONSE_PREFERENCES_KEY, json.dumps(after, separators=(",", ":")))
    emit_audit_event(
        "settings.agent_response_preferences.updated",
        "permission",
        "succeeded",
        actor=actor,
        before_summary=_agent_response_preferences_audit_summary(before),
        after_summary=_agent_response_preferences_audit_summary(after),
    )
    return after
