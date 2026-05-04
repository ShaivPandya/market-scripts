"""Live app settings endpoints."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError as PydanticValidationError

from api.audit import emit_audit_event
from api.exceptions import ValidationError
from api.llm_settings import get_setting, set_llm_provider_setting, set_setting
from api.routers.auth import require_actor
from llm_utils import (
    MODEL_HIGH,
    MODEL_LOW,
    MODEL_MID,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    api_key_env,
    get_api_key,
    model_for_tier,
    require_api_key,
    selected_provider,
)
from ontology.policy import Actor

router = APIRouter()
ActorDep = Annotated[Actor, Depends(require_actor)]

Provider = Literal["anthropic", "openai"]
PreferenceLevel = Literal["less", "balanced", "more"]
Personality = Literal["friendly", "pragmatic"]
CustomInstructionText = Annotated[str, Field(max_length=2000)]
AGENT_RESPONSE_PREFERENCES_KEY = "agent.response_preferences"


class LLMSettingsUpdate(BaseModel):
    provider: Provider


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
    return "Claude" if provider == PROVIDER_ANTHROPIC else "OpenAI"


def _provider_status(provider: str) -> dict:
    return {
        "provider": provider,
        "label": _provider_label(provider),
        "configured": get_api_key(provider) is not None,
        "api_key_env": api_key_env(provider),
    }


def _settings_response() -> dict:
    provider = selected_provider()
    return {
        "provider": provider,
        "available_providers": [
            _provider_status(PROVIDER_ANTHROPIC),
            _provider_status(PROVIDER_OPENAI),
        ],
        "models": {
            MODEL_LOW: model_for_tier(MODEL_LOW, provider),
            MODEL_MID: model_for_tier(MODEL_MID, provider),
            MODEL_HIGH: model_for_tier(MODEL_HIGH, provider),
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
    set_llm_provider_setting(body.provider)
    after = _settings_response()
    emit_audit_event(
        "settings.llm_provider.updated",
        "permission",
        "succeeded",
        actor=actor,
        before_summary={"provider": before.get("provider")},
        after_summary={"provider": after.get("provider")},
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
