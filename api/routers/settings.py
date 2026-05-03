"""Live app settings endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.audit import emit_audit_event
from api.exceptions import ValidationError
from api.llm_settings import set_llm_provider_setting
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


class LLMSettingsUpdate(BaseModel):
    provider: Provider


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
