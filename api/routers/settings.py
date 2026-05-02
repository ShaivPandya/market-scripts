"""Live app settings endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from api.exceptions import ValidationError
from api.llm_settings import set_llm_provider_setting
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

router = APIRouter()

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
def update_llm_settings(body: LLMSettingsUpdate):
    try:
        require_api_key(body.provider)
    except RuntimeError as exc:
        raise ValidationError(str(exc)) from exc

    set_llm_provider_setting(body.provider)
    return _settings_response()
