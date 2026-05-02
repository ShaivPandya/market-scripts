from __future__ import annotations

import pytest

import llm_utils


@pytest.fixture
def temp_llm_settings(tmp_path, monkeypatch):
    from api import llm_settings

    if llm_settings._conn is not None:
        llm_settings._conn.close()
    llm_settings._conn = None
    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setattr(llm_settings, "DB_PATH", tmp_path / "app_settings.db")

    yield llm_settings

    if llm_settings._conn is not None:
        llm_settings._conn.close()
    llm_settings._conn = None


def test_selected_provider_uses_env_fallback_without_persisted_setting(temp_llm_settings, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    assert not temp_llm_settings.DB_PATH.exists()
    assert llm_utils.selected_provider() == "openai"
    assert not temp_llm_settings.DB_PATH.exists()


def test_selected_provider_uses_persisted_setting_before_env(temp_llm_settings, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    temp_llm_settings.set_llm_provider_setting("openai")

    assert llm_utils.selected_provider() == "openai"


def test_get_llm_settings_returns_env_fallback(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = auth_client.get("/api/v1/settings/llm")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "anthropic"
    assert payload["models"]["low"] == "claude-haiku-4-5"
    anthropic = next(item for item in payload["available_providers"] if item["provider"] == "anthropic")
    openai = next(item for item in payload["available_providers"] if item["provider"] == "openai")
    assert anthropic == {
        "provider": "anthropic",
        "label": "Claude",
        "configured": True,
        "api_key_env": "ANTHROPIC_API_KEY",
    }
    assert openai["configured"] is False
    assert "sk-ant-test" not in response.text


def test_put_llm_settings_persists_provider(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    response = auth_client.put("/api/v1/settings/llm", json={"provider": "openai"})

    assert response.status_code == 200
    assert response.json()["provider"] == "openai"
    assert temp_llm_settings.get_llm_provider_setting() == "openai"


def test_put_llm_settings_rejects_invalid_provider(temp_llm_settings, auth_client):
    response = auth_client.put("/api/v1/settings/llm", json={"provider": "other"})

    assert response.status_code == 422


def test_put_llm_settings_rejects_missing_provider_key(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = auth_client.put("/api/v1/settings/llm", json={"provider": "openai"})

    assert response.status_code == 422
    assert "OPENAI_API_KEY" in response.text
