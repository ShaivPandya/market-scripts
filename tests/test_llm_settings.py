from __future__ import annotations

import json
from types import SimpleNamespace

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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    response = auth_client.get("/api/settings/llm")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "anthropic"
    assert payload["provider_mode"] == "single"
    assert payload["provider_by_tier"] == {
        "low": "anthropic",
        "mid": "anthropic",
        "high": "anthropic",
    }
    assert payload["models"]["low"] == "claude-haiku-4-5"
    assert payload["models_by_provider"]["openai"]["mid"] == "gpt-5.4"
    assert payload["models_by_provider"]["gemini"]["mid"] == "gemini-3.1-pro-preview-customtools"
    assert "local" not in payload["models_by_provider"]
    assert payload["reasoning_efforts"]["anthropic"] == {
        "low": "medium",
        "mid": "high",
        "high": "max",
    }
    assert payload["reasoning_efforts"]["gemini"] == {
        "low": "low",
        "mid": "medium",
        "high": "high",
    }
    assert [item["effort"] for item in payload["reasoning_options"]["anthropic"]["high"]] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert [item["effort"] for item in payload["reasoning_options"]["openai"]["mid"]] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    ]
    assert [item["effort"] for item in payload["reasoning_options"]["gemini"]["low"]] == [
        "minimal",
        "low",
        "medium",
        "high",
    ]
    assert [item["effort"] for item in payload["reasoning_options"]["gemini"]["mid"]] == [
        "low",
        "medium",
        "high",
    ]
    anthropic = next(item for item in payload["available_providers"] if item["provider"] == "anthropic")
    openai = next(item for item in payload["available_providers"] if item["provider"] == "openai")
    gemini = next(item for item in payload["available_providers"] if item["provider"] == "gemini")
    assert [item["provider"] for item in payload["available_providers"]] == [
        "anthropic",
        "openai",
        "gemini",
        "talisman",
    ]
    assert anthropic == {
        "provider": "anthropic",
        "label": "Claude",
        "configured": True,
        "api_key_env": "ANTHROPIC_API_KEY",
    }
    assert openai["configured"] is False
    assert gemini == {
        "provider": "gemini",
        "label": "Gemini",
        "configured": False,
        "api_key_env": "GEMINI_API_KEY",
    }
    talisman = next(item for item in payload["available_providers"] if item["provider"] == "talisman")
    assert talisman == {
        "provider": "talisman",
        "label": "Talisman",
        "configured": False,
        "api_key_env": "TALISMAN_API_KEY",
        "base_url_env": "TALISMAN_BASE_URL",
        "base_url_configured": False,
    }
    assert payload["gateway_policy"]["private_egress_mode"] == "allow_with_warning"
    assert payload["gateway_policy"]["provider_lifecycle"]["talisman"] == "draft"
    assert payload["gateway_policy"]["owned_model_rollout"]["enabled"] is False
    assert payload["gateway_policy"]["owned_model_rollout"]["candidate_provider"] == "talisman"
    assert "local" not in payload["gateway_policy"]["provider_lifecycle"]
    assert "local_provider" not in payload
    assert "sk-ant-test" not in response.text


def test_get_llm_settings_uses_bulk_settings_fetch(auth_client, monkeypatch):
    from api.routers import settings

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    calls = []

    def fake_get_settings(keys):
        calls.append(list(keys))
        return {}

    def fail_get_setting(_key):
        raise AssertionError("GET /settings/llm should not call individual get_setting")

    monkeypatch.setattr(settings, "get_settings", fake_get_settings)
    monkeypatch.setattr(settings, "get_setting", fail_get_setting)
    monkeypatch.setattr("api.llm_settings.get_setting", fail_get_setting)

    response = auth_client.get("/api/settings/llm")

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0] == [
        "llm.provider",
        "llm.provider_mode",
        "llm.provider_by_tier",
        "llm.gateway_policy",
        "llm.reasoning_effort.anthropic.low",
        "llm.reasoning_effort.anthropic.mid",
        "llm.reasoning_effort.anthropic.high",
        "llm.reasoning_effort.openai.low",
        "llm.reasoning_effort.openai.mid",
        "llm.reasoning_effort.openai.high",
        "llm.reasoning_effort.gemini.low",
        "llm.reasoning_effort.gemini.mid",
        "llm.reasoning_effort.gemini.high",
        "llm.reasoning_effort.talisman.low",
        "llm.reasoning_effort.talisman.mid",
        "llm.reasoning_effort.talisman.high",
    ]


def test_get_settings_returns_rows_without_creating_missing_sqlite_db(temp_llm_settings):
    assert not temp_llm_settings.DB_PATH.exists()

    rows = temp_llm_settings.get_settings(["llm.provider"])

    assert rows == {}
    assert not temp_llm_settings.DB_PATH.exists()


def test_get_settings_returns_unique_persisted_rows(temp_llm_settings):
    temp_llm_settings.set_setting("llm.provider", "openai")
    temp_llm_settings.set_setting("llm.reasoning_effort.openai.mid", "xhigh")

    rows = temp_llm_settings.get_settings(
        [
            "llm.provider",
            "llm.provider",
            "llm.reasoning_effort.openai.mid",
            "missing",
        ]
    )

    assert set(rows) == {"llm.provider", "llm.reasoning_effort.openai.mid"}
    assert rows["llm.provider"]["value"] == "openai"
    assert rows["llm.reasoning_effort.openai.mid"]["value"] == "xhigh"
    assert temp_llm_settings.get_settings([]) == {}


def test_put_llm_settings_persists_provider(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    response = auth_client.put(
        "/api/settings/llm",
        json={
            "provider": "openai",
            "reasoning_efforts": {
                "low": "none",
                "mid": "xhigh",
                "high": "medium",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "openai"
    assert response.json()["reasoning_efforts"]["openai"] == {
        "low": "none",
        "mid": "xhigh",
        "high": "medium",
    }
    assert temp_llm_settings.get_llm_provider_setting() == "openai"
    assert temp_llm_settings.get_llm_reasoning_effort_setting("openai", "mid") == "xhigh"


def test_put_llm_settings_persists_gemini_provider(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-12345678901234567890")

    response = auth_client.put(
        "/api/settings/llm",
        json={
            "provider": "gemini",
            "reasoning_efforts": {
                "low": "minimal",
                "mid": "high",
                "high": "medium",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "gemini"
    assert response.json()["models_by_provider"]["gemini"]["high"] == "gemini-3.1-pro-preview-customtools"
    assert response.json()["reasoning_efforts"]["gemini"] == {
        "low": "minimal",
        "mid": "high",
        "high": "medium",
    }
    assert temp_llm_settings.get_llm_provider_setting() == "gemini"
    assert temp_llm_settings.get_llm_reasoning_effort_setting("gemini", "low") == "minimal"


def test_put_llm_settings_persists_custom_provider_by_tier(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-12345678901234567890")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    response = auth_client.put(
        "/api/settings/llm",
        json={
            "provider": "gemini",
            "provider_mode": "custom",
            "provider_by_tier": {
                "low": "gemini",
                "mid": "gemini",
                "high": "openai",
            },
            "reasoning_efforts_by_provider": {
                "gemini": {
                    "low": "minimal",
                    "mid": "medium",
                    "high": "high",
                },
                "openai": {
                    "low": "none",
                    "mid": "medium",
                    "high": "xhigh",
                },
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "gemini"
    assert payload["provider_mode"] == "custom"
    assert payload["provider_by_tier"] == {
        "low": "gemini",
        "mid": "gemini",
        "high": "openai",
    }
    assert payload["models"] == {
        "low": "gemini-3.1-flash-lite",
        "mid": "gemini-3.1-pro-preview-customtools",
        "high": "gpt-5.5",
    }
    assert payload["reasoning_efforts"]["gemini"]["mid"] == "medium"
    assert payload["reasoning_efforts"]["openai"]["high"] == "xhigh"
    assert temp_llm_settings.get_llm_provider_mode_setting() == "custom"
    assert temp_llm_settings.get_llm_provider_by_tier_setting(fallback_provider="gemini") == {
        "low": "gemini",
        "mid": "gemini",
        "high": "openai",
    }


def test_custom_provider_by_tier_routes_model_helpers(temp_llm_settings, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-12345678901234567890")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    temp_llm_settings.set_llm_provider_setting("gemini")
    temp_llm_settings.set_llm_provider_mode_setting("custom")
    temp_llm_settings.set_llm_provider_by_tier_setting(
        {"low": "gemini", "mid": "gemini", "high": "openai"},
        fallback_provider="gemini",
    )

    assert llm_utils.selected_provider() == "gemini"
    assert llm_utils.selected_provider_for_tier(llm_utils.MODEL_MID) == "gemini"
    assert llm_utils.selected_provider_for_tier(llm_utils.MODEL_HIGH) == "openai"
    assert llm_utils.model_for_tier(llm_utils.MODEL_MID) == "gemini-3.1-pro-preview-customtools"
    assert llm_utils.model_for_tier(llm_utils.MODEL_HIGH) == "gpt-5.5"
    assert not llm_utils.has_llm_api_key()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm_utils.has_llm_api_key()


def test_custom_provider_by_tier_routes_call_llm_text(temp_llm_settings, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    temp_llm_settings.set_llm_provider_setting("gemini")
    temp_llm_settings.set_llm_provider_mode_setting("custom")
    temp_llm_settings.set_llm_provider_by_tier_setting(
        {"low": "gemini", "mid": "gemini", "high": "openai"},
        fallback_provider="gemini",
    )
    monkeypatch.setattr(
        llm_utils,
        "_prepare_text_egress",
        lambda **kwargs: (kwargs["prompt"], kwargs["system"]),
    )
    captured = {}

    def fake_openai_response(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text="answer")

    monkeypatch.setattr(llm_utils, "_call_openai_response", fake_openai_response)

    text, _citations, _response = llm_utils.call_llm_text(prompt="hello", model=llm_utils.MODEL_HIGH)

    assert text == "answer"
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-5.5"


def test_put_llm_settings_rejects_invalid_provider(temp_llm_settings, auth_client):
    response = auth_client.put("/api/settings/llm", json={"provider": "other"})

    assert response.status_code == 422


def test_put_llm_settings_rejects_local_provider(temp_llm_settings, auth_client):
    response = auth_client.put("/api/settings/llm", json={"provider": "local"})

    assert response.status_code == 422


def test_put_llm_settings_rejects_missing_provider_key(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = auth_client.put("/api/settings/llm", json={"provider": "openai"})

    assert response.status_code == 422
    assert "OPENAI_API_KEY" in response.text


def test_put_llm_settings_rejects_missing_gemini_key(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    response = auth_client.put("/api/settings/llm", json={"provider": "gemini"})

    assert response.status_code == 422
    assert "GEMINI_API_KEY" in response.text


def test_put_llm_settings_gateway_policy_requires_note(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    policy = temp_llm_settings.default_gateway_policy()
    policy["denied_rules"] = [
        {"provider": "anthropic", "model": "claude-test", "data_sensitivity": "portfolio_private"}
    ]

    response = auth_client.put("/api/settings/llm", json={"provider": "anthropic", "gateway_policy": policy})

    assert response.status_code == 422
    assert "gateway_note" in response.text


def test_put_llm_settings_gateway_policy_persists_with_audit(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    policy = temp_llm_settings.default_gateway_policy()
    policy["denied_rules"] = [
        {"provider": "anthropic", "model": "claude-test", "data_sensitivity": "portfolio_private"}
    ]

    response = auth_client.put(
        "/api/settings/llm",
        json={"provider": "anthropic", "gateway_policy": policy, "gateway_note": "Block this model for tests."},
    )

    assert response.status_code == 200
    assert response.json()["gateway_policy"]["denied_rules"] == policy["denied_rules"]
    assert temp_llm_settings.get_gateway_policy_setting()["denied_rules"] == policy["denied_rules"]


def test_put_llm_settings_rejects_invalid_gateway_policy(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    policy = temp_llm_settings.default_gateway_policy()
    policy["denied_rules"] = [{"provider": "unknown", "model": "*", "data_sensitivity": "portfolio_private"}]

    response = auth_client.put(
        "/api/settings/llm",
        json={"provider": "anthropic", "gateway_policy": policy, "gateway_note": "Invalid policy."},
    )

    assert response.status_code == 422


def test_get_financial_policy_matrix_returns_default(temp_llm_settings, auth_client):
    response = auth_client.get("/api/settings/financial-policy-matrix")

    assert response.status_code == 200
    payload = response.json()
    assert payload["policy"]["schema_version"] == 1
    assert payload["policy"]["rules"][0]["id"] == "default.current_checks"
    assert "max_position_weight_pct" in payload["limit_defaults"]
    assert "blocked" in payload["metadata"]["outcomes"]


def test_validate_financial_policy_matrix_reports_errors(temp_llm_settings, auth_client):
    response = auth_client.post(
        "/api/settings/financial-policy-matrix/validate",
        json={
            "policy": {
                "schema_version": 1,
                "policy_id": "bad-policy",
                "rules": [{"id": "bad", "match": {"risk_levels": ["extreme"]}}],
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert "risk_levels" in response.text


def test_put_financial_policy_matrix_requires_note(temp_llm_settings, auth_client):
    policy = auth_client.get("/api/settings/financial-policy-matrix").json()["policy"]

    response = auth_client.put("/api/settings/financial-policy-matrix", json={"policy": policy})

    assert response.status_code == 422
    assert "note" in response.text


def test_put_financial_policy_matrix_persists_with_audit(temp_llm_settings, auth_client, monkeypatch):
    from api.routers import settings

    audit_events = []
    monkeypatch.setattr(settings, "emit_audit_event", lambda *args, **kwargs: audit_events.append((args, kwargs)))
    policy = auth_client.get("/api/settings/financial-policy-matrix").json()["policy"]
    policy["rules"].append(
        {
            "id": "test.block_self_apply",
            "enabled": True,
            "priority": 100,
            "match": {"request_modes": ["self_apply"]},
            "limits": {},
            "outcome": "blocked",
            "approval_mode": None,
            "reason": "No self apply in tests.",
            "remediation": "Use proposal review.",
        }
    )

    response = auth_client.put(
        "/api/settings/financial-policy-matrix",
        json={"policy": policy, "note": "Add test self-apply guard."},
    )

    assert response.status_code == 200
    assert response.json()["policy"]["rules"][-1]["id"] == "test.block_self_apply"
    assert temp_llm_settings.get_setting("financial.policy_matrix") is not None
    assert audit_events
    assert audit_events[0][0][0] == "settings.financial_policy_matrix.updated"


def test_put_financial_policy_matrix_rejects_invalid_rule(temp_llm_settings, auth_client):
    response = auth_client.put(
        "/api/settings/financial-policy-matrix",
        json={
            "note": "Invalid rule.",
            "policy": {
                "schema_version": 1,
                "policy_id": "bad-policy",
                "rules": [{"id": "bad", "limits": {"bad_limit": 1}}],
            },
        },
    )

    assert response.status_code == 422
    assert "unsupported limit key" in response.text


def test_normalize_gateway_policy_accepts_deny_mode(temp_llm_settings):
    policy = temp_llm_settings.default_gateway_policy()
    policy["private_egress_mode"] = "deny"

    normalized = temp_llm_settings.normalize_gateway_policy(policy)

    assert normalized["private_egress_mode"] == "deny"


def test_normalize_gateway_policy_accepts_allow_with_warning_mode(temp_llm_settings):
    policy = temp_llm_settings.default_gateway_policy()
    policy["private_egress_mode"] = "allow_with_warning"

    normalized = temp_llm_settings.normalize_gateway_policy(policy)

    assert normalized["private_egress_mode"] == "allow_with_warning"


def test_normalize_gateway_policy_rejects_invalid_mode(temp_llm_settings):
    policy = temp_llm_settings.default_gateway_policy()
    policy["private_egress_mode"] = "block_everything"

    with pytest.raises(ValueError, match="private_egress_mode"):
        temp_llm_settings.normalize_gateway_policy(policy)


def test_normalize_gateway_policy_env_override_takes_precedence(temp_llm_settings, monkeypatch):
    monkeypatch.setenv("PRIVATE_EGRESS_MODE", "deny")
    policy = temp_llm_settings.default_gateway_policy()
    policy["private_egress_mode"] = "allow_with_warning"

    normalized = temp_llm_settings.normalize_gateway_policy(policy)

    assert normalized["private_egress_mode"] == "deny"


def test_put_llm_settings_rejects_unsupported_reasoning_effort(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    response = auth_client.put(
        "/api/settings/llm",
        json={
            "provider": "anthropic",
            "reasoning_efforts": {
                "low": "xhigh",
                "mid": "high",
                "high": "high",
            },
        },
    )

    assert response.status_code == 422
    assert "claude-haiku-4-5" in response.text


def test_put_llm_settings_rejects_unsupported_gemini_reasoning_effort(temp_llm_settings, auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-12345678901234567890")

    response = auth_client.put(
        "/api/settings/llm",
        json={
            "provider": "gemini",
            "reasoning_efforts": {
                "low": "minimal",
                "mid": "minimal",
                "high": "high",
            },
        },
    )

    assert response.status_code == 422
    assert "gemini-3.1-pro-preview-customtools" in response.text


def test_get_agent_response_preferences_returns_defaults(temp_llm_settings, auth_client):
    response = auth_client.get("/api/settings/agent-response-preferences")

    assert response.status_code == 200
    assert response.json() == {
        "personality": "pragmatic",
        "warmth": "less",
        "enthusiasm": "less",
        "headers_lists": "less",
        "emoji": "less",
        "fast_answers": True,
        "thinking_enabled": False,
        "custom_instructions": "",
    }
    assert not temp_llm_settings.DB_PATH.exists()


def test_put_agent_response_preferences_persists_preferences(temp_llm_settings, auth_client):
    response = auth_client.put(
        "/api/settings/agent-response-preferences",
        json={
            "thinking_enabled": True,
            "custom_instructions": "  End responses after answering. Do not ask follow-up questions.  ",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["thinking_enabled"] is True
    assert payload["custom_instructions"] == "End responses after answering. Do not ask follow-up questions."

    row = temp_llm_settings.get_setting("agent.response_preferences")
    assert row is not None
    saved = json.loads(row["value"])
    assert saved["thinking_enabled"] is True
    assert saved["custom_instructions"] == "End responses after answering. Do not ask follow-up questions."

    get_response = auth_client.get("/api/settings/agent-response-preferences")
    assert get_response.status_code == 200
    assert get_response.json()["thinking_enabled"] is True
    assert (
        get_response.json()["custom_instructions"] == "End responses after answering. Do not ask follow-up questions."
    )


def test_put_agent_response_preferences_rejects_invalid_values(temp_llm_settings, auth_client):
    response = auth_client.put(
        "/api/settings/agent-response-preferences",
        json={"personality": "other"},
    )

    assert response.status_code == 422
