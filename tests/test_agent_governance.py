from __future__ import annotations

import pytest

from api.agent_governance import (
    AgentBudgetExceeded,
    AgentBudgetState,
    ModelGatewayDenied,
    prepare_model_egress,
    redact_secrets,
)
from ontology.policy import admin_actor, agent_actor


def test_redact_secrets_removes_provider_keys_bearer_tokens_and_credentials():
    payload = {
        "message": "use sk-ant-1234567890abcdefghijklmnop and AIza1234567890abcdefghijklmnop and Bearer abcdefghijklmnop",
        "headers": {"authorization": "Bearer should-not-leak"},
        "note": "password = supersecret123",
    }

    redacted, findings = redact_secrets(payload)

    text = str(redacted)
    assert "sk-ant-1234567890abcdefghijklmnop" not in text
    assert "AIza1234567890abcdefghijklmnop" not in text
    assert "Bearer abcdefghijklmnop" not in text
    assert "supersecret123" not in text
    assert findings


def test_prepare_model_egress_understands_gemini_contents_and_config():
    actor = agent_actor(admin_actor())
    budget = AgentBudgetState(max_model_calls=2, max_input_tokens=10_000, max_cost_usd=10.0)

    kwargs, manifest = prepare_model_egress(
        provider="gemini",
        purpose="agent_chat",
        stream_kwargs={
            "model": "gemini-test",
            "contents": [{"role": "user", "parts": [{"text": "Analyze my portfolio positions."}]}],
            "config": {
                "max_output_tokens": 256,
                "system_instruction": "system instructions",
                "tools": [{"function_declarations": [{"name": "get_portfolio"}]}],
            },
        },
        actor=actor,
        budget=budget,
        session_id="s1",
    )

    assert kwargs["contents"][0]["parts"][0]["text"] == "Analyze my portfolio positions."
    assert manifest["provider_egress"] == "external_allowed_raw_private"
    assert manifest["decision"] == "allowed_with_warning"
    assert manifest["decision_reason"] == "private_external_egress_allowed_with_warning"
    assert manifest["data_sensitivity"] == "portfolio_private"
    assert budget.model_calls == 1


def test_prepare_model_egress_allows_private_payload_with_manifest_and_budget():
    actor = agent_actor(admin_actor())
    budget = AgentBudgetState(max_model_calls=2, max_input_tokens=10_000, max_cost_usd=10.0)

    kwargs, manifest = prepare_model_egress(
        provider="anthropic",
        purpose="agent_chat",
        stream_kwargs={
            "model": "claude-test",
            "max_tokens": 256,
            "system": "system instructions",
            "messages": [{"role": "user", "content": "Analyze my portfolio positions."}],
        },
        actor=actor,
        budget=budget,
        session_id="s1",
    )

    assert kwargs["messages"][0]["content"] == "Analyze my portfolio positions."
    assert manifest["provider_egress"] == "external_allowed_raw_private"
    assert manifest["decision"] == "allowed_with_warning"
    assert manifest["data_sensitivity"] == "portfolio_private"
    assert manifest["policy_decision_id"]
    assert budget.model_calls == 1


def test_prepare_model_egress_public_payload_is_allowed():
    _kwargs, manifest = prepare_model_egress(
        provider="openai",
        purpose="agent_chat",
        stream_kwargs={
            "model": "gpt-test",
            "max_output_tokens": 16,
            "instructions": "instructions",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "What moved the S&P 500?"}]}],
        },
        actor=agent_actor(admin_actor()),
    )

    assert manifest["provider_egress"] == "external_allowed"
    assert manifest["decision"] == "allowed"
    assert manifest["data_sensitivity"] == "public_market"


def test_explicit_denied_rule_blocks_before_budget(monkeypatch):
    from api import llm_settings

    monkeypatch.setattr(
        llm_settings,
        "get_gateway_policy_setting",
        lambda: {
            **llm_settings.default_gateway_policy(),
            "denied_rules": [
                {"provider": "anthropic", "model": "claude-test", "data_sensitivity": "portfolio_private"}
            ],
        },
    )
    budget = AgentBudgetState(max_model_calls=2)

    with pytest.raises(ModelGatewayDenied) as exc_info:
        prepare_model_egress(
            provider="anthropic",
            purpose="agent_chat",
            stream_kwargs={
                "model": "claude-test",
                "max_tokens": 256,
                "system": "system instructions",
                "messages": [{"role": "user", "content": "Analyze my portfolio positions."}],
            },
            actor=agent_actor(admin_actor()),
            budget=budget,
        )

    assert exc_info.value.manifest["decision"] == "blocked"
    assert exc_info.value.manifest["decision_reason"] == "explicit_denied_rule"
    assert budget.model_calls == 0


def test_local_only_model_egress_blocks_external_and_allows_local():
    with pytest.raises(ModelGatewayDenied) as exc_info:
        prepare_model_egress(
            provider="openai",
            purpose="agent_chat",
            stream_kwargs={
                "model": "gpt-test",
                "max_output_tokens": 16,
                "local_only_required": True,
                "instructions": "instructions",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            },
            actor=agent_actor(admin_actor()),
        )

    assert exc_info.value.manifest["decision_reason"] == "local_only_required"

    _kwargs, manifest = prepare_model_egress(
        provider="local",
        purpose="agent_chat",
        stream_kwargs={
            "model": "local-mid",
            "max_output_tokens": 16,
            "local_only_required": True,
            "instructions": "instructions",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        },
        actor=agent_actor(admin_actor()),
    )

    assert manifest["decision"] == "allowed"
    assert manifest["provider_egress"] == "local_only"
    assert manifest["local_only_required"] is True


def test_model_lifecycle_disabled_blocks_and_deprecated_warns(monkeypatch):
    from api import llm_settings

    base = llm_settings.default_gateway_policy()
    monkeypatch.setattr(
        llm_settings,
        "get_gateway_policy_setting",
        lambda: {**base, "model_lifecycle": {"gpt-test": "disabled"}},
    )
    with pytest.raises(ModelGatewayDenied) as exc_info:
        prepare_model_egress(
            provider="openai",
            purpose="agent_chat",
            stream_kwargs={"model": "gpt-test", "max_output_tokens": 16, "input": "hello"},
            actor=agent_actor(admin_actor()),
        )
    assert exc_info.value.manifest["decision_reason"] == "model_or_provider_disabled"

    monkeypatch.setattr(
        llm_settings,
        "get_gateway_policy_setting",
        lambda: {**base, "model_lifecycle": {"gpt-test": "deprecated"}},
    )
    _kwargs, manifest = prepare_model_egress(
        provider="openai",
        purpose="agent_chat",
        stream_kwargs={"model": "gpt-test", "max_output_tokens": 16, "input": "hello"},
        actor=agent_actor(admin_actor()),
    )
    assert manifest["decision"] == "allowed_with_warning"
    assert manifest["decision_reason"] == "model_or_provider_deprecated"


def test_model_budget_fails_closed_before_provider_call():
    budget = AgentBudgetState(max_model_calls=0)

    with pytest.raises(AgentBudgetExceeded):
        prepare_model_egress(
            provider="openai",
            purpose="agent_chat",
            stream_kwargs={
                "model": "gpt-test",
                "max_output_tokens": 16,
                "instructions": "instructions",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            },
            actor=agent_actor(admin_actor()),
            budget=budget,
        )
