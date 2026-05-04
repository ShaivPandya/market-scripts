from __future__ import annotations

import pytest

from api.agent_governance import AgentBudgetExceeded, AgentBudgetState, prepare_model_egress, redact_secrets
from ontology.policy import admin_actor, agent_actor


def test_redact_secrets_removes_provider_keys_bearer_tokens_and_credentials():
    payload = {
        "message": "use sk-ant-1234567890abcdefghijklmnop and Bearer abcdefghijklmnop",
        "headers": {"authorization": "Bearer should-not-leak"},
        "note": "password = supersecret123",
    }

    redacted, findings = redact_secrets(payload)

    text = str(redacted)
    assert "sk-ant-1234567890abcdefghijklmnop" not in text
    assert "Bearer abcdefghijklmnop" not in text
    assert "supersecret123" not in text
    assert findings


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
    assert manifest["data_sensitivity"] == "portfolio_private"
    assert manifest["policy_decision_id"]
    assert budget.model_calls == 1


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
