"""Integration-style tests for agent owned-model rollout wiring (TL-92)."""

from __future__ import annotations

from api.llm_settings import default_gateway_policy
from api.owned_model_rollout import resolve_rollout_decision
from api.routers import agent as agent_router
from decision_quality.intent_router import RouteDecision


def _route() -> RouteDecision:
    return RouteDecision(
        intent_class="general_research",
        run_hidden_dq=False,
        run_opportunity_preflight=False,
        workflow_name=None,
        workflow_ticker=None,
        tool_names=["get_portfolio"],
        confidence=0.9,
        source="regex",
        tool_pack="default",
    )


def test_resolve_owned_model_rollout_helper():
    decision, telemetry = agent_router._resolve_owned_model_rollout(
        route_decision=_route(),
        baseline_provider="anthropic",
        session_id="session-1",
        client_turn_id="turn-1",
        path="agent_chat",
        confidence=0.9,
    )
    assert decision.mode == "off"
    assert telemetry.enabled is False


def test_apply_owned_model_rollout_provider_keeps_baseline_in_shadow(monkeypatch, tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        """
        {
          "registry_version": 1,
          "active_candidate_id": "abc123",
          "candidates": {
            "abc123": {
              "candidate_id": "abc123",
              "lifecycle_state": "approved",
              "artifact_path": "data/test"
            }
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("TALISMAN_MODEL_MID", "qwen-test")
    monkeypatch.setenv("TALISMAN_MODEL_REGISTRY_PATH", str(registry_path))
    policy = default_gateway_policy()
    policy["owned_model_rollout"].update(
        {
            "enabled": True,
            "shadow_enabled": True,
            "canary_enabled": False,
            "approved_candidate_id": "abc123",
        }
    )
    policy["provider_lifecycle"]["talisman"] = "enabled"

    decision, _telemetry = resolve_rollout_decision(
        task_class="agent_turn",
        baseline_provider="anthropic",
        session_id="session-1",
        gateway_policy=policy,
    )
    provider, _api_key, _client, _conversation = agent_router._apply_owned_model_rollout_provider(
        rollout_decision=decision,
        baseline_provider="anthropic",
        api_key="sk-ant-test",
        raw_conversation=[{"role": "user", "content": "hello"}],
    )
    assert decision.mode == "shadow"
    assert provider == "anthropic"


def test_canary_fallback_helper_switches_back_to_baseline():
    from api.owned_model_rollout import RolloutDecision, RolloutTelemetry

    decision = RolloutDecision(
        mode="canary",
        baseline_provider="anthropic",
        candidate_provider="talisman",
        applied_provider="talisman",
        task_class="agent_turn",
        confidence=1.0,
        canary_selected=True,
        candidate_id="abc123",
        candidate_model="qwen-test",
        rule_version="owned_model_rollout_v1",
    )
    telemetry = RolloutTelemetry()
    updated, provider, _api_key, _client, _conversation = agent_router._owned_model_apply_fallback(
        rollout_decision=decision,
        rollout_telemetry=telemetry,
        fallback_reason="endpoint_failure",
        baseline_provider="anthropic",
        api_key="sk-ant-test",
        raw_conversation=[{"role": "user", "content": "hello"}],
    )
    assert updated.applied_provider == "anthropic"
    assert updated.fallback_reason == "endpoint_failure"
    assert provider == "anthropic"


def test_owned_rollout_done_payload_shape():
    from api.owned_model_rollout import RolloutDecision, RolloutTelemetry

    decision = RolloutDecision(
        mode="off",
        baseline_provider="anthropic",
        candidate_provider="talisman",
        applied_provider="anthropic",
        task_class="agent_turn",
        confidence=1.0,
        canary_selected=False,
        candidate_id=None,
        candidate_model=None,
        rule_version="owned_model_rollout_v1",
        fallback_reason="rollout_disabled",
    )
    telemetry = RolloutTelemetry()
    payload = agent_router._owned_rollout_done_payload(
        rollout_decision=decision,
        rollout_telemetry=telemetry,
        timings={"models": [], "tools": []},
    )
    assert payload["owned_model_rollout"]["applied"]["mode"] == "off"
    assert payload["owned_model_rollout"]["reporting"]["fallback_count"] == 1
