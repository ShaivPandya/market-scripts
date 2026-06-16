"""Tests for owned-model rollout controls (TL-92)."""

from __future__ import annotations

from api.llm_settings import default_gateway_policy, normalize_gateway_policy
from api.owned_model_rollout import (
    RolloutDecision,
    RolloutTelemetry,
    apply_canary_fallback,
    classify_task_class,
    finalize_shadow_comparison,
    map_exception_to_fallback_reason,
    normalize_owned_model_rollout,
    owned_model_force_baseline,
    owned_model_rollout_kill_switch,
    resolve_rollout_decision,
    rollout_policy_from_gateway,
)
from decision_quality.intent_router import RouteDecision


def _route(**overrides) -> RouteDecision:
    base = {
        "intent_class": "general_research",
        "run_hidden_dq": False,
        "run_opportunity_preflight": False,
        "workflow_name": None,
        "workflow_ticker": None,
        "tool_names": ["get_portfolio"],
        "confidence": 0.9,
        "source": "regex",
        "tool_pack": "default",
    }
    base.update(overrides)
    return RouteDecision(**base)


def test_normalize_owned_model_rollout_defaults():
    policy = normalize_owned_model_rollout(None)
    assert policy["enabled"] is False
    assert policy["shadow_enabled"] is True
    assert policy["candidate_provider"] == "talisman"
    assert "agent_turn" in policy["approved_task_classes"]


def test_gateway_policy_includes_owned_model_rollout():
    policy = normalize_gateway_policy(default_gateway_policy())
    assert "owned_model_rollout" in policy
    assert policy["owned_model_rollout"]["rule_version"] == "owned_model_rollout_v1"


def test_classify_task_class_paths():
    assert classify_task_class(path="portfolio_summary") == "synthesis"
    assert classify_task_class(path="decision_quality_chat") == "synthesis"
    assert (
        classify_task_class(
            route_decision=_route(run_hidden_dq=True),
            path="agent_chat",
        )
        == "structured_output"
    )


def test_resolve_rollout_disabled_by_default():
    decision, telemetry = resolve_rollout_decision(
        task_class="agent_turn",
        baseline_provider="anthropic",
        session_id="session-1",
        client_turn_id="turn-1",
        gateway_policy=default_gateway_policy(),
    )
    assert decision.mode == "off"
    assert decision.applied_provider == "anthropic"
    assert decision.fallback_reason == "rollout_disabled"
    assert telemetry.enabled is False


def test_kill_switch_and_force_baseline(monkeypatch):
    policy = default_gateway_policy()
    policy["owned_model_rollout"]["enabled"] = True
    monkeypatch.setenv("AGENT_OWNED_MODEL_ROLLOUT_KILL_SWITCH", "true")
    decision, telemetry = resolve_rollout_decision(
        task_class="agent_turn",
        baseline_provider="anthropic",
        session_id="session-1",
        gateway_policy=policy,
    )
    assert decision.mode == "off"
    assert decision.fallback_reason == "kill_switch_active"
    assert telemetry.kill_switch_active is True

    monkeypatch.delenv("AGENT_OWNED_MODEL_ROLLOUT_KILL_SWITCH", raising=False)
    monkeypatch.setenv("AGENT_OWNED_MODEL_FORCE_BASELINE", "true")
    decision, _telemetry = resolve_rollout_decision(
        task_class="agent_turn",
        baseline_provider="anthropic",
        session_id="session-1",
        gateway_policy=policy,
    )
    assert decision.fallback_reason == "force_baseline_active"


def test_task_class_not_eligible():
    policy = default_gateway_policy()
    policy["owned_model_rollout"]["enabled"] = True
    policy["owned_model_rollout"]["approved_task_classes"] = ["synthesis"]
    decision, _telemetry = resolve_rollout_decision(
        task_class="agent_turn",
        baseline_provider="anthropic",
        session_id="session-1",
        gateway_policy=policy,
    )
    assert decision.fallback_reason == "task_class_not_eligible"


def test_shadow_mode_when_candidate_missing():
    policy = default_gateway_policy()
    rollout = dict(policy["owned_model_rollout"])
    rollout.update(
        {
            "enabled": True,
            "shadow_enabled": True,
            "canary_enabled": False,
            "approved_candidate_id": "missing-candidate",
        }
    )
    policy["owned_model_rollout"] = rollout
    decision, _telemetry = resolve_rollout_decision(
        task_class="synthesis",
        baseline_provider="anthropic",
        session_id="session-1",
        gateway_policy=policy,
    )
    assert decision.mode == "off"
    assert decision.fallback_reason == "candidate_not_approved"


def test_shadow_mode_with_approved_candidate(monkeypatch, tmp_path):
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
    rollout = dict(policy["owned_model_rollout"])
    rollout.update(
        {
            "enabled": True,
            "shadow_enabled": True,
            "canary_enabled": False,
            "approved_candidate_id": "abc123",
            "approved_model_ids": [],
        }
    )
    policy["owned_model_rollout"] = rollout
    policy["provider_lifecycle"]["talisman"] = "enabled"

    decision, telemetry = resolve_rollout_decision(
        task_class="synthesis",
        baseline_provider="anthropic",
        session_id="session-1",
        client_turn_id="turn-1",
        gateway_policy=policy,
    )
    assert decision.mode == "shadow"
    assert decision.applied_provider == "anthropic"
    assert decision.candidate_provider == "talisman"
    assert decision.candidate_id == "abc123"
    assert telemetry.shadow_mode is True


def test_canary_selection_is_deterministic(monkeypatch, tmp_path):
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
    rollout = dict(policy["owned_model_rollout"])
    rollout.update(
        {
            "enabled": True,
            "shadow_enabled": False,
            "canary_enabled": True,
            "canary_percent": 100,
            "approved_candidate_id": "abc123",
        }
    )
    policy["owned_model_rollout"] = rollout
    policy["provider_lifecycle"]["talisman"] = "enabled"

    first, _ = resolve_rollout_decision(
        task_class="synthesis",
        baseline_provider="anthropic",
        session_id="session-canary",
        client_turn_id="turn-canary",
        gateway_policy=policy,
    )
    second, _ = resolve_rollout_decision(
        task_class="synthesis",
        baseline_provider="anthropic",
        session_id="session-canary",
        client_turn_id="turn-canary",
        gateway_policy=policy,
    )
    assert first.mode == "canary"
    assert first == second
    assert first.applied_provider == "talisman"


def test_lifecycle_disablement_blocks_candidate():
    policy = default_gateway_policy()
    rollout = dict(policy["owned_model_rollout"])
    rollout.update({"enabled": True, "approved_candidate_id": "abc123"})
    policy["owned_model_rollout"] = rollout
    policy["provider_lifecycle"]["talisman"] = "disabled"
    decision, _telemetry = resolve_rollout_decision(
        task_class="synthesis",
        baseline_provider="anthropic",
        session_id="session-1",
        gateway_policy=policy,
    )
    assert decision.fallback_reason in {"candidate_not_approved", "provider_lifecycle_disabled"}


def test_map_exception_to_fallback_reason():
    class FakeDenied(Exception):
        pass

    FakeDenied.__name__ = "ModelGatewayDenied"
    assert map_exception_to_fallback_reason(FakeDenied("blocked")) == "policy_denied"
    assert map_exception_to_fallback_reason(TimeoutError("timeout")) == "endpoint_timeout"
    assert map_exception_to_fallback_reason(ValueError("schema validation failed")) == "schema_failure"


def test_compare_model_outcomes_and_canary_fallback():
    baseline = {
        "provider": "anthropic",
        "model": "claude-test",
        "output_text": "hello",
        "tool_names": ["get_portfolio"],
        "latency_ms": 100,
        "usage": {"input_tokens": 10},
        "status": "ok",
    }
    candidate = {
        "provider": "talisman",
        "model": "qwen-test",
        "output_text": "hello world",
        "tool_names": ["search_web"],
        "latency_ms": 120,
        "usage": {"input_tokens": 12},
        "status": "ok",
    }
    telemetry = RolloutTelemetry()
    comparison = finalize_shadow_comparison(telemetry, baseline=baseline, candidate=candidate)
    assert comparison["output_text_match"] is False
    assert "get_portfolio" in comparison["tool_only_in_baseline"]

    decision = RolloutDecision(
        mode="canary",
        baseline_provider="anthropic",
        candidate_provider="talisman",
        applied_provider="talisman",
        task_class="synthesis",
        confidence=1.0,
        canary_selected=True,
        candidate_id="abc123",
        candidate_model="qwen-test",
        rule_version="owned_model_rollout_v1",
    )
    updated = apply_canary_fallback(decision, telemetry, fallback_reason="endpoint_failure")
    assert updated.applied_provider == "anthropic"
    assert updated.fallback_reason == "endpoint_failure"


def test_env_flag_helpers(monkeypatch):
    monkeypatch.delenv("AGENT_OWNED_MODEL_ROLLOUT_KILL_SWITCH", raising=False)
    monkeypatch.delenv("AGENT_OWNED_MODEL_FORCE_BASELINE", raising=False)
    assert owned_model_rollout_kill_switch() is False
    assert owned_model_force_baseline() is False


def test_rollout_policy_from_gateway():
    gateway = default_gateway_policy()
    assert rollout_policy_from_gateway(gateway)["enabled"] is False
