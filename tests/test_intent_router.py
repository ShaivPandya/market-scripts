from __future__ import annotations

import json

from decision_quality.intent_router import (
    RouteDecision,
    build_regex_route_decision,
    build_route_context,
    compare_route_decisions,
    resolve_agent_route,
    training_row_from_telemetry,
)


def _regex_baseline(user_text: str, screen_context=None) -> RouteDecision:
    import api.routers.agent as agent_router

    return build_regex_route_decision(
        user_text=user_text,
        select_tool_names=agent_router._select_tool_names,
        detect_workflow=agent_router._detect_workflow,
        should_run_hidden_dq=agent_router._should_run_decision_quality_chat,
        should_run_opportunity_preflight=agent_router._should_run_opportunity_candidate_preflight,
        screen_context=screen_context,
    )


def test_regex_baseline_lower_case_thesis_routes_hidden_dq():
    decision = _regex_baseline("what do you think about nvidia as a long?")

    assert decision.run_hidden_dq is True
    assert decision.run_opportunity_preflight is True
    assert decision.intent_class == "thesis_review"
    assert "get_thesis" in decision.tool_names
    assert "run_chart" in decision.tool_names


def test_regex_baseline_discovery_without_full_dq():
    decision = _regex_baseline("Scan semiconductors for anything interesting right now?")

    assert decision.run_hidden_dq is False
    assert decision.run_opportunity_preflight is True
    assert decision.intent_class == "opportunity_discovery"


def test_regex_baseline_general_research_skips_hidden_passes():
    decision = _regex_baseline("How is the yield curve shaping macro risk right now?")

    assert decision.run_hidden_dq is False
    assert decision.run_opportunity_preflight is False
    assert "get_yield_curve" in decision.tool_names


def test_resolve_agent_route_disabled_uses_regex(monkeypatch):
    monkeypatch.delenv("AGENT_INTENT_ROUTER_ENABLED", raising=False)
    baseline = _regex_baseline("what do you think about nvidia as a long?")
    context = build_route_context(user_text="what do you think about nvidia as a long?")

    effective, meta = resolve_agent_route(context=context, regex_baseline=baseline)

    assert effective.source == "regex"
    assert meta["enabled"] is False
    assert effective.run_hidden_dq is True


def test_resolve_agent_route_falls_back_when_llm_unavailable(monkeypatch):
    monkeypatch.setenv("AGENT_INTENT_ROUTER_ENABLED", "true")
    monkeypatch.setenv("AGENT_INTENT_ROUTER_SHADOW_MODE", "false")
    baseline = _regex_baseline("what do you think about nvidia as a long?")
    context = build_route_context(user_text="what do you think about nvidia as a long?")

    effective, meta = resolve_agent_route(
        context=context,
        regex_baseline=baseline,
        provider="anthropic",
        api_key="test",
        system_prompt="route",
    )

    assert effective.source == "regex"
    assert meta.get("fallback_reason") == "llm_parse_or_call_failed"


def test_resolve_agent_route_shadow_mode_keeps_regex(monkeypatch):
    monkeypatch.setenv("AGENT_INTENT_ROUTER_ENABLED", "true")
    monkeypatch.setenv("AGENT_INTENT_ROUTER_SHADOW_MODE", "true")

    baseline = _regex_baseline("Scan semiconductors for anything interesting right now?")
    context = build_route_context(user_text="Scan semiconductors for anything interesting right now?")
    llm_candidate = RouteDecision(
        intent_class="thesis_review",
        run_hidden_dq=True,
        run_opportunity_preflight=False,
        workflow_name=None,
        workflow_ticker=None,
        tool_names=["get_thesis"],
        confidence=0.95,
        source="llm",
        tool_pack="thesis_review",
    )

    def fake_llm(**_kwargs):
        return llm_candidate

    import decision_quality.intent_router as intent_router

    monkeypatch.setattr(intent_router, "run_llm_route_decision", fake_llm)

    effective, meta = resolve_agent_route(
        context=context,
        regex_baseline=baseline,
        provider="anthropic",
        api_key="test",
        system_prompt="route",
    )

    assert effective.source == "regex"
    assert meta["shadow_mode"] is True
    assert meta["shadow_comparison"]["hidden_dq_match"] is False


def test_resolve_agent_route_applies_llm_when_confident(monkeypatch):
    monkeypatch.setenv("AGENT_INTENT_ROUTER_ENABLED", "true")
    monkeypatch.setenv("AGENT_INTENT_ROUTER_SHADOW_MODE", "false")
    monkeypatch.setenv("AGENT_INTENT_ROUTER_CONFIDENCE_THRESHOLD", "0.70")

    baseline = _regex_baseline("How is the yield curve shaping macro risk right now?")
    context = build_route_context(user_text="How is the yield curve shaping macro risk right now?")
    llm_candidate = RouteDecision(
        intent_class="portfolio_query",
        run_hidden_dq=False,
        run_opportunity_preflight=False,
        workflow_name=None,
        workflow_ticker=None,
        tool_names=["get_yield_curve", "search_agent_capabilities"],
        confidence=0.91,
        source="llm",
        tool_pack="portfolio_query",
    )

    import decision_quality.intent_router as intent_router

    monkeypatch.setattr(intent_router, "run_llm_route_decision", lambda **_kwargs: llm_candidate)

    effective, meta = resolve_agent_route(
        context=context,
        regex_baseline=baseline,
        provider="anthropic",
        api_key="test",
        system_prompt="route",
    )

    assert effective.source == "llm"
    assert meta["applied_source"] == "llm"
    assert effective.intent_class == "portfolio_query"


def test_safety_floor_keeps_hidden_dq_when_regex_requires(monkeypatch):
    monkeypatch.setenv("AGENT_INTENT_ROUTER_ENABLED", "true")
    monkeypatch.setenv("AGENT_INTENT_ROUTER_SHADOW_MODE", "false")

    baseline = _regex_baseline("Should I buy NVDA here? What do you think?")
    context = build_route_context(user_text="Should I buy NVDA here? What do you think?")
    llm_candidate = RouteDecision(
        intent_class="general_research",
        run_hidden_dq=False,
        run_opportunity_preflight=False,
        workflow_name=None,
        workflow_ticker=None,
        tool_names=["get_signal_aggregator"],
        confidence=0.95,
        source="llm",
        tool_pack="general_research",
    )

    import decision_quality.intent_router as intent_router

    def _fake_llm(**kwargs):
        return intent_router._enforce_safety_floor(
            llm_candidate,
            regex_baseline=kwargs["regex_baseline"],
            user_text=kwargs["context"].user_text,
        )

    monkeypatch.setattr(intent_router, "run_llm_route_decision", _fake_llm)

    effective, _meta = resolve_agent_route(
        context=context,
        regex_baseline=baseline,
        provider="anthropic",
        api_key="test",
        system_prompt="route",
    )

    assert effective.run_hidden_dq is True


def test_compare_route_decisions_and_training_row():
    applied = _regex_baseline("Scan semiconductors for anything interesting right now?")
    candidate = RouteDecision(
        intent_class="thesis_review",
        run_hidden_dq=True,
        run_opportunity_preflight=False,
        workflow_name=None,
        workflow_ticker=None,
        tool_names=["get_thesis"],
        confidence=0.88,
        source="llm",
    )
    comparison = compare_route_decisions(applied=applied, candidate=candidate)
    assert comparison["hidden_dq_match"] is False

    row = training_row_from_telemetry(
        user_text="Scan semiconductors",
        route_meta={
            "regex_baseline": applied.to_meta(),
            "llm_candidate": candidate.to_meta(),
            "shadow_comparison": comparison,
            "applied_source": "regex_shadow",
        },
        session_id="sess-1",
    )
    assert row["session_id"] == "sess-1"
    assert row["schema_version"] == 1
    assert json.loads(json.dumps(row))["applied_source"] == "regex_shadow"


def test_routing_expectation_scoring_unit():
    from decision_quality.chat_eval_runner import AgentChatRun, ChatEvalCase, _routing_expectation_checks

    case = ChatEvalCase(
        path=__file__,  # type: ignore[arg-type]
        data={
            "routing_expectations": {
                "intent_class": "thesis_review",
                "run_hidden_dq": True,
                "required_tool_names": ["get_thesis"],
            }
        },
    )
    checks: list[dict] = []

    run = AgentChatRun(
        final_text="answer",
        events=[],
        tool_names=["get_thesis"],
        done_payload={
            "intent_router": {
                "applied": {
                    "intent_class": "thesis_review",
                    "run_hidden_dq": True,
                    "tool_names": ["get_thesis", "run_chart"],
                }
            }
        },
    )
    _routing_expectation_checks(checks, case=case, run=run)
    assert all(item["passed"] for item in checks)


def test_routing_expectation_scoring_flags_missing_meta():
    from decision_quality.chat_eval_runner import AgentChatRun, ChatEvalCase, _routing_expectation_checks

    case = ChatEvalCase(
        path=__file__,  # type: ignore[arg-type]
        data={"routing_expectations": {"intent_class": "thesis_review"}},
    )
    checks: list[dict] = []
    run = AgentChatRun(final_text="", events=[], tool_names=[], done_payload={})
    _routing_expectation_checks(checks, case=case, run=run)
    assert checks[0]["name"] == "intent_router_meta_present"
    assert checks[0]["passed"] is False
