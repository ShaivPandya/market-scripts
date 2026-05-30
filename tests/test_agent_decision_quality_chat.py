from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import api.routers.agent as agent_router
from decision_quality.gates import apply_decision_quality_gates
from decision_quality.models import DecisionQuality


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for chunk in raw.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        event_name = None
        payload = None
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
        if event_name and isinstance(payload, dict):
            events.append((event_name, payload))
    return events


def _event_text_delta(text: str):
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


class _FakeStream:
    def __init__(self, events: list[Any], final_message: Any):
        self._events = events
        self._final_message = final_message

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final_message


class _FakeStreamManager:
    def __init__(self, stream: _FakeStream):
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMessages:
    def __init__(self, streams: list[tuple[list[Any], Any]]):
        self._streams = streams
        self.calls = 0
        self.kwargs_history: list[dict[str, Any]] = []

    def stream(self, **kwargs):
        self.kwargs_history.append(dict(kwargs))
        events, final_message = self._streams[self.calls]
        self.calls += 1
        return _FakeStreamManager(_FakeStream(events, final_message))


class _FakeClient:
    def __init__(self, streams: list[tuple[list[Any], Any]]):
        self.messages = _FakeMessages(streams)


def _install_fake_anthropic(monkeypatch, text: str) -> _FakeClient:
    fake = _FakeClient(
        [
            (
                [_event_text_delta(text)],
                SimpleNamespace(
                    content=[{"type": "text", "text": text}],
                    stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=10, output_tokens=20),
                ),
            )
        ]
    )
    monkeypatch.setattr("anthropic.Anthropic", lambda *args, **kwargs: fake)
    return fake


def _decision_quality(actionability_status: str = "actionable") -> DecisionQuality:
    return DecisionQuality.model_validate(
        {
            "simple_thesis": "Meta can compound ads if AI capex translates into better monetization.",
            "opportunity_type": "quality_compounder",
            "embedded_macro_exposure": "Digital advertising cycle.",
            "mispricing": {
                "consensus_view": "The market discounts AI capex and regulatory risk.",
                "variant_view": "Ad efficiency and smart glasses can offset the investment drag.",
                "pricing_evidence": "Deck valuation frames bear/base/bull upside.",
                "why_consensus_is_wrong": "Consensus may underweight AI ad conversion gains.",
            },
            "catalyst_or_reason_now": {
                "event_or_condition": "Q2 guide and annual meeting.",
                "expected_timeframe": "Next quarter.",
                "why_now": "AI capex and monetization evidence are becoming observable.",
                "source_evidence": ["deck"],
            },
            "invalidation": {
                "observable": "AI capex ROI",
                "metric_or_event": "Capex rises without ad monetization improvement",
                "threshold": "Capex above plan and no ad price/impression improvement",
                "timeframe": "Two quarters",
                "implication": "The thesis should stay research/watch rather than add.",
            },
            "evidence_for": [{"claim": "Scale", "support": "3.56B Family DAP", "source_refs": ["deck"]}],
            "evidence_against": [
                {"claim": "Reality Labs losses", "support": "Wide losses remain", "source_refs": ["deck"]}
            ],
            "price_action_read": {
                "observed_behavior": "Current chart context is required.",
                "interpretation": "Do not add without confirming price action.",
                "confirms_thesis": None,
                "data_needed": ["current META chart"],
            },
            "actionability": {
                "status": actionability_status,
                "reason": "Needs current price action before sizing.",
                "missing_inputs": ["portfolio size", "current chart"],
            },
            "recommended_action": "buy",
            "expression": {
                "primary": "META common stock",
                "instrument_type": "equity",
                "directness": "direct",
                "alternatives": [],
                "follow_on": "Add only after catalyst confirmation.",
            },
            "conviction": {
                "level": 3,
                "max_level": 5,
                "raw_target_weight": 0.02,
                "upgrade_condition": "Ad ROI evidence.",
            },
            "confidence": 0.62,
            "confidence_reason": "Good evidence but capex and chart confirmation are unresolved.",
            "sizing_context": {
                "starting_size": "No position supplied.",
                "add_conditions": "Use a starter only after price action confirms.",
                "liquidity_constraints": "Large-cap liquid equity.",
                "portfolio_constraints": "Need current portfolio exposure.",
                "sizing_delta": {
                    "direction": "increase",
                    "amount": 0.02,
                    "unit": "portfolio_weight",
                    "basis": "target_weight",
                    "condition": "Only after catalyst and chart confirmation.",
                },
            },
            "trade_after_trade": {
                "if_right": "Add after monetization evidence.",
                "if_wrong": "Do not add and revisit capex risk.",
                "next_review_trigger": "Q2 results or annual meeting update.",
            },
        }
    )


def _dq_result(actionability_status: str = "actionable") -> dict[str, Any]:
    dq = _decision_quality(actionability_status)
    gate = apply_decision_quality_gates(dq, current_action=dq.recommended_action, recommendation_status="clear")
    return {"decision_quality": dq, "parse_errors": [], "gate": gate, "usage": {"input_tokens": 1, "output_tokens": 2}}


def test_serious_thesis_prompt_triggers_hidden_pass_and_metadata(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(agent_router, "_run_decision_quality_structured_pass", lambda **_kwargs: _dq_result())
    fake = _install_fake_anthropic(
        monkeypatch,
        "Bottom line: research it before adding. The thesis is specific, but price action and sizing matter.",
    )
    seen_tools: list[str] = []

    def fake_execute_tool(name: str, args: dict, **_kwargs):
        seen_tools.append(name)
        return json.dumps({"name": name, "args": args, "_meta": {"cache": "test"}})

    monkeypatch.setattr(agent_router, "execute_tool", fake_execute_tool)

    resp = auth_client.post(
        "/api/agent/chat",
        json={
            "message": "Here is my Meta thesis? What do you think?",
            "screen_context": {"page_name": "Research", "route": "/research/meta", "ticker": "META"},
            "finalize_synchronously": True,
        },
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    done = [payload for event, payload in parsed if event == "done"][-1]
    final_text = "".join(payload["text"] for event, payload in parsed if event == "delta")
    assert done["decision_quality_chat"]["ran"] is True
    assert done["decision_quality_chat"]["final_action"] == "buy"
    assert "run_chart" in seen_tools
    assert "get_position_valuation" in seen_tools
    assert "simple_thesis" not in final_text
    assert fake.messages.kwargs_history[0]["messages"][0]["content"]


def test_casual_prompt_does_not_trigger_hidden_pass(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        agent_router,
        "_run_decision_quality_structured_pass",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    resp = auth_client.post("/api/agent/chat", json={"message": "hey", "finalize_synchronously": True})

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    final_text = "".join(payload["text"] for event, payload in parsed if event == "delta")
    done = [payload for event, payload in parsed if event == "done"][-1]
    assert final_text.startswith("Hey")
    assert "decision_quality_chat" not in done


def test_lower_case_company_thesis_selects_price_action_tools():
    tools = agent_router._select_tool_names("Here is my Meta thesis? What do you think?")

    assert "get_thesis" in tools
    assert "run_chart" in tools
    assert "get_position_valuation" in tools
    assert agent_router._should_run_decision_quality_chat("Here is my Meta thesis? What do you think?") is True


def test_gate_downgrade_is_visible_to_synthesis_prompt(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        agent_router, "_run_decision_quality_structured_pass", lambda **_kwargs: _dq_result("missing_inputs")
    )
    fake = _install_fake_anthropic(
        monkeypatch,
        "Bottom line: watch it until the missing inputs are solved. No raw JSON here.",
    )
    monkeypatch.setattr(
        agent_router,
        "execute_tool",
        lambda name, args, **_kwargs: json.dumps({"name": name, "args": args, "_meta": {"cache": "test"}}),
    )

    resp = auth_client.post(
        "/api/agent/chat",
        json={
            "message": "Here is my Meta thesis? What do you think?",
            "screen_context": {"page_name": "Research", "route": "/research/meta", "ticker": "META"},
            "finalize_synchronously": True,
        },
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    done = [payload for event, payload in parsed if event == "done"][-1]
    synthesis_prompt = fake.messages.kwargs_history[0]["messages"][0]["content"]
    final_text = "".join(payload["text"] for event, payload in parsed if event == "delta")
    assert done["decision_quality_chat"]["gate_status"] == "downgraded"
    assert done["decision_quality_chat"]["final_action"] == "watch"
    assert '"final_action": "watch"' in synthesis_prompt
    assert "recommended_action" not in final_text
