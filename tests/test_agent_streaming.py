from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import api.agent_chat_worker as agent_chat_worker
import api.routers.agent as agent_router
from api.agent_domain_policy import DOMAIN_CLARIFY_RESPONSE, AgentDomainClassification


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


def test_agent_delta_flush_uses_env_coalescing_and_force(monkeypatch):
    events: list[tuple[str, str, dict]] = []
    times = iter([0.10, 0.20, 0.30])

    monkeypatch.setenv("AGENT_DELTA_FLUSH_INTERVAL_MS", "500")
    monkeypatch.setenv("AGENT_DELTA_FLUSH_BYTES", "1024")
    monkeypatch.setattr(agent_chat_worker.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        agent_chat_worker,
        "append_job_event",
        lambda job_id, event_type, payload: events.append((job_id, event_type, payload)),
    )

    state = {"last_delta_flush": 0.0}
    buffer = ["small"]
    agent_chat_worker._append_agent_delta("job-1", buffer, state=state)
    assert events == []
    assert buffer == ["small"]

    buffer.append("x" * 1024)
    agent_chat_worker._append_agent_delta("job-1", buffer, state=state)
    assert events == [("job-1", "delta", {"text": "small" + ("x" * 1024)})]
    assert buffer == []

    buffer.append("tail")
    agent_chat_worker._append_agent_delta("job-1", buffer, force=True, state=state)
    assert events[-1] == ("job-1", "delta", {"text": "tail"})
    assert buffer == []


def _event_tool_use_start(name: str, call_id: str):
    return SimpleNamespace(
        type="content_block_start",
        content_block=SimpleNamespace(type="tool_use", name=name, id=call_id),
    )


class _FakeStream:
    def __init__(self, events: list[Any], final_message: Any):
        self._events = events
        self._final_message = final_message

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final_message

    def get_final_response(self):
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
        if self.calls >= len(self._streams):
            raise AssertionError("Unexpected extra messages.stream() call")
        events, final_message = self._streams[self.calls]
        self.calls += 1
        return _FakeStreamManager(_FakeStream(events, final_message))


class _FakeClient:
    def __init__(self, streams: list[tuple[list[Any], Any]]):
        self.messages = _FakeMessages(streams)


def _install_fake_anthropic(monkeypatch, streams: list[tuple[list[Any], Any]]):
    fake_client = _FakeClient(streams)
    monkeypatch.setattr("anthropic.Anthropic", lambda *args, **kwargs: fake_client)
    return fake_client


def _openai_event_text_delta(text: str):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _openai_event_function_call(name: str, call_id: str):
    return SimpleNamespace(
        type="response.output_item.added",
        item=SimpleNamespace(type="function_call", name=name, call_id=call_id),
    )


class _FakeResponses:
    def __init__(self, streams: list[tuple[list[Any], Any]]):
        self._streams = streams
        self.calls = 0
        self.kwargs_history: list[dict[str, Any]] = []

    def stream(self, **kwargs):
        self.kwargs_history.append(dict(kwargs))
        if self.calls >= len(self._streams):
            raise AssertionError("Unexpected extra responses.stream() call")
        events, final_response = self._streams[self.calls]
        self.calls += 1
        return _FakeStreamManager(_FakeStream(events, final_response))


class _FakeOpenAIClient:
    def __init__(self, streams: list[tuple[list[Any], Any]]):
        self.responses = _FakeResponses(streams)


def _install_fake_openai(monkeypatch, streams: list[tuple[list[Any], Any]]):
    fake_client = _FakeOpenAIClient(streams)
    monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: fake_client)
    return fake_client


def _gemini_chunk(parts: list[Any], usage: Any | None = None):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
        usage_metadata=usage,
    )


def _gemini_text_part(text: str):
    return SimpleNamespace(text=text)


def _gemini_function_call_part(name: str, call_id: str, args: dict):
    return SimpleNamespace(function_call=SimpleNamespace(name=name, id=call_id, args=args))


class _FakeGeminiModels:
    def __init__(self, streams: list[list[Any]]):
        self._streams = streams
        self.calls = 0
        self.kwargs_history: list[dict[str, Any]] = []

    def generate_content_stream(self, **kwargs):
        self.kwargs_history.append(dict(kwargs))
        if self.calls >= len(self._streams):
            raise AssertionError("Unexpected extra generate_content_stream() call")
        stream = self._streams[self.calls]
        self.calls += 1
        return iter(stream)


class _FakeGeminiClient:
    def __init__(self, streams: list[list[Any]]):
        self.models = _FakeGeminiModels(streams)


def _install_fake_gemini_agent(monkeypatch, streams: list[list[Any]]):
    fake_client = _FakeGeminiClient(streams)
    monkeypatch.setattr(agent_router, "get_llm_client", lambda _provider, api_key=None: fake_client)
    return fake_client


class _RaiseInStreamMessages:
    def stream(self, **_kwargs):
        raise RuntimeError(
            "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', "
            "'message': 'invalid x-api-key'}}"
        )


class _RaiseInStreamClient:
    def __init__(self):
        self.messages = _RaiseInStreamMessages()


def test_agent_stream_tracks_args_per_call_id(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")

    streams = [
        (
            [
                _event_tool_use_start("query_ontology", "call-1"),
                _event_tool_use_start("query_ontology", "call-2"),
            ],
            SimpleNamespace(
                content=[
                    {"type": "tool_use", "name": "query_ontology", "id": "call-1", "input": {"query": "A"}},
                    {"type": "tool_use", "name": "query_ontology", "id": "call-2", "input": {"query": "B"}},
                ],
                stop_reason="tool_use",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            ),
        ),
        (
            [_event_text_delta("analysis")],
            SimpleNamespace(
                content=[{"type": "text", "text": "analysis"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    fake_client = _install_fake_anthropic(monkeypatch, streams)

    seen_args: list[dict] = []

    def fake_execute_tool(_name: str, args: dict):
        seen_args.append(args)
        return json.dumps({"ok": True})

    monkeypatch.setattr(agent_router, "execute_tool", fake_execute_tool)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "portfolio test"},
    )

    assert resp.status_code == 200
    assert fake_client.messages.kwargs_history[0].get("tool_choice") == {"type": "any"}
    assert "thinking" not in fake_client.messages.kwargs_history[0]
    assert "tool_choice" not in fake_client.messages.kwargs_history[1]
    assert seen_args == [{"query": "A"}, {"query": "B"}]
    parsed = _parse_sse(resp.text)
    tool_results = [p for e, p in parsed if e == "tool_result"]
    assert len(tool_results) == 2
    assert all(p["status"] == "ok" for p in tool_results)


def test_agent_stream_openai_function_call_roundtrip(auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")

    streams = [
        (
            [_openai_event_function_call("query_ontology", "call-1")],
            SimpleNamespace(
                output=[
                    {
                        "type": "function_call",
                        "name": "query_ontology",
                        "call_id": "call-1",
                        "arguments": json.dumps({"query": "A"}),
                    }
                ],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            ),
        ),
        (
            [_openai_event_text_delta("analysis")],
            SimpleNamespace(
                output=[{"type": "message", "content": [{"type": "output_text", "text": "analysis"}]}],
                usage=SimpleNamespace(input_tokens=2, output_tokens=3),
            ),
        ),
    ]
    fake_client = _install_fake_openai(monkeypatch, streams)

    seen_args: list[dict] = []

    def fake_execute_tool(_name: str, args: dict):
        seen_args.append(args)
        return json.dumps({"ok": True})

    monkeypatch.setattr(agent_router, "execute_tool", fake_execute_tool)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "portfolio test"},
    )

    assert resp.status_code == 200
    assert fake_client.responses.kwargs_history[0]["tool_choice"] == "required"
    assert "reasoning" not in fake_client.responses.kwargs_history[0]
    assert "tool_choice" not in fake_client.responses.kwargs_history[1]
    assert fake_client.responses.kwargs_history[0]["tools"][0]["type"] == "function"
    assert fake_client.responses.kwargs_history[1]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": json.dumps({"ok": True}),
    }
    assert seen_args == [{"query": "A"}]
    parsed = _parse_sse(resp.text)
    assert any(e == "tool_call" and p["name"] == "query_ontology" for e, p in parsed)
    assert any(e == "tool_result" and p["status"] == "ok" for e, p in parsed)
    assert any(e == "delta" and p["text"] == "analysis" for e, p in parsed)


def test_agent_stream_gemini_function_call_roundtrip(auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-12345678901234567890")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")

    streams = [
        [
            _gemini_chunk(
                [_gemini_function_call_part("query_ontology", "call-1", {"query": "A"})],
                usage=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
            )
        ],
        [
            _gemini_chunk(
                [_gemini_text_part("analysis")],
                usage=SimpleNamespace(prompt_token_count=2, candidates_token_count=3),
            )
        ],
    ]
    fake_client = _install_fake_gemini_agent(monkeypatch, streams)

    seen_args: list[dict] = []

    def fake_execute_tool(_name: str, args: dict):
        seen_args.append(args)
        return json.dumps({"ok": True})

    monkeypatch.setattr(agent_router, "execute_tool", fake_execute_tool)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "portfolio test"},
    )

    assert resp.status_code == 200
    first_config = fake_client.models.kwargs_history[0]["config"]
    second_config = fake_client.models.kwargs_history[1]["config"]
    assert first_config["tool_config"]["function_calling_config"]["mode"] == "ANY"
    assert second_config["tool_config"]["function_calling_config"]["mode"] == "AUTO"
    assert first_config["tools"][0]["function_declarations"][0]["name"]
    assert fake_client.models.kwargs_history[0]["contents"] == [{"role": "user", "parts": [{"text": "portfolio test"}]}]
    assert fake_client.models.kwargs_history[1]["contents"][-1] == {
        "role": "tool",
        "parts": [{"function_response": {"name": "query_ontology", "response": {"result": json.dumps({"ok": True})}}}],
    }
    assert seen_args == [{"query": "A"}]
    parsed = _parse_sse(resp.text)
    assert any(e == "tool_call" and p["name"] == "query_ontology" for e, p in parsed)
    assert any(e == "tool_result" and p["status"] == "ok" for e, p in parsed)
    assert any(e == "delta" and p["text"] == "analysis" for e, p in parsed)


def test_agent_gemini_v2_conversation_conversion_and_kwargs():
    conversation = agent_router._gemini_conversation_from_context(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )

    assert conversation == [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"text": "hi"}]},
    ]

    kwargs = agent_router._model_stream_kwargs(
        provider=agent_router.PROVIDER_GEMINI,
        instructions="instructions",
        conversation=conversation,
        max_tokens=123,
        tool_defs=[{"name": "get_portfolio", "parameters": {"type": "object"}}],
        force_tool_use=True,
        reasoning_effort="high",
    )

    assert kwargs["model"] == "gemini-3.1-pro-preview-customtools"
    assert kwargs["contents"] == conversation
    assert kwargs["config"]["max_output_tokens"] == 123
    assert kwargs["config"]["system_instruction"] == "instructions"
    assert kwargs["config"]["thinking_config"] == {"thinking_level": "high"}
    assert kwargs["config"]["tool_config"]["function_calling_config"]["mode"] == "ANY"


def test_agent_stream_openai_thinking_keeps_required_tool_choice(auth_client, monkeypatch):
    from api import llm_settings

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    llm_settings.set_llm_reasoning_effort_settings(
        "openai",
        {
            "low": "none",
            "mid": "xhigh",
            "high": "medium",
        },
    )

    streams = [
        (
            [_openai_event_function_call("query_ontology", "call-1")],
            SimpleNamespace(
                output=[
                    {
                        "type": "function_call",
                        "name": "query_ontology",
                        "call_id": "call-1",
                        "arguments": json.dumps({"query": "A"}),
                    }
                ],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            ),
        ),
        (
            [_openai_event_text_delta("analysis")],
            SimpleNamespace(
                output=[{"type": "message", "content": [{"type": "output_text", "text": "analysis"}]}],
                usage=SimpleNamespace(input_tokens=2, output_tokens=3),
            ),
        ),
    ]
    fake_client = _install_fake_openai(monkeypatch, streams)
    monkeypatch.setattr(agent_router, "execute_tool", lambda _name, _args: json.dumps({"ok": True}))

    resp = auth_client.post(
        "/api/agent/chat",
        json={
            "messages": [{"role": "user", "content": "portfolio test"}],
            "response_preferences": {"thinking_enabled": True},
        },
    )

    assert resp.status_code == 200
    assert fake_client.responses.kwargs_history[0]["reasoning"] == {"effort": "xhigh"}
    assert fake_client.responses.kwargs_history[0]["tool_choice"] == "required"
    assert fake_client.responses.kwargs_history[1]["reasoning"] == {"effort": "xhigh"}


def test_agent_stream_anthropic_thinking_relaxes_forced_tool_choice(auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")

    streams = [
        (
            [_event_text_delta("analysis")],
            SimpleNamespace(
                content=[{"type": "text", "text": "analysis"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    fake_client = _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={
            "messages": [{"role": "user", "content": "portfolio test"}],
            "response_preferences": {"thinking_enabled": True},
        },
    )

    assert resp.status_code == 200
    assert fake_client.messages.kwargs_history[0]["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert fake_client.messages.kwargs_history[0]["output_config"] == {"effort": "high"}
    assert "tool_choice" not in fake_client.messages.kwargs_history[0]
    assert fake_client.messages.kwargs_history[0]["tools"]


def test_agent_stream_marks_tool_result_error(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")

    streams = [
        (
            [_event_tool_use_start("query_ontology", "call-1")],
            SimpleNamespace(
                content=[{"type": "tool_use", "name": "query_ontology", "id": "call-1", "input": {}}],
                stop_reason="tool_use",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            ),
        ),
        (
            [_event_text_delta("analysis")],
            SimpleNamespace(
                content=[{"type": "text", "text": "analysis"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    _install_fake_anthropic(monkeypatch, streams)
    monkeypatch.setattr(agent_router, "execute_tool", lambda _name, _args: json.dumps({"error": "boom"}))

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "portfolio test"},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    tool_results = [p for e, p in parsed if e == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["status"] == "error"
    assert "boom" in tool_results[0]["message"]


def test_agent_stream_synthesizes_when_tool_loop_limit_is_reached(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(agent_router, "execute_tool", lambda _name, _args: json.dumps({"ok": True}))

    streams: list[tuple[list[Any], Any]] = []
    for i in range(agent_router.MAX_TOOL_CONTINUATION_ROUNDS):
        call_id = f"call-{i}"
        streams.append(
            (
                [_event_tool_use_start("query_ontology", call_id)],
                SimpleNamespace(
                    content=[{"type": "tool_use", "name": "query_ontology", "id": call_id, "input": {}}],
                    stop_reason="tool_use",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                ),
            )
        )
    streams.append(
        (
            [_event_text_delta("I gathered the available data and stopped calling tools.")],
            SimpleNamespace(
                content=[{"type": "text", "text": "I gathered the available data and stopped calling tools."}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=8),
            ),
        )
    )

    fake_client = _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "portfolio test"},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    assert not any(e == "error" and "loop limit" in str(p.get("message", "")).lower() for e, p in parsed)
    assert any(e == "delta" and "stopped calling tools" in str(p.get("text", "")) for e, p in parsed)
    assert any(e == "done" for e, _p in parsed)
    assert fake_client.messages.calls == agent_router.MAX_TOOL_CONTINUATION_ROUNDS + 1
    assert "tools" not in fake_client.messages.kwargs_history[-1]


def test_agent_stream_auth_error_is_user_friendly(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr("anthropic.Anthropic", lambda *args, **kwargs: _RaiseInStreamClient())

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "portfolio test"},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    errors = [p for e, p in parsed if e == "error"]
    assert len(errors) == 1
    assert "set a valid anthropic api key" in str(errors[0].get("message", "")).lower()
    assert any(e == "done" for e, _p in parsed)


def test_agent_chat_rejects_non_anthropic_key(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-proj-not-anthropic")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "portfolio test"},
    )

    assert resp.status_code == 503
    assert "must be an anthropic key" in str(resp.json()).lower()


def test_agent_stream_skips_forced_tools_for_casual_prompt(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")

    streams = [
        (
            [_event_text_delta("hi there")],
            SimpleNamespace(
                content=[{"type": "text", "text": "hi there"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    fake_client = _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "hello"},
    )

    assert resp.status_code == 200
    assert fake_client.messages.calls == 0
    parsed = _parse_sse(resp.text)
    assert any(e == "done" for e, _p in parsed)


def test_agent_stream_dedupes_identical_tool_calls(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")

    streams = [
        (
            [
                _event_tool_use_start("query_ontology", "call-1"),
                _event_tool_use_start("query_ontology", "call-2"),
            ],
            SimpleNamespace(
                content=[
                    {"type": "tool_use", "name": "query_ontology", "id": "call-1", "input": {"query": "A"}},
                    {"type": "tool_use", "name": "query_ontology", "id": "call-2", "input": {"query": "A"}},
                ],
                stop_reason="tool_use",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            ),
        ),
        (
            [_event_text_delta("analysis")],
            SimpleNamespace(
                content=[{"type": "text", "text": "analysis"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    _install_fake_anthropic(monkeypatch, streams)

    call_count = 0

    def fake_execute_tool(_name: str, _args: dict):
        nonlocal call_count
        call_count += 1
        return json.dumps({"ok": True})

    monkeypatch.setattr(agent_router, "execute_tool", fake_execute_tool)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "portfolio test"},
    )

    assert resp.status_code == 200
    assert call_count == 1
    parsed = _parse_sse(resp.text)
    tool_results = [p for e, p in parsed if e == "tool_result"]
    assert len(tool_results) == 2
    assert all(p["status"] == "ok" for p in tool_results)


def test_agent_stream_handles_sentiment_quality_failure_without_tool_error(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")

    streams = [
        (
            [_event_tool_use_start("get_sentiment", "call-1")],
            SimpleNamespace(
                content=[{"type": "tool_use", "name": "get_sentiment", "id": "call-1", "input": {}}],
                stop_reason="tool_use",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            ),
        ),
        (
            [_event_text_delta("Sentiment section unavailable due to data quality checks.")],
            SimpleNamespace(
                content=[{"type": "text", "text": "Sentiment section unavailable due to data quality checks."}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    _install_fake_anthropic(monkeypatch, streams)
    monkeypatch.setattr(
        agent_router,
        "execute_tool",
        lambda _name, _args: json.dumps(
            {
                "as_of": "2026-03-08",
                "quality": {
                    "ok": False,
                    "mode": "fail_closed",
                    "allow_sentiment_conclusion": False,
                    "issues": ["AAII feed stale"],
                },
            }
        ),
    )

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "How is sentiment?"},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    tool_results = [p for e, p in parsed if e == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["status"] == "ok"
    assert any(e == "done" for e, _p in parsed)


def test_agent_chat_sends_initial_ping_and_disables_gzip(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: ([{"role": "user", "content": new_user_message}], "session-1"),
    )
    monkeypatch.setattr("api.memory_manager.finalize_turn_async", lambda *_args, **_kwargs: None)

    streams = [
        (
            [_event_text_delta("hi there")],
            SimpleNamespace(
                content=[{"type": "text", "text": "hi there"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "How is liquidity?"},
    )

    assert resp.status_code == 200
    assert resp.headers.get("content-encoding") == "identity"
    assert "no-transform" in resp.headers.get("cache-control", "")
    parsed = _parse_sse(resp.text)
    assert parsed[0][0] == "ping"
    assert any(e == "phase" and p.get("phase") == "model_thinking" for e, p in parsed)
    assert any(e == "delta" and p.get("text") == "hi there" for e, p in parsed)
    done_events = [p for e, p in parsed if e == "done"]
    assert done_events[-1]["session_id"] == "session-1"
    assert done_events[-1]["timings"]["total_ms"] >= 0
    assert done_events[-1]["timings"]["models"][0]["phase"] == "model_thinking"
    assert "agent instructions" not in json.dumps(done_events[-1]["timings"])


def test_agent_chat_fast_paths_simple_portfolio_summary(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "session-portfolio",
        ),
    )
    finalized: list[dict] = []
    monkeypatch.setattr(
        "api.memory_manager.finalize_turn_async",
        lambda _sid, _user_msg, assistant_msg: finalized.append(assistant_msg),
    )

    streams = [
        (
            [_event_text_delta("Portfolio summary")],
            SimpleNamespace(
                content=[{"type": "text", "text": "Portfolio summary"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    fake_client = _install_fake_anthropic(monkeypatch, streams)
    seen_tools: list[tuple[str, dict]] = []

    def fake_execute_tool(name: str, args: dict, **_kwargs):
        seen_tools.append((name, args))
        return json.dumps(
            {
                "summary": {"position_count": 1},
                "_meta": {"duration_ms": 12.3, "cache": "hit", "status": "ok"},
            }
        )

    monkeypatch.setattr(agent_router, "execute_tool", fake_execute_tool)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "Summarize my portfolio's performance"},
    )

    assert resp.status_code == 200
    assert seen_tools == [("get_portfolio", {})]
    assert fake_client.messages.calls == 1
    assert "tools" not in fake_client.messages.kwargs_history[0]
    assert fake_client.messages.kwargs_history[0]["max_tokens"] == agent_router.PORTFOLIO_SUMMARY_MAX_TOKENS
    parsed = _parse_sse(resp.text)
    assert any(
        e == "phase" and p.get("phase") == "tool_running" and p.get("label") == "Reading portfolio..."
        for e, p in parsed
    )
    assert any(e == "phase" and p.get("phase") == "model_writing" for e, p in parsed)
    assert any(
        e == "tool_result" and p.get("name") == "get_portfolio" and p.get("elapsed_ms") == 12.3 for e, p in parsed
    )
    done_events = [p for e, p in parsed if e == "done"]
    assert done_events[-1]["tools_used"] == ["get_portfolio"]
    assert done_events[-1]["timings"]["tools"][0]["cache"] == "hit"
    assert done_events[-1]["timings"]["models"][0]["purpose"] == "portfolio_summary_synthesis"
    assert finalized[0]["content"] == "Portfolio summary"


def test_agent_chat_portfolio_summary_openai_backfills_final_text(auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "session-portfolio-openai",
        ),
    )
    finalized: list[dict] = []
    monkeypatch.setattr(
        "api.memory_manager.finalize_turn_async",
        lambda _sid, _user_msg, assistant_msg: finalized.append(assistant_msg),
    )

    streams = [
        (
            [],
            SimpleNamespace(
                output=[{"type": "message", "content": [{"type": "output_text", "text": "Portfolio summary"}]}],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            ),
        ),
    ]
    fake_client = _install_fake_openai(monkeypatch, streams)
    monkeypatch.setattr(
        agent_router,
        "execute_tool",
        lambda _name, _args, **_kwargs: json.dumps(
            {
                "summary": {"position_count": 1, "long_count": 1, "short_count": 0},
                "positions": [{"ticker": "MU", "monthly_contribution_pct": 1.2}],
                "_meta": {"duration_ms": 10.0, "cache": "hit", "status": "ok"},
            }
        ),
    )

    resp = auth_client.post(
        "/api/agent/chat",
        json={
            "message": "Summarize my portfolio's performance",
            "response_preferences": {"thinking_enabled": True},
        },
    )

    assert resp.status_code == 200
    kwargs = fake_client.responses.kwargs_history[0]
    assert kwargs["max_output_tokens"] == agent_router.PORTFOLIO_SUMMARY_MAX_TOKENS
    assert kwargs["max_output_tokens"] >= 2048
    assert "reasoning" in kwargs
    parsed = _parse_sse(resp.text)
    assert any(e == "delta" and p.get("text") == "Portfolio summary" for e, p in parsed)
    assert finalized[0]["content"] == "Portfolio summary"


def test_agent_chat_portfolio_summary_never_finalizes_blank(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "session-portfolio-fallback",
        ),
    )
    finalized: list[dict] = []
    monkeypatch.setattr(
        "api.memory_manager.finalize_turn_async",
        lambda _sid, _user_msg, assistant_msg: finalized.append(assistant_msg),
    )

    streams = [
        (
            [],
            SimpleNamespace(
                content=[],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=10, output_tokens=0),
            ),
        ),
    ]
    _install_fake_anthropic(monkeypatch, streams)
    monkeypatch.setattr(
        agent_router,
        "execute_tool",
        lambda _name, _args, **_kwargs: json.dumps(
            {
                "summary": {
                    "position_count": 2,
                    "long_count": 1,
                    "short_count": 1,
                    "monthly_portfolio_return_pct": 2.3,
                },
                "positions": [
                    {"ticker": "MU", "monthly_contribution_pct": 1.2},
                    {"ticker": "OKLO", "monthly_contribution_pct": -0.4},
                ],
                "_meta": {"duration_ms": 10.0, "cache": "hit", "status": "ok"},
            }
        ),
    )

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "Summarize my portfolio's performance"},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    fallback_delta = [
        p.get("text")
        for e, p in parsed
        if e == "delta" and isinstance(p.get("text"), str) and "Portfolio read complete" in p.get("text", "")
    ]
    assert fallback_delta
    assert finalized[0]["content"] == fallback_delta[-1]


def test_agent_chat_normal_path_never_finalizes_blank(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "session-empty-normal",
        ),
    )
    finalized: list[dict] = []
    monkeypatch.setattr(
        "api.memory_manager.finalize_turn_async",
        lambda _sid, _user_msg, assistant_msg: finalized.append(assistant_msg),
    )

    streams = [
        (
            [],
            SimpleNamespace(
                content=[],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=10, output_tokens=0),
            ),
        ),
    ]
    _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "How is global liquidity affecting risk assets?"},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    fallback_delta = [
        p.get("text")
        for e, p in parsed
        if e == "delta" and p.get("text") == agent_router.EMPTY_AGENT_RESPONSE_TEXT
    ]
    assert fallback_delta
    assert finalized[0]["content"] == agent_router.EMPTY_AGENT_RESPONSE_TEXT


def test_agent_chat_portfolio_risk_uses_normal_agent_loop(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "session-risk",
        ),
    )
    monkeypatch.setattr("api.memory_manager.finalize_turn_async", lambda *_args, **_kwargs: None)

    streams = [
        (
            [_event_tool_use_start("get_portfolio", "call-portfolio")],
            SimpleNamespace(
                content=[{"type": "tool_use", "name": "get_portfolio", "id": "call-portfolio", "input": {}}],
                stop_reason="tool_use",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            ),
        ),
        (
            [_event_text_delta("Risk-aware answer")],
            SimpleNamespace(
                content=[{"type": "text", "text": "Risk-aware answer"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    fake_client = _install_fake_anthropic(monkeypatch, streams)
    monkeypatch.setattr(agent_router, "execute_tool", lambda _name, _args, **_kwargs: json.dumps({"ok": True}))

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "Summarize my portfolio risk"},
    )

    assert resp.status_code == 200
    assert fake_client.messages.calls == 2
    assert fake_client.messages.kwargs_history[0].get("tools")
    parsed = _parse_sse(resp.text)
    assert any(e == "phase" and p.get("phase") == "model_thinking" for e, p in parsed)
    assert any(e == "phase" and p.get("phase") == "tool_running" for e, p in parsed)
    assert any(e == "phase" and p.get("phase") == "model_writing" for e, p in parsed)


def test_agent_chat_casual_prompt_skips_anthropic_tools_and_retrieval(auth_client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        agent_router,
        "_build_agent_instructions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no prompt")),
    )
    monkeypatch.setattr(
        "anthropic.Anthropic", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no client"))
    )
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no retrieval/context build")),
    )
    monkeypatch.setattr(
        "api.memory_db.get_or_create_session",
        lambda _session_id=None: {"session_id": "casual-session", "server_messages": [], "rolling_summary": None},
    )
    finalized: list[tuple[dict, dict]] = []
    monkeypatch.setattr(
        "api.memory_manager.finalize_turn_async",
        lambda _sid, user_msg, assistant_msg: finalized.append((user_msg, assistant_msg)),
    )

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "hello"},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    assert parsed[0][0] == "ping"
    assert any(e == "delta" and "portfolio" not in str(p.get("text", "")).lower() for e, p in parsed)
    done_events = [p for e, p in parsed if e == "done"]
    assert done_events[-1]["session_id"] == "casual-session"
    assert finalized


def test_agent_chat_soft_allows_off_domain_text_to_model_with_guardrail_instruction(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "soft-allow-session",
        ),
    )
    monkeypatch.setattr("api.memory_manager.finalize_turn_async", lambda *_args, **_kwargs: None)
    streams = [
        (
            [_event_text_delta("I can only handle the investing side here.")],
            SimpleNamespace(
                content=[{"type": "text", "text": "I can only handle the investing side here."}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    fake_client = _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "give me a chicken recipe"},
    )

    assert resp.status_code == 200
    assert fake_client.messages.calls == 1
    assert "Domain Guardrail" in fake_client.messages.kwargs_history[0]["system"]
    parsed = _parse_sse(resp.text)
    assert any(e == "delta" and "investing side" in str(p.get("text", "")) for e, p in parsed)


def test_agent_chat_allows_operational_proposal_status_followup(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "proposal-session",
        ),
    )
    monkeypatch.setattr("api.memory_manager.finalize_turn_async", lambda *_args, **_kwargs: None)
    streams = [
        (
            [_event_text_delta("I’ll stage the status proposals.")],
            SimpleNamespace(
                content=[{"type": "text", "text": "I’ll stage the status proposals."}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    fake_client = _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "can you make the proposals to update the status?"},
    )

    assert resp.status_code == 200
    assert fake_client.messages.calls == 1
    parsed = _parse_sse(resp.text)
    assert any(e == "delta" and "status proposals" in str(p.get("text", "")) for e, p in parsed)


def test_agent_chat_clarifies_empty_prompt_before_provider_context_or_tools(auth_client, monkeypatch):
    monkeypatch.setattr(
        agent_router,
        "selected_provider",
        lambda: (_ for _ in ()).throw(AssertionError("no provider selection")),
    )
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no retrieval/context build")),
    )
    monkeypatch.setattr(
        "api.memory_db.get_or_create_session",
        lambda _session_id=None: {"session_id": "clarify-session", "server_messages": [], "rolling_summary": None},
    )
    finalized: list[dict] = []
    monkeypatch.setattr(
        "api.memory_manager.finalize_turn_async",
        lambda _sid, _user_msg, assistant_msg: finalized.append(assistant_msg),
    )

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": " "},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    assert any(e == "delta" and p.get("text") == DOMAIN_CLARIFY_RESPONSE for e, p in parsed)
    done_events = [p for e, p in parsed if e == "done"]
    assert done_events[-1]["session_id"] == "clarify-session"
    assert done_events[-1]["domain_decision"] == "clarify"
    assert finalized[0]["content"] == DOMAIN_CLARIFY_RESPONSE


def test_agent_chat_soft_allows_off_domain_text_to_model(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    streams = [
        (
            [_event_text_delta("I can help if there is an investing angle.")],
            SimpleNamespace(
                content=[{"type": "text", "text": "I can help if there is an investing angle."}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    fake_client = _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "plan a trip to Tokyo"},
    )

    assert resp.status_code == 200
    assert fake_client.messages.calls == 1
    assert "Domain Guardrail" in fake_client.messages.kwargs_history[0]["system"]
    parsed = _parse_sse(resp.text)
    assert any(e == "delta" and "investing angle" in str(p.get("text", "")) for e, p in parsed)


def test_agent_tool_execution_blocks_when_domain_decision_is_not_allow(monkeypatch):
    monkeypatch.setattr(
        agent_router,
        "execute_tool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tool should not execute")),
    )

    result = agent_router._execute_tool_for_actor(
        "get_portfolio",
        {},
        actor=None,
        domain_classification=AgentDomainClassification("block", "unsupported_domain"),
    )

    payload = json.loads(result)
    assert payload["type"] == "RuntimeError"
    assert payload["_meta"]["status"] == "blocked"


def test_agent_chat_workflow_done_includes_tool_metadata(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "session-wf",
        ),
    )
    finalized: list[dict] = []
    monkeypatch.setattr(
        "api.memory_manager.finalize_turn_async",
        lambda _sid, _user_msg, assistant_msg: finalized.append(assistant_msg),
    )
    monkeypatch.setattr(
        agent_router,
        "execute_workflow",
        lambda *_args, **_kwargs: (
            "run-wf",
            "synthesis prompt",
            [
                {"tool": "get_thesis", "data": {}, "duration_ms": 1.0},
                {"tool": "query_ontology", "data": {}, "duration_ms": 2.0},
            ],
        ),
    )
    monkeypatch.setattr("api.workflow_artifacts.extract_artifacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("api.workflow_artifacts.persist_artifacts", lambda *_args, **_kwargs: 0)

    streams = [
        (
            [_event_text_delta('Analysis\n```artifacts\n{"evaluation_draft": {"ticker": "NVDA"}}\n```')],
            SimpleNamespace(
                content=[{"type": "text", "text": "Analysis"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "/workflow:thesis_review:NVDA", "allow_workflow_handoff": False},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    done_events = [p for e, p in parsed if e == "done"]
    assert done_events[-1]["session_id"] == "session-wf"
    assert done_events[-1]["tools_used"] == ["get_thesis", "query_ontology"]
    assert done_events[-1]["tool_calls"][0]["status"] == "ok"
    assert finalized[0]["toolCalls"] == done_events[-1]["tool_calls"]


def test_agent_chat_position_dossier_pressure_test_workflow(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "session-pressure",
        ),
    )
    finalized: list[dict] = []
    monkeypatch.setattr(
        "api.memory_manager.finalize_turn_async",
        lambda _sid, _user_msg, assistant_msg: finalized.append(assistant_msg),
    )
    monkeypatch.setattr(
        agent_router,
        "execute_workflow",
        lambda *_args, **_kwargs: (
            "run-pressure",
            "synthesis prompt",
            [
                {"tool": "get_dossier", "data": {}, "duration_ms": 1.0},
                {"tool": "get_thesis", "data": {}, "duration_ms": 2.0},
            ],
        ),
    )
    monkeypatch.setattr("api.workflow_artifacts.extract_artifacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("api.workflow_artifacts.persist_artifacts", lambda *_args, **_kwargs: 0)

    streams = [
        (
            [_event_text_delta('Pressure test\n```artifacts\n{"evaluation_draft": {"ticker": "MU"}}\n```')],
            SimpleNamespace(
                content=[{"type": "text", "text": "Pressure test"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "/workflow:position_dossier_pressure_test:MU", "allow_workflow_handoff": False},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    done_events = [p for e, p in parsed if e == "done"]
    assert done_events[-1]["session_id"] == "session-pressure"
    assert done_events[-1]["tools_used"] == ["get_dossier", "get_thesis"]


def test_agent_chat_workflow_hands_off_to_durable_job(auth_client, monkeypatch):
    from api import async_job_runner, cache

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    monkeypatch.setenv("AGENT_CHAT_DISPATCH_BACKEND", "warm_worker")
    monkeypatch.setattr(
        async_job_runner,
        "_enqueue_cloud_run_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no Cloud Run dispatch")),
    )

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "/workflow:thesis_review:NVDA", "client_turn_id": "handoff-turn"},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    handoffs = [payload for event, payload in parsed if event == "handoff"]
    assert handoffs
    assert handoffs[-1]["status"] == "queued"
    assert handoffs[-1]["job_id"]
    assert any(
        event["event_type"] == "status" and event["payload"].get("status") == "starting"
        for event in handoffs[-1].get("events") or []
    )
    assert not any(event == "tool_call" for event, _payload in parsed)


def test_agent_chat_async_returns_replayable_events_and_finalizes(auth_client, monkeypatch):
    from api import cache

    cache.invalidate_all()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "async-session",
        ),
    )
    finalized: list[tuple[dict, dict]] = []
    monkeypatch.setattr(
        "api.memory_manager.finalize_turn",
        lambda _sid, user_msg, assistant_msg: finalized.append((user_msg, assistant_msg)),
    )

    streams = [
        (
            [_event_text_delta("async answer")],
            SimpleNamespace(
                content=[{"type": "text", "text": "async answer"}],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]
    _install_fake_anthropic(monkeypatch, streams)

    started = auth_client.post(
        "/api/agent/chat/async",
        json={"message": "How is liquidity?", "client_turn_id": "turn-1"},
    )

    assert started.status_code in (200, 202)
    job_id = started.json()["job_id"]
    deadline = time.time() + 4
    after_seq = 0
    seen_events: list[dict] = []
    while time.time() < deadline:
        resp = auth_client.get(f"/api/agent/chat/async/{job_id}/events", params={"after_seq": after_seq})
        assert resp.status_code == 200
        body = resp.json()
        seen_events.extend(body.get("events") or [])
        after_seq = body.get("next_seq", after_seq)
        if body["status"] == "done":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("agent async job did not complete")

    assert any(event["event_type"] == "delta" and "async answer" in event["payload"]["text"] for event in seen_events)
    assert any(
        event["event_type"] == "phase" and event["payload"].get("phase") == "model_thinking" for event in seen_events
    )
    assert any(event["event_type"] == "done" for event in seen_events)
    done_event = next(event for event in seen_events if event["event_type"] == "done")
    assert done_event["payload"]["timings"]["models"][0]["phase"] == "model_thinking"
    assert finalized and finalized[0][1]["content"] == "async answer"


def test_agent_chat_async_created_cloud_run_job_returns_starting_event(auth_client, monkeypatch):
    from api import async_job_runner, cache

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    dispatched: list[str] = []
    monkeypatch.setattr(
        async_job_runner,
        "_enqueue_cloud_run_job",
        lambda _job_type, job_id: dispatched.append(job_id),
    )

    resp = auth_client.post(
        "/api/agent/chat/async",
        json={"message": "How is liquidity?", "client_turn_id": "starting-turn"},
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert dispatched == [body["job_id"]]
    assert any(
        event["event_type"] == "status" and event["payload"].get("status") == "starting"
        for event in body.get("events") or []
    )


def test_agent_chat_async_reuses_duplicate_active_job(auth_client, monkeypatch):
    from api import cache

    cache.invalidate_all()
    started = threading.Event()
    release = threading.Event()

    def slow_agent_job(req, *, job_id):
        started.set()
        assert release.wait(timeout=2)
        return {"status": "done", "session_id": req.session_id}

    monkeypatch.setattr(agent_chat_worker, "_run_agent_chat_turn_job", slow_agent_job)

    body = {"session_id": "dup-session", "message": "same turn", "client_turn_id": "dup-turn"}
    first = auth_client.post("/api/agent/chat/async", json=body)
    assert first.status_code == 202
    assert started.wait(timeout=2)

    second = auth_client.post("/api/agent/chat/async", json=body)
    assert second.status_code == 202
    assert second.json()["job_id"] == first.json()["job_id"]

    release.set()


def test_agent_chat_async_cancel_marks_job_cancelled(auth_client, monkeypatch):
    from api import cache

    cache.invalidate_all()
    started = threading.Event()
    release = threading.Event()

    def slow_agent_job(req, *, job_id):
        started.set()
        release.wait(timeout=2)
        return {"status": "done", "session_id": req.session_id}

    monkeypatch.setattr(agent_chat_worker, "_run_agent_chat_turn_job", slow_agent_job)

    started_resp = auth_client.post(
        "/api/agent/chat/async",
        json={"session_id": "cancel-session", "message": "cancel me", "client_turn_id": "cancel-turn"},
    )
    assert started_resp.status_code == 202
    job_id = started_resp.json()["job_id"]
    assert started.wait(timeout=2)

    cancel_resp = auth_client.post(f"/api/agent/chat/async/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    body = cancel_resp.json()
    assert body["status"] == "cancelled"
    assert any(event["event_type"] == "error" for event in body["events"])
    release.set()


def test_workflow_execution_emits_keepalive_while_blocked(monkeypatch):
    monkeypatch.setattr(agent_router, "SSE_KEEPALIVE_INTERVAL_S", 0.001)

    def slow_execute_workflow(*_args, **_kwargs):
        time.sleep(0.02)
        return "run-slow", "synthesis prompt", []

    monkeypatch.setattr(agent_router, "execute_workflow", slow_execute_workflow)

    gen = agent_router._execute_workflow_keepalive("thesis_review", "NVDA", actor=None)
    frames: list[str] = []
    while True:
        try:
            frames.append(next(gen))
        except StopIteration as stop:
            result = stop.value
            break

    assert any("event: ping" in frame for frame in frames)
    assert result == ("run-slow", "synthesis prompt", [])


# ---------------------------------------------------------------------------
# Provider history shape: assistant turns must use output_text for OpenAI
# (regression: the Responses API rejects input_text on assistant content)
# ---------------------------------------------------------------------------


def test_openai_initial_conversation_uses_output_text_for_assistant():
    msgs = [
        agent_router.ChatMessage(role="user", content="hello"),
        agent_router.ChatMessage(role="assistant", content="hi back"),
        agent_router.ChatMessage(role="user", content="what model are you?"),
    ]
    convo = agent_router._initial_conversation(agent_router.PROVIDER_OPENAI, msgs)
    assert [m["content"][0]["type"] for m in convo] == ["input_text", "output_text", "input_text"]
    assert [m["role"] for m in convo] == ["user", "assistant", "user"]


def test_openai_conversation_from_context_uses_output_text_for_assistant():
    raw = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi back"},
        {"role": "user", "content": "follow-up"},
    ]
    convo = agent_router._openai_conversation_from_context(raw)
    assert [m["content"][0]["type"] for m in convo] == ["input_text", "output_text", "input_text"]


def test_anthropic_initial_conversation_passes_strings_through():
    msgs = [
        agent_router.ChatMessage(role="user", content="hello"),
        agent_router.ChatMessage(role="assistant", content="hi"),
    ]
    convo = agent_router._initial_conversation(agent_router.PROVIDER_ANTHROPIC, msgs)
    assert convo == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_agent_chat_openai_replays_assistant_history_as_output_text(auth_client, monkeypatch):
    """Multi-turn replay: prior assistant message must be sent as output_text, not input_text."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hey. What are you looking at?"},
                {"role": "user", "content": new_user_message},
            ],
            "session-multiturn",
        ),
    )
    monkeypatch.setattr("api.memory_manager.finalize_turn_async", lambda *_args, **_kwargs: None)

    streams = [
        (
            [_openai_event_text_delta("I'm Stan.")],
            SimpleNamespace(
                output=[{"type": "message", "content": [{"type": "output_text", "text": "I'm Stan."}]}],
                usage=SimpleNamespace(input_tokens=2, output_tokens=3),
            ),
        ),
    ]
    fake_client = _install_fake_openai(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "what model are you?"},
    )

    assert resp.status_code == 200
    sent_input = fake_client.responses.kwargs_history[0]["input"]
    types = [(m["role"], m["content"][0]["type"]) for m in sent_input]
    assert types == [
        ("user", "input_text"),
        ("assistant", "output_text"),
        ("user", "input_text"),
    ]


def test_agent_chat_openai_continues_after_output_token_limit(auth_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr(agent_router, "_select_tool_names", lambda _message: [])
    monkeypatch.setattr(
        "api.memory_manager.build_conversation_context",
        lambda _session_id, new_user_message, **_kwargs: (
            [{"role": "user", "content": new_user_message}],
            "session-output-limit",
        ),
    )
    finalized: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "api.memory_manager.finalize_turn_async",
        lambda _session_id, _user_msg, assistant_msg: finalized.append(assistant_msg),
    )

    streams = [
        (
            [_openai_event_text_delta("The BOJ should hike because ")],
            SimpleNamespace(
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                output=[
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "The BOJ should hike because "}],
                    }
                ],
                usage=SimpleNamespace(input_tokens=20, output_tokens=2048),
            ),
        ),
        (
            [_openai_event_text_delta("imported inflation is accelerating.")],
            SimpleNamespace(
                status="completed",
                output=[
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "imported inflation is accelerating."}],
                    }
                ],
                usage=SimpleNamespace(input_tokens=10, output_tokens=12),
            ),
        ),
    ]
    fake_client = _install_fake_openai(monkeypatch, streams)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "What does higher Japanese inflation mean for BOJ rates?"},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    assert "".join(p["text"] for e, p in parsed if e == "delta") == (
        "The BOJ should hike because imported inflation is accelerating."
    )
    assert any(e == "done" for e, _p in parsed)
    assert fake_client.responses.calls == 2
    continuation_input = fake_client.responses.kwargs_history[1]["input"]
    assert continuation_input[-1]["role"] == "user"
    assert "Continue exactly from where" in continuation_input[-1]["content"][0]["text"]
    assert finalized[-1]["content"] == "The BOJ should hike because imported inflation is accelerating."
