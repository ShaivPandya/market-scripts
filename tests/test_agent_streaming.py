from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import api.routers.agent as agent_router


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
        "/api/v1/agent/chat",
        json={"messages": [{"role": "user", "content": "test"}]},
    )

    assert resp.status_code == 200
    assert fake_client.messages.kwargs_history[0].get("tool_choice") == {"type": "any"}
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
        "/api/v1/agent/chat",
        json={"messages": [{"role": "user", "content": "test"}]},
    )

    assert resp.status_code == 200
    assert fake_client.responses.kwargs_history[0]["tool_choice"] == "required"
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
        "/api/v1/agent/chat",
        json={"messages": [{"role": "user", "content": "test"}]},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    tool_results = [p for e, p in parsed if e == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["status"] == "error"
    assert "boom" in tool_results[0]["message"]


def test_agent_stream_enforces_tool_loop_limit(auth_client, monkeypatch):
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

    fake_client = _install_fake_anthropic(monkeypatch, streams)

    resp = auth_client.post(
        "/api/v1/agent/chat",
        json={"messages": [{"role": "user", "content": "test"}]},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    assert any(e == "error" and "loop limit" in str(p.get("message", "")).lower() for e, p in parsed)
    assert any(e == "done" for e, _p in parsed)
    assert fake_client.messages.calls == agent_router.MAX_TOOL_CONTINUATION_ROUNDS


def test_agent_stream_auth_error_is_user_friendly(auth_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")
    monkeypatch.setattr("anthropic.Anthropic", lambda *args, **kwargs: _RaiseInStreamClient())

    resp = auth_client.post(
        "/api/v1/agent/chat",
        json={"messages": [{"role": "user", "content": "test"}]},
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
        "/api/v1/agent/chat",
        json={"messages": [{"role": "user", "content": "test"}]},
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
        "/api/v1/agent/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert resp.status_code == 200
    assert "tool_choice" not in fake_client.messages.kwargs_history[0]
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
        "/api/v1/agent/chat",
        json={"messages": [{"role": "user", "content": "test"}]},
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
        "/api/v1/agent/chat",
        json={"messages": [{"role": "user", "content": "How is sentiment?"}]},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    tool_results = [p for e, p in parsed if e == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["status"] == "ok"
    assert any(e == "done" for e, _p in parsed)


def test_agent_chat_v2_sends_initial_ping_and_disables_gzip(auth_client, monkeypatch):
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
        "/api/v1/agent/chat/v2",
        json={"message": "How is liquidity?"},
    )

    assert resp.status_code == 200
    assert resp.headers.get("content-encoding") == "identity"
    assert "no-transform" in resp.headers.get("cache-control", "")
    parsed = _parse_sse(resp.text)
    assert parsed[0][0] == "ping"
    assert any(e == "delta" and p.get("text") == "hi there" for e, p in parsed)
    done_events = [p for e, p in parsed if e == "done"]
    assert done_events[-1]["session_id"] == "session-1"


def test_agent_chat_v2_casual_prompt_skips_anthropic_tools_and_retrieval(auth_client, monkeypatch):
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
        "/api/v1/agent/chat/v2",
        json={"message": "hello"},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    assert parsed[0][0] == "ping"
    assert any(e == "delta" and "portfolio" not in str(p.get("text", "")).lower() for e, p in parsed)
    done_events = [p for e, p in parsed if e == "done"]
    assert done_events[-1]["session_id"] == "casual-session"
    assert finalized
