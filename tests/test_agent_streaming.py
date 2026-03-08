from __future__ import annotations

import json
from types import SimpleNamespace

import api.routers.agent as agent_router


def _event(event_type: str, **kwargs):
    return SimpleNamespace(type=event_type, **kwargs)


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


class _FakeResponses:
    def __init__(self, streams):
        self.streams = streams
        self.calls = 0
        self.kwargs_history: list[dict] = []

    def create(self, **_kwargs):
        self.kwargs_history.append(dict(_kwargs))
        if self.calls >= len(self.streams):
            raise AssertionError("Unexpected extra responses.create() call")
        out = self.streams[self.calls]
        self.calls += 1
        return iter(out)


class _FakeClient:
    def __init__(self, streams):
        self.responses = _FakeResponses(streams)


def _install_fake_openai(monkeypatch, streams):
    fake_client = _FakeClient(streams)
    monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: fake_client)
    return fake_client


def test_agent_stream_tracks_args_per_call_id(auth_client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda: "agent instructions")

    streams = [
        [
            _event("response.created", response=SimpleNamespace(id="resp-1")),
            _event(
                "response.output_item.added",
                item=SimpleNamespace(type="function_call", name="query_ontology", call_id="call-1", id="item-1"),
            ),
            _event("response.function_call_arguments.delta", call_id="call-1", delta='{"query":"A'),
            _event(
                "response.output_item.added",
                item=SimpleNamespace(type="function_call", name="query_ontology", call_id="call-2", id="item-2"),
            ),
            _event("response.function_call_arguments.delta", call_id="call-2", delta='{"query":"B'),
            _event("response.function_call_arguments.delta", call_id="call-1", delta='"}'),
            _event("response.function_call_arguments.delta", call_id="call-2", delta='"}'),
            _event(
                "response.output_item.done",
                item=SimpleNamespace(type="function_call", name="query_ontology", call_id="call-1"),
            ),
            _event(
                "response.output_item.done",
                item=SimpleNamespace(type="function_call", name="query_ontology", call_id="call-2"),
            ),
            _event("response.completed", response=SimpleNamespace(usage=None)),
        ],
        [
            _event("response.created", response=SimpleNamespace(id="resp-2")),
            _event("response.output_text.delta", delta="analysis"),
            _event(
                "response.completed",
                response=SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=2)),
            ),
        ],
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
    assert fake_client.responses.kwargs_history[0].get("tool_choice") == "required"
    assert seen_args == [{"query": "A"}, {"query": "B"}]
    parsed = _parse_sse(resp.text)
    tool_results = [p for e, p in parsed if e == "tool_result"]
    assert len(tool_results) == 2
    assert all(p["status"] == "ok" for p in tool_results)


def test_agent_stream_marks_tool_result_error(auth_client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda: "agent instructions")

    streams = [
        [
            _event("response.created", response=SimpleNamespace(id="resp-1")),
            _event(
                "response.output_item.added",
                item=SimpleNamespace(type="function_call", name="query_ontology", call_id="call-1", id="item-1"),
            ),
            _event("response.function_call_arguments.delta", call_id="call-1", delta="{}"),
            _event(
                "response.output_item.done",
                item=SimpleNamespace(type="function_call", name="query_ontology", call_id="call-1"),
            ),
            _event("response.completed", response=SimpleNamespace(usage=None)),
        ],
        [
            _event("response.created", response=SimpleNamespace(id="resp-2")),
            _event("response.output_text.delta", delta="analysis"),
            _event(
                "response.completed",
                response=SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=2)),
            ),
        ],
    ]
    _install_fake_openai(monkeypatch, streams)
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
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda: "agent instructions")
    monkeypatch.setattr(agent_router, "execute_tool", lambda _name, _args: json.dumps({"ok": True}))

    streams = []
    for i in range(agent_router.MAX_TOOL_CONTINUATION_ROUNDS + 1):
        call_id = f"call-{i}"
        streams.append(
            [
                _event("response.created", response=SimpleNamespace(id=f"resp-{i}")),
                _event(
                    "response.output_item.added",
                    item=SimpleNamespace(type="function_call", name="query_ontology", call_id=call_id, id=f"item-{i}"),
                ),
                _event("response.function_call_arguments.delta", call_id=call_id, delta="{}"),
                _event(
                    "response.output_item.done",
                    item=SimpleNamespace(type="function_call", name="query_ontology", call_id=call_id),
                ),
                _event("response.completed", response=SimpleNamespace(usage=None)),
            ]
        )

    fake_client = _install_fake_openai(monkeypatch, streams)

    resp = auth_client.post(
        "/api/v1/agent/chat",
        json={"messages": [{"role": "user", "content": "test"}]},
    )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.text)
    assert any(e == "error" and "loop limit" in str(p.get("message", "")).lower() for e, p in parsed)
    assert any(e == "done" for e, _p in parsed)
    assert fake_client.responses.calls == agent_router.MAX_TOOL_CONTINUATION_ROUNDS + 1
