from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import api.routers.agent as agent_router
import llm_utils
from api.agent_governance import prepare_model_egress
from talisman_openai_compat import (
    call_chat_completions_text,
    extract_chat_tool_calls,
    openai_compatible_client,
    stream_chat_completions_events,
)


def _talisman_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "talisman")
    monkeypatch.setenv("TALISMAN_BASE_URL", "http://talisman.test/v1")
    monkeypatch.setenv("TALISMAN_API_KEY", "talisman-secret-key")
    monkeypatch.setenv("TALISMAN_MODEL_MID", "owned-mid")
    llm_utils._CLIENT_CACHE.clear()
    llm_utils._CLIENT_FACTORY_CACHE.clear()


def test_talisman_provider_resolution_and_configuration(monkeypatch):
    _talisman_env(monkeypatch)

    assert llm_utils.selected_provider() == "talisman"
    assert llm_utils.model_for_tier(llm_utils.MODEL_MID, "talisman") == "owned-mid"
    assert llm_utils.is_provider_configured("talisman") is True
    assert llm_utils.require_api_key("talisman") == "talisman-secret-key"

    monkeypatch.delenv("TALISMAN_BASE_URL", raising=False)
    assert llm_utils.is_provider_configured("talisman") is False
    with pytest.raises(RuntimeError, match="TALISMAN_BASE_URL"):
        llm_utils.require_api_key("talisman")


class _FakeChatCompletions:
    def __init__(self):
        self.kwargs_history: list[dict] = []
        self.stream_calls = 0

    def create(self, **kwargs):
        self.kwargs_history.append(kwargs)
        if kwargs.get("stream"):
            self.stream_calls += 1
            if self.stream_calls == 1:
                return _FakeChatStream(tool_round=True)
            return _FakeChatStream(tool_round=False)
        message = SimpleNamespace(
            content='{"ok": true}',
            tool_calls=None,
            parsed=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
        )


class _FakeChatStream:
    def __init__(self, *, tool_round: bool):
        self._tool_round = tool_round

    def __iter__(self):
        if self._tool_round:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call-1",
                                    function=SimpleNamespace(name="query_ontology", arguments='{"query":"A"}'),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ]
            )
        else:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="owned ", tool_calls=None),
                        finish_reason=None,
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="answer", tool_calls=None),
                        finish_reason=None,
                    )
                ]
            )
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop")]
        )


class _FakeChat:
    def __init__(self):
        self.completions = _FakeChatCompletions()


class _FakeTalismanClient:
    def __init__(self):
        self.chat = _FakeChat()


def test_talisman_text_and_json_calls_use_chat_completions(monkeypatch):
    _talisman_env(monkeypatch)
    fake_client = _FakeTalismanClient()
    monkeypatch.setattr(llm_utils, "get_llm_client", lambda *_args, **_kwargs: fake_client)

    text, citations, response = llm_utils.call_llm_text(
        prompt="hello",
        model=llm_utils.MODEL_MID,
        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        json_schema_name="test_schema",
    )
    assert text == '{"ok": true}'
    assert citations == []
    assert fake_client.chat.completions.kwargs_history[0]["model"] == "owned-mid"
    assert fake_client.chat.completions.kwargs_history[0]["response_format"]["type"] == "json_schema"

    parsed, _citations, _response, diagnostics = llm_utils.call_llm_json(
        prompt="hello",
        model=llm_utils.MODEL_MID,
        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )
    assert parsed == {"ok": True}
    assert diagnostics["status"] == "ok"
    assert diagnostics["provider"] == "talisman"


def test_talisman_adapter_helpers_roundtrip_tool_calls():
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "query_ontology", "arguments": json.dumps({"query": "A"})},
            }
        ],
    }
    calls = extract_chat_tool_calls(message)
    assert calls == [{"name": "query_ontology", "call_id": "call-1", "args": {"query": "A"}}]


def test_talisman_stream_events_emit_delta_and_tool_call(monkeypatch):
    _talisman_env(monkeypatch)

    class _SingleRoundChatCompletions:
        def create(self, **kwargs):
            return _FakeChatStream(tool_round=False)

    class _SingleRoundClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_SingleRoundChatCompletions())

    events = list(
        stream_chat_completions_events(
            client=_SingleRoundClient(),
            stream_kwargs={
                "model": "owned-mid",
                "system": "instructions",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "name": "query_ontology", "parameters": {"type": "object"}}],
                "tool_choice": "required",
                "max_tokens": 256,
            },
            text_parts=[],
        )
    )
    delta_text = "".join(
        json.loads(line[len("data: ") :])["text"]
        for event in events
        if "event: delta" in event
        for line in event.splitlines()
        if line.startswith("data: ")
    )
    assert "owned answer" in delta_text


def test_prepare_model_egress_allows_talisman_when_local_only_required(monkeypatch):
    _talisman_env(monkeypatch)
    sanitized, manifest = prepare_model_egress(
        provider="talisman",
        purpose="test",
        stream_kwargs={
            "model": "owned-mid",
            "messages": [{"role": "user", "content": "hello"}],
            "local_only_required": True,
        },
        actor=None,
    )
    assert sanitized["messages"][0]["content"] == "hello"
    assert manifest["decision"] == "allowed"
    assert manifest["provider_egress"] == "first_party_allowed"


def test_llm_settings_do_not_expose_talisman_secrets(auth_client, monkeypatch):
    _talisman_env(monkeypatch)
    response = auth_client.get("/api/settings/llm")
    assert response.status_code == 200
    assert "talisman-secret-key" not in response.text
    talisman = next(item for item in response.json()["available_providers"] if item["provider"] == "talisman")
    assert talisman["configured"] is True
    assert talisman["base_url_configured"] is True


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


def test_agent_stream_talisman_function_call_roundtrip(auth_client, monkeypatch):
    _talisman_env(monkeypatch)
    monkeypatch.setattr(agent_router, "_build_agent_instructions", lambda screen_context=None: "agent instructions")

    fake_client = _FakeTalismanClient()
    monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: fake_client)

    seen_args: list[dict] = []

    def fake_execute_tool(_name: str, args: dict):
        seen_args.append(args)
        return json.dumps({"ok": True})

    monkeypatch.setattr(agent_router, "execute_tool", fake_execute_tool)

    resp = auth_client.post(
        "/api/agent/chat",
        json={"message": "What ontology nodes exist for semiconductors?"},
    )

    assert resp.status_code == 200
    tool_round = next(kwargs for kwargs in fake_client.chat.completions.kwargs_history if kwargs.get("tools"))
    assert tool_round["tools"][0]["function"]["name"] == "query_ontology"
    assert seen_args == [{"query": "A"}]
    parsed = _parse_sse(resp.text)
    assert any(e == "tool_call" and p["name"] == "query_ontology" for e, p in parsed)
    assert any(e == "tool_result" and p["status"] == "ok" for e, p in parsed)
    assert fake_client.chat.completions.stream_calls >= 2
    delta_text = "".join(p.get("text", "") for e, p in parsed if e == "delta")
    done_text = next((p.get("content", "") for e, p in parsed if e == "done"), "")
    assert "owned answer" in (delta_text or done_text)
