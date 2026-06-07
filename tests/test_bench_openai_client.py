from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import llm_utils
from decision_quality.bench_openai_client import (
    BenchOpenAIConfig,
    activate_bench_openai,
    call_openai_compatible_json,
    call_openai_compatible_tools,
    estimate_cost_usd,
    stream_bench_openai_compatible,
)


def test_estimate_cost_usd_computes_input_and_output():
    cost = estimate_cost_usd(
        {"input_tokens": 2000, "output_tokens": 1000},
        cost_per_1k_input_tokens_usd=0.0002,
        cost_per_1k_output_tokens_usd=0.0004,
    )
    assert cost == pytest.approx(0.0008)


def test_call_openai_compatible_json_records_usage_and_cost():
    config = BenchOpenAIConfig(
        base_url="http://localhost:8000/v1",
        api_key="test",
        model="qwen2.5-7b-instruct",
        cost_per_1k_input_tokens_usd=0.0002,
        cost_per_1k_output_tokens_usd=0.0004,
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}', parsed=None))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
    )
    with patch("decision_quality.bench_openai_client._openai_client") as mock_client_factory:
        mock_client_factory.return_value.chat.completions.create.return_value = response
        parsed, _citations, _raw, diagnostics = call_openai_compatible_json(
            config=config,
            prompt="return json",
            json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        )
    assert parsed == {"ok": True}
    assert diagnostics["usage"] == {"input_tokens": 100, "output_tokens": 50}
    assert diagnostics["estimated_cost_usd"] == pytest.approx(0.00004)


def test_call_openai_compatible_tools_extracts_tool_calls():
    config = BenchOpenAIConfig(base_url="http://localhost:8000/v1", api_key="test", model="bench-model")
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="get_price", arguments='{"ticker":"NVDA"}'),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tool_call]))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    with patch("decision_quality.bench_openai_client._openai_client") as mock_client_factory:
        mock_client_factory.return_value.chat.completions.create.return_value = response
        message, diagnostics = call_openai_compatible_tools(
            config=config,
            messages=[{"role": "user", "content": "price for NVDA"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_price",
                        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}},
                    },
                }
            ],
        )
    assert message.tool_calls[0].function.name == "get_price"
    assert diagnostics["tool_calls"][0]["name"] == "get_price"


def test_activate_bench_openai_agent_mode_patches_provider_resolution(monkeypatch):
    original_provider = llm_utils.selected_provider
    config = BenchOpenAIConfig(base_url="http://localhost:8000/v1", api_key="test", model="owned-candidate")
    with activate_bench_openai(config, agent_mode=True):
        assert llm_utils.selected_provider() == llm_utils.PROVIDER_OPENAI
        assert llm_utils.model_for_tier("mid") == "owned-candidate"
    assert llm_utils.selected_provider is not original_provider or llm_utils.selected_provider() == original_provider()


def test_stream_bench_openai_compatible_emits_delta_and_tool_call():
    config = BenchOpenAIConfig(base_url="http://localhost:8000/v1", api_key="test", model="bench-model")

    class FakeStream:
        def __iter__(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Hello",
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    function=SimpleNamespace(name="get_price", arguments='{"ticker":"NVDA"}'),
                                )
                            ],
                        )
                    )
                ]
            )

    with patch("decision_quality.bench_openai_client._openai_client") as mock_client_factory:
        mock_client_factory.return_value.chat.completions.create.return_value = FakeStream()
        events = list(
            stream_bench_openai_compatible(
                config=config,
                stream_kwargs={
                    "instructions": "system",
                    "input": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "get_price", "parameters": {}}}],
                },
                text_parts=[],
            )
        )
    payload = "".join(events)
    assert '"text": "Hello"' in payload
    assert '"name": "get_price"' in payload
