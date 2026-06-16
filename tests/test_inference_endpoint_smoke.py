"""Env-gated live contract smoke against a provisioned Talisman inference endpoint."""

from __future__ import annotations

import json
import os

import pytest

from decision_quality.bench_openai_client import BenchOpenAIConfig, stream_bench_openai_compatible
from talisman_openai_compat import (
    TalismanEndpointConfig,
    call_chat_completions_text,
    call_chat_completions_tools,
    openai_compatible_client,
    openai_function_tools,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TALISMAN_INFERENCE_SMOKE", "").strip().lower() not in {"1", "true", "yes", "on"},
    reason="Set TALISMAN_INFERENCE_SMOKE=1 with TALISMAN_BASE_URL and TALISMAN_API_KEY to run live smoke",
)


def _require_endpoint_env() -> TalismanEndpointConfig:
    base_url = (os.environ.get("TALISMAN_BASE_URL") or "").strip()
    if not base_url:
        pytest.skip("TALISMAN_BASE_URL is required for live inference smoke")
    return TalismanEndpointConfig.from_env()


def _model_id() -> str:
    return (os.environ.get("TALISMAN_MODEL_MID") or os.environ.get("TALISMAN_BENCH_CANDIDATE_MODEL") or "").strip()


@pytest.fixture(scope="module")
def endpoint_client():
    config = _require_endpoint_env()
    return openai_compatible_client(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout_s=config.timeout_s,
    )


def test_live_models_endpoint_lists_configured_alias(endpoint_client):
    model_id = _model_id()
    if not model_id:
        pytest.skip("TALISMAN_MODEL_MID or TALISMAN_BENCH_CANDIDATE_MODEL required")
    models = endpoint_client.models.list()
    ids = {getattr(item, "id", None) for item in models.data}
    assert model_id in ids


def test_live_text_completion(endpoint_client):
    model_id = _model_id()
    if not model_id:
        pytest.skip("TALISMAN_MODEL_MID or TALISMAN_BENCH_CANDIDATE_MODEL required")
    text, _, _response = call_chat_completions_text(
        client=endpoint_client,
        model=model_id,
        prompt="Reply with the single word READY.",
        max_tokens=32,
    )
    assert "READY" in text.upper()


def test_live_tool_call_roundtrip(endpoint_client):
    model_id = _model_id()
    if not model_id:
        pytest.skip("TALISMAN_MODEL_MID or TALISMAN_BENCH_CANDIDATE_MODEL required")
    tools = openai_function_tools(
        [
            {
                "name": "echo_status",
                "description": "Return a status string",
                "parameters": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                },
            }
        ]
    )
    message = call_chat_completions_tools(
        client=endpoint_client,
        model=model_id,
        messages=[{"role": "user", "content": "Call echo_status with status=ok"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=128,
    )
    tool_calls = getattr(message, "tool_calls", None) or []
    assert tool_calls


def test_live_structured_json_output(endpoint_client):
    model_id = _model_id()
    if not model_id:
        pytest.skip("TALISMAN_MODEL_MID or TALISMAN_BENCH_CANDIDATE_MODEL required")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    text, _, _response = call_chat_completions_text(
        client=endpoint_client,
        model=model_id,
        prompt="Return JSON with answer=smoke",
        json_schema=schema,
        json_schema_name="smoke_schema",
        max_tokens=64,
    )
    payload = json.loads(text)
    assert payload["answer"]


def test_live_streaming_deltas():
    base_url = (os.environ.get("TALISMAN_BASE_URL") or "").strip()
    model_id = _model_id()
    if not base_url or not model_id:
        pytest.skip("TALISMAN_BASE_URL and model id required")
    config = BenchOpenAIConfig(
        base_url=base_url,
        api_key=os.environ.get("TALISMAN_API_KEY") or "talisman-placeholder",
        model=model_id,
        timeout_s=float(os.environ.get("TALISMAN_TIMEOUT_S") or "120"),
    )
    text_parts: list[str] = []
    events = list(
        stream_bench_openai_compatible(
            config=config,
            stream_kwargs={
                "messages": [{"role": "user", "content": "Stream one short sentence."}],
                "max_tokens": 64,
            },
            text_parts=text_parts,
        )
    )
    assert text_parts or any("delta" in event for event in events)
