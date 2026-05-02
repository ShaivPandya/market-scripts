from __future__ import annotations

from types import SimpleNamespace

import llm_utils


def test_provider_and_model_resolution_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_utils.selected_provider() == "anthropic"
    assert llm_utils.model_for_tier(llm_utils.MODEL_LOW, "anthropic") == "claude-haiku-4-5"
    assert llm_utils.model_for_tier(llm_utils.MODEL_MID, "openai") == "gpt-5.4"
    assert llm_utils.resolve_model("claude-opus-4-7", "openai") == "gpt-5.5"


def test_provider_model_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL_HIGH", "gpt-custom-high")
    assert llm_utils.model_for_tier(llm_utils.MODEL_HIGH, "openai") == "gpt-custom-high"


def test_required_key_validation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        llm_utils.require_api_key()
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("missing OpenAI key should fail")

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-proj-wrong")
    try:
        llm_utils.require_api_key()
    except RuntimeError as exc:
        assert "Anthropic key" in str(exc)
    else:
        raise AssertionError("OpenAI key in Anthropic env should fail")


def test_anthropic_text_request_shape_and_citations(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    class FakeMessages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text="answer",
                        citations=[SimpleNamespace(title="Source", url="https://example.com")],
                    )
                ],
                stop_reason="end_turn",
            )

    fake_messages = FakeMessages()

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = fake_messages

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    text, citations, _response = llm_utils.call_llm_text(
        prompt="hello",
        model=llm_utils.MODEL_LOW,
        api_key=None,
        max_tokens=123,
        system="system",
        allowed_domains=["example.com"],
        max_web_search_uses=2,
    )

    assert text == "answer"
    assert citations == [("Source", "https://example.com")]
    assert fake_messages.kwargs["model"] == "claude-haiku-4-5"
    assert fake_messages.kwargs["max_tokens"] == 123
    assert fake_messages.kwargs["system"] == "system"
    assert fake_messages.kwargs["tools"] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 2,
            "allowed_domains": ["example.com"],
        }
    ]
    assert "thinking" not in fake_messages.kwargs


def test_openai_text_request_shape_and_citations(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeResponses:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[
                            SimpleNamespace(
                                text="openai answer",
                                annotations=[
                                    SimpleNamespace(
                                        type="url_citation",
                                        title="OpenAI Source",
                                        url="https://example.com/openai",
                                    )
                                ],
                            )
                        ],
                    )
                ]
            )

    fake_responses = FakeResponses()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = fake_responses

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    text, citations, _response = llm_utils.call_llm_text(
        prompt="hello",
        model=llm_utils.MODEL_LOW,
        api_key=None,
        max_tokens=456,
        system="instructions",
        allowed_domains=["example.com"],
    )

    assert text == "openai answer"
    assert citations == [("OpenAI Source", "https://example.com/openai")]
    assert fake_responses.kwargs["model"] == "gpt-5.4-mini"
    assert fake_responses.kwargs["max_output_tokens"] == 456
    assert fake_responses.kwargs["instructions"] == "instructions"
    assert fake_responses.kwargs["tools"] == [
        {
            "type": "web_search",
            "filters": {"allowed_domains": ["example.com"]},
            "search_context_size": "medium",
        }
    ]
    assert "reasoning" not in fake_responses.kwargs


def test_openai_reasoning_effort_request_shape(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeResponses:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(output_text="reasoned")

    fake_responses = FakeResponses()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = fake_responses

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    text, _citations, _response = llm_utils.call_llm_text(
        prompt="hello",
        model=llm_utils.MODEL_HIGH,
        api_key=None,
        max_tokens=456,
        reasoning_effort=llm_utils.REASONING_HIGH,
    )

    assert text == "reasoned"
    assert fake_responses.kwargs["model"] == "gpt-5.5"
    assert fake_responses.kwargs["reasoning"] == {"effort": "high"}


def test_anthropic_adaptive_thinking_request_shape(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    class FakeMessages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="reasoned")], stop_reason="end_turn")

    fake_messages = FakeMessages()

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = fake_messages

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    text, _citations, _response = llm_utils.call_llm_text(
        prompt="hello",
        model=llm_utils.MODEL_MID,
        api_key=None,
        max_tokens=4096,
        reasoning_effort=llm_utils.REASONING_MEDIUM,
    )

    assert text == "reasoned"
    assert fake_messages.kwargs["model"] == "claude-sonnet-4-6"
    assert fake_messages.kwargs["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert fake_messages.kwargs["output_config"] == {"effort": "medium"}


def test_anthropic_manual_thinking_request_shape(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    class FakeMessages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="reasoned")], stop_reason="end_turn")

    fake_messages = FakeMessages()

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = fake_messages

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    text, _citations, _response = llm_utils.call_llm_text(
        prompt="hello",
        model=llm_utils.MODEL_LOW,
        api_key=None,
        max_tokens=4096,
        reasoning_effort=llm_utils.REASONING_HIGH,
    )

    assert text == "reasoned"
    assert fake_messages.kwargs["model"] == "claude-haiku-4-5"
    assert fake_messages.kwargs["thinking"] == {
        "type": "enabled",
        "budget_tokens": 2048,
        "display": "omitted",
    }
    assert "output_config" not in fake_messages.kwargs


def test_pdf_input_shapes(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeResponses:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(output_text="pdf answer")

    fake_responses = FakeResponses()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = fake_responses

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    text, _citations, _response = llm_utils.call_llm_pdf_text(
        pdf_bytes=b"%PDF-test",
        prompt="read",
        model=llm_utils.MODEL_MID,
        api_key=None,
        system="system",
        filename="deck.pdf",
    )

    assert text == "pdf answer"
    content = fake_responses.kwargs["input"][0]["content"]
    assert content[0]["type"] == "input_file"
    assert content[0]["filename"] == "deck.pdf"
    assert content[0]["file_data"].startswith("data:application/pdf;base64,")
    assert content[1] == {"type": "input_text", "text": "read"}
