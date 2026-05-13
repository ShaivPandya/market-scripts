from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import llm_utils


def test_provider_and_model_resolution_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_utils.selected_provider() == "anthropic"
    assert llm_utils.model_for_tier(llm_utils.MODEL_LOW, "anthropic") == "claude-haiku-4-5"
    assert llm_utils.model_for_tier(llm_utils.MODEL_MID, "openai") == "gpt-5.4"
    assert llm_utils.model_for_tier(llm_utils.MODEL_LOW, "gemini") == "gemini-3.1-flash-lite"
    assert llm_utils.model_for_tier(llm_utils.MODEL_MID, "gemini") == "gemini-3.1-pro-preview-customtools"
    assert llm_utils.resolve_model("claude-opus-4-7", "openai") == "gpt-5.5"


def test_provider_model_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL_HIGH", "gpt-custom-high")
    assert llm_utils.model_for_tier(llm_utils.MODEL_HIGH, "openai") == "gpt-custom-high"

    monkeypatch.setenv("GEMINI_MODEL_MID", "gemini-custom-mid")
    assert llm_utils.model_for_tier(llm_utils.MODEL_MID, "gemini") == "gemini-custom-mid"


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

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        llm_utils.require_api_key()
    except RuntimeError as exc:
        assert "GEMINI_API_KEY" in str(exc)
    else:
        raise AssertionError("missing Gemini key should fail")

    monkeypatch.setenv("LLM_PROVIDER", "local")
    with pytest.raises(ValueError, match="anthropic.*openai.*gemini"):
        llm_utils.require_api_key()


def _install_fake_gemini(monkeypatch, fake_client):
    import google

    fake_genai = SimpleNamespace(Client=lambda *args, **kwargs: fake_client)
    monkeypatch.setattr(google, "genai", fake_genai, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    llm_utils._CLIENT_CACHE.clear()
    llm_utils._CLIENT_FACTORY_CACHE.clear()


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
            "search_context_size": "medium",
        }
    ]
    assert "reasoning" not in fake_responses.kwargs


def test_gemini_text_request_shape_reasoning_and_citations(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-12345678901234567890")

    class FakeModels:
        def __init__(self):
            self.kwargs = None

        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                text="gemini answer",
                candidates=[
                    SimpleNamespace(
                        grounding_metadata=SimpleNamespace(
                            grounding_chunks=[
                                SimpleNamespace(web=SimpleNamespace(title="Gemini Source", uri="https://example.com/g"))
                            ]
                        )
                    )
                ],
            )

    fake_models = FakeModels()
    _install_fake_gemini(monkeypatch, SimpleNamespace(models=fake_models))

    text, citations, _response = llm_utils.call_llm_text(
        prompt="hello",
        model=llm_utils.MODEL_MID,
        api_key=None,
        max_tokens=789,
        system="system",
        reasoning_effort=llm_utils.REASONING_HIGH,
    )

    assert text == "gemini answer"
    assert citations == [("Gemini Source", "https://example.com/g")]
    assert fake_models.kwargs["model"] == "gemini-3.1-pro-preview-customtools"
    assert fake_models.kwargs["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert fake_models.kwargs["config"] == {
        "max_output_tokens": 789,
        "system_instruction": "system",
        "thinking_config": {"thinking_level": "high"},
    }


def test_gemini_allowed_domains_enable_unrestricted_search(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-12345678901234567890")

    class FakeModels:
        def __init__(self):
            self.kwargs = None

        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(text="grounded")

    fake_models = FakeModels()
    _install_fake_gemini(monkeypatch, SimpleNamespace(models=fake_models))

    text, _citations, _response = llm_utils.call_llm_text(
        prompt="latest",
        model=llm_utils.MODEL_LOW,
        allowed_domains=["example.com"],
        max_tokens=128,
    )

    assert text == "grounded"
    assert fake_models.kwargs["config"]["tools"] == [{"google_search": {}}]


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
        reasoning_effort=llm_utils.REASONING_XHIGH,
    )

    assert text == "reasoned"
    assert fake_responses.kwargs["model"] == "gpt-5.5"
    assert fake_responses.kwargs["reasoning"] == {"effort": "xhigh"}


def test_openai_none_reasoning_effort_request_shape(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeResponses:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(output_text="fast")

    fake_responses = FakeResponses()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = fake_responses

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    text, _citations, _response = llm_utils.call_llm_text(
        prompt="hello",
        model=llm_utils.MODEL_MID,
        api_key=None,
        max_tokens=456,
        reasoning_effort=llm_utils.REASONING_NONE,
    )

    assert text == "fast"
    assert fake_responses.kwargs["model"] == "gpt-5.4"
    assert fake_responses.kwargs["reasoning"] == {"effort": "none"}


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
        reasoning_effort=llm_utils.REASONING_HIGH,
    )

    assert text == "reasoned"
    assert fake_messages.kwargs["model"] == "claude-sonnet-4-6"
    assert fake_messages.kwargs["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert fake_messages.kwargs["output_config"] == {"effort": "high"}


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
    assert fake_messages.kwargs["output_config"] == {"effort": "high"}


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


def test_gemini_pdf_input_shape(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-12345678901234567890")

    class FakeModels:
        def __init__(self):
            self.kwargs = None

        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(text="pdf answer")

    fake_models = FakeModels()
    _install_fake_gemini(monkeypatch, SimpleNamespace(models=fake_models))

    text, _citations, _response = llm_utils.call_llm_pdf_text(
        pdf_bytes=b"%PDF-test",
        prompt="read",
        model=llm_utils.MODEL_LOW,
        api_key=None,
        system="system",
        filename="deck.pdf",
        reasoning_effort=llm_utils.REASONING_MINIMAL,
    )

    assert text == "pdf answer"
    assert fake_models.kwargs["model"] == "gemini-3.1-flash-lite"
    assert fake_models.kwargs["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "application/pdf"
    assert fake_models.kwargs["contents"][0]["parts"][0]["inline_data"]["data"].startswith("JVBERi10ZXN0")
    assert fake_models.kwargs["contents"][0]["parts"][1] == {"text": "read"}
    assert fake_models.kwargs["config"]["thinking_config"] == {"thinking_level": "minimal"}
