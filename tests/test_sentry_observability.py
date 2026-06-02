"""Tests for optional Sentry observability and event scrubbing."""

from __future__ import annotations

import pytest

from api import observability


@pytest.fixture(autouse=True)
def _reset_sentry_state(monkeypatch):
    observability._INITIALIZED = False
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.setenv("SENTRY_ENABLED", "true")
    yield
    observability._INITIALIZED = False


def test_sentry_disabled_without_dsn(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert observability.sentry_enabled() is False
    assert observability.init_sentry(component="api") is False


def test_sentry_disabled_when_explicitly_off(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "https://example@o0.ingest.sentry.io/0")
    monkeypatch.setenv("SENTRY_ENABLED", "false")
    assert observability.sentry_enabled() is False


def test_init_sentry_is_non_fatal_with_invalid_dsn(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "not-a-valid-dsn")
    # Should not raise even if SDK rejects the DSN.
    result = observability.init_sentry(component="api")
    assert result in {True, False}


def test_scrub_event_payload_redacts_sensitive_keys():
    event = {
        "request": {
            "url": "https://app.example/api/agent/chat?token=secret",
            "headers": {
                "Authorization": "Bearer abc",
                "X-CSRF-Token": "csrf-value",
                "Accept": "application/json",
            },
            "data": {"prompt": "private thesis text", "messages": [{"role": "user", "content": "holdings"}]},
            "cookies": {"__session": "session-id"},
        },
        "extra": {
            "payload_json": {"positions": [{"ticker": "MSFT", "weight": 12}]},
            "job_id": "job-123",
        },
    }
    scrubbed = observability.scrub_event_payload(event)
    assert scrubbed is not None
    assert "secret" not in str(scrubbed["request"]["url"])
    assert "data" not in scrubbed["request"]
    assert "cookies" not in scrubbed["request"]
    assert "REDACTED" in str(scrubbed["request"]["headers"])
    assert "MSFT" not in str(scrubbed["extra"])


def test_capture_exception_noops_when_uninitialized(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert observability.capture_exception(RuntimeError("boom")) is None


def test_capture_exception_scrubs_context(monkeypatch):
    captured: dict = {}

    class FakeScope:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def set_tag(self, key, value):
            captured.setdefault("tags", {})[key] = value

        def set_context(self, key, value):
            captured.setdefault("contexts", {})[key] = value

    def fake_capture_exception(exc):
        captured["exc"] = exc
        return "event-id"

    monkeypatch.setattr(observability, "_INITIALIZED", True)
    monkeypatch.setattr("sentry_sdk.push_scope", lambda: FakeScope())
    monkeypatch.setattr("sentry_sdk.capture_exception", fake_capture_exception)

    observability.capture_exception(
        RuntimeError("failed"),
        tags={"job_type": "analyzer"},
        context={"job_id": "job-1", "password": "secret", "prompt": "private"},
    )

    assert captured["exc"].args == ("failed",)
    assert captured["tags"]["job_type"] == "analyzer"
    assert "secret" not in str(captured["contexts"]["talisman"])
    assert "private" not in str(captured["contexts"]["talisman"])


def test_capture_message_noops_when_uninitialized():
    assert observability.capture_message("workflow failed") is None
