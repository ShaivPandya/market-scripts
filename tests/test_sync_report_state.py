from __future__ import annotations

import pytest
import requests

from auto_report import sync_report_state


class _FakeResponse:
    status_code = 200
    text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"ok": True}


def test_sync_payload_sends_report_sync_schema_headers(monkeypatch):
    captured: dict = {}

    def fake_post(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse()

    monkeypatch.setenv("TALISMAN_API_URL", "https://example.test")
    monkeypatch.setenv("REPORT_SYNC_SECRET", "sync-secret")
    monkeypatch.setenv("API_PROXY_SECRET", "proxy-secret")
    monkeypatch.setattr(sync_report_state.requests, "post", fake_post)

    assert sync_report_state.sync_payload("daily", {"as_of": "2026-05-03"}) == {"ok": True}

    assert captured["url"] == "https://example.test/api/report-sync/daily"
    assert captured["headers"] == {
        "Content-Type": "application/json",
        "X-Report-Sync-Secret": "sync-secret",
        "X-Request-Schema-Name": "post:/api/report-sync/{report_type}",
        "X-Request-Schema-Version": "1",
        "X-Api-Proxy-Secret": "proxy-secret",
    }
    assert captured["json"] == {"as_of": "2026-05-03"}


def test_sync_payload_surfaces_api_error_detail(monkeypatch):
    class ErrorResponse:
        status_code = 422
        text = ""

        def raise_for_status(self) -> None:
            raise requests.HTTPError("422", response=self)

        def json(self):
            return {"detail": "Node audit_event:abc has non-canonical identity"}

    monkeypatch.setenv("TALISMAN_API_URL", "https://example.test")
    monkeypatch.setenv("REPORT_SYNC_SECRET", "sync-secret")
    monkeypatch.setattr(sync_report_state.requests, "post", lambda *_args, **_kwargs: ErrorResponse())

    with pytest.raises(RuntimeError, match="non-canonical identity"):
        sync_report_state.sync_payload("daily", {"as_of": "2026-05-03"})
