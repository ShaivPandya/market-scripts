from __future__ import annotations

from typing import Any

import pytest

import api.watch_trigger_monitor as monitor


class _FakeCommandService:
    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = calls

    def propose_action(self, action_id, payload, context, *, reason):
        self.calls.append(
            {
                "action_id": action_id,
                "payload": payload,
                "source_id": context.source_id,
                "source_type": context.source_type,
                "reason": reason,
            }
        )
        return {"id": f"approval:{len(self.calls)}"}


class _FakeReads:
    def __init__(self, triggers: list[dict[str, Any]]):
        self.triggers = triggers

    def watch_triggers(self, *, status):
        assert status == "active"
        return self.triggers


def _install_monitor_fakes(monkeypatch: pytest.MonkeyPatch, triggers: list[dict[str, Any]]):
    calls: list[dict[str, Any]] = []

    import ontology.command_service as command_service
    import ontology.runtime_read_service as runtime_read_service

    monkeypatch.setattr(command_service, "OntologyCommandService", lambda: _FakeCommandService(calls))
    monkeypatch.setattr(runtime_read_service, "OntologyRuntimeReadService", lambda: _FakeReads(triggers))
    return calls


def test_monitor_uses_normalized_source_id_for_inferred_and_skipped_checks(monkeypatch):
    calls = _install_monitor_fakes(
        monkeypatch,
        [{"object_uid": "watch_trigger:abc", "status": "active", "condition": "Watch A"}],
    )
    result = {
        "fired": False,
        "skipped": True,
        "evidence": "Trigger inferred but skipped",
        "inferred_definition": {"type": "price_level", "ticker": "ABC"},
    }
    monkeypatch.setattr(monitor, "evaluate_trigger", lambda _trigger: result)

    summary = monitor.run_watch_trigger_monitor()

    assert summary == {"checked": 1, "fired": 0, "skipped": 1, "errors": 0}
    assert [call["action_id"] for call in calls] == [
        "update_watch_trigger_definition",
        "update_watch_trigger_check",
    ]
    assert [call["source_id"] for call in calls] == ["watch_trigger:abc", "watch_trigger:abc"]
    assert calls[0]["payload"]["trigger_id"] == "watch_trigger:abc"
    assert calls[1]["payload"]["trigger_id"] == "watch_trigger:abc"


def test_monitor_fired_source_id_has_single_trigger_prefix_and_fingerprint(monkeypatch):
    calls = _install_monitor_fakes(
        monkeypatch,
        [{"object_uid": "watch_trigger:abc", "status": "active", "condition": "Watch A", "ticker": "ABC"}],
    )
    result = {
        "fired": True,
        "type": "price_level",
        "actual": 101,
        "expected": 100,
        "as_of": "2026-05-14T14:30:00Z",
        "evidence": "ABC crossed 100",
    }
    monkeypatch.setattr(monitor, "evaluate_trigger", lambda _trigger: result)

    summary = monitor.run_watch_trigger_monitor()

    expected_source_id = f"watch_trigger:abc:{monitor._result_fingerprint(result)}"
    assert summary == {"checked": 1, "fired": 1, "skipped": 0, "errors": 0}
    assert [call["action_id"] for call in calls] == ["fire_watch_trigger"]
    assert [call["source_id"] for call in calls] == [expected_source_id]
    assert "watch_trigger:watch_trigger:" not in expected_source_id


def test_monitor_error_path_uses_normalized_source_id(monkeypatch):
    calls = _install_monitor_fakes(
        monkeypatch,
        [{"object_uid": "watch_trigger:abc", "status": "active", "condition": "Watch A"}],
    )

    def fail(_trigger):
        raise RuntimeError("boom")

    monkeypatch.setattr(monitor, "evaluate_trigger", fail)

    summary = monitor.run_watch_trigger_monitor()

    assert summary == {"checked": 1, "fired": 0, "skipped": 0, "errors": 1}
    assert [call["action_id"] for call in calls] == ["update_watch_trigger_check"]
    assert calls[0]["source_id"] == "watch_trigger:abc"
    assert calls[0]["payload"]["trigger_id"] == "watch_trigger:abc"
    assert calls[0]["payload"]["result"] == {"error": "boom", "fired": False}


def test_monitor_normalizes_legacy_numeric_id_source_id(monkeypatch):
    calls = _install_monitor_fakes(
        monkeypatch,
        [{"id": "123", "status": "active", "condition": "Watch legacy"}],
    )
    monkeypatch.setattr(
        monitor,
        "evaluate_trigger",
        lambda _trigger: {"fired": False, "skipped": False, "evidence": "Still active"},
    )

    summary = monitor.run_watch_trigger_monitor()

    assert summary == {"checked": 1, "fired": 0, "skipped": 0, "errors": 0}
    assert [call["action_id"] for call in calls] == ["update_watch_trigger_check"]
    assert calls[0]["source_id"] == "watch_trigger:123"
    assert calls[0]["payload"]["trigger_id"] == "123"
