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


def test_builder_monitor_preview_wraps_watch_trigger_evaluation(monkeypatch):
    import api.mission_runner as runner

    monkeypatch.setattr(
        monitor,
        "evaluate_trigger",
        lambda trigger: {
            "fired": True,
            "type": trigger["definition"]["type"],
            "evidence": f"{trigger['ticker']} matched",
            "source_ids": ["trusted_news"],
        },
    )

    result = runner.evaluate_monitor_definition(
        {
            "object_uid": "monitor_definition:mu",
            "name": "MU thesis monitor",
            "scope": {"ticker": "MU"},
            "trigger_type": "fundamental_news",
            "condition": "Watch MU thesis evidence",
            "definition": {"query": "MU demand"},
            "source_requirements": [{"source_name": "trusted_news", "required": True}],
        }
    )

    assert result["fired"] is True
    assert result["type"] == "fundamental_news"
    assert result["source_requirement_review"]["status"] == "ok"


def test_builder_runner_records_hits_and_review_approvals(monkeypatch):
    import api.mission_runner as runner
    from api import workflows

    class _FakeReads:
        def monitor_definitions(self, *, status, limit):
            assert status == "active"
            return [
                {
                    "object_uid": "monitor_definition:mu",
                    "name": "MU thesis monitor",
                    "scope": {"ticker": "MU"},
                    "trigger_type": "custom",
                    "condition": "Watch MU",
                    "definition_hash": "hash1",
                }
            ]

        def mission_definitions(self, *, status, limit):
            assert status == "active"
            return []

    class _FakeService:
        def __init__(self):
            pass

        def propose_action(self, action_id, payload, context, *, reason):
            calls.append({"action_id": action_id, "payload": payload, "source_id": context.source_id, "reason": reason})
            return {"id": f"approval:{len(calls)}"}

        def resolve_approval(self, approval_id, status, note, context):
            return {"id": approval_id, "status": status, "application_status": "applied"}

    calls: list[dict[str, Any]] = []
    artifacts: list[tuple[str, str, Any]] = []
    monkeypatch.setattr(runner, "OntologyRuntimeReadService", lambda: _FakeReads())
    monkeypatch.setattr(runner, "OntologyCommandService", _FakeService)
    monkeypatch.setattr(
        runner, "evaluate_monitor_definition", lambda _definition: {"fired": True, "evidence": "matched"}
    )
    monkeypatch.setattr(
        workflows, "create_workflow_run", lambda *_args, **_kwargs: {"run_id": "workflow:monitor_mission_runner:test"}
    )
    monkeypatch.setattr(
        workflows,
        "complete_workflow_run",
        lambda run_id, synthesis, artifacts_arg=None, *_args, **_kwargs: {"run_id": run_id},
    )
    monkeypatch.setattr(workflows, "fail_workflow_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner, "_write_workflow_artifact", lambda run_id, key, value: artifacts.append((run_id, key, value))
    )

    summary = runner.run_monitor_mission_runner()

    assert summary["checked"] == 1
    assert summary["hits"] == 1
    assert [call["action_id"] for call in calls] == ["create_monitor_hit", "create_action_item"]
    assert calls[0]["payload"]["entity_type"] == "monitor_definition"
    assert "monitor_definition:mu" in calls[0]["source_id"]
    assert artifacts[0][1] == "monitor_mission_results"
