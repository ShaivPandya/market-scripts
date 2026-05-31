from __future__ import annotations

from datetime import UTC, date, timedelta
from typing import Any

import pytest

import api.catalyst_kill_monitor as monitor


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
    def __init__(
        self,
        *,
        catalysts: list[dict[str, Any]] | None = None,
        kill_conditions: list[dict[str, Any]] | None = None,
        hits: list[dict[str, Any]] | None = None,
    ):
        self.catalysts_data = catalysts or []
        self.kill_conditions_data = kill_conditions or []
        self.hits = hits or []

    def catalysts(self, status=None, limit=100):
        assert status == "pending"
        return self.catalysts_data

    def kill_conditions(self, status=None, limit=100):
        assert status == "active"
        return self.kill_conditions_data

    def monitor_hits(self, *, entity_id=None, limit=100, **kwargs):
        return [hit for hit in self.hits if not entity_id or hit.get("entity_id") == entity_id][:limit]


def _install_monitor_fakes(monkeypatch: pytest.MonkeyPatch, reads: _FakeReads):
    calls: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []

    import api.action_execution as action_execution
    import ontology.command_service as command_service
    import ontology.runtime_read_service as runtime_read_service

    monkeypatch.setattr(command_service, "OntologyCommandService", lambda: _FakeCommandService(calls))
    monkeypatch.setattr(runtime_read_service, "OntologyRuntimeReadService", lambda: reads)

    def fake_execute(action_id, payload, *, source_id, actor, request_mode):
        applied.append(
            {
                "action_id": action_id,
                "payload": payload,
                "source_id": source_id,
                "request_mode": request_mode,
            }
        )
        return {"status": "applied"}

    monkeypatch.setattr(action_execution, "execute_api_action", fake_execute)
    return calls, applied


def test_evaluate_catalyst_approaching_target_date():
    target = (date.today() + timedelta(days=7)).isoformat()
    result = monitor.evaluate_catalyst(
        {
            "object_uid": "catalyst:abc",
            "ticker": "MU",
            "description": "Earnings beat",
            "status": "pending",
            "target_date": target,
        }
    )
    assert result["hit_type"] == "approaching"
    assert result["confidence"] == 0.7


def test_evaluate_kill_condition_triggered():
    monkeypatch_price = pytest.MonkeyPatch()
    monkeypatch_price.setattr(monitor, "_metric_value", lambda _metric, _ticker: 90.0)
    try:
        result = monitor.evaluate_kill_condition(
            {
                "object_uid": "kill_condition:abc",
                "ticker": "MU",
                "condition": "Price below support",
                "status": "active",
                "metric": "price",
                "threshold": "<= 95",
            }
        )
    finally:
        monkeypatch_price.undo()

    assert result["hit_type"] == "triggered"
    assert result["suggested_status"] == "triggered"


def test_monitor_records_hit_and_proposes_status(monkeypatch):
    target = (date.today() - timedelta(days=2)).isoformat()
    reads = _FakeReads(
        catalysts=[
            {
                "object_uid": "catalyst:abc",
                "ticker": "MU",
                "description": "Earnings beat",
                "status": "pending",
                "target_date": target,
            }
        ]
    )
    calls, applied = _install_monitor_fakes(monkeypatch, reads)

    summary = monitor.run_catalyst_kill_monitor()

    assert summary["checked"] == 1
    assert summary["hits"] == 1
    assert summary["proposals"] == 1
    assert len(applied) == 1
    assert applied[0]["action_id"] == "create_monitor_hit"
    assert applied[0]["payload"]["hit_type"] == "needs_review"
    assert [call["action_id"] for call in calls] == ["update_catalyst_status"]


def test_monitor_skips_duplicate_fingerprint(monkeypatch):
    target = (date.today() + timedelta(days=3)).isoformat()
    catalyst = {
        "object_uid": "catalyst:abc",
        "ticker": "MU",
        "description": "Earnings beat",
        "status": "pending",
        "target_date": target,
    }
    fingerprint = monitor._result_fingerprint(monitor.evaluate_catalyst(catalyst))
    reads = _FakeReads(
        catalysts=[catalyst],
        hits=[{"entity_id": "catalyst:abc", "fingerprint": fingerprint}],
    )
    calls, applied = _install_monitor_fakes(monkeypatch, reads)

    summary = monitor.run_catalyst_kill_monitor()

    assert summary == {"checked": 1, "hits": 0, "skipped": 1, "proposals": 0, "errors": 0}
    assert calls == []
    assert applied == []


def test_monitor_skips_unparseable_kill_condition(monkeypatch):
    reads = _FakeReads(
        kill_conditions=[
            {
                "object_uid": "kill_condition:abc",
                "ticker": "MU",
                "condition": "Management credibility collapses",
                "status": "active",
                "metric": None,
                "threshold": None,
            }
        ]
    )
    calls, applied = _install_monitor_fakes(monkeypatch, reads)

    summary = monitor.run_catalyst_kill_monitor()

    assert summary == {"checked": 1, "hits": 0, "skipped": 1, "proposals": 0, "errors": 0}
    assert calls == []
    assert applied == []
