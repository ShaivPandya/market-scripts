from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _isolate_trajectory_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setenv("TALISMAN_ALLOW_SQLITE_STATE", "true")
    monkeypatch.setenv("DATABASE_URL", "")
    from api import agent_trajectories

    monkeypatch.setattr(agent_trajectories, "_SQLITE_PATH", tmp_path / "agent_trajectories.sqlite3")
    yield
    agent_trajectories.reset_agent_trajectory_store_for_tests()


def _sample_payload(**overrides):
    payload = {
        "session_id": "sess-1",
        "client_turn_id": "turn-1",
        "final_disposition": "succeeded",
        "provider": "talisman",
        "model": "talisman-test",
        "messages": [
            {"role": "user", "content": "Review NVDA using sk-test-secret-1234567890abcdef"},
            {"role": "assistant", "content": "Done"},
        ],
        "steps": [
            {
                "step_id": "step-0",
                "index": 0,
                "kind": "route",
                "name": "intent_router",
                "status": "applied",
                "payload": {"intent_class": "thesis_review"},
            },
            {
                "step_id": "step-1",
                "index": 1,
                "kind": "tool_call",
                "name": "get_portfolio",
                "status": "ok",
                "payload": {
                    "arguments": {"ticker": "NVDA", "api_key": "secret-value"},
                    "result": {"position": "private"},
                },
            },
        ],
        "consent_state": "granted",
        "training_eligible": True,
    }
    payload.update(overrides)
    return payload


def test_insert_export_sanitizes_training_view():
    from api.agent_trajectories import export_sanitized_trajectories, get_trajectory, insert_trajectory

    trajectory_id = insert_trajectory(_sample_payload())
    assert trajectory_id

    stored = get_trajectory(trajectory_id)
    assert stored is not None
    assert stored["training_eligible"] is True
    assert stored["redaction_manifest"]["policy"] == "agent_trajectory_training_v1"

    exported = export_sanitized_trajectories()
    assert len(exported) == 1
    rendered = json.dumps(exported[0], sort_keys=True)
    assert "sk-test-secret" not in rendered
    assert "secret-value" not in rendered
    assert "[REDACTED" in rendered
    assert '"raw_payload_ref"' in rendered
    tool_payload = exported[0]["steps"][1]["payload"]
    assert tool_payload["arguments"]["redacted"] is True
    assert tool_payload["result"]["redacted"] is True


def test_tombstone_removes_training_export_eligibility():
    from api.agent_trajectories import (
        export_sanitized_trajectories,
        get_trajectory,
        insert_trajectory,
        tombstone_trajectory,
    )

    trajectory_id = insert_trajectory(_sample_payload(client_turn_id="turn-delete"))
    assert trajectory_id
    assert tombstone_trajectory(trajectory_id, reason="user_delete")

    stored = get_trajectory(trajectory_id)
    assert stored is not None
    assert stored["tombstoned_at"]
    assert stored["training_eligible"] is False
    assert export_sanitized_trajectories() == []


def test_export_rejects_unknown_schema_and_unredacted_payloads():
    from api.agent_trajectories import TrajectoryExportError, _exportable_payload

    with pytest.raises(TrajectoryExportError, match="Unsupported trajectory schema"):
        _exportable_payload(
            {
                "trajectory_id": "traj-unknown",
                "schema_version": 99,
                "training_eligible": True,
                "redaction_manifest": {"policy": "agent_trajectory_training_v1"},
                "sanitized_payload": {"messages": [{}], "steps": [{}]},
            }
        )

    with pytest.raises(TrajectoryExportError, match="restricted fields"):
        _exportable_payload(
            {
                "trajectory_id": "traj-secret",
                "schema_version": 1,
                "training_eligible": True,
                "redaction_manifest": {"policy": "agent_trajectory_training_v1"},
                "sanitized_payload": {
                    "messages": [{"role": "user", "content": "sk-live-secret-1234567890abcdef"}],
                    "steps": [{"step_id": "step", "index": 0}],
                },
            }
        )


def test_dataset_split_group_is_stable_for_session_turn():
    from api.agent_trajectories import dataset_split_group_for

    first = dataset_split_group_for(session_id="sess", client_turn_id="turn", messages=[{"content": "A"}])
    second = dataset_split_group_for(session_id="sess", client_turn_id="turn", messages=[{"content": "B"}])
    assert first == second


def test_migration_contract_contains_training_boundaries():
    from pathlib import Path

    migration = Path("migrations/versions/20260607_0001_agent_trajectories.py").read_text(encoding="utf-8")
    assert "agent_trajectories" in migration
    assert "raw_payload_json" in migration
    assert "sanitized_payload_json" in migration
    assert "redaction_manifest_json" in migration
    assert "dataset_split_group" in migration


def test_casual_agent_turn_captures_trajectory_without_sse_contract_change(auth_client):
    from api.agent_trajectories import list_trajectories

    response = auth_client.post(
        "/api/agent/chat",
        json={"message": "hello", "session_id": "traj-casual-session", "client_turn_id": "turn-casual"},
    )
    assert response.status_code == 200
    assert "event: trajectory" not in response.text
    assert "event: done" in response.text

    rows = list_trajectories(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == "traj-casual-session"
    assert row["client_turn_id"] == "turn-casual"
    assert row["sanitized_payload"]["final_disposition"] == "succeeded"
    assert any(step["kind"] == "final" for step in row["sanitized_payload"]["steps"])
