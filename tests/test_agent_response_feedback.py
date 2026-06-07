from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_feedback_stores(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setenv("TALISMAN_ALLOW_SQLITE_STATE", "true")
    monkeypatch.setenv("DATABASE_URL", "")
    from api import agent_response_feedback, agent_trajectories

    monkeypatch.setattr(agent_trajectories, "_SQLITE_PATH", tmp_path / "agent_trajectories.sqlite3")
    monkeypatch.setattr(agent_response_feedback, "_SQLITE_PATH", tmp_path / "agent_response_feedback.sqlite3")
    yield
    agent_trajectories.reset_agent_trajectory_store_for_tests()
    agent_response_feedback.reset_agent_response_feedback_store_for_tests()


def _sample_trajectory(**overrides):
    from api.agent_trajectories import insert_trajectory

    payload = {
        "session_id": "sess-feedback",
        "client_turn_id": "turn-feedback",
        "final_disposition": "succeeded",
        "provider": "talisman",
        "model": "talisman-test",
        "messages": [
            {"role": "user", "content": "Review NVDA"},
            {"role": "assistant", "content": "NVDA looks constructive."},
        ],
        "steps": [
            {
                "step_id": "step-0",
                "index": 0,
                "kind": "final",
                "name": "assistant_response",
                "status": "ok",
                "payload": {},
            }
        ],
    }
    payload.update(overrides)
    trajectory_id = insert_trajectory(payload)
    assert trajectory_id
    from api.agent_trajectories import get_trajectory

    return get_trajectory(trajectory_id)


def test_feedback_upsert_is_idempotent_for_same_reviewer_and_response():
    from api.agent_response_feedback import list_feedback_for_trajectory, upsert_feedback

    trajectory = _sample_trajectory()
    first = upsert_feedback(
        {
            "feedback_id": "fb-1",
            "trajectory_id": trajectory["trajectory_id"],
            "session_id": trajectory["session_id"],
            "client_turn_id": trajectory["client_turn_id"],
            "response_version": "resp-v1",
            "decision": "approve",
            "reviewer_actor_id": "reviewer-1",
            "reviewed_at": "2026-06-07T12:00:00+00:00",
            "training_eligible": False,
        }
    )
    second = upsert_feedback(
        {
            **first,
            "decision": "reject",
            "notes": "Tool choice was wrong",
            "failure_tags": ["tools"],
            "reviewed_at": "2026-06-07T12:05:00+00:00",
        }
    )

    assert second["decision"] == "reject"
    assert second["notes"] == "Tool choice was wrong"
    rows = list_feedback_for_trajectory(trajectory["trajectory_id"])
    assert len(rows) == 1


def test_export_human_reviewed_feedback_excludes_unreviewed_labels():
    from api.agent_response_feedback import export_human_reviewed_feedback, upsert_feedback

    trajectory = _sample_trajectory(client_turn_id="turn-export")
    upsert_feedback(
        {
            "feedback_id": "fb-eligible",
            "trajectory_id": trajectory["trajectory_id"],
            "session_id": trajectory["session_id"],
            "client_turn_id": trajectory["client_turn_id"],
            "response_version": "resp-v1",
            "decision": "reject",
            "reviewer_actor_id": "reviewer-1",
            "reviewed_at": "2026-06-07T12:00:00+00:00",
            "training_eligible": True,
            "failure_tags": ["synthesis"],
        }
    )
    upsert_feedback(
        {
            "feedback_id": "fb-ineligible",
            "trajectory_id": trajectory["trajectory_id"],
            "session_id": trajectory["session_id"],
            "client_turn_id": trajectory["client_turn_id"],
            "response_version": "resp-v2",
            "decision": "approve",
            "reviewer_actor_id": "reviewer-2",
            "reviewed_at": "2026-06-07T12:01:00+00:00",
            "training_eligible": False,
        }
    )

    exported = export_human_reviewed_feedback()
    assert len(exported) == 1
    assert exported[0]["decision"] == "reject"
    assert exported[0]["human_reviewed"] is True
    assert exported[0]["signal_source"] == "human_reviewed"


def test_tombstone_trajectory_cascades_feedback_deletion():
    from api.agent_response_feedback import list_feedback, upsert_feedback
    from api.agent_trajectories import tombstone_trajectory

    trajectory = _sample_trajectory(client_turn_id="turn-delete")
    upsert_feedback(
        {
            "feedback_id": "fb-delete",
            "trajectory_id": trajectory["trajectory_id"],
            "session_id": trajectory["session_id"],
            "client_turn_id": trajectory["client_turn_id"],
            "response_version": "resp-v1",
            "decision": "approve",
            "reviewer_actor_id": "reviewer-1",
            "reviewed_at": "2026-06-07T12:00:00+00:00",
            "training_eligible": True,
        }
    )

    assert tombstone_trajectory(trajectory["trajectory_id"], reason="user_delete")
    remaining = list_feedback(training_eligible_only=True)
    assert remaining == []


def test_promote_trajectory_for_training_sets_consent_and_eligibility():
    from api.agent_trajectories import export_sanitized_trajectories, promote_trajectory_for_training

    trajectory = _sample_trajectory(client_turn_id="turn-promote")
    promoted = promote_trajectory_for_training(
        trajectory["trajectory_id"],
        reviewer_actor_id="reviewer-1",
    )
    assert promoted is not None
    assert promoted["consent_state"] == "granted"
    assert promoted["training_eligible"] is True
    assert "missing_training_consent" not in promoted["exclusion_reasons"]
    assert len(export_sanitized_trajectories()) == 1


def test_feedback_api_submit_and_list(auth_client):
    from api.agent_trajectories import insert_trajectory

    trajectory_id = insert_trajectory(
        {
            "session_id": "sess-api",
            "client_turn_id": "turn-api",
            "final_disposition": "succeeded",
            "provider": "openai",
            "model": "gpt-test",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "steps": [
                {
                    "step_id": "step-0",
                    "index": 0,
                    "kind": "final",
                    "name": "assistant_response",
                    "status": "ok",
                    "payload": {},
                }
            ],
        }
    )
    assert trajectory_id

    submit = auth_client.post(
        "/api/agent/feedback",
        json={
            "session_id": "sess-api",
            "client_turn_id": "turn-api",
            "decision": "reject",
            "failure_tags": ["routing"],
            "notes": "Wrong route",
            "eligible_for_training": True,
        },
    )
    assert submit.status_code == 200
    body = submit.json()
    assert body["feedback"]["decision"] == "reject"
    assert body["feedback"]["human_reviewed"] is True
    assert "governed training datasets" in body["disclosure"]

    listed = auth_client.get(
        "/api/agent/feedback",
        params={"session_id": "sess-api", "client_turn_id": "turn-api"},
    )
    assert listed.status_code == 200
    assert len(listed.json()["feedback"]) == 1


def test_feedback_api_requires_auth(client):
    response = client.post(
        "/api/agent/feedback",
        json={
            "session_id": "sess",
            "client_turn_id": "turn",
            "decision": "approve",
        },
    )
    assert response.status_code == 401


def test_feedback_migration_contract_contains_human_review_boundaries():
    from pathlib import Path

    migration = Path("migrations/versions/20260607_0002_agent_response_feedback.py").read_text(encoding="utf-8")
    assert "agent_response_feedback" in migration
    assert "training_eligible" in migration
    assert "signal_source" in migration
    assert "uq_agent_response_feedback_reviewer_turn" in migration


def test_correct_feedback_requires_corrected_response(auth_client):
    from api.agent_trajectories import insert_trajectory

    insert_trajectory(
        {
            "session_id": "sess-correct",
            "client_turn_id": "turn-correct",
            "final_disposition": "succeeded",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "steps": [
                {"step_id": "step-0", "index": 0, "kind": "final", "name": "final", "status": "ok", "payload": {}}
            ],
        }
    )

    response = auth_client.post(
        "/api/agent/feedback",
        json={
            "session_id": "sess-correct",
            "client_turn_id": "turn-correct",
            "decision": "correct",
        },
    )
    assert response.status_code == 400
