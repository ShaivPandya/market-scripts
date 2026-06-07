from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _isolate_training_stores(tmp_path, monkeypatch):
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
    from api.agent_trajectories import insert_trajectory, promote_trajectory_for_training

    payload = {
        "session_id": "sess-dataset",
        "client_turn_id": "turn-dataset",
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
        "consent_state": "granted",
        "training_eligible": True,
    }
    payload.update(overrides)
    trajectory_id = insert_trajectory(payload)
    assert trajectory_id
    promote_trajectory_for_training(trajectory_id, reviewer_actor_id="reviewer-test")
    from api.agent_trajectories import get_trajectory

    return get_trajectory(trajectory_id)


def test_build_training_dataset_joins_approve_feedback_to_trajectory():
    from api.agent_response_feedback import response_version_for_trajectory, upsert_feedback
    from decision_quality.agent_training_datasets import build_training_dataset

    trajectory = _sample_trajectory(client_turn_id="turn-sft")
    response_version = response_version_for_trajectory(trajectory)
    upsert_feedback(
        {
            "feedback_id": "fb-sft",
            "trajectory_id": trajectory["trajectory_id"],
            "session_id": trajectory["session_id"],
            "client_turn_id": trajectory["client_turn_id"],
            "response_version": response_version,
            "decision": "approve",
            "reviewer_actor_id": "reviewer-1",
            "reviewed_at": "2026-06-07T12:00:00+00:00",
            "training_eligible": True,
        }
    )

    bundle = build_training_dataset(include_eval_fixtures=False, include_seeds=False)
    assert len(bundle["sft_rows"]) == 1
    assert bundle["sft_rows"][0]["source_type"] == "trajectory"
    assert bundle["sft_rows"][0]["signal_source"] == "human_reviewed"
    assert bundle["preference_rows"] == []


def test_build_training_dataset_creates_preference_rows_for_reject_and_correct():
    from api.agent_response_feedback import response_version_for_trajectory, upsert_feedback
    from decision_quality.agent_training_datasets import build_training_dataset

    reject_trajectory = _sample_trajectory(client_turn_id="turn-reject")
    reject_version = response_version_for_trajectory(reject_trajectory)
    upsert_feedback(
        {
            "feedback_id": "fb-reject",
            "trajectory_id": reject_trajectory["trajectory_id"],
            "session_id": reject_trajectory["session_id"],
            "client_turn_id": reject_trajectory["client_turn_id"],
            "response_version": reject_version,
            "decision": "reject",
            "reviewer_actor_id": "reviewer-1",
            "reviewed_at": "2026-06-07T12:00:00+00:00",
            "training_eligible": True,
            "failure_tags": ["synthesis"],
        }
    )

    correct_trajectory = _sample_trajectory(client_turn_id="turn-correct")
    correct_version = response_version_for_trajectory(correct_trajectory)
    upsert_feedback(
        {
            "feedback_id": "fb-correct",
            "trajectory_id": correct_trajectory["trajectory_id"],
            "session_id": correct_trajectory["session_id"],
            "client_turn_id": correct_trajectory["client_turn_id"],
            "response_version": correct_version,
            "decision": "correct",
            "reviewer_actor_id": "reviewer-1",
            "reviewed_at": "2026-06-07T12:01:00+00:00",
            "training_eligible": True,
            "corrected_response": "NVDA looks constructive with tighter risk framing.",
            "failure_tags": ["calibration"],
        }
    )

    bundle = build_training_dataset(include_eval_fixtures=False, include_seeds=False)
    assert len(bundle["preference_rows"]) == 2
    decisions = {row["decision"] for row in bundle["preference_rows"]}
    assert decisions == {"reject", "correct"}
    corrected = next(row for row in bundle["preference_rows"] if row["decision"] == "correct")
    assert corrected["chosen"] == "NVDA looks constructive with tighter risk framing."


def test_export_is_deterministic_for_fixed_version():
    from decision_quality.agent_training_datasets import export_training_dataset

    first = export_training_dataset(
        export_version="fixed-version",
        include_eval_fixtures=False,
        include_seeds=False,
        dry_run=True,
    )
    second = export_training_dataset(
        export_version="fixed-version",
        include_eval_fixtures=False,
        include_seeds=False,
        dry_run=True,
    )
    assert first["content_hashes"] == second["content_hashes"]
    assert first["sft_count"] == second["sft_count"]
    assert first["preference_count"] == second["preference_count"]


def test_release_gate_case_collision_fails_export(tmp_path):
    from decision_quality.agent_training_datasets import (
        AgentTrainingDatasetError,
        build_training_dataset,
        release_gate_case_ids,
        seed_row_to_sft_row,
    )

    blocked_case_id = next(iter(release_gate_case_ids()))
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    poison = seed_row_to_sft_row(
        {
            "source_type": "synthetic",
            "source_id": blocked_case_id,
            "example_id": f"sft:synthetic:{blocked_case_id}",
            "messages": [{"role": "user", "content": "test"}],
            "target_output": "blocked",
            "provenance": {"case_id": blocked_case_id},
        }
    )
    (seeds_dir / "poison.jsonl").write_text(json.dumps(poison) + "\n", encoding="utf-8")

    with pytest.raises(AgentTrainingDatasetError, match="Release-gate contamination"):
        build_training_dataset(
            include_eval_fixtures=False,
            include_seeds=True,
            seeds_dir=seeds_dir,
        )


def test_write_training_dataset_reproduces_manifest_hashes(tmp_path):
    from decision_quality.agent_training_datasets import export_training_dataset

    manifest = export_training_dataset(
        output_dir=tmp_path / "outputs",
        export_version="hash-check",
        include_eval_fixtures=False,
        include_seeds=False,
    )
    manifest_path = tmp_path / "outputs" / "hash-check" / "manifest.json"
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["content_hashes"] == manifest["content_hashes"]
    assert stored["version"] == "hash-check"


def test_admin_export_requires_admin(client):
    response = client.post("/api/admin/agent/training-datasets/export?dry_run=true")
    assert response.status_code == 401


def test_admin_export_returns_manifest(auth_client, monkeypatch):
    monkeypatch.setattr(
        "decision_quality.agent_training_datasets.export_training_dataset",
        lambda **kwargs: {
            "version": "test-version",
            "sft_count": 1,
            "preference_count": 0,
            "exclusion_count": 0,
            "leakage_check_passed": True,
            "content_hashes": {"sft.jsonl": "abc", "preference.jsonl": "def", "manifest.json": "ghi"},
            "split_counts": {"train": 1},
            "source_counts": {"sft": {"trajectory": 1}, "preference": {}},
        },
    )
    response = auth_client.post("/api/admin/agent/training-datasets/export?dry_run=true")
    assert response.status_code == 200
    body = response.json()
    assert body["manifest"]["version"] == "test-version"
    assert body["manifest"]["sft_count"] == 1
