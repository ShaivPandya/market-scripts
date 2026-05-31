"""Decision outcome API endpoint tests."""

from __future__ import annotations


def test_finalize_decision_outcome_endpoint_uses_typed_actor(auth_client, monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_finalize_decision_outcome(decision_outcome_id: str, **kwargs):
        calls.append({"decision_outcome_id": decision_outcome_id, **kwargs})
        return {
            "uid": f"decision_outcome:{decision_outcome_id}",
            "outcome_status": "evaluated",
            "final_label_status": "confirmed",
        }

    monkeypatch.setattr(
        "api.routers.decision_outcomes.finalize_decision_outcome",
        fake_finalize_decision_outcome,
    )

    response = auth_client.post(
        "/api/decision-outcomes/outcome-1/finalize",
        json={"decision": "confirm", "note": "reviewed"},
    )

    assert response.status_code == 200
    assert calls == [
        {
            "decision_outcome_id": "outcome-1",
            "decision": "confirm",
            "note": "reviewed",
            "corrected_postmortem": None,
            "lessons_learned": None,
            "actor_id": "admin",
        }
    ]
    assert response.json()["learning_state"] == "confirmed"
