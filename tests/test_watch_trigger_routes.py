from __future__ import annotations


def test_cancel_trigger_route_stages_uid(auth_client, monkeypatch):
    from api.routers import triggers

    calls: list[dict] = []

    def fake_stage(action_id, payload, **kwargs):
        calls.append({"action_id": action_id, "payload": payload, **kwargs})
        return {
            "status": "pending_approval_created",
            "approval_id": "approval:cancel",
            "application_status": "pending",
            "action_id": action_id,
            "entity_type": "watch_trigger_status",
            "ticker": None,
            "proposed_change": payload,
        }

    monkeypatch.setattr(triggers, "stage_api_action", fake_stage)

    response = auth_client.put("/api/triggers/watch_trigger%3Aoklo-break/cancel", json={})

    assert response.status_code == 200
    assert calls[0]["action_id"] == "cancel_watch_trigger"
    assert calls[0]["payload"] == {"trigger_id": "watch_trigger:oklo-break"}
    assert calls[0]["entity_id"] == "watch_trigger:oklo-break"


def test_replace_trigger_route_stages_uid(auth_client, monkeypatch):
    from api.routers import triggers

    calls: list[dict] = []

    def fake_stage(action_id, payload, **kwargs):
        calls.append({"action_id": action_id, "payload": payload, **kwargs})
        return {
            "status": "pending_approval_created",
            "approval_id": "approval:replace",
            "application_status": "pending",
            "action_id": action_id,
            "entity_type": "watch_trigger",
            "ticker": "OKLO",
            "proposed_change": payload,
        }

    monkeypatch.setattr(triggers, "stage_api_action", fake_stage)

    response = auth_client.put(
        "/api/triggers/watch_trigger%3Aoklo-break/replace",
        json={
            "condition": "Watch OKLO breadth reversal",
            "trigger_type": "technical",
            "ticker": "OKLO",
            "definition": {"type": "technical"},
        },
    )

    assert response.status_code == 200
    assert calls[0]["action_id"] == "replace_watch_trigger"
    assert calls[0]["payload"]["trigger_id"] == "watch_trigger:oklo-break"
    assert calls[0]["payload"]["condition"] == "Watch OKLO breadth reversal"
    assert calls[0]["entity_id"] == "watch_trigger:oklo-break"


def test_replace_pending_watch_trigger_approval_route(auth_client, monkeypatch):
    from api.routers import approvals

    class FakeService:
        def get_approval(self, approval_id, actor=None):
            assert approval_id == "approval:old"
            return {
                "id": "approval:old",
                "status": "pending",
                "application_status": "pending",
                "entity_type": "watch_trigger",
                "action_id": "create_watch_trigger",
                "ticker": "OKLO",
                "reason": "old proposal",
                "proposed_change": {"condition": "Old", "trigger_type": "custom", "ticker": "OKLO"},
            }

        def propose_action(self, action_id, payload, context, **kwargs):
            assert action_id == "create_watch_trigger"
            assert payload["condition"] == "New"
            assert kwargs["supersedes_approval_id"] == "approval:old"
            return {
                "id": "approval:new",
                "status": "pending",
                "application_status": "pending",
                "entity_type": "watch_trigger",
                "action_id": action_id,
                "ticker": payload.get("ticker"),
                "reason": kwargs.get("reason"),
                "proposed_change": payload,
            }

    def fake_execute(action_id, payload, **kwargs):
        assert action_id == "resolve_approval"
        assert payload["status"] == "rejected"
        return {
            "id": "approval:old",
            "status": "rejected",
            "application_status": "not_applicable",
            "entity_type": "watch_trigger",
            "action_id": "create_watch_trigger",
            "ticker": "OKLO",
            "reason": "old proposal",
            "proposed_change": {"condition": "Old", "trigger_type": "custom", "ticker": "OKLO"},
        }

    monkeypatch.setattr(approvals, "OntologyCommandService", FakeService)
    monkeypatch.setattr(approvals, "execute_api_action", fake_execute)

    response = auth_client.post(
        "/api/approvals/approval%3Aold/replace",
        json={"condition": "New", "trigger_type": "technical", "ticker": "OKLO"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "replacement_created"
    assert body["original"]["status"] == "rejected"
    assert body["replacement"]["id"] == "approval:new"
    assert body["replacement"]["proposed_change"]["condition"] == "New"
