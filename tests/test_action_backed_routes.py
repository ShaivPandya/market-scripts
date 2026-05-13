from __future__ import annotations

import pytest

import portfolio.core_db as core_db


@pytest.fixture(autouse=True)
def _temp_core_db(tmp_path, monkeypatch):
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    monkeypatch.setattr("portfolio.thesis_sync.sync_markdown_from_entities", lambda _ticker: None)
    yield
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "_conn", None)


def test_process_entity_routes_stage_by_default_and_self_apply_through_approval(auth_client):
    staged_catalyst_resp = auth_client.post(
        "/api/v1/catalysts",
        json={"ticker": "mu", "description": "HBM ramp", "category": "fundamental"},
    )
    assert staged_catalyst_resp.status_code == 200
    staged = staged_catalyst_resp.json()
    assert staged["status"] == "pending_approval_created"
    assert staged["action_id"] == "create_catalyst"
    assert core_db.get_catalysts("MU") == []

    catalyst_resp = auth_client.post(
        "/api/v1/catalysts",
        json={
            "ticker": "mu",
            "description": "HBM ramp",
            "category": "fundamental",
            "apply": True,
            "approval_note": "Apply in test",
        },
    )
    assert catalyst_resp.status_code == 200
    catalyst_payload = catalyst_resp.json()
    assert catalyst_payload["status"] == "applied"
    catalyst = core_db.get_catalysts("MU")[0]

    catalyst_update = auth_client.put(
        f"/api/v1/catalysts/{catalyst['id']}/status",
        json={"status": "played_out", "evidence": "Confirmed", "apply": True, "approval_note": "Apply in test"},
    )
    assert catalyst_update.status_code == 200
    assert catalyst_update.json()["status"] == "applied"
    assert core_db.get_catalysts("MU")[0]["status"] == "played_out"

    catalyst_uid_update = auth_client.put(
        f"/api/v1/catalysts/catalyst:{catalyst['id']}/status",
        json={"status": "failed", "evidence": "Disconfirmed", "apply": True, "approval_note": "Apply in test"},
    )
    assert catalyst_uid_update.status_code == 200
    assert catalyst_uid_update.json()["status"] == "applied"
    assert core_db.get_catalysts("MU")[0]["status"] == "failed"

    kill_resp = auth_client.post(
        "/api/v1/kill-conditions",
        json={
            "ticker": "mu",
            "condition": "Demand rolls",
            "metric": "orders",
            "apply": True,
            "approval_note": "Apply in test",
        },
    )
    assert kill_resp.status_code == 200
    kill_condition = core_db.get_kill_conditions("MU")[0]
    assert kill_condition["ticker"] == "MU"
    assert kill_condition["status"] == "active"

    kill_update = auth_client.put(
        f"/api/v1/kill-conditions/{kill_condition['id']}/status",
        json={"status": "triggered", "apply": True, "approval_note": "Apply in test"},
    )
    assert kill_update.status_code == 200
    assert core_db.get_kill_conditions("MU")[0]["status"] == "triggered"

    kill_uid_update = auth_client.put(
        f"/api/v1/kill-conditions/kill_condition:{kill_condition['id']}/status",
        json={"status": "retired", "apply": True, "approval_note": "Apply in test"},
    )
    assert kill_uid_update.status_code == 200
    assert core_db.get_kill_conditions("MU")[0]["status"] == "retired"

    claim_resp = auth_client.post(
        "/api/v1/thesis-claims",
        json={
            "ticker": "mu",
            "claim": "HBM remains supply constrained",
            "source_requirements": ["earnings"],
            "apply": True,
            "approval_note": "Apply in test",
        },
    )
    assert claim_resp.status_code == 200
    claim = core_db.get_thesis_claims(ticker="MU")[0]
    assert claim["ticker"] == "MU"
    assert claim["source_requirements"][0]["description"] == "earnings"

    claim_update = auth_client.put(
        f"/api/v1/thesis-claims/{claim['id']}",
        json={"status": "supported", "confidence": 0.8, "apply": True, "approval_note": "Apply in test"},
    )
    assert claim_update.status_code == 200
    assert core_db.get_thesis_claims(ticker="MU")[0]["status"] == "supported"

    assert core_db.get_action_runs("create_catalyst")[0]["status"] == "succeeded"
    assert core_db.get_action_runs("update_thesis_claim")[0]["status"] == "succeeded"


def test_action_item_and_trigger_routes_stage_and_self_apply(auth_client):
    staged_action_resp = auth_client.post(
        "/api/v1/actions",
        json={"description": "Review MU", "action_type": "review", "ticker": "mu", "urgency": "high"},
    )
    assert staged_action_resp.status_code == 200
    assert staged_action_resp.json()["status"] == "pending_approval_created"
    assert core_db.get_action_items(ticker="MU") == []

    action_resp = auth_client.post(
        "/api/v1/actions",
        json={
            "description": "Review MU",
            "action_type": "review",
            "ticker": "mu",
            "urgency": "high",
            "apply": True,
            "approval_note": "Apply in test",
        },
    )
    assert action_resp.status_code == 200
    action = core_db.get_action_items(ticker="MU")[0]
    assert action["ticker"] == "MU"
    assert action["status"] == "open"

    complete_resp = auth_client.put(
        f"/api/v1/actions/{action['id']}/complete",
        json={"resolution_note": "Done", "apply": True, "approval_note": "Apply in test"},
    )
    assert complete_resp.status_code == 200
    assert core_db.get_action_items(status="completed", ticker="MU")[0]["status"] == "completed"

    auth_client.post(
        "/api/v1/actions", json={"description": "Dismiss me", "apply": True, "approval_note": "Apply in test"}
    )
    dismiss_action = core_db.get_action_items(status="open")[0]
    dismiss_resp = auth_client.put(
        f"/api/v1/actions/{dismiss_action['id']}/dismiss",
        json={"apply": True, "approval_note": "Apply in test"},
    )
    assert dismiss_resp.status_code == 200
    assert core_db.get_action_items(status="dismissed")[0]["status"] == "dismissed"

    trigger_resp = auth_client.post(
        "/api/v1/triggers",
        json={
            "condition": "MU > 150",
            "trigger_type": "price_level",
            "ticker": "mu",
            "apply": True,
            "approval_note": "Apply in test",
        },
    )
    assert trigger_resp.status_code == 200
    trigger = core_db.get_watch_triggers(status="active", ticker="MU")[0]
    assert trigger["ticker"] == "MU"
    assert trigger["status"] == "active"

    fire_resp = auth_client.put(
        f"/api/v1/triggers/{trigger['id']}/fire", json={"apply": True, "approval_note": "Apply in test"}
    )
    assert fire_resp.status_code == 200
    assert core_db.get_watch_triggers(status="fired", ticker="MU")[0]["status"] == "fired"

    auth_client.post(
        "/api/v1/triggers", json={"condition": "Cancel me", "apply": True, "approval_note": "Apply in test"}
    )
    cancel_trigger = core_db.get_watch_triggers(status="active")[0]
    cancel_resp = auth_client.put(
        f"/api/v1/triggers/{cancel_trigger['id']}/cancel", json={"apply": True, "approval_note": "Apply in test"}
    )
    assert cancel_resp.status_code == 200
    assert core_db.get_watch_triggers(status="cancelled")[0]["status"] == "cancelled"

    assert core_db.get_action_runs("create_action_item")[0]["status"] == "succeeded"
    assert core_db.get_action_runs("create_watch_trigger")[0]["status"] == "succeeded"


def test_approval_routes_resolve_through_action_registry(auth_client):
    from portfolio.action_registry import ActionContext, propose_action

    approval = propose_action(
        "create_action_item",
        {"description": "Review MU", "ticker": "MU"},
        ActionContext(actor_type="workflow", source_type="workflow", source_id="run-1"),
        reason="Workflow generated",
    )

    resp = auth_client.post(f"/api/v1/approvals/{approval['id']}/approve", json={"note": "Apply"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "approved"
    assert payload["application_status"] == "applied"
    assert core_db.get_action_items(ticker="MU")[0]["description"] == "Review MU"
    assert core_db.get_action_runs("resolve_approval")[0]["status"] == "succeeded"
    assert core_db.get_action_runs("create_action_item", approval_id=approval["id"])[0]["status"] == "succeeded"
