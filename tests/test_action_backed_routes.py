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


def test_process_entity_routes_preserve_response_shapes(auth_client):
    catalyst_resp = auth_client.post(
        "/api/v1/catalysts",
        json={"ticker": "mu", "description": "HBM ramp", "category": "fundamental"},
    )
    assert catalyst_resp.status_code == 200
    catalyst = catalyst_resp.json()
    assert catalyst["ticker"] == "MU"
    assert catalyst["status"] == "pending"

    catalyst_update = auth_client.put(
        f"/api/v1/catalysts/{catalyst['id']}/status",
        json={"status": "played_out", "evidence": "Confirmed"},
    )
    assert catalyst_update.status_code == 200
    assert catalyst_update.json()["status"] == "played_out"

    kill_resp = auth_client.post(
        "/api/v1/kill-conditions",
        json={"ticker": "mu", "condition": "Demand rolls", "metric": "orders"},
    )
    assert kill_resp.status_code == 200
    kill_condition = kill_resp.json()
    assert kill_condition["ticker"] == "MU"
    assert kill_condition["status"] == "active"

    kill_update = auth_client.put(
        f"/api/v1/kill-conditions/{kill_condition['id']}/status",
        json={"status": "triggered"},
    )
    assert kill_update.status_code == 200
    assert kill_update.json()["status"] == "triggered"

    claim_resp = auth_client.post(
        "/api/v1/thesis-claims",
        json={"ticker": "mu", "claim": "HBM remains supply constrained", "source_requirements": ["earnings"]},
    )
    assert claim_resp.status_code == 200
    claim = claim_resp.json()
    assert claim["ticker"] == "MU"
    assert claim["source_requirements"][0]["description"] == "earnings"

    claim_update = auth_client.put(
        f"/api/v1/thesis-claims/{claim['id']}",
        json={"status": "supported", "confidence": 0.8},
    )
    assert claim_update.status_code == 200
    assert claim_update.json()["status"] == "supported"

    assert core_db.get_action_runs("create_catalyst")[0]["status"] == "succeeded"
    assert core_db.get_action_runs("update_thesis_claim")[0]["status"] == "succeeded"


def test_action_item_and_trigger_routes_preserve_response_shapes(auth_client):
    action_resp = auth_client.post(
        "/api/v1/actions",
        json={"description": "Review MU", "action_type": "review", "ticker": "mu", "urgency": "high"},
    )
    assert action_resp.status_code == 200
    action = action_resp.json()
    assert action["ticker"] == "MU"
    assert action["status"] == "open"

    complete_resp = auth_client.put(
        f"/api/v1/actions/{action['id']}/complete",
        json={"resolution_note": "Done"},
    )
    assert complete_resp.status_code == 200
    assert complete_resp.json()["status"] == "completed"

    dismiss_action = auth_client.post("/api/v1/actions", json={"description": "Dismiss me"}).json()
    dismiss_resp = auth_client.put(f"/api/v1/actions/{dismiss_action['id']}/dismiss")
    assert dismiss_resp.status_code == 200
    assert dismiss_resp.json()["status"] == "dismissed"

    trigger_resp = auth_client.post(
        "/api/v1/triggers",
        json={"condition": "MU > 150", "trigger_type": "price_level", "ticker": "mu"},
    )
    assert trigger_resp.status_code == 200
    trigger = trigger_resp.json()
    assert trigger["ticker"] == "MU"
    assert trigger["status"] == "active"

    fire_resp = auth_client.put(f"/api/v1/triggers/{trigger['id']}/fire")
    assert fire_resp.status_code == 200
    assert fire_resp.json()["status"] == "fired"

    cancel_trigger = auth_client.post("/api/v1/triggers", json={"condition": "Cancel me"}).json()
    cancel_resp = auth_client.put(f"/api/v1/triggers/{cancel_trigger['id']}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

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
