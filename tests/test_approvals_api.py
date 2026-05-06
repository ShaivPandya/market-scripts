from __future__ import annotations

from urllib.parse import quote

import portfolio.core_db as core_db
from portfolio.action_registry import ActionContext, compute_action_base_state_hash, propose_action


def _reset_core_db(tmp_path, monkeypatch):
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "approvals_api.db")
    monkeypatch.setattr(core_db, "_conn", None)


def _approval(
    *,
    ticker: str,
    action_id: str = "create_action_item",
    recommendation_id: int | None = None,
) -> dict:
    proposed_change = {
        "ticker": ticker,
        "description": f"{ticker} action",
        "action_type": "review",
        "urgency": "normal",
    }
    if recommendation_id is not None:
        proposed_change["recommendation_id"] = recommendation_id
    return core_db.create_pending_approval(
        entity_type="action_item",
        proposed_change=proposed_change,
        ticker=ticker,
        action_id=action_id,
        source_type="user",
        source_id=f"test-{ticker}-{action_id}",
        reason=f"Review {ticker}",
    )


def test_approval_summary_limits_items_and_counts_recommendation_approvals(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    _approval(ticker="MU", recommendation_id=101)
    _approval(ticker="TSM")
    _approval(ticker="NVDA", recommendation_id=202)

    resp = auth_client.get("/api/v1/approvals/summary?limit=1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert len(data["items"]) == 1
    assert data["recommendation_approval_count"] == 2
    assert data["has_more"] is True
    assert data["status"] == "pending"
    assert data["ticker"] is None
    assert data["application_status"] is None
    assert data["limit"] == 1


def test_approval_summary_filters_ticker_and_application_status(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    _approval(ticker="MU", recommendation_id=101)
    failed = _approval(ticker="mu", action_id="create_watch_trigger")
    _approval(ticker="TSM")

    conn = core_db._get_conn()
    with core_db._lock:
        conn.execute("UPDATE pending_approvals SET application_status = 'failed' WHERE id = ?", (failed["id"],))
        conn.commit()

    resp = auth_client.get(
        "/api/v1/approvals/summary", params={"ticker": "mu", "application_status": "failed", "limit": 50}
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["items"][0]["id"] == failed["id"]
    assert data["items"][0]["ticker"] == "MU"
    assert data["ticker"] == "MU"
    assert data["application_status"] == "failed"
    assert data["has_more"] is False


def test_approval_summary_all_filters_preserve_existing_semantics(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    rejected = _approval(ticker="MU")
    pending = _approval(ticker="TSM")
    core_db.resolve_approval(rejected["id"], "rejected", "No longer needed")

    resp = auth_client.get(
        "/api/v1/approvals/summary", params={"status": "all", "application_status": "all", "limit": 50}
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert {item["id"] for item in data["items"]} == {rejected["id"], pending["id"]}
    assert data["status"] is None
    assert data["application_status"] is None


def test_approval_summary_route_is_not_treated_as_approval_id(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)

    resp = auth_client.get("/api/v1/approvals/summary")

    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_approve_stale_action_backed_approval_returns_conflict_without_applying(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    item = core_db.create_action_item("Review MU thesis", "review", ticker="MU")
    approval = propose_action(
        "complete_action_item",
        {"item_id": item["id"], "resolution_note": "Done"},
        ActionContext(actor_type="workflow", source_type="workflow", source_id="run-stale"),
        reason="Complete action item",
    )
    core_db.dismiss_action_item(item["id"])

    resp = auth_client.post(f"/api/v1/approvals/{approval['id']}/approve", json={"note": "Apply"})

    assert resp.status_code == 409
    assert "base state changed" in str(resp.json()).lower()
    current = core_db.get_action_items()[0]
    assert current["id"] == item["id"]
    assert current["status"] == "dismissed"


def test_approve_ontology_recommendation_id_resolves_through_command_service(auth_client, monkeypatch):
    from api import action_execution
    from ontology.command_service import OntologyCommandContext, OntologyCommandService
    from ontology.object_service import OntologyObjectService
    from ontology.policy import admin_actor
    from tests.test_ontology_command_service import NormalizingTemporalRepo

    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))
    approval = service.propose_action(
        "create_recommendation",
        {
            "record": {
                "action": "rebalance",
                "instrument": "hedge_overlay",
                "report_type": "daily",
                "as_of": "2026-05-06",
                "confidence": 0.65,
                "horizon": "1 trading day",
                "rationale": "Rebalance hedge overlay.",
                "critical_data_quality": "ok",
                "idempotency_key": "daily:2026-05-06:hedge-overlay",
            }
        },
        OntologyCommandContext(actor=admin_actor(source="test"), source_type="workflow", source_id="daily"),
        reason="Daily recommendation for hedge_overlay",
    )
    encoded_id = quote(approval["id"], safe="")
    monkeypatch.setattr(action_execution, "ontology_primary_writes_enabled", lambda: True)
    monkeypatch.setattr(action_execution, "OntologyCommandService", lambda: service)

    resp = auth_client.post(
        f"/api/v1/approvals/{encoded_id}/approve",
        json={"note": "approved"},
        headers={
            "X-Request-Schema-Name": f"post:/api/v1/approvals/{encoded_id}/approve",
            "X-Request-Schema-Version": "1",
        },
    )

    assert resp.status_code == 200
    resolved = resp.json()
    assert resolved["id"] == approval["id"]
    assert resolved["application_status"] == "applied"
    assert "recommendation:daily_2026_05_06_hedge_overlay" in repo.objects


def test_reject_and_restage_stale_action_backed_approval_creates_replacement(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    item = core_db.create_action_item("Review MU thesis", "review", ticker="MU")
    approval = propose_action(
        "complete_action_item",
        {"item_id": item["id"], "resolution_note": "Done"},
        ActionContext(actor_type="workflow", source_type="workflow", source_id="run-restage"),
        reason="Complete action item",
    )
    core_db.dismiss_action_item(item["id"])

    resp = auth_client.post(
        f"/api/v1/approvals/{approval['id']}/reject-and-restage",
        json={"note": "Restage from current state"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "replacement_created"
    assert data["original"]["status"] == "rejected"
    assert data["original"]["application_status"] == "not_applicable"
    replacement = data["replacement"]
    assert replacement["status"] == "pending"
    assert replacement["supersedes_approval_id"] == approval["id"]
    assert replacement["reason_code"] == "state_changed"
    assert replacement["base_state_status"] == "valid"
    assert replacement["base_state_hash"] == compute_action_base_state_hash(
        "complete_action_item",
        {"item_id": item["id"], "resolution_note": "Done"},
    )


def test_reject_and_restage_rejects_missing_non_stale_and_non_pending(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)

    missing = auth_client.post("/api/v1/approvals/999/reject-and-restage", json={})
    assert missing.status_code == 404

    item = core_db.create_action_item("Review MU thesis", "review", ticker="MU")
    approval = propose_action(
        "complete_action_item",
        {"item_id": item["id"], "resolution_note": "Done"},
        ActionContext(actor_type="workflow", source_type="workflow", source_id="run-not-stale"),
        reason="Complete action item",
    )

    non_stale = auth_client.post(f"/api/v1/approvals/{approval['id']}/reject-and-restage", json={})
    assert non_stale.status_code == 409

    core_db.resolve_approval(approval["id"], "rejected", "Skip")
    non_pending = auth_client.post(f"/api/v1/approvals/{approval['id']}/reject-and-restage", json={})
    assert non_pending.status_code == 409
