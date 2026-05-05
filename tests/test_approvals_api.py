from __future__ import annotations

import portfolio.core_db as core_db


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

    resp = auth_client.get("/api/v1/approvals/summary", params={"ticker": "mu", "application_status": "failed", "limit": 50})

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

    resp = auth_client.get("/api/v1/approvals/summary", params={"status": "all", "application_status": "all", "limit": 50})

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
