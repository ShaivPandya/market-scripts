from __future__ import annotations

import api.routers.approvals as approvals_router


def test_approval_summary_includes_source_health_review_and_blocks_approval(auth_client, monkeypatch):
    approval = {
        "id": "approval:source-blocked",
        "status": "pending",
        "application_status": "pending",
        "entity_type": "action_item",
        "ticker": "MU",
        "reason": "Review MU",
        "created_at": "2026-05-14T18:00:00",
        "proposed_change": {"description": "Review MU", "action_type": "review", "ticker": "MU"},
    }
    source_health = {
        "generated_at": "2026-05-14T18:00:00",
        "domains": [
            {
                "domain": "portfolio",
                "sources": [
                    {
                        "id": "portfolio",
                        "source_name": "portfolio",
                        "domain": "portfolio",
                        "status": "missing",
                        "quality_state": "missing",
                        "required": True,
                        "freshness_timestamp": None,
                        "detail": "source has no freshness record yet",
                    }
                ],
            }
        ],
    }

    monkeypatch.setattr(approvals_router, "_list_approval_records", lambda **_kwargs: [approval])
    monkeypatch.setattr(approvals_router, "_current_source_health_context", lambda: source_health)

    response = auth_client.get("/api/approvals/summary")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["source_health_review"]["status"] == "blocked"
    assert item["source_health_review"]["blockers"][0]["id"] == "portfolio"
    assert item["can_approve"] is False
