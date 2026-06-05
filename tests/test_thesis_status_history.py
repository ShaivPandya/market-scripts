from __future__ import annotations

from typing import Any

import pytest

from ontology.runtime_read_service import OntologyRuntimeReadService, object_props


def _thesis_row(
    *,
    ticker: str,
    status: str,
    created_at: str,
    updated_at: str,
    version_id: str,
    tx_from: str,
) -> dict[str, Any]:
    return {
        "object_uid": f"thesis:{ticker}",
        "object_type": "Thesis",
        "properties": {
            "ticker": ticker,
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
            "instrument_id": f"instrument:{ticker}",
            "ontology_run_id": "operational",
        },
        "_meta": {"temporal": {"version_id": version_id, "tx_from": tx_from, "valid_from": tx_from}},
    }


def _approval_row(
    *,
    approval_id: str,
    ticker: str,
    new_status: str,
    reason: str,
    resolved_at: str,
    source_type: str = "user",
    source_id: str = "thesis.change_thesis_status",
) -> dict[str, Any]:
    return {
        "object_uid": f"approval:{approval_id}",
        "object_type": "Approval",
        "properties": {
            "entity_type": "thesis_status",
            "ticker": ticker,
            "action_id": "change_thesis_status",
            "application_status": "applied",
            "proposed_change": {"ticker": ticker, "new_status": new_status, "reason": reason},
            "reason": reason,
            "source_type": source_type,
            "source_id": source_id,
            "resolved_by_actor_id": "analyst:test",
            "resolved_at": resolved_at,
            "application_completed_at": resolved_at,
            "status": "approved",
            "ontology_run_id": "operational",
        },
    }


def _action_run_row(*, approval_uid: str, provenance_event_id: str) -> dict[str, Any]:
    return {
        "object_uid": "action_run:change_thesis_status:2026-06-05T10:00:00+00:00",
        "object_type": "ActionRun",
        "properties": {
            "action_id": "change_thesis_status",
            "approval_id": approval_uid,
            "provenance_event_id": provenance_event_id,
            "status": "succeeded",
            "ontology_run_id": "operational",
        },
    }


class _HistoryObjectService:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = list(rows)

    def get_object(self, object_uid: str, **kwargs: Any) -> dict[str, Any] | None:
        matches = [row for row in self.rows if str(row.get("object_uid")) == str(object_uid)]
        if not matches:
            return None
        return sorted(matches, key=lambda row: str(row.get("_meta", {}).get("temporal", {}).get("tx_from") or ""))[-1]

    def query_objects(
        self,
        object_type: str | None = None,
        filters: dict[str, Any] | None = None,
        *,
        include_history: bool = False,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        rows = [row for row in self.rows if object_type is None or row.get("object_type") == object_type]
        if filters:
            filtered: list[dict[str, Any]] = []
            for row in rows:
                props = row.get("properties") if isinstance(row.get("properties"), dict) else {}
                if all(str(props.get(key) or "") == str(value) for key, value in filters.items()):
                    filtered.append(row)
            rows = filtered
        if not include_history and object_type == "Thesis":
            by_uid: dict[str, dict[str, Any]] = {}
            for row in rows:
                uid = str(row.get("object_uid") or "")
                current = by_uid.get(uid)
                row_tx = str(row.get("_meta", {}).get("temporal", {}).get("tx_from") or "")
                current_tx = str(current.get("_meta", {}).get("temporal", {}).get("tx_from") or "") if current else ""
                if current is None or row_tx >= current_tx:
                    by_uid[uid] = row
            rows = list(by_uid.values())
        return rows[:limit]

    def query_relations(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []


@pytest.fixture
def history_reads() -> OntologyRuntimeReadService:
    rows = [
        _thesis_row(
            ticker="MU",
            status="active",
            created_at="2026-06-01T10:00:00+00:00",
            updated_at="2026-06-01T10:00:00+00:00",
            version_id="version:1",
            tx_from="2026-06-01T10:00:00+00:00",
        ),
        _thesis_row(
            ticker="MU",
            status="under_review",
            created_at="2026-06-05T10:00:00+00:00",
            updated_at="2026-06-05T10:00:00+00:00",
            version_id="version:2",
            tx_from="2026-06-05T10:00:00+00:00",
        ),
        _thesis_row(
            ticker="MU",
            status="invalidated",
            created_at="2026-06-05T12:00:00+00:00",
            updated_at="2026-06-05T12:00:00+00:00",
            version_id="version:3",
            tx_from="2026-06-05T12:00:00+00:00",
        ),
        _approval_row(
            approval_id="101",
            ticker="MU",
            new_status="under_review",
            reason="monitoring deterioration",
            resolved_at="2026-06-05T10:00:00+00:00",
        ),
        _approval_row(
            approval_id="102",
            ticker="MU",
            new_status="invalidated",
            reason="thesis broken",
            resolved_at="2026-06-05T12:00:00+00:00",
        ),
        _action_run_row(approval_uid="approval:101", provenance_event_id="pv:101"),
    ]
    return OntologyRuntimeReadService(object_service=_HistoryObjectService(rows))


def test_thesis_status_history_returns_empty_without_thesis():
    reads = OntologyRuntimeReadService(object_service=_HistoryObjectService([]))
    assert reads.thesis_status_history("MU") == []


def test_thesis_status_history_returns_current_status_fallback():
    rows = [
        _thesis_row(
            ticker="MU",
            status="active",
            created_at="2026-06-01T10:00:00+00:00",
            updated_at="2026-06-01T10:00:00+00:00",
            version_id="version:1",
            tx_from="2026-06-01T10:00:00+00:00",
        )
    ]
    reads = OntologyRuntimeReadService(object_service=_HistoryObjectService(rows))
    history = reads.thesis_status_history("MU")
    assert len(history) == 1
    assert history[0]["old_status"] is None
    assert history[0]["new_status"] == "active"
    assert history[0]["changed_at"] == "2026-06-01T10:00:00+00:00"


def test_thesis_status_history_includes_governed_change_with_refs(history_reads: OntologyRuntimeReadService):
    history = history_reads.thesis_status_history("MU")
    under_review = next(row for row in history if row["new_status"] == "under_review")
    assert under_review["old_status"] == "active"
    assert under_review["reason"] == "monitoring deterioration"
    assert under_review["approval_id"] == "approval:101"
    assert under_review["actor"] == "analyst:test"
    assert under_review["source"] == "user:thesis.change_thesis_status"
    assert under_review["action_run_id"] == "action_run:change_thesis_status:2026-06-05T10:00:00+00:00"
    assert under_review["provenance_event_id"] == "pv:101"


def test_thesis_status_history_is_newest_first_and_bounded(history_reads: OntologyRuntimeReadService):
    history = history_reads.thesis_status_history("MU", limit=1)
    assert len(history) == 1
    full_history = history_reads.thesis_status_history("MU", limit=20)
    assert len(full_history) == 2
    assert full_history[0]["changed_at"] >= full_history[1]["changed_at"]


def test_get_thesis_detail_returns_status_history(monkeypatch):
    from api.routers import thesis as thesis_router

    class _Reads:
        def thesis(self, ticker: str):
            return {"ticker": ticker, "status": "under_review", "updated_at": "2026-06-05T10:00:00+00:00"}

        def thesis_status_history(self, ticker: str, *, limit: int = 20):
            return [
                {
                    "id": 101,
                    "ticker": ticker,
                    "old_status": "active",
                    "new_status": "under_review",
                    "reason": "monitoring deterioration",
                    "changed_at": "2026-06-05T10:00:00+00:00",
                }
            ]

        def evaluations(self, ticker: str, *, limit: int = 1000):
            return [{"id": 1, "ticker": ticker, "evaluated_at": "2026-06-05T09:00:00+00:00"}]

    monkeypatch.setattr(thesis_router, "OntologyRuntimeReadService", lambda: _Reads())
    monkeypatch.setattr(thesis_router, "_thesis_exists", lambda ticker: True)
    monkeypatch.setattr(thesis_router, "_read_thesis", lambda ticker: "thesis body")

    detail = thesis_router.get_thesis_detail("MU")
    assert detail["status_history"][0]["new_status"] == "under_review"
    assert detail["evaluations"][0]["ticker"] == "MU"
    assert detail["evaluations"][0]["evaluated_at"] == "2026-06-05T09:00:00+00:00"


def test_fetch_thesis_evaluations_returns_status_history(monkeypatch):
    from api import agent_tools

    class _Reads:
        def thesis(self, ticker: str):
            return {"ticker": ticker, "status": "under_review"}

        def evaluations(self, ticker: str, *, limit: int = 1000):
            return [{"id": 1, "ticker": ticker, "evaluated_at": "2026-06-05T09:00:00+00:00"}]

        def thesis_status_history(self, ticker: str, *, limit: int = 20):
            return [
                {
                    "id": 101,
                    "ticker": ticker,
                    "old_status": "active",
                    "new_status": "under_review",
                    "reason": "monitoring deterioration",
                    "changed_at": "2026-06-05T10:00:00+00:00",
                }
            ]

    monkeypatch.setattr("ontology.runtime_read_service.OntologyRuntimeReadService", lambda: _Reads())

    payload = agent_tools._fetch_thesis_evaluations("MU", limit=10)
    assert payload["status_history"][0]["new_status"] == "under_review"
    assert payload["evaluation_count"] == 1
    assert payload["evaluations"][0]["evaluated_at"] == "2026-06-05T09:00:00+00:00"


def test_object_props_preserves_temporal_meta():
    row = _thesis_row(
        ticker="MU",
        status="active",
        created_at="2026-06-01T10:00:00+00:00",
        updated_at="2026-06-01T10:00:00+00:00",
        version_id="version:1",
        tx_from="2026-06-01T10:00:00+00:00",
    )
    props = object_props(row)
    assert props["_meta"]["temporal"]["version_id"] == "version:1"
