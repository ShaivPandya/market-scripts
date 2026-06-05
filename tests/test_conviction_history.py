from __future__ import annotations

from typing import Any

import pytest

from ontology.conviction_history import (
    backfill_conviction_history,
    build_conviction_history_props,
    compact_conviction_history_entry,
    conviction_transitions,
    deterministic_entry_key,
    record_conviction_change,
    record_position_conviction_changes,
)
from ontology.runtime_read_service import OntologyRuntimeReadService, object_props


def _position_row(
    *,
    ticker: str,
    conviction: int | None,
    group_conviction: int | None = None,
    group_name: str | None = None,
    created_at: str,
    updated_at: str,
    version_id: str,
    tx_from: str,
) -> dict[str, Any]:
    return {
        "object_uid": f"position:{ticker}",
        "object_type": "Position",
        "properties": {
            "ticker": ticker,
            "asset": "equity",
            "direction": "long",
            "conviction": conviction,
            "group_conviction": group_conviction,
            "group_name": group_name,
            "created_at": created_at,
            "updated_at": updated_at,
            "ontology_run_id": "operational",
        },
        "_meta": {"temporal": {"version_id": version_id, "tx_from": tx_from, "valid_from": tx_from}},
    }


def _approval_row(
    *,
    approval_id: str,
    ticker: str,
    reason: str,
    resolved_at: str,
    conviction_before: int,
    conviction_after: int,
) -> dict[str, Any]:
    return {
        "object_uid": f"approval:{approval_id}",
        "object_type": "Approval",
        "properties": {
            "entity_type": "portfolio_positions",
            "ticker": ticker,
            "action_id": "update_portfolio_positions",
            "application_status": "applied",
            "proposed_change": {
                "position_changes": [
                    {
                        "ticker": ticker,
                        "field_changes": [
                            {
                                "field": "conviction",
                                "before": conviction_before,
                                "after": conviction_after,
                            }
                        ],
                    }
                ]
            },
            "reason": reason,
            "source_type": "user",
            "source_id": "portfolio.edit",
            "resolved_by_actor_id": "analyst:test",
            "resolved_at": resolved_at,
            "application_completed_at": resolved_at,
            "status": "approved",
            "ontology_run_id": "operational",
        },
    }


class _HistoryObjectService:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = list(rows)
        self.written: list[tuple[str, str, dict[str, Any]]] = []

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
        if not include_history and object_type in {"Position", "HedgePosition", "InvestmentIdea"}:
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

    def write_object(
        self,
        object_type: str,
        business_key: str,
        properties: dict[str, Any],
        now: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        uid = f"conviction_history_entry:{business_key}"
        row = {
            "object_uid": uid,
            "object_type": object_type,
            "properties": {**properties, "id": uid, "object_uid": uid},
        }
        self.rows.append(row)
        self.written.append((object_type, business_key, properties))
        return row

    def query_relations(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []


@pytest.fixture
def history_reads() -> OntologyRuntimeReadService:
    rows = [
        _position_row(
            ticker="MU",
            conviction=3,
            created_at="2026-06-01T10:00:00+00:00",
            updated_at="2026-06-01T10:00:00+00:00",
            version_id="version:1",
            tx_from="2026-06-01T10:00:00+00:00",
        ),
        _position_row(
            ticker="MU",
            conviction=4,
            created_at="2026-06-05T10:00:00+00:00",
            updated_at="2026-06-05T10:00:00+00:00",
            version_id="version:2",
            tx_from="2026-06-05T10:00:00+00:00",
        ),
        _approval_row(
            approval_id="101",
            ticker="MU",
            reason="earnings beat",
            resolved_at="2026-06-05T10:00:00+00:00",
            conviction_before=3,
            conviction_after=4,
        ),
        {
            "object_uid": "conviction_history_entry:abc123",
            "object_type": "ConvictionHistoryEntry",
            "properties": {
                "entry_id": "abc123",
                "ticker": "MU",
                "entity_type": "position",
                "entity_id": "position:MU",
                "conviction_field": "conviction",
                "previous_conviction": 3,
                "new_conviction": 4,
                "changed_at": "2026-06-05T10:00:00+00:00",
                "reason": "earnings beat",
                "approval_id": "approval:101",
                "conviction_source_kind": "portfolio_update",
            },
        },
    ]
    return OntologyRuntimeReadService(object_service=_HistoryObjectService(rows))


def test_conviction_transitions_detects_position_changes():
    rows = [
        _position_row(
            ticker="MU",
            conviction=3,
            created_at="2026-06-01T10:00:00+00:00",
            updated_at="2026-06-01T10:00:00+00:00",
            version_id="version:1",
            tx_from="2026-06-01T10:00:00+00:00",
        ),
        _position_row(
            ticker="MU",
            conviction=5,
            created_at="2026-06-05T10:00:00+00:00",
            updated_at="2026-06-05T10:00:00+00:00",
            version_id="version:2",
            tx_from="2026-06-05T10:00:00+00:00",
        ),
    ]
    transitions = conviction_transitions([object_props(row) for row in rows], field_name="conviction")
    assert len(transitions) == 1
    assert transitions[0]["previous_conviction"] == 3
    assert transitions[0]["new_conviction"] == 5


def test_deterministic_entry_key_is_stable():
    first = deterministic_entry_key(
        entity_type="position",
        entity_id="position:MU",
        conviction_field="conviction",
        changed_at="2026-06-05T10:00:00+00:00",
        previous_conviction=3,
        new_conviction=4,
    )
    second = deterministic_entry_key(
        entity_type="position",
        entity_id="position:MU",
        conviction_field="conviction",
        changed_at="2026-06-05T10:00:00+00:00",
        previous_conviction=3,
        new_conviction=4,
    )
    assert first == second


def test_build_conviction_history_props_keeps_ai_confidence_separate():
    props = build_conviction_history_props(
        entity_type="investment_idea",
        entity_id="investment_idea:mu_idea",
        ticker="MU",
        conviction_field="conviction",
        previous_conviction=2,
        new_conviction=4,
        changed_at="2026-06-05T10:00:00+00:00",
        conviction_source_kind="idea_update",
        ai_confidence=0.82,
        ai_confidence_reason="model score from evaluation",
    )
    assert props["new_conviction"] == 4
    assert props["ai_confidence"] == 0.82
    assert props["ai_confidence_reason"] == "model score from evaluation"


def test_record_conviction_change_skips_no_op():
    service = _HistoryObjectService([])
    result = record_conviction_change(
        service,
        entity_type="position",
        entity_id="position:MU",
        ticker="MU",
        conviction_field="conviction",
        previous_conviction=4,
        new_conviction=4,
        changed_at="2026-06-05T10:00:00+00:00",
        conviction_source_kind="portfolio_update",
    )
    assert result is None
    assert service.written == []


def test_record_position_conviction_changes_writes_position_and_group_entries():
    service = _HistoryObjectService([])
    refs = record_position_conviction_changes(
        service,
        before_row={"ticker": "MU", "conviction": 3, "group_name": "Semis", "group_conviction": 3},
        after_row={"ticker": "MU", "conviction": 4, "group_name": "Semis", "group_conviction": 5},
        entity_type="position",
        entity_id="position:MU",
        changed_at="2026-06-05T10:00:00+00:00",
        conviction_source_kind="portfolio_update",
    )
    assert len(refs) == 2
    fields = {row["conviction_field"] for _, _, row in service.written}
    assert fields == {"conviction", "group_conviction"}


def test_backfill_conviction_history_is_idempotent():
    rows = [
        _position_row(
            ticker="MU",
            conviction=3,
            created_at="2026-06-01T10:00:00+00:00",
            updated_at="2026-06-01T10:00:00+00:00",
            version_id="version:1",
            tx_from="2026-06-01T10:00:00+00:00",
        ),
        _position_row(
            ticker="MU",
            conviction=4,
            created_at="2026-06-05T10:00:00+00:00",
            updated_at="2026-06-05T10:00:00+00:00",
            version_id="version:2",
            tx_from="2026-06-05T10:00:00+00:00",
        ),
    ]
    service = _HistoryObjectService(rows)
    first = backfill_conviction_history(service, ticker="MU", now="2026-06-05T12:00:00+00:00")
    second = backfill_conviction_history(service, ticker="MU", now="2026-06-05T12:00:00+00:00")
    assert first == 1
    assert second == 0


def test_conviction_history_returns_materialized_entries(history_reads: OntologyRuntimeReadService):
    history = history_reads.conviction_history("MU", backfill=False)
    assert len(history) == 1
    assert history[0]["previous_conviction"] == 3
    assert history[0]["new_conviction"] == 4
    assert history[0]["approval_id"] == "approval:101"


def test_conviction_summary_includes_current_and_timeline(history_reads: OntologyRuntimeReadService):
    summary = history_reads.conviction_summary("MU")
    assert summary["timeline"][0]["new_conviction"] == 4


def test_get_dossier_returns_conviction(monkeypatch):
    from api.routers import dossier as dossier_router

    class _Reads:
        def dossier_bundle(self, ticker: str):
            return {
                "position": {"ticker": ticker, "conviction": 4, "group_conviction": 3, "group_name": "Semis"},
                "thesis_meta": {"ticker": ticker, "status": "active"},
                "evaluations": [],
                "catalysts": [],
                "kill_conditions": [],
                "thesis_claims": [],
                "workflow_runs": [],
                "action_items": [],
                "watch_triggers": [],
                "monitor_hits": [],
                "pending_approvals": [],
                "decision_outcomes": [],
            }

        def positions(self, *, include_hedges: bool = False, limit: int = 1000):
            return [{"ticker": "MU", "conviction": 4, "group_conviction": 3, "group_name": "Semis"}]

        def thesis_status_history(self, ticker: str, *, limit: int = 20):
            return []

        def conviction_summary(self, ticker: str):
            return {
                "current": 4,
                "group_current": 3,
                "group_name": "Semis",
                "timeline": [
                    {
                        "id": 1,
                        "ticker": ticker,
                        "previous_conviction": 3,
                        "new_conviction": 4,
                        "changed_at": "2026-06-05T10:00:00+00:00",
                    }
                ],
            }

        def evidence_ledger(self, ticker: str):
            return {"ticker": ticker, "items": []}

    monkeypatch.setattr(dossier_router, "OntologyRuntimeReadService", lambda: _Reads())

    payload = dossier_router.get_dossier("MU")
    assert payload["conviction"]["current"] == 4
    assert payload["conviction"]["timeline"][0]["new_conviction"] == 4


def test_fetch_thesis_evaluations_returns_conviction(monkeypatch):
    from api import agent_tools

    class _Reads:
        def thesis(self, ticker: str):
            return {"ticker": ticker, "status": "active"}

        def evaluations(self, ticker: str, *, limit: int = 1000):
            return [{"id": 1, "ticker": ticker, "confidence": "0.8", "evaluated_at": "2026-06-05T09:00:00+00:00"}]

        def thesis_status_history(self, ticker: str, *, limit: int = 20):
            return []

        def conviction_summary(self, ticker: str):
            return {
                "current": 4,
                "group_current": None,
                "group_name": None,
                "timeline": [
                    {
                        "id": 1,
                        "ticker": ticker,
                        "previous_conviction": 3,
                        "new_conviction": 4,
                        "changed_at": "2026-06-05T10:00:00+00:00",
                    }
                ],
            }

    monkeypatch.setattr("ontology.runtime_read_service.OntologyRuntimeReadService", lambda: _Reads())

    payload = agent_tools._fetch_thesis_evaluations("MU", limit=10)
    assert payload["conviction"]["current"] == 4
    assert payload["conviction"]["timeline"][0]["new_conviction"] == 4
    assert payload["evaluations"][0]["confidence"] == "0.8"


def test_compact_conviction_history_entry_shape():
    row = compact_conviction_history_entry(
        {
            "entry_id": "entry-1",
            "ticker": "MU",
            "entity_type": "position",
            "entity_id": "position:MU",
            "conviction_field": "conviction",
            "previous_conviction": 3,
            "new_conviction": 4,
            "changed_at": "2026-06-05T10:00:00+00:00",
            "ai_confidence": 0.75,
            "ai_confidence_reason": "evaluation confidence",
        },
        entry_id=1,
    )
    assert row["new_conviction"] == 4
    assert row["ai_confidence"] == 0.75
    assert row["ai_confidence_reason"] == "evaluation confidence"
