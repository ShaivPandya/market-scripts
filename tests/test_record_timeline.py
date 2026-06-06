from __future__ import annotations

from typing import Any

import pytest

from ontology.record_timeline import (
    build_idea_record_timeline,
    build_position_record_timeline,
    build_record_timeline,
)


class _TimelineReads:
    def __init__(self, **handlers: Any):
        self._handlers = handlers
        self._ideas: dict[str, dict[str, Any]] = handlers.get("ideas", {})
        self._objects: dict[str, dict[str, Any]] = handlers.get("objects", {})

    @staticmethod
    def idea_uid(value: Any) -> str:
        text = str(value or "").strip()
        return text if text.startswith("investment_idea:") else f"investment_idea:{text}"

    def get(self, object_uid: str) -> dict[str, Any] | None:
        return self._objects.get(str(object_uid))

    def idea_by_id(self, idea_id: Any) -> dict[str, Any] | None:
        uid = self.idea_uid(idea_id)
        return self._ideas.get(uid) or self._ideas.get(str(idea_id))

    def conviction_history(
        self,
        ticker: str,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit: int = 20,
        backfill: bool = True,
    ):
        return self._handlers.get("conviction_history", lambda *args, **kwargs: [])()

    def thesis_status_history(self, ticker: str, *, limit: int = 20):
        return self._handlers.get("thesis_status_history", lambda *args, **kwargs: [])()

    def evaluations(self, ticker: str | None = None, *, limit: int = 1000):
        return self._handlers.get("evaluations", lambda *args, **kwargs: [])()

    def recommendations(
        self,
        *,
        report_type: str | None = None,
        status: str | None = None,
        ticker: str | None = None,
        approval_status: str | None = None,
        outcome_status: str | None = None,
        limit: int = 100,
    ):
        return self._handlers.get("recommendations", lambda *args, **kwargs: [])()

    def approvals(
        self,
        *,
        ticker: str | None = None,
        status: str | None = None,
        application_status: str | None = None,
        limit: int = 200,
    ):
        return self._handlers.get("approvals", lambda *args, **kwargs: [])()

    def idea_lifecycle_events(self, idea_id: Any, *, limit: int = 100):
        return self._handlers.get("idea_lifecycle_events", lambda *args, **kwargs: [])()

    def idea_evaluations(self, idea_id: Any, *, limit: int = 100):
        return self._handlers.get("idea_evaluations", lambda *args, **kwargs: [])()


def test_position_timeline_merges_streams_newest_first():
    reads = _TimelineReads(
        conviction_history=lambda: [
            {
                "id": 1,
                "ticker": "MU",
                "entity_type": "position",
                "previous_conviction": 3,
                "new_conviction": 4,
                "changed_at": "2026-06-05T10:00:00+00:00",
                "approval_id": "approval:101",
            }
        ],
        thesis_status_history=lambda: [
            {
                "id": 2,
                "ticker": "MU",
                "old_status": "active",
                "new_status": "under_review",
                "changed_at": "2026-06-04T10:00:00+00:00",
                "approval_id": "approval:99",
            }
        ],
        evaluations=lambda: [
            {
                "id": 3,
                "ticker": "MU",
                "action": "hold",
                "confidence": "medium",
                "thesis_status": "active",
                "evaluated_at": "2026-06-03T10:00:00+00:00",
            }
        ],
        recommendations=lambda: [],
        approvals=lambda: [],
    )

    timeline = build_position_record_timeline(reads, "MU", limit=10)

    assert len(timeline) == 3
    assert timeline[0]["kind"] == "conviction_change"
    assert timeline[0]["label"] == "Conviction"
    assert timeline[1]["kind"] == "thesis_status_change"
    assert timeline[2]["kind"] == "evaluation"
    assert timeline[0]["changed_at"] >= timeline[1]["changed_at"] >= timeline[2]["changed_at"]
    assert timeline[0]["refs"]["approval_id"] == "approval:101"


def test_position_timeline_dedupes_shared_approval_refs():
    reads = _TimelineReads(
        conviction_history=lambda: [
            {
                "id": 1,
                "ticker": "MU",
                "previous_conviction": 3,
                "new_conviction": 4,
                "changed_at": "2026-06-05T10:00:00+00:00",
                "approval_id": "approval:101",
            }
        ],
        thesis_status_history=lambda: [
            {
                "id": 2,
                "ticker": "MU",
                "old_status": "active",
                "new_status": "under_review",
                "changed_at": "2026-06-05T10:00:00+00:00",
                "approval_id": "approval:101",
            }
        ],
        evaluations=lambda: [],
        recommendations=lambda: [],
        approvals=lambda: [
            {
                "id": "approval:101",
                "ticker": "MU",
                "action_id": "update_portfolio_positions",
                "application_status": "applied",
                "resolved_at": "2026-06-05T10:00:00+00:00",
            }
        ],
    )

    timeline = build_position_record_timeline(reads, "MU", limit=10)
    kinds = [entry["kind"] for entry in timeline]

    assert "conviction_change" in kinds
    assert "thesis_status_change" in kinds
    assert "approval_applied" not in kinds


def test_idea_timeline_includes_lifecycle_conviction_and_evaluations():
    reads = _TimelineReads(
        ideas={
            "investment_idea:test": {
                "id": "investment_idea:test",
                "ticker": "TEST",
                "accepted_recommendation_id": None,
            }
        },
        idea_lifecycle_events=lambda: [
            {
                "id": "event-1",
                "idea_id": "investment_idea:test",
                "ticker": "TEST",
                "event_type": "status_changed",
                "changed_at": "2026-06-05T11:00:00+00:00",
                "changed_fields": ["status"],
                "before": {"status": "watching"},
                "after": {"status": "ready_for_review"},
            }
        ],
        conviction_history=lambda: [
            {
                "id": 1,
                "ticker": "TEST",
                "entity_type": "investment_idea",
                "entity_id": "investment_idea:test",
                "previous_conviction": 2,
                "new_conviction": 3,
                "changed_at": "2026-06-05T10:00:00+00:00",
            }
        ],
        idea_evaluations=lambda: [
            {
                "id": "idea_evaluation:1",
                "evaluation_id": "idea_evaluation:1",
                "idea_id": "investment_idea:test",
                "ticker": "TEST",
                "action": "watch",
                "score": 0.7,
                "evaluated_at": "2026-06-04T10:00:00+00:00",
            }
        ],
    )

    timeline = build_idea_record_timeline(reads, idea_id="test", limit=10)

    assert [entry["kind"] for entry in timeline] == [
        "lifecycle_event",
        "conviction_change",
        "idea_evaluation",
    ]
    assert timeline[0]["label"] == "Idea lifecycle"


def test_build_record_timeline_routes_by_context():
    reads = _TimelineReads(
        conviction_history=lambda: [
            {
                "id": 1,
                "ticker": "MU",
                "previous_conviction": 2,
                "new_conviction": 3,
                "changed_at": "2026-06-05T10:00:00+00:00",
            }
        ],
        thesis_status_history=lambda: [],
        evaluations=lambda: [],
        recommendations=lambda: [],
        approvals=lambda: [],
        ideas={"investment_idea:test": {"id": "investment_idea:test", "ticker": "TEST"}},
        idea_lifecycle_events=lambda: [
            {
                "id": "event-1",
                "idea_id": "investment_idea:test",
                "ticker": "TEST",
                "event_type": "status_changed",
                "changed_at": "2026-06-05T11:00:00+00:00",
                "changed_fields": ["status"],
            }
        ],
        idea_evaluations=lambda: [],
    )

    position_timeline = build_record_timeline(reads, context="position", ticker="MU", limit=5)
    idea_timeline = build_record_timeline(reads, context="idea", idea_id="test", ticker="TEST", limit=5)

    assert position_timeline[0]["kind"] == "conviction_change"
    assert idea_timeline[0]["kind"] == "lifecycle_event"


def test_fetch_record_evolution_timeline_agent_helper(monkeypatch):
    from api import agent_tools

    class _Reads:
        def record_timeline(
            self, *, context: str, ticker: str | None = None, idea_id: str | None = None, limit: int = 30
        ):
            return [
                {
                    "id": 1,
                    "kind": "conviction_change",
                    "label": "Conviction",
                    "summary": "conviction 3 → 4",
                    "changed_at": "2026-06-05T10:00:00+00:00",
                    "ticker": ticker,
                    "refs": {"approval_id": "approval:101"},
                    "payload": {},
                }
            ]

    monkeypatch.setattr("ontology.runtime_read_service.OntologyRuntimeReadService", lambda: _Reads())

    payload = agent_tools._fetch_record_evolution_timeline(
        ticker="MU",
        idea_id=None,
        entity_type="position",
        limit=20,
    )

    assert payload["context"] == "position"
    assert payload["ticker"] == "MU"
    assert payload["entry_count"] == 1
    assert payload["timeline"][0]["kind"] == "conviction_change"


def test_execute_tool_record_evolution_timeline(monkeypatch):
    import json

    from api.agent_tools import execute_tool
    from ontology.policy import admin_actor, agent_actor

    class _Reads:
        def record_timeline(
            self, *, context: str, ticker: str | None = None, idea_id: str | None = None, limit: int = 30
        ):
            return [
                {
                    "id": 1,
                    "kind": "thesis_status_change",
                    "label": "Thesis status",
                    "summary": "active → under_review",
                    "changed_at": "2026-06-05T09:00:00+00:00",
                    "ticker": ticker,
                    "refs": {},
                    "payload": {},
                }
            ]

    monkeypatch.setattr("ontology.runtime_read_service.OntologyRuntimeReadService", lambda: _Reads())

    payload = json.loads(
        execute_tool(
            "get_record_evolution_timeline",
            {"ticker": "MU", "entity_type": "thesis", "limit": 10},
            actor=agent_actor(admin_actor()),
        )
    )

    assert payload["timeline"][0]["kind"] == "thesis_status_change"
    assert payload["entry_count"] == 1
    assert payload["_meta"]["status"] == "ok"


def test_agent_tool_is_registered():
    from api.agent_tools import TOOL_DEFINITIONS

    names = {tool.get("name") for tool in TOOL_DEFINITIONS}
    assert "get_record_evolution_timeline" in names


def test_context_pack_includes_record_timeline_tool():
    from decision_quality.context_packs import build_context_pack_tool_calls, get_context_pack

    pack = get_context_pack("quality_entry")
    calls = build_context_pack_tool_calls(
        pack=pack,
        user_text="How has NVDA conviction evolved?",
        screen_context={"ticker": "NVDA"},
        allowed_tool_names={
            "get_portfolio",
            "get_dossier",
            "get_thesis",
            "get_position_valuation",
            "run_chart",
            "get_thesis_evaluations",
            "get_record_evolution_timeline",
            "search_knowledge_base",
        },
    )
    names = [call["name"] for call in calls]
    assert "get_record_evolution_timeline" in names
