from __future__ import annotations

from typing import Any

import pytest


def _sample_idea(**overrides: Any) -> dict[str, Any]:
    idea = {
        "id": "investment_idea:testlc",
        "idea_id": "investment_idea:testlc",
        "ticker": "TESTLC",
        "status": "watching",
        "user_notes": "initial note",
        "tags": ["quality"],
        "metadata": {
            "analyzer_direction": "long",
            "use_portfolio_context": True,
        },
        "asset": "equity",
        "instrument_type": "security",
        "price_symbol": "TESTLC",
        "contract_multiplier": 1.0,
    }
    idea.update(overrides)
    return idea


class _InMemoryOntology:
    def __init__(self, seed: dict[str, dict[str, Any]] | None = None):
        self.rows: dict[str, dict[str, Any]] = dict(seed or {})

    def write_object(self, object_type: str, business_key: str, properties: dict[str, Any], valid_from: str, **kwargs):
        from ontology.schemas.registry import NODE_SCHEMAS

        props = dict(properties)
        uid = str(business_key)
        if object_type == "InvestmentIdea":
            props.setdefault("idea_id", props.get("id") or uid)
        elif object_type == "IdeaLifecycleEvent":
            props.setdefault("event_id", props.get("id") or uid)
        NODE_SCHEMAS[object_type].model_validate(props)
        row = {
            "object_uid": uid,
            "object_type": object_type,
            "properties": props,
        }
        self.rows[uid] = row
        return row

    def get_object(self, object_uid: str, **kwargs):
        return self.rows.get(str(object_uid))

    def query_objects(self, object_type: str | None = None, filters: dict[str, Any] | None = None, limit: int = 100):
        rows = [
            row
            for row in self.rows.values()
            if object_type is None or str(row.get("object_type") or "") == str(object_type)
        ]
        if filters:
            filtered: list[dict[str, Any]] = []
            for row in rows:
                props = row.get("properties") if isinstance(row.get("properties"), dict) else {}
                if all(str(props.get(key) or "") == str(value) for key, value in filters.items()):
                    filtered.append(row)
            rows = filtered
        return rows[:limit]


@pytest.fixture
def ideas_store(monkeypatch):
    from api.routers import ideas as ideas_router
    from ontology.runtime_read_service import OntologyRuntimeReadService

    store = _InMemoryOntology(
        {
            "investment_idea:testlc": {
                "object_uid": "investment_idea:testlc",
                "object_type": "InvestmentIdea",
                "properties": _sample_idea(),
            }
        }
    )

    class _ServiceFactory:
        def __call__(self):
            return store

    read_service = lambda object_service=None: OntologyRuntimeReadService(object_service=store)  # noqa: E731

    monkeypatch.setattr(ideas_router, "OntologyObjectService", _ServiceFactory())
    monkeypatch.setattr(ideas_router, "OntologyRuntimeReadService", read_service)
    return store


def test_diff_idea_lifecycle_changes_detects_tracked_fields():
    from api.routers.ideas import _diff_idea_lifecycle_changes, _idea_lifecycle_snapshot

    before = _idea_lifecycle_snapshot(_sample_idea())
    after = _idea_lifecycle_snapshot(_sample_idea(status="ready_for_review", user_notes="updated note"))

    changed_fields, before_values, after_values = _diff_idea_lifecycle_changes(before, after)

    assert changed_fields == ["status", "user_notes"]
    assert before_values["status"] == "watching"
    assert after_values["status"] == "ready_for_review"
    assert before_values["user_notes"] == "initial note"
    assert after_values["user_notes"] == "updated note"


def test_reject_idea_preserves_metadata_and_emits_lifecycle_event(ideas_store):
    from api.routers.ideas import IdeaRejectRequest, _get_idea, reject_idea

    response = reject_idea("testlc", IdeaRejectRequest(note="not compelling"))

    idea = response["idea"]
    metadata = idea["metadata"]
    assert metadata["analyzer_direction"] == "long"
    assert metadata["use_portfolio_context"] is True
    assert metadata["rejection_note"] == "not compelling"
    assert metadata["rejected_at"]

    detail = _get_idea("testlc")
    assert detail is not None
    assert detail["status"] == "rejected"

    lifecycle_rows = [
        row for row in ideas_store.rows.values() if str(row.get("object_type") or "") == "IdeaLifecycleEvent"
    ]
    assert len(lifecycle_rows) == 1
    event = lifecycle_rows[0]["properties"]
    assert event["event_type"] == "rejected"
    assert "status" in event["changed_fields"]
    assert event["reason"] == "not compelling"


def test_update_idea_emits_ordered_lifecycle_history(ideas_store):
    from api.routers.ideas import IdeaUpdateRequest, _idea_detail, update_idea

    update_idea("testlc", IdeaUpdateRequest(status="ready_for_review"))
    update_idea("testlc", IdeaUpdateRequest(user_notes="second edit", tags=["quality", "cyclical"]))

    detail = _idea_detail("testlc")
    history = detail["lifecycle_history"]

    assert len(history) == 2
    assert history[0]["changed_at"] >= history[1]["changed_at"]
    assert history[0]["event_type"] in {"tags_edited", "notes_edited", "idea_updated"}
    assert history[1]["event_type"] == "status_changed"


def test_write_idea_lifecycle_event_records_accept_decision(ideas_store):
    from api.routers.ideas import _write_idea_lifecycle_event

    idea = _sample_idea()
    event = _write_idea_lifecycle_event(
        idea,
        event_type="evaluation_accepted",
        changed_fields=["evaluation_accepted"],
        before={},
        after={
            "evaluation_id": "idea_evaluation:test",
            "action": "buy",
            "recommendation_id": "recommendation:test",
        },
        reason="looks good",
        evaluation_id="idea_evaluation:test",
        recommendation_id="recommendation:test",
        approval_id="approval:test",
        source_id="ideas.accept:testlc:test",
    )

    assert event["event_type"] == "evaluation_accepted"
    assert event["reason"] == "looks good"
    assert event["evaluation_id"] == "idea_evaluation:test"
    assert event["recommendation_id"] == "recommendation:test"
    assert event["approval_id"] == "approval:test"
