from __future__ import annotations

from typing import Any

from api.generated_approval_filters import should_suppress_generated_review_approval


class _FakeCommandService:
    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = calls

    def propose_action(self, action_id, payload, context, *, reason, entity_id=None):
        self.calls.append(
            {
                "action_id": action_id,
                "payload": payload,
                "source_type": context.source_type,
                "source_id": context.source_id,
                "reason": reason,
                "entity_id": entity_id,
            }
        )
        return {"id": f"approval:{len(self.calls)}"}


class _FakeObjectService:
    def __init__(self, writes: list[dict[str, Any]]):
        self.writes = writes

    def write_object(self, object_type, business_key, properties, valid_from, **kwargs):
        self.writes.append(
            {
                "object_type": object_type,
                "business_key": business_key,
                "properties": dict(properties),
                "valid_from": valid_from,
                "kwargs": kwargs,
            }
        )
        return {"object_uid": f"{object_type.lower()}:{business_key}"}


def test_generated_review_approval_predicate_is_limited_to_automated_review_action_items():
    assert should_suppress_generated_review_approval(
        "create_action_item",
        {"action_type": " Review "},
        source_type="workflow",
    )
    assert should_suppress_generated_review_approval(
        "create_action_item",
        {"action_type": "review"},
        source_type="system",
    )
    assert not should_suppress_generated_review_approval(
        "create_action_item",
        {"action_type": "review"},
        source_type="user",
    )
    assert not should_suppress_generated_review_approval(
        "create_action_item",
        {"action_type": "research"},
        source_type="workflow",
    )
    assert not should_suppress_generated_review_approval(
        "create_watch_trigger",
        {"action_type": "review"},
        source_type="workflow",
    )


def test_report_sync_positions_flagged_do_not_create_review_action_item_approvals(monkeypatch):
    import ontology.command_service as command_service
    from api import report_sync

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(command_service, "OntologyCommandService", lambda: _FakeCommandService(calls))

    count = report_sync._create_report_action_items(  # noqa: SLF001 - focused regression coverage.
        "daily",
        "2026-05-15",
        "daily:2026-05-15",
        {
            "summary": {
                "positions_flagged": ["META", "CRWD"],
                "thesis_monitoring": {"positions_needing_reassessment": ["EWY"]},
            }
        },
    )

    assert count == 0
    assert calls == []


def test_continuous_optimizer_skips_review_action_item_approval(monkeypatch):
    import ontology.command_service as command_service
    from api import continuous_optimizer

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(command_service, "OntologyCommandService", lambda: _FakeCommandService(calls))

    approval_id = continuous_optimizer._stage_action_item(  # noqa: SLF001 - focused regression coverage.
        {"id": "alert:1", "run_id": "run-1", "severity": "normal", "change_summary": "MU needs review."},
        {"ticker": "MU", "action": "Review", "evidence": {}},
    )

    assert approval_id is None
    assert calls == []


def test_continuous_optimizer_still_stages_non_review_action_items(monkeypatch):
    import ontology.command_service as command_service
    from api import continuous_optimizer

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(command_service, "OntologyCommandService", lambda: _FakeCommandService(calls))
    monkeypatch.setenv("PROACTIVE_ALERT_DQ_GATE_ENABLED", "true")

    approval_id = continuous_optimizer._stage_action_item(  # noqa: SLF001 - focused regression coverage.
        {"id": "alert:2", "run_id": "run-1", "severity": "high", "change_summary": "Trim MU."},
        {"ticker": "MU", "action": "Trim Long", "evidence": {}},
    )

    assert approval_id == "approval:1"
    assert [call["action_id"] for call in calls] == ["create_action_item"]
    assert calls[0]["payload"]["action_type"] == "resize"
    assert calls[0]["payload"]["alert_context"]["change_summary"] == "Trim MU."
    updated, gate = __import__(
        "decision_quality.proactive_alert_gate", fromlist=["apply_proactive_alert_gate"]
    ).apply_proactive_alert_gate(
        "create_action_item",
        calls[0]["payload"],
        source_type="workflow",
        alert_context=calls[0]["payload"].get("alert_context"),
    )
    assert updated["action_type"] == "research"
    assert gate.scout.status == "pass"
    assert gate.skeptic.status == "fail"
    assert "scout_skeptic_sizer_gate" in updated


def test_workflow_artifacts_skip_review_action_items_but_stage_research_items(monkeypatch):
    from api import workflow_artifacts

    calls: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(workflow_artifacts, "OntologyCommandService", lambda: _FakeCommandService(calls))
    monkeypatch.setattr(workflow_artifacts, "OntologyObjectService", lambda: _FakeObjectService(writes))
    monkeypatch.setattr(workflow_artifacts, "emit_audit_event", lambda *args, **kwargs: None)

    count = workflow_artifacts.persist_artifacts(
        "run-1",
        "MU",
        {
            "action_items": [
                {"description": "Review MU setup", "action_type": "review", "urgency": "normal"},
                {"description": "Research MU memory pricing", "action_type": "research", "urgency": "normal"},
            ]
        },
    )

    assert count == 1
    assert len(writes) == 2
    assert [call["action_id"] for call in calls] == ["create_action_item"]
    assert calls[0]["payload"]["action_type"] == "research"


def test_workflow_artifacts_persist_metadata_and_proposal_context(monkeypatch):
    from api import workflow_artifacts

    calls: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(workflow_artifacts, "OntologyCommandService", lambda: _FakeCommandService(calls))
    monkeypatch.setattr(workflow_artifacts, "OntologyObjectService", lambda: _FakeObjectService(writes))
    monkeypatch.setattr(workflow_artifacts, "emit_audit_event", lambda *args, **kwargs: None)

    count = workflow_artifacts.persist_artifacts(
        "workflow:thesis_review:unit",
        "MU",
        {
            "evaluation_draft": {"ticker": "MU", "thesis_status": "watch", "confidence": 0.4},
            "action_items": [
                {"description": "Research MU HBM pricing into earnings", "action_type": "research", "urgency": "normal"}
            ],
        },
    )

    assert count == 2
    assert len(writes) == 2
    assert [call["action_id"] for call in calls] == ["save_evaluation", "create_action_item"]
    for write, call in zip(writes, calls, strict=True):
        props = write["properties"]
        assert write["object_type"] == "WorkflowArtifact"
        assert write["business_key"].startswith("workflow_artifact:")
        assert props["workflow_run_id"] == "workflow:thesis_review:unit"
        assert props["state"] == "proposed"
        assert props["artifact_hash"]
        assert props["provenance_event_id"].startswith("pv:workflow_artifact:workflow:thesis_review:unit:")
        assert props["metadata"]["ticker"] == "MU"
        assert "payload" in props["metadata"]
        assert write["kwargs"]["provenance"] == props["provenance_event_id"]
        assert call["source_type"] == "workflow"
        assert call["source_id"] == "workflow:thesis_review:unit"
        assert call["entity_id"] == write["business_key"]


def test_workflow_artifacts_fail_closed_for_malformed_or_incomplete_artifacts(monkeypatch):
    from api import workflow_artifacts

    calls: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(workflow_artifacts, "OntologyCommandService", lambda: _FakeCommandService(calls))
    monkeypatch.setattr(workflow_artifacts, "OntologyObjectService", lambda: _FakeObjectService(writes))
    monkeypatch.setattr(workflow_artifacts, "emit_audit_event", lambda *args, **kwargs: None)

    count = workflow_artifacts.persist_artifacts(
        "workflow:thesis_review:unit",
        "MU",
        {
            "evaluation_draft": {"ticker": "MU"},
            "action_items": [{"action_type": "research"}],
            "watch_triggers": "not-a-list",
        },
    )

    assert count == 0
    assert calls == []
    assert writes == []
