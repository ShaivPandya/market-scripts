from __future__ import annotations

from unittest.mock import patch

from api.decision_state import normalize_course_of_action, normalize_decision_outcome, normalize_recommendation
from ontology.decision_outcome_service import finalize_decision_outcome, record_recommendation_outcome
from ontology.schemas.identity import decision_outcome_id
from ontology.schemas.objects import DecisionOutcome


def test_decision_outcome_schema_validation():
    model = DecisionOutcome(
        decision_outcome_id="rec:test-1",
        source_kind="recommendation",
        recommendation_id="recommendation:test-1",
        ticker="MU",
        as_of="2026-01-01",
        outcome_status="evaluated",
        final_label_status="draft",
        draft_postmortem="Draft learning note.",
    )
    assert decision_outcome_id(model.decision_outcome_id).startswith("decision_outcome:")


def test_normalize_decision_outcome_flags_draft_review():
    normalized = normalize_decision_outcome(
        {
            "decision_outcome_id": "rec:test-1",
            "outcome_status": "evaluated",
            "final_label_status": "draft",
            "draft_postmortem": "Needs review",
        }
    )
    assert normalized is not None
    assert normalized["requires_review"] is True
    assert normalized["learning_state"] == "draft"
    assert normalized["decision_state"] == "draft"


def test_normalize_recommendation_exposes_payload_outcome_fields():
    normalized = normalize_recommendation(
        {
            "action": "buy",
            "approval_required": False,
            "outcome_status": "evaluated",
            "payload": {
                "outcome": {
                    "draft_postmortem": "Draft text",
                    "final_label_status": "draft",
                    "process_label": "good_process_bad_outcome",
                }
            },
        }
    )
    assert normalized is not None
    assert normalized["draft_postmortem"] == "Draft text"
    assert normalized["process_label"] == "good_process_bad_outcome"


def test_normalize_course_of_action_uses_outcome_status():
    normalized = normalize_course_of_action(
        {
            "action": "add",
            "approval_required": True,
            "status": "open",
            "outcome_status": "pending",
        }
    )
    assert normalized is not None
    assert normalized["outcome_state"] == "pending"


@patch("ontology.decision_outcome_service.OntologyRuntimeReadService")
@patch("ontology.decision_outcome_service.OntologyObjectService")
def test_finalize_decision_outcome_updates_parent_payload(mock_objects_cls, mock_reads_cls):
    reads = mock_reads_cls.return_value
    objects = mock_objects_cls.return_value
    outcome_uid = decision_outcome_id("rec:test-1")
    reads.get.side_effect = [
        {
            "object_uid": outcome_uid,
            "decision_outcome_id": "rec:test-1",
            "source_kind": "recommendation",
            "recommendation_id": "test-1",
            "outcome_status": "evaluated",
            "final_label_status": "draft",
            "draft_postmortem": "Draft",
        },
        {
            "object_uid": "recommendation:test-1",
            "recommendation_id": "test-1",
            "payload": {"outcome": {"draft_postmortem": "Draft", "final_label_status": "draft"}},
        },
    ]
    objects.write_object.return_value = {"object_uid": outcome_uid, "final_label_status": "confirmed"}

    result = finalize_decision_outcome(outcome_uid, decision="confirm", actor_id="tester")

    assert result["final_label_status"] == "confirmed"
    assert objects.write_object.call_count >= 2


@patch("ontology.decision_outcome_service._write_decision_outcome")
@patch("ontology.decision_outcome_service.OntologyObjectService")
def test_record_recommendation_outcome_writes_legacy_and_first_class_objects(mock_objects_cls, mock_write_outcome):
    objects = mock_objects_cls.return_value
    mock_write_outcome.return_value = decision_outcome_id("rec:1")

    uid = record_recommendation_outcome(
        {
            "object_uid": "recommendation:1",
            "recommendation_id": "1",
            "action": "buy",
            "ticker": "MU",
            "as_of": "2026-01-01",
            "payload": {},
        },
        "evaluated",
        {"draft_postmortem": "Draft", "final_label_status": "draft"},
        objects=objects,
        actor={"actor_id": "test"},
    )

    assert uid.startswith("decision_outcome:")
    objects.write_object.assert_called_once()
    mock_write_outcome.assert_called_once()
