from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ontology.command_service import (
    OntologyCommandConflict,
    OntologyCommandContext,
    OntologyCommandService,
    OntologyCommandValidationError,
)
from ontology.object_service import OntologyObjectService
from ontology.policy import admin_actor
from ontology.temporal_repository import ObjectVersionWrite, RelationVersionWrite


class FakeObjectService:
    def __init__(self):
        self.objects: dict[str, dict[str, Any]] = {}
        self.relations: list[dict[str, Any]] = []

    def write_object(self, object_type, business_key, properties, valid_from, **kwargs):
        object_uid = str(business_key)
        if not object_uid.startswith(
            (
                "account:",
                "approval:",
                "audit_event:",
                "executed_decision_record:",
                "document_artifact:",
                "instrument:",
                "issuer:",
                "management_quality_accomplishment:",
                "management_quality_assessment:",
                "management_quality_scorecard_row:",
                "management_quality_setback:",
                "portfolio:",
                "position:",
                "recommendation:",
                "thesis:",
            )
        ):
            object_uid = f"{object_type.lower()}:{object_uid}"
        row = {
            "object_uid": object_uid,
            "object_type": object_type,
            "properties": dict(properties),
            "_meta": {"temporal": {"version_id": f"version:{len(self.objects) + 1}", "valid_from": str(valid_from)}},
        }
        self.objects[object_uid] = row
        return row

    def write_relation(self, source_uid, target_uid, relation_type, properties, valid_from, **kwargs):
        row = {
            "relation_uid": f"{relation_type}:{source_uid}:{target_uid}",
            "source_object_uid": source_uid,
            "target_object_uid": target_uid,
            "relation_type": relation_type,
            "properties": dict(properties or {}),
            "_meta": {"temporal": {"valid_from": str(valid_from)}},
        }
        self.relations.append(row)
        return row

    def get_object(self, object_uid, **kwargs):
        return self.objects.get(str(object_uid))

    def query_objects(self, object_type=None, filters=None, **kwargs):
        rows = [row for row in self.objects.values() if object_type is None or row["object_type"] == object_type]
        for key, value in (filters or {}).items():
            rows = [row for row in rows if row["properties"].get(key) == value]
        return rows


class NormalizingTemporalRepo:
    def __init__(self):
        self.objects: dict[str, dict[str, Any]] = {}
        self.relations: list[dict[str, Any]] = []
        self.version = 0

    def write_object_version(self, write: ObjectVersionWrite):
        self.version += 1
        row = {
            "version_id": f"version:{self.version}",
            "object_uid": write.object_uid,
            "object_type": write.object_type,
            "business_key": write.business_key,
            "schema_name": write.schema_name,
            "schema_version": write.schema_version,
            "properties_json": dict(write.properties),
            "valid_from": datetime(2026, 5, 6, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 6, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }
        self.objects[write.object_uid] = row
        return row

    def write_relation_version(self, write: RelationVersionWrite):
        row = {
            "version_id": f"relation:{len(self.relations) + 1}",
            "relation_uid": write.relation_uid,
            "source_object_uid": write.source_object_uid,
            "target_object_uid": write.target_object_uid,
            "relation_type": write.relation_type,
            "relation_schema_name": write.relation_schema_name,
            "relation_schema_version": write.relation_schema_version,
            "properties_json": dict(write.properties),
            "valid_from": datetime(2026, 5, 6, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 6, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }
        self.relations.append(row)
        return row

    def get_object(self, object_uid, **kwargs):
        return self.objects.get(str(object_uid))

    def query_objects(self, object_type=None, filters=None, **kwargs):
        rows = [row for row in self.objects.values() if object_type is None or row["object_type"] == object_type]
        for key, value in (filters or {}).items():
            rows = [row for row in rows if row["properties_json"].get(key) == value]
        return rows


def test_propose_and_apply_position_update_writes_only_ontology_objects():
    service = OntologyCommandService(FakeObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(
        actor=admin_actor(source="test"),
        source_type="test",
        source_id="unit",
    )

    approval = service.propose_action(
        "update_portfolio_positions",
        {"positions": [{"ticker": "MU", "asset": "equity", "direction": "long", "shares": 10}]},
        context,
        reason="unit",
    )
    assert approval["id"].startswith("approval:")
    assert approval["status"] == "pending"

    applied = service.resolve_approval(approval["id"], "approved", "apply", context)
    assert applied["application_status"] == "applied"
    assert "position:MU" in service.objects.objects  # type: ignore[attr-defined]
    assert any(rel["relation_type"] == "position_references_instrument" for rel in service.objects.relations)  # type: ignore[attr-defined]
    assert any(rel["relation_type"] == "executed_decision_applies_approval" for rel in service.objects.relations)  # type: ignore[attr-defined]


def test_create_recommendation_approval_applies_with_real_schema_normalization():
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="workflow", source_id="daily")

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
        context,
        reason="Daily recommendation for hedge_overlay",
    )

    applied = service.resolve_approval(approval["id"], "approved", "approved", context)

    assert applied["application_status"] == "applied"
    assert "recommendation:daily_2026_05_06_hedge_overlay" in repo.objects
    assert any(row["object_type"] == "ActionRun" for row in repo.objects.values())
    assert any(row["object_type"] == "ExecutedDecisionRecord" for row in repo.objects.values())


@pytest.mark.parametrize(
    ("action_id", "payload", "expected_uid"),
    [
        (
            "create_action_item",
            {
                "description": "Review daily report flag for OKLO 2026-05-06",
                "action_type": "review",
                "ticker": "OKLO",
                "urgency": "normal",
            },
            "action_item:review_daily_report_flag_for_oklo_2026_05_06",
        ),
        (
            "create_watch_trigger",
            {
                "condition": "Watch OKLO breadth reversal",
                "trigger_type": "custom",
                "ticker": "OKLO",
            },
            "watch_trigger:watch_oklo_breadth_reversal",
        ),
        (
            "create_research_note",
            {
                "title": "OKLO daily report flag",
                "ticker": "OKLO",
                "note": "Review daily report flag for OKLO.",
                "document_id": "daily:2026-05-06:oklo-flag",
            },
            "document_artifact:research_note:daily_2026_05_06_oklo_flag",
        ),
    ],
)
def test_research_object_approvals_apply_with_schema_canonical_ids(action_id, payload, expected_uid):
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="workflow", source_id="daily")

    approval = service.propose_action(action_id, payload, context, reason="Apply research object")
    applied = service.resolve_approval(approval["id"], "approved", "approved", context)

    assert applied["application_status"] == "applied"
    assert expected_uid in repo.objects


def test_unsupported_action_is_rejected_before_any_write():
    fake = FakeObjectService()
    service = OntologyCommandService(fake)  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    with pytest.raises(OntologyCommandValidationError):
        service.propose_action("legacy_unregistered_write", {}, context)
    assert fake.objects == {}


def test_restaged_approval_uses_distinct_uid_and_survives_original_rejection(monkeypatch):
    import portfolio.action_registry as action_registry

    monkeypatch.setattr(action_registry, "compute_action_base_state_hash", lambda _action_id, _payload: "base")
    service = OntologyCommandService(FakeObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")
    payload = {"item_id": 1, "resolution_note": "Done"}

    original = service.propose_action("complete_action_item", payload, context, reason="Complete item")
    replacement = service.propose_action(
        "complete_action_item",
        payload,
        context,
        reason="Restage item",
        supersedes_approval_id=original["id"],
    )

    assert replacement["id"] != original["id"]
    assert replacement["supersedes_approval_id"] == original["id"]

    rejected = service.resolve_approval(original["id"], "rejected", "Superseded", context)

    assert rejected["status"] == "rejected"
    assert service.get_approval(replacement["id"], actor=context.actor)["status"] == "pending"


def test_approve_rejects_stale_ontology_base_state(monkeypatch):
    import portfolio.action_registry as action_registry

    current_hash = {"value": "old"}
    monkeypatch.setattr(
        action_registry,
        "compute_action_base_state_hash",
        lambda _action_id, _payload: current_hash["value"],
    )
    service = OntologyCommandService(FakeObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "complete_action_item",
        {"item_id": 1, "resolution_note": "Done"},
        context,
        reason="Complete item",
    )
    current_hash["value"] = "new"

    with pytest.raises(OntologyCommandConflict, match="base state changed"):
        service.resolve_approval(approval["id"], "approved", "Apply", context)

    assert service.get_approval(approval["id"], actor=context.actor)["status"] == "pending"


def test_apply_failure_keeps_ontology_approval_retryable(monkeypatch):
    service = OntologyCommandService(FakeObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "create_action_item",
        {"ticker": "MU", "description": "Review MU thesis", "action_type": "review"},
        context,
        reason="Create action item",
    )

    def fail_apply(*args, **kwargs):
        raise RuntimeError("cannot apply")

    monkeypatch.setattr(service, "_write_action_targets", fail_apply)

    with pytest.raises(OntologyCommandConflict, match="cannot apply"):
        service.resolve_approval(approval["id"], "approved", "Apply", context)

    failed = service.get_approval(approval["id"], actor=context.actor)
    assert failed["status"] == "pending"
    assert failed["resolution_state"] == "pending"
    assert failed["application_status"] == "failed"
    assert failed["application_state"] == "failed"
    assert failed["application_attempts"] == 1
    assert failed["application_error"] == "cannot apply"


def test_audit_write_failure_does_not_break_ontology_rejection():
    class AuditFailObjectService(FakeObjectService):
        def write_object(self, object_type, business_key, properties, valid_from, **kwargs):
            if object_type == "AuditEvent":
                raise RuntimeError("audit down")
            return super().write_object(object_type, business_key, properties, valid_from, **kwargs)

    service = OntologyCommandService(AuditFailObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "create_action_item",
        {"ticker": "MU", "description": "Review MU thesis", "action_type": "review"},
        context,
        reason="Create action item",
    )
    rejected = service.resolve_approval(approval["id"], "rejected", "Skip", context)

    assert rejected["status"] == "rejected"
    assert rejected["application_status"] == "not_applicable"


def test_save_management_quality_content_writes_ontology_children_and_markdown(monkeypatch, tmp_path):
    import portfolio.management_quality_content as management_quality_content

    indexed: list[dict[str, Any]] = []
    mgmt_dir = tmp_path / "investment_management_quality"
    mgmt_dir.mkdir()
    monkeypatch.setattr(management_quality_content, "MANAGEMENT_QUALITY_DIR", mgmt_dir)
    monkeypatch.setattr("api.retrieval.index_document", lambda **kwargs: indexed.append(kwargs))

    fake = FakeObjectService()
    service = OntologyCommandService(fake)  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "save_management_quality_content",
        {
            "ticker": "mu",
            "content": """# MU Management Quality

## Executive Summary
- **Overall Rating**: Strong
- **Bottom Line**: Good operator.
- **Owner Mindset**: Strong - Disciplined capital allocation.
- **Business Value Understanding**: Mixed - Some gaps.
- **Follow-through / Character**: Strong - Targets met.

## Management Scorecard
| Question | Rating | Evidence |
|----------|--------|----------|
| Do managers think and act like owners? | Strong | Buybacks were disciplined. |

## Most Impressive Accomplishments
- **HBM ramp (2025)**: Executed well.

## Biggest Setbacks and Responses
- **Inventory cycle (2023)**: Downturn. **Response**: Mixed - Costs were reset.
""",
        },
        context,
        reason="unit",
    )

    applied = service.resolve_approval(approval["id"], "approved", "apply", context)

    assert applied["application_status"] == "applied"
    assert (mgmt_dir / "MU.md").read_text(encoding="utf-8").endswith("\n")
    object_types = {row["object_type"] for row in fake.objects.values()}
    assert "ManagementQualityAssessment" in object_types
    assert "ManagementQualityScorecardRow" in object_types
    assert "ManagementQualityAccomplishment" in object_types
    assert "ManagementQualitySetback" in object_types
    assert any(rel["relation_type"] == "management_quality_assesses_issuer" for rel in fake.relations)
    assert any(rel["relation_type"] == "research_object_uses_document" for rel in fake.relations)
    assert indexed and indexed[0]["doc_type"] == "management_quality"
