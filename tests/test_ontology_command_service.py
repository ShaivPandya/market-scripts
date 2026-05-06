from __future__ import annotations

from typing import Any

import pytest

from ontology.command_service import (
    OntologyCommandContext,
    OntologyCommandService,
    OntologyCommandValidationError,
)
from ontology.policy import admin_actor


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
                "instrument:",
                "portfolio:",
                "position:",
                "recommendation:",
                "research_note:",
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


def test_unsupported_action_is_rejected_before_any_write():
    fake = FakeObjectService()
    service = OntologyCommandService(fake)  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    with pytest.raises(OntologyCommandValidationError):
        service.propose_action("legacy_unregistered_write", {}, context)
    assert fake.objects == {}
