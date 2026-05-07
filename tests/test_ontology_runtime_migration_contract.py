from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


class _FakeObjectService:
    def write_object(self, object_type: str, business_key: str, properties: dict[str, Any], valid_from: str, **kwargs):
        return {
            "object_uid": f"{object_type.lower()}:{business_key}",
            "object_type": object_type,
            "properties": dict(properties),
            "valid_from": datetime(2026, 5, 6, tzinfo=UTC),
            "_meta": {
                "temporal": {
                    "object_uid": f"{object_type.lower()}:{business_key}",
                    "version_id": f"version:{object_type}",
                    "valid_from": str(valid_from),
                }
            },
        }


class _ValidatingFakeObjectService:
    def __init__(self):
        self.object_writes: list[dict[str, Any]] = []

    def write_object(self, object_type: str, business_key: str, properties: dict[str, Any], valid_from: str, **kwargs):
        from ontology.schemas.registry import NODE_SCHEMAS

        NODE_SCHEMAS[object_type].model_validate(properties)
        self.object_writes.append(dict(properties))
        return {
            "object_uid": f"{object_type.lower()}:{business_key}",
            "object_type": object_type,
            "properties": {**properties, "schema_version": 1},
            "valid_from": datetime(2026, 5, 6, tzinfo=UTC),
            "_meta": {
                "temporal": {
                    "object_uid": f"{object_type.lower()}:{business_key}",
                    "version_id": f"version:{object_type}:{len(self.object_writes)}",
                    "valid_from": str(valid_from),
                }
            },
        }

    def write_relation(self, *args, **kwargs):
        return {}


def test_idea_runtime_write_preserves_temporal_meta(monkeypatch):
    from api.routers import ideas

    monkeypatch.setattr(ideas, "ontology_primary_writes_enabled", lambda: True)
    monkeypatch.setattr(ideas, "OntologyObjectService", _FakeObjectService)

    row = ideas._write_runtime_object(  # noqa: SLF001 - migration contract helper coverage.
        "InvestmentIdea",
        "investment_idea:MU",
        {"ticker": "MU", "status": "watching"},
    )

    assert isinstance(row["id"], str)
    assert row["object_uid"] == row["id"]
    assert row["_meta"]["temporal"]["version_id"] == "version:InvestmentIdea"


def test_optimizer_runtime_write_preserves_temporal_meta(monkeypatch):
    import api.continuous_optimizer as optimizer

    monkeypatch.setattr(optimizer, "ontology_primary_writes_enabled", lambda: True)
    monkeypatch.setattr(optimizer, "OntologyObjectService", _FakeObjectService)

    row = optimizer._write_runtime_object(  # noqa: SLF001 - migration contract helper coverage.
        "OptimizationAlert",
        "optimization_alert:run-1:MU",
        {
            "mission_id": "optimization_mission:default",
            "run_id": "optimization_run:run-1",
            "ticker": "MU",
            "alert_type": "action_changed",
            "severity": "normal",
            "status": "open",
            "change_summary": "MU: action changed.",
        },
    )

    assert isinstance(row["id"], str)
    assert row["object_uid"] == row["id"]
    assert row["_meta"]["temporal"]["version_id"] == "version:OptimizationAlert"


def test_optimizer_runtime_write_strips_envelope_fields_before_rewrite(monkeypatch):
    import api.continuous_optimizer as optimizer

    service = _ValidatingFakeObjectService()
    monkeypatch.setattr(optimizer, "ontology_primary_writes_enabled", lambda: True)
    monkeypatch.setattr(optimizer, "OntologyObjectService", lambda: service)

    run = optimizer._write_runtime_object(  # noqa: SLF001 - migration contract helper coverage.
        "OptimizationRun",
        "optimization_run:test",
        {
            "id": "optimization_run:test",
            "run_id": "optimization_run:test",
            "mission_id": "optimization_mission:default",
            "mission_name": "Daily Command Center",
            "status": "running",
            "started_at": "2026-05-06T14:15:00+00:00",
            "input_hash": "input",
        },
    )

    completed = optimizer._write_runtime_object(  # noqa: SLF001 - migration contract helper coverage.
        "OptimizationRun",
        str(run["id"]),
        {
            **run,
            "status": "completed",
            "completed_at": "2026-05-06T14:16:00+00:00",
            "summary": {"alerts_created": 0},
            "source_freshness": {},
            "output_hash": "output",
        },
    )

    submitted = service.object_writes[-1]
    assert completed["object_uid"] == completed["id"]
    assert "object_uid" not in submitted
    assert "_meta" not in submitted
    assert "object_type" not in submitted
    assert "schema_name" not in submitted
    assert "business_key" not in submitted
    assert "schema_version" not in submitted
