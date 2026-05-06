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
