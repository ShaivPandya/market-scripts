from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ontology.object_service import OntologyObjectService, OntologyWriteContractError, relation_uid_for
from ontology.temporal_repository import ObjectVersionWrite, RelationVersionWrite


class _FakeTemporalRepo:
    def __init__(self):
        self.object_writes: list[ObjectVersionWrite] = []
        self.relation_writes: list[RelationVersionWrite] = []

    def write_object_version(self, write: ObjectVersionWrite):
        self.object_writes.append(write)
        return {
            "version_id": "00000000-0000-0000-0000-000000000001",
            "object_uid": write.object_uid,
            "object_type": write.object_type,
            "business_key": write.business_key,
            "schema_name": write.schema_name,
            "schema_version": write.schema_version,
            "properties_json": write.properties,
            "valid_from": datetime(2026, 5, 1, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 3, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }

    def write_relation_version(self, write: RelationVersionWrite):
        self.relation_writes.append(write)
        return {
            "version_id": "00000000-0000-0000-0000-000000000002",
            "relation_uid": write.relation_uid,
            "source_object_uid": write.source_object_uid,
            "target_object_uid": write.target_object_uid,
            "relation_type": write.relation_type,
            "relation_schema_name": write.relation_schema_name,
            "relation_schema_version": write.relation_schema_version,
            "properties_json": write.properties,
            "valid_from": datetime(2026, 5, 1, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 3, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }


def test_object_service_writes_normalized_position_with_temporal_meta():
    repo = _FakeTemporalRepo()
    service = OntologyObjectService(repository=repo)

    row = service.write_object(
        "Position",
        "MU",
        {
            "ticker": "MU",
            "asset": "Equity",
            "direction": "Long",
            "timeframe": "Daily",
            "risk_score": 0.62,
            "risk_level": "medium",
            "volatility_cluster": 0.5,
            "breadth_stress": 0.6,
            "sector_stress": 0.6,
            "macro_regime": 0.7,
            "ontology_run_id": "run-1",
        },
        valid_from="2026-05-01T00:00:00Z",
        actor=SimpleNamespace(actor_type="user", actor_id="u1"),
        provenance={"provenance_event_id": "pv:test"},
    )

    write = repo.object_writes[0]
    assert write.object_uid == "position:MU"
    assert write.schema_name == "Position"
    assert write.properties["asset"] == "equity"
    assert write.actor_type == "user"
    assert row["_meta"]["temporal"]["object_uid"] == "position:MU"
    assert row["_meta"]["temporal"]["version_id"] == "00000000-0000-0000-0000-000000000001"


def test_object_service_writes_relation_with_uid_and_temporal_meta():
    repo = _FakeTemporalRepo()
    service = OntologyObjectService(repository=repo)

    row = service.write_relation(
        "position:MU",
        "asset:MU",
        "references_asset",
        {"ontology_run_id": "run-1"},
        valid_from="2026-05-01T00:00:00Z",
        actor={"actor_type": "system", "actor_id": "test"},
        provenance="pv:test",
    )

    write = repo.relation_writes[0]
    assert write.relation_uid == "references_asset:position:MU->asset:MU"
    assert write.relation_schema_name == "references_asset"
    assert row["_meta"]["temporal"]["relation_uid"] == "references_asset:position:MU->asset:MU"


def test_object_service_rejects_writes_without_provenance():
    repo = _FakeTemporalRepo()
    service = OntologyObjectService(repository=repo)

    with pytest.raises(OntologyWriteContractError, match="requires provenance"):
        service.write_object(
            "Position",
            "MU",
            {
                "ticker": "MU",
                "asset": "equity",
                "direction": "long",
                "risk_score": 0.0,
                "risk_level": "low",
                "ontology_run_id": "run-1",
            },
            valid_from="2026-05-01T00:00:00Z",
        )

    with pytest.raises(OntologyWriteContractError, match="requires provenance"):
        service.write_relation(
            "position:MU",
            "asset:MU",
            "references_asset",
            {"ontology_run_id": "run-1"},
            valid_from="2026-05-01T00:00:00Z",
        )

    with pytest.raises(OntologyWriteContractError, match="requires provenance"):
        service.correct_object_version(
            "00000000-0000-0000-0000-000000000001",
            properties={"ticker": "MU"},
        )


@pytest.mark.parametrize("env_name", ["ENVIRONMENT", "ONTOLOGY_PRIMARY_WRITES"])
def test_object_service_rejects_unregistered_object_types_in_primary_runtime(monkeypatch, env_name):
    repo = _FakeTemporalRepo()
    service = OntologyObjectService(repository=repo)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("ONTOLOGY_PRIMARY_WRITES", raising=False)
    if env_name == "ENVIRONMENT":
        monkeypatch.setenv("ENVIRONMENT", "production")
    else:
        monkeypatch.setenv("ONTOLOGY_PRIMARY_WRITES", "true")

    with pytest.raises(OntologyWriteContractError, match="Unregistered ontology object type"):
        service.write_object(
            "UnknownType",
            "unknown:1",
            {"value": 1},
            valid_from="2026-05-01T00:00:00Z",
            provenance="pv:test",
        )

    assert repo.object_writes == []


def test_object_service_writes_registered_runtime_migration_types():
    repo = _FakeTemporalRepo()
    service = OntologyObjectService(repository=repo)

    idea = service.write_object(
        "InvestmentIdea",
        "investment_idea:MU:seed",
        {
            "ticker": "MU",
            "company_name": "Micron",
            "status": "watching",
            "source_type": "user",
            "source_id": "ideas.create",
        },
        valid_from="2026-05-01T00:00:00Z",
        provenance="pv:idea",
    )
    alert = service.write_object(
        "OptimizationAlert",
        "optimization_alert:run-1:MU:action_changed",
        {
            "mission_id": "optimization_mission:default",
            "run_id": "optimization_run:run-1",
            "ticker": "MU",
            "alert_type": "action_changed",
            "severity": "normal",
            "status": "open",
            "change_summary": "MU: action changed.",
        },
        valid_from="2026-05-01T00:00:00Z",
        provenance="pv:optimizer",
    )
    event = service.write_object(
        "ProvenanceEvent",
        "pv:unit",
        {
            "event_type": "unit",
            "event_name": "test",
            "status": "started",
            "lineage_root_id": "pv:unit",
            "redaction_policy": "audit_summary_v1",
            "retention_class": "provenance_365d",
        },
        valid_from="2026-05-01T00:00:00Z",
        provenance="pv:unit",
    )

    assert idea["schema_name"] == "InvestmentIdea"
    assert idea["object_uid"] == "investment_idea:mu_seed"
    assert alert["schema_name"] == "OptimizationAlert"
    assert alert["object_uid"] == "optimization_alert:run_1_mu_action_changed"
    assert event["schema_name"] == "ProvenanceEvent"
    assert event["object_uid"] == "provenance_event:pv_unit"


def test_typed_provenance_relation_is_registered_and_temporal():
    repo = _FakeTemporalRepo()
    service = OntologyObjectService(repository=repo)

    row = service.write_relation(
        "provenance_event:pv_unit",
        "object_version_ref:version_1",
        "provenance_produced",
        {
            "event_id": "pv:unit",
            "ontology_run_id": "operational",
            "source_ref_type": "producer_event",
            "source_ref_id": "pv:unit",
            "target_ref_type": "ontology_object_version",
            "target_ref_id": "version-1",
            "redaction_policy": "audit_summary_v1",
            "retention_class": "provenance_365d",
        },
        valid_from="2026-05-01T00:00:00Z",
        provenance="pv:unit",
    )

    write = repo.relation_writes[0]
    assert write.relation_type == "provenance_produced"
    assert write.relation_schema_name == "provenance_produced"
    assert row["_meta"]["temporal"]["relation_uid"].startswith("provenance_produced:")


def test_provenance_relation_uid_uses_full_ref_tuple():
    props = {
        "event_id": "pv:unit",
        "source_ref_type": "producer_event",
        "source_ref_id": "pv:unit",
        "source_ref_version": "event-v1",
        "target_ref_type": "ontology_object_version",
        "target_ref_id": "version-1",
        "target_ref_version": "object-v1",
    }

    uid = relation_uid_for("provenance_event:pv_unit", "object_version_ref:version_1", "provenance_produced", props)
    same = relation_uid_for(
        "provenance_event:pv_unit",
        "object_version_ref:version_1",
        "provenance_produced",
        dict(props),
    )
    different_event = relation_uid_for(
        "provenance_event:pv_unit",
        "object_version_ref:version_1",
        "provenance_produced",
        {**props, "event_id": "pv:other"},
    )
    different_target_version = relation_uid_for(
        "provenance_event:pv_unit",
        "object_version_ref:version_1",
        "provenance_produced",
        {**props, "target_ref_version": "object-v2"},
    )

    assert uid == same
    assert uid != different_event
    assert uid != different_target_version


def test_typed_provenance_relation_requires_redaction_retention_and_refs():
    service = OntologyObjectService(repository=_FakeTemporalRepo())

    with pytest.raises(OntologyWriteContractError, match="redaction_policy"):
        service.write_relation(
            "provenance_event:pv_unit",
            "object_version_ref:version_1",
            "provenance_produced",
            {
                "event_id": "pv:unit",
                "ontology_run_id": "operational",
                "source_ref_type": "producer_event",
                "source_ref_id": "pv:unit",
                "target_ref_type": "ontology_object_version",
                "target_ref_id": "version-1",
                "retention_class": "provenance_365d",
            },
            valid_from="2026-05-01T00:00:00Z",
            provenance="pv:unit",
        )


def test_postgres_snapshot_success_uses_temporal_versions(monkeypatch):
    from api import snapshot_store

    captured = {}

    class _Repo:
        def write_computed_snapshot_version(self, write):
            captured["write"] = write
            return {
                "snapshot_key": write.snapshot_key,
                "payload_json": write.payload,
                "as_of": datetime(2026, 5, 1, tzinfo=UTC),
                "load_time": datetime(2026, 5, 3, tzinfo=UTC),
                "status": write.status,
                "error": write.error,
                "artifact_uri": write.artifact_uri,
            }

    monkeypatch.setattr(snapshot_store, "use_postgres_state", lambda: True)
    monkeypatch.setattr(snapshot_store, "TemporalOntologyRepository", lambda: _Repo())

    record = snapshot_store.write_snapshot_success("unit:temporal", {"value": 1}, as_of_date="2026-05-01")

    assert captured["write"].snapshot_key == "unit:temporal"
    assert captured["write"].status == "ok"
    assert record.payload == {"value": 1}
