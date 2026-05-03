from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ontology.object_service import OntologyObjectService
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
