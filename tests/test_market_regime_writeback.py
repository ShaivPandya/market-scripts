from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from api.snapshot_keys import SNAPSHOT_SIGNAL_AGGREGATOR
from ontology.market_regime_writeback import materialize_signal_aggregator_snapshot
from ontology.object_service import OntologyObjectService
from ontology.temporal_repository import ObjectVersionWrite, RelationVersionWrite


class _FakeTemporalRepo:
    def __init__(self):
        self.object_writes: list[ObjectVersionWrite] = []
        self.relation_writes: list[RelationVersionWrite] = []

    def write_object_version(self, write: ObjectVersionWrite):
        self.object_writes.append(write)
        return {
            "version_id": f"version-{len(self.object_writes)}",
            "object_uid": write.object_uid,
            "object_type": write.object_type,
            "business_key": write.business_key,
            "schema_name": write.schema_name,
            "schema_version": write.schema_version,
            "properties_json": write.properties,
            "valid_from": datetime(2026, 5, 6, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 6, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }

    def write_relation_version(self, write: RelationVersionWrite):
        self.relation_writes.append(write)
        return {
            "version_id": f"relation-{len(self.relation_writes)}",
            "relation_uid": write.relation_uid,
            "source_object_uid": write.source_object_uid,
            "target_object_uid": write.target_object_uid,
            "relation_type": write.relation_type,
            "relation_schema_name": write.relation_schema_name,
            "relation_schema_version": write.relation_schema_version,
            "properties_json": write.properties,
            "valid_from": datetime(2026, 5, 6, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 6, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }


def test_signal_aggregator_materializes_typed_regime_state_with_lineage():
    repo = _FakeTemporalRepo()
    payload: dict[str, Any] = {
        "as_of": "2026-05-06",
        "regime": {"label": "risk_on", "score": 73, "confidence": 0.82, "history_percentile": 68},
        "weights": {"breadth": 0.3, "liquidity": 0.2},
        "module_status": {"breadth": {"status": "ok"}, "liquidity": {"status": "stale"}},
        "failed_modules": ["positioning"],
        "factors": [
            {"key": "breadth", "name": "Market breadth", "score": 70, "weight": 0.3, "status": "ok"},
            {"key": "liquidity", "name": "Liquidity", "score": 55, "weight": 0.2, "status": "stale"},
        ],
        "forward_outlook": {"label": "constructive", "detail": "Breadth is firm.", "basis": ["breadth"]},
        "history": {"episodes": [{"regime": "risk_on", "start_date": "2026-04-01", "weeks": 5, "avg_score": 71}]},
    }

    rows = materialize_signal_aggregator_snapshot(
        snapshot_key=SNAPSHOT_SIGNAL_AGGREGATOR,
        snapshot_version_id="snapshot-version-1",
        payload=payload,
        as_of_date="2026-05-06",
        fetched_at="2026-05-06T12:00:00+00:00",
        status="error",
        quality="degraded",
        error="positioning timeout",
        object_service=OntologyObjectService(repository=repo),
        provenance_id="pv:test-regime",
    )

    object_types = {write.object_type for write in repo.object_writes}
    relation_types = {write.relation_type for write in repo.relation_writes}
    regime = next(write for write in repo.object_writes if write.object_type == "MarketRegimeSnapshot")

    assert rows
    assert {
        "ComputedSnapshotRef",
        "ObjectVersionRef",
        "MarketRegimeSnapshot",
        "SignalFactorScore",
        "ForwardOutlook",
        "RegimeEpisode",
    } <= object_types
    assert {
        "computed_snapshot_materializes_object_version",
        "market_regime_has_factor_score",
        "factor_score_uses_computed_snapshot",
        "market_regime_has_forward_outlook",
        "market_regime_has_episode",
    } <= relation_types
    assert regime.properties["status"] == "error"
    assert regime.properties["quality"] == "degraded"
    assert regime.properties["error"] == "positioning timeout"
    assert regime.properties["source_status"]["liquidity"]["status"] == "stale"
    assert len([write for write in repo.object_writes if write.object_type == "SignalFactorScore"]) == 2
