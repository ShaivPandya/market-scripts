from __future__ import annotations

from datetime import datetime, timedelta

from api.snapshot_store import SnapshotRecord
from ontology.sources.reliability import derive_reliability_tier, gate_action_for_tier, sla_seconds_for_registry
from ontology.sources.source_registry import all_source_registry_entries


def test_derive_reliability_tier_defaults():
    entries = all_source_registry_entries()

    assert derive_reliability_tier(entries["market_breadth"]) == "critical"
    assert derive_reliability_tier(entries["market_regime"]) == "standard"
    assert derive_reliability_tier(entries["momentum"]) == "supplemental"
    assert derive_reliability_tier(entries["portfolio_news_digest"]) == "ad_hoc"


def test_gate_action_for_tier_blocks_critical_stale():
    assert gate_action_for_tier("critical", "stale") == "block"
    assert gate_action_for_tier("critical", "failed") == "block"
    assert gate_action_for_tier("standard", "stale") == "warn"
    assert gate_action_for_tier("ad_hoc", "stale") == "inform"


def test_sla_seconds_for_registry_uses_per_source_value():
    registry = {"freshness_sla_seconds": 3600}
    assert sla_seconds_for_registry(registry) == 3600


def test_snapshot_staleness_honors_per_source_sla():
    from api.source_health import _snapshot_is_stale

    now = datetime(2026, 5, 14, 18, 0)
    record = SnapshotRecord(
        snapshot_key="market_breadth:sp500:1y",
        payload={"value": 1},
        as_of_date=now.date().isoformat(),
        fetched_at=(now - timedelta(hours=2)).isoformat(),
        status="ok",
        error=None,
        version=1,
        artifact_uri=None,
        quality="ok",
    )

    assert _snapshot_is_stale(record, now=now, sla_seconds=3600) is True
    assert _snapshot_is_stale(record, now=now, sla_seconds=129600) is False
