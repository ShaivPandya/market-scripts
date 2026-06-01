from __future__ import annotations

import pytest

from decision_quality.opportunity_scout import maybe_create_candidate_from_monitor_hit


class _FakeWriteback:
    calls: list[dict] = []

    def record_opportunity_candidate(self, *, record, actor, provenance_id, valid_from=None, approval_id=None):
        self.__class__.calls.append({"record": dict(record), "provenance_id": provenance_id})
        return [{"object_type": "OpportunityCandidate", "object_uid": "opportunity_candidate:test"}]


def test_maybe_create_candidate_from_monitor_hit_persists_actionable_hit(monkeypatch):
    _FakeWriteback.calls = []
    monkeypatch.setattr("decision_quality.opportunity_scout.DecisionOntologyWriteback", _FakeWriteback)
    result = maybe_create_candidate_from_monitor_hit(
        {
            "ticker": "MU",
            "entity_type": "kill_condition",
            "entity_id": "kill_condition:1",
            "entity_label": "Threshold breach",
            "hit_type": "triggered",
            "severity": "high",
            "evidence": "Price crossed threshold",
            "fingerprint": "fingerprint-1",
        },
        source_id="monitor:fingerprint-1",
    )
    assert result is not None
    assert _FakeWriteback.calls
    assert _FakeWriteback.calls[0]["record"]["ticker"] == "MU"


def test_maybe_create_candidate_from_monitor_hit_skips_ok_hits(monkeypatch):
    _FakeWriteback.calls = []
    monkeypatch.setattr("decision_quality.opportunity_scout.DecisionOntologyWriteback", _FakeWriteback)
    result = maybe_create_candidate_from_monitor_hit(
        {"ticker": "MU", "hit_type": "ok", "fingerprint": "fingerprint-2"},
        source_id="monitor:fingerprint-2",
    )
    assert result is None
    assert _FakeWriteback.calls == []


def test_maybe_create_candidate_from_monitor_hit_skips_duplicate_idempotency(monkeypatch):
    _FakeWriteback.calls = []
    monkeypatch.setattr("decision_quality.opportunity_scout.DecisionOntologyWriteback", _FakeWriteback)
    hit = {
        "ticker": "MU",
        "entity_type": "catalyst",
        "entity_id": "catalyst:1",
        "entity_label": "Earnings",
        "hit_type": "needs_review",
        "severity": "medium",
        "evidence": "Approaching",
        "fingerprint": "fingerprint-3",
    }
    record = maybe_create_candidate_from_monitor_hit(hit, source_id="monitor:fingerprint-3")
    assert record is not None
    idempotency_key = record["idempotency_key"]
    duplicate = maybe_create_candidate_from_monitor_hit(
        hit,
        source_id="monitor:fingerprint-3",
        existing_idempotency_keys={idempotency_key},
    )
    assert duplicate is None
    assert len(_FakeWriteback.calls) == 1
