from __future__ import annotations

from decision_quality.opportunity_scout import (
    build_candidate_from_monitor_hit,
    compute_candidate_rank_score,
    normalize_candidate_queue_item,
    rank_opportunity_candidates,
)


def test_build_candidate_from_monitor_hit_includes_required_fields():
    record = build_candidate_from_monitor_hit(
        {
            "ticker": "NVDA",
            "entity_type": "kill_condition",
            "entity_id": "kill_condition:1",
            "entity_label": "Margin compression threshold",
            "hit_type": "triggered",
            "severity": "high",
            "evidence": "Price crossed threshold",
            "fingerprint": "abc123",
            "confidence": 0.9,
        },
        source_id="monitor:abc123",
    )
    assert record["ticker"] == "NVDA"
    assert record["source_kind"] == "monitor_hit"
    assert record["trigger"]
    assert record["why_now"]
    assert record["missing_inputs"]
    assert record["next_action"] == "research"
    assert record["status"] == "open"
    assert record["opportunity_candidate_gate"]["status"] in {"pass", "downgraded", "blocked"}


def test_rank_opportunity_candidates_orders_by_score():
    rows = [
        {"candidate_id": "low", "severity": "low", "missing_inputs": ["a", "b", "c"], "next_action": "research"},
        {
            "candidate_id": "high",
            "severity": "high",
            "rank_signals": {"severity": "high", "hit_type": "triggered", "confidence": 0.9},
            "missing_inputs": [],
            "next_action": "graduate_to_decision_quality",
            "updated_at": "2026-06-01T12:00:00Z",
        },
    ]
    ranked = rank_opportunity_candidates(rows)
    assert ranked[0]["candidate_id"] == "high"
    assert ranked[0]["rank_score"] > ranked[1]["rank_score"]


def test_compute_candidate_rank_score_is_deterministic():
    row = {
        "severity": "high",
        "rank_signals": {"severity": "high", "hit_type": "triggered", "confidence": 0.8},
        "missing_inputs": ["one"],
        "next_action": "research",
        "opportunity_candidate_gate": {"status": "pass"},
    }
    assert compute_candidate_rank_score(row) == compute_candidate_rank_score(row)


def test_normalize_candidate_queue_item_projects_api_shape():
    item = normalize_candidate_queue_item(
        {
            "candidate_id": "candidate:nvda:pullback",
            "object_uid": "opportunity_candidate:candidate-nvda-pullback",
            "ticker": "NVDA",
            "source_kind": "monitor_hit",
            "trigger": "Monitor hit",
            "opportunity_type": "quality_compounder",
            "consensus": "Crowded",
            "variant_view": "Pullback",
            "why_now": "Now",
            "price_confirmation": "Needs chart",
            "missing_inputs": ["Chart"],
            "next_action": "research",
            "summary": "Research",
            "status": "open",
            "decision_state": "generated",
            "opportunity_candidate_gate": {"status": "pass", "final_action": "research", "should_graduate": False},
            "updated_at": "2026-06-01T12:00:00Z",
        }
    )
    assert item["candidate_id"] == "candidate:nvda:pullback"
    assert item["gate_status"] == "pass"
    assert item["rank_score"] > 0
