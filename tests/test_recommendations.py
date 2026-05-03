from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

import portfolio.core_db as core_db
from auto_report.recommendations import (
    RECOMMENDATIONS_SEPARATOR,
    RecommendationValidationError,
    assess_report_data_quality,
    parse_recommendations_response,
    persist_recommendations,
    validate_recommendations_payload,
)


@pytest.fixture
def data_quality_ok():
    return {
        "critical_data_quality": "ok",
        "recommendations_blocked": False,
        "blocked_reasons": [],
        "sources": [],
    }


@pytest.fixture
def temp_core_db(tmp_path, monkeypatch):
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "recommendations.db")
    monkeypatch.setattr(core_db, "_conn", None)
    yield
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "_conn", None)


def _valid_payload(action: str = "do_nothing") -> dict:
    return {
        "report_type": "daily",
        "as_of": "2026-05-02",
        "stance": "Neutral / Watchful",
        "recommendation_status": "clear",
        "critical_data_quality": "ok",
        "blocked_reasons": [],
        "do_nothing_rationale": "No fat pitch today.",
        "what_changed": ["Breadth and liquidity are mixed."],
        "recommended_actions": [
            {
                "action": action,
                "ticker": "MU" if action == "buy" else None,
                "instrument": "MU" if action == "buy" else "portfolio",
                "horizon": "1 trading day",
                "target_change": "start one-third size" if action == "buy" else "none",
                "rationale": "Evidence does not justify forcing a trade." if action != "buy" else "Validated setup.",
                "evidence": ["price action confirms"],
                "disconfirming_evidence": ["liquidity mixed"],
                "catalyst": "earnings",
                "invalidation": "breaks support",
                "expected_onset_window": "1 week",
                "confidence": 0.64,
                "source_quality": "ok",
                "approval_required": action == "buy",
            }
        ],
        "alternatives": [],
        "opportunity_cost": [],
    }


def test_validate_recommendations_payload_accepts_valid_contract(data_quality_ok):
    payload = validate_recommendations_payload(
        _valid_payload("buy"),
        report_type="daily",
        as_of="2026-05-02",
        stance="Neutral / Watchful",
        data_quality=data_quality_ok,
    )

    assert payload["stance"] == "Neutral / Watchful"
    assert payload["recommended_actions"][0]["action"] == "buy"
    assert payload["recommended_actions"][0]["approval_required"] is True


def test_validate_recommendations_payload_rejects_invalid_stance(data_quality_ok):
    payload = _valid_payload()
    payload["stance"] = "bullish"

    with pytest.raises(RecommendationValidationError, match="stance"):
        validate_recommendations_payload(
            payload,
            report_type="daily",
            as_of="2026-05-02",
            stance="Neutral / Watchful",
            data_quality=data_quality_ok,
        )


def test_validate_recommendations_payload_rejects_invalid_action(data_quality_ok):
    payload = _valid_payload()
    payload["recommended_actions"][0]["action"] = "trim"

    with pytest.raises(RecommendationValidationError, match="action"):
        validate_recommendations_payload(
            payload,
            report_type="daily",
            as_of="2026-05-02",
            stance="Neutral / Watchful",
            data_quality=data_quality_ok,
        )


def test_validate_recommendations_payload_rejects_missing_required_fields(data_quality_ok):
    payload = _valid_payload()
    del payload["do_nothing_rationale"]
    del payload["recommended_actions"][0]["invalidation"]

    with pytest.raises(RecommendationValidationError, match="missing required"):
        validate_recommendations_payload(
            payload,
            report_type="daily",
            as_of="2026-05-02",
            stance="Neutral / Watchful",
            data_quality=data_quality_ok,
        )


def test_parse_recommendations_response_handles_fenced_json(data_quality_ok):
    raw = "Memo\n" + RECOMMENDATIONS_SEPARATOR + "\n```json\n" + json.dumps(_valid_payload()) + "\n```"

    memo, payload = parse_recommendations_response(
        raw,
        report_type="daily",
        as_of="2026-05-02",
        stance="Neutral / Watchful",
        data_quality=data_quality_ok,
    )

    assert memo == "Memo"
    assert payload["recommended_actions"][0]["action"] == "do_nothing"


def test_parse_recommendations_response_requires_separator(data_quality_ok):
    with pytest.raises(RecommendationValidationError, match="separator"):
        parse_recommendations_response(
            json.dumps(_valid_payload()),
            report_type="daily",
            as_of="2026-05-02",
            stance="Neutral / Watchful",
            data_quality=data_quality_ok,
        )


def test_data_quality_blocks_failed_critical_source():
    today = datetime.now(UTC).date().isoformat()
    raw = {
        "portfolio_positions": [{"ticker": "MU"}],
        "risk_data": {"timestamp": today},
        "sizer_summary": {"timestamp": today},
        "market_data": {
            "indices": {"timestamp": today},
            "breadth": {"timestamp": today},
            "vix": {"timestamp": today},
            "liquidity": {"error": "timeout"},
            "yield_curve": {"timestamp": today},
        },
    }

    quality = assess_report_data_quality(raw, "daily")

    assert quality["recommendations_blocked"] is True
    assert quality["critical_data_quality"] == "failed"
    assert any("liquidity" in reason for reason in quality["blocked_reasons"])


def test_data_quality_allows_noncritical_degraded_source():
    today = datetime.now(UTC).date().isoformat()
    raw = {
        "portfolio_positions": [{"ticker": "MU"}],
        "risk_data": {"timestamp": today},
        "sizer_summary": {"timestamp": today},
        "market_data": {
            "indices": {"timestamp": today},
            "breadth": {"timestamp": today},
            "vix": {"timestamp": today},
            "liquidity": {"timestamp": today},
            "yield_curve": {"timestamp": today},
            "sentiment": {"error": "survey timeout"},
        },
    }

    quality = assess_report_data_quality(raw, "daily")

    assert quality["recommendations_blocked"] is False
    assert quality["overall_status"] == "degraded"


def test_data_quality_blocks_stale_critical_source():
    stale = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
    today = datetime.now(UTC).date().isoformat()
    raw = {
        "portfolio_positions": [{"ticker": "MU"}],
        "risk_data": {"timestamp": today},
        "sizer_summary": {"timestamp": today},
        "market_data": {
            "indices": {"timestamp": stale},
            "breadth": {"timestamp": today},
            "vix": {"timestamp": today},
            "liquidity": {"timestamp": today},
            "yield_curve": {"timestamp": today},
        },
    }

    quality = assess_report_data_quality(raw, "daily")

    assert quality["recommendations_blocked"] is True
    assert quality["critical_data_quality"] == "stale"


def test_persist_actionable_recommendation_creates_pending_approval(temp_core_db):
    rows = persist_recommendations(
        _valid_payload("buy"),
        source_report_path="/tmp/recommendations.md",
        source_json_path="/tmp/recommendations.json",
        prompt_metadata={"model": "test", "prompt_hash": "p", "input_hash": "i", "validation_status": "ok"},
    )

    assert len(rows) == 1
    assert rows[0]["approval_status"] == "pending"
    approvals = core_db.get_pending_approvals(status="pending")
    assert len(approvals) == 1
    assert approvals[0]["proposed_change"]["recommendation_id"] == rows[0]["id"]

    core_db.resolve_approval(approvals[0]["id"], "approved")
    updated = core_db.get_recommendation(rows[0]["id"])
    assert updated["approval_status"] == "approved"
    resolved = core_db.get_pending_approval(approvals[0]["id"])
    assert resolved["status"] == "approved"
    assert resolved["application_status"] == "applied"
    assert core_db.get_action_items(status="open")[0]["source_id"] == "daily:2026-05-02"


def test_recommendation_approval_failure_keeps_state_pending_and_retryable(temp_core_db, monkeypatch):
    rows = persist_recommendations(
        _valid_payload("buy"),
        source_report_path="/tmp/recommendations.md",
        source_json_path="/tmp/recommendations.json",
        prompt_metadata={"model": "test", "prompt_hash": "p", "input_hash": "i", "validation_status": "ok"},
    )
    approval = core_db.get_pending_approvals(status="pending")[0]
    original = core_db._APPROVAL_SIDE_EFFECT_HANDLERS["action_item"]

    def fail_after_insert(conn, current, change, callbacks):
        original(conn, current, change, callbacks)
        raise RuntimeError("recommendation apply failed")

    monkeypatch.setitem(core_db._APPROVAL_SIDE_EFFECT_HANDLERS, "action_item", fail_after_insert)
    with pytest.raises(core_db.ApprovalApplicationError, match="recommendation apply failed"):
        core_db.resolve_approval(approval["id"], "approved")

    failed_approval = core_db.get_pending_approval(approval["id"])
    failed_recommendation = core_db.get_recommendation(rows[0]["id"])
    assert failed_approval["status"] == "pending"
    assert failed_approval["application_status"] == "failed"
    assert failed_recommendation["approval_status"] == "pending"
    assert core_db.get_action_items(status="open") == []

    monkeypatch.setitem(core_db._APPROVAL_SIDE_EFFECT_HANDLERS, "action_item", original)
    core_db.resolve_approval(approval["id"], "approved")

    updated_recommendation = core_db.get_recommendation(rows[0]["id"])
    assert updated_recommendation["approval_status"] == "approved"
    assert len(core_db.get_action_items(status="open")) == 1


def test_persist_do_nothing_recommendation_does_not_create_approval(temp_core_db):
    rows = persist_recommendations(
        _valid_payload("do_nothing"),
        source_report_path="/tmp/recommendations.md",
        source_json_path="/tmp/recommendations.json",
    )

    assert len(rows) == 1
    assert rows[0]["approval_status"] == "none"
    assert core_db.get_pending_approvals(status="pending") == []
