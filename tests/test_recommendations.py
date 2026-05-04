from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

import portfolio.core_db as core_db
from auto_report import auto_daily_report
from auto_report.recommendations import (
    ACTION_OPTIONS,
    MAX_RECOMMENDATIONS_COMMENTARY_CHARS,
    MAX_RECOMMENDATIONS_EVIDENCE_CHARS,
    MAX_RECOMMENDATIONS_EXTRA_CONTEXT_CHARS,
    RECOMMENDATIONS_SEPARATOR,
    RecommendationValidationError,
    assess_report_data_quality,
    build_recommendations_user_message,
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


def test_validate_recommendations_payload_normalizes_review_action_to_watch(data_quality_ok):
    payload = _valid_payload()
    payload["recommended_actions"][0]["action"] = "review"
    payload["recommended_actions"][0]["approval_required"] = True
    payload["recommended_actions"][0]["rationale"] = "Review the setup but do not trade."

    normalized = validate_recommendations_payload(
        payload,
        report_type="daily",
        as_of="2026-05-02",
        stance="Neutral / Watchful",
        data_quality=data_quality_ok,
    )

    action = normalized["recommended_actions"][0]
    assert action["action"] == "watch"
    assert action["approval_required"] is False
    assert action["rationale"] == "Review the setup but do not trade."


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


def test_build_recommendations_user_message_compacts_large_context(data_quality_ok):
    evidence_bundle = {
        "market_data": {
            "indices": {
                "data": {
                    "indices": {
                        "SPX": [
                            {
                                "date": f"2026-05-{(idx % 28) + 1:02d}",
                                "value": idx,
                                "raw_payload": "x" * 10_000,
                            }
                            for idx in range(250)
                        ]
                    }
                }
            },
            "central_banks": {
                "items": [{"title": f"speech {idx}", "content_preview": "y" * 20_000} for idx in range(80)]
            },
        },
        "portfolio_positions": [{"ticker": f"T{idx}", "notes": "z" * 5_000} for idx in range(100)],
        "data_quality": data_quality_ok,
    }
    commentary_md = (
        "# Daily Commentary Report\n\n"
        + ("market context\n" * 20_000)
        + "\n## Sources\n- [source](https://example.com)"
    )
    extra_context_md = "## Deterministic Risk Tables\n\n" + ("risk table row\n" * 10_000)

    message = build_recommendations_user_message(
        report_type="daily",
        as_of="2026-05-02",
        stance="Neutral / Watchful",
        data_quality=data_quality_ok,
        evidence_bundle=evidence_bundle,
        commentary_md=commentary_md,
        extra_context_md=extra_context_md,
    )

    assert len(message) < (
        MAX_RECOMMENDATIONS_EVIDENCE_CHARS
        + MAX_RECOMMENDATIONS_COMMENTARY_CHARS
        + MAX_RECOMMENDATIONS_EXTRA_CONTEXT_CHARS
        + 20_000
    )
    assert "truncated" in message
    assert "## Sources" not in message
    assert "x" * 10_000 not in message


def test_build_recommendations_user_message_uses_only_legal_action_values(data_quality_ok):
    message = build_recommendations_user_message(
        report_type="daily",
        as_of="2026-05-02",
        stance="Neutral / Watchful",
        data_quality=data_quality_ok,
        evidence_bundle={},
        commentary_md="",
    )

    assert "or review" not in message
    assert "- If the expected onset window failed, recommend reduce, exit, or watch." in message
    assert f"- Use only these actions: {' | '.join(ACTION_OPTIONS)}." in message
    assert "`review` is not an action" in message


def test_daily_recommendation_generation_accepts_review_action_as_watch(data_quality_ok, monkeypatch):
    payload = _valid_payload()
    payload["recommended_actions"][0]["action"] = "review"
    payload["recommended_actions"][0]["approval_required"] = True
    raw_text = "Decision memo\n" + RECOMMENDATIONS_SEPARATOR + "\n" + json.dumps(payload)

    def fake_call_report_llm(**_kwargs):
        return raw_text, []

    def fail_repair(*_args, **_kwargs):
        raise AssertionError("review action should normalize before repair fallback")

    monkeypatch.setattr(auto_daily_report, "call_report_llm", fake_call_report_llm)
    monkeypatch.setattr(auto_daily_report, "repair_recommendations_response", fail_repair)

    recommendations_md, recommendations_payload = auto_daily_report._generate_daily_recommendations(
        today_str="2026-05-02",
        stance_dict={"stance": "Neutral / Watchful"},
        data_quality=data_quality_ok,
        evidence_bundle={},
        commentary_md="",
        risk_summary_md="",
        adjustments_md="",
    )

    assert recommendations_payload["recommendation_status"] == "clear"
    assert recommendations_payload["recommended_actions"][0]["action"] == "watch"
    assert recommendations_payload["recommended_actions"][0]["approval_required"] is False
    assert "Status: **error**" not in recommendations_md


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
    assert rows[0]["status"] == "pending_approval_created"
    approvals = core_db.get_pending_approvals(status="pending")
    assert len(approvals) == 1
    assert approvals[0]["action_id"] == "create_recommendation"

    core_db.resolve_approval(approvals[0]["id"], "approved", "Apply recommendation")
    updated = core_db.get_recommendations(report_type="daily")[0]
    assert updated["approval_status"] == "pending"
    resolved = core_db.get_pending_approval(approvals[0]["id"])
    assert resolved["status"] == "approved"
    assert resolved["application_status"] == "applied"
    action_approval = [
        approval
        for approval in core_db.get_pending_approvals(status="pending")
        if approval["entity_type"] == "action_item"
    ][0]
    assert action_approval["proposed_change"]["recommendation_id"] == updated["id"]
    core_db.resolve_approval(action_approval["id"], "approved", "Apply action item")
    assert core_db.get_action_items(status="open")[0]["source_id"] == "daily:2026-05-02"


def test_persist_recommendations_supersedes_removed_report_actions(temp_core_db):
    payload = _valid_payload("buy")
    payload["recommended_actions"].append(
        {
            **payload["recommended_actions"][0],
            "ticker": "AAPL",
            "instrument": "AAPL",
            "target_change": "start pilot size",
            "rationale": "Second validated setup.",
        }
    )
    metadata = {
        "report_id": "daily-report-2026-05-02",
        "model": "test",
        "prompt_hash": "p",
        "input_hash": "i",
        "validation_status": "ok",
    }
    persist_recommendations(
        payload,
        source_report_path="/tmp/recommendations.md",
        source_json_path="/tmp/recommendations.json",
        prompt_metadata=metadata,
    )
    for approval in [
        approval
        for approval in core_db.get_pending_approvals(status="pending")
        if approval["action_id"] == "create_recommendation"
    ]:
        core_db.resolve_approval(approval["id"], "approved", "Apply recommendation")

    assert {row["ticker"] for row in core_db.get_recommendations(report_type="daily", status="open")} == {
        "AAPL",
        "MU",
    }

    rerun_payload = json.loads(json.dumps(payload))
    rerun_payload["recommended_actions"] = [rerun_payload["recommended_actions"][0]]
    persist_recommendations(
        rerun_payload,
        source_report_path="/tmp/recommendations.md",
        source_json_path="/tmp/recommendations.json",
        prompt_metadata=metadata,
    )

    assert {row["ticker"] for row in core_db.get_recommendations(report_type="daily", status="open")} == {"MU"}
    superseded = core_db.get_recommendations(report_type="daily", status="superseded")
    assert [row["ticker"] for row in superseded] == ["AAPL"]


def test_recommendation_approval_failure_keeps_state_pending_and_retryable(temp_core_db, monkeypatch):
    persist_recommendations(
        _valid_payload("buy"),
        source_report_path="/tmp/recommendations.md",
        source_json_path="/tmp/recommendations.json",
        prompt_metadata={"model": "test", "prompt_hash": "p", "input_hash": "i", "validation_status": "ok"},
    )
    recommendation_approval = core_db.get_pending_approvals(status="pending")[0]
    core_db.resolve_approval(recommendation_approval["id"], "approved", "Apply recommendation")
    approval = [
        approval
        for approval in core_db.get_pending_approvals(status="pending")
        if approval["entity_type"] == "action_item"
    ][0]
    original_create_action_item = core_db.create_action_item

    def fail_create_action_item(*args, **kwargs):
        raise RuntimeError("recommendation apply failed")

    monkeypatch.setattr(core_db, "create_action_item", fail_create_action_item)
    with pytest.raises(core_db.ApprovalApplicationError, match="recommendation apply failed"):
        core_db.resolve_approval(approval["id"], "approved", "Apply action item")

    failed_approval = core_db.get_pending_approval(approval["id"])
    failed_recommendation = core_db.get_recommendations(report_type="daily")[0]
    assert failed_approval["status"] == "pending"
    assert failed_approval["application_status"] == "failed"
    assert failed_recommendation["approval_status"] == "pending"
    assert core_db.get_action_items(status="open") == []

    monkeypatch.setattr(core_db, "create_action_item", original_create_action_item)
    core_db.resolve_approval(approval["id"], "approved", "Apply action item")

    updated_recommendation = core_db.get_recommendations(report_type="daily")[0]
    assert updated_recommendation["approval_status"] == "approved"
    assert len(core_db.get_action_items(status="open")) == 1


def test_persist_do_nothing_recommendation_does_not_create_approval(temp_core_db):
    rows = persist_recommendations(
        _valid_payload("do_nothing"),
        source_report_path="/tmp/recommendations.md",
        source_json_path="/tmp/recommendations.json",
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "pending_approval_created"
    approvals = core_db.get_pending_approvals(status="pending")
    assert len(approvals) == 1
    core_db.resolve_approval(approvals[0]["id"], "approved", "Apply recommendation")
    assert core_db.get_recommendations(report_type="daily")[0]["approval_status"] == "none"
    assert core_db.get_pending_approvals(status="pending") == []
