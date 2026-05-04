from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

import portfolio.core_db as core_db
import portfolio.thesis_db as thesis_db


@pytest.fixture
def temp_investing_dbs(tmp_path, monkeypatch):
    if core_db._conn:
        core_db._conn.close()
    if thesis_db._conn:
        thesis_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    monkeypatch.setattr(thesis_db, "DB_PATH", tmp_path / "thesis.db")
    monkeypatch.setattr(thesis_db, "_conn", None)
    yield
    if core_db._conn:
        core_db._conn.close()
    if thesis_db._conn:
        thesis_db._conn.close()
    monkeypatch.setattr(core_db, "_conn", None)
    monkeypatch.setattr(thesis_db, "_conn", None)


def _recommendations_payload(action: str = "buy") -> dict:
    return {
        "report_type": "daily",
        "as_of": "2026-05-02",
        "stance": "Neutral / Watchful",
        "recommendation_status": "clear",
        "critical_data_quality": "ok",
        "blocked_reasons": [],
        "do_nothing_rationale": "No forced trade.",
        "what_changed": ["Breadth improved."],
        "recommended_actions": [
            {
                "action": action,
                "ticker": "MU" if action == "buy" else None,
                "instrument": "MU" if action == "buy" else "portfolio",
                "horizon": "1 trading day",
                "target_change": "start one-third size" if action == "buy" else "none",
                "rationale": "Validated setup." if action == "buy" else "No fat pitch.",
                "evidence": ["price action confirms"],
                "disconfirming_evidence": [],
                "catalyst": "earnings",
                "invalidation": "breaks support",
                "expected_onset_window": "1 week",
                "confidence": 0.7,
                "source_quality": "ok",
                "approval_required": action == "buy",
            }
        ],
        "alternatives": [],
        "opportunity_cost": [],
    }


def test_report_sync_requires_secret(client, monkeypatch):
    monkeypatch.setenv("REPORT_SYNC_SECRET", "sync-secret")

    resp = client.post("/api/v1/report-sync/daily", json={})

    assert resp.status_code == 401


def test_report_sync_rejects_malformed_payload(client, monkeypatch, temp_investing_dbs):
    monkeypatch.setenv("REPORT_SYNC_SECRET", "sync-secret")

    resp = client.post(
        "/api/v1/report-sync/daily",
        headers={"X-Report-Sync-Secret": "sync-secret"},
        json={"summary": {}},
    )

    assert resp.status_code == 422


def test_daily_report_sync_is_idempotent_and_visible(auth_client, monkeypatch, temp_investing_dbs):
    monkeypatch.setenv("REPORT_SYNC_SECRET", "sync-secret")
    payload = {
        "as_of": "2026-05-02",
        "report_md": "# Daily",
        "summary": {"positions_flagged": ["MU"], "watchlist_triggers": ["VIX > 30"], "data_quality": {}},
        "recommendations": _recommendations_payload("buy"),
        "bundle": {"inputs": "same"},
        "metadata": {"github_run_id": "123", "source_url": "https://github.com/o/r/actions/runs/123"},
    }

    for _ in range(2):
        resp = auth_client.post(
            "/api/v1/report-sync/daily",
            headers={"X-Report-Sync-Secret": "sync-secret"},
            json=payload,
        )
        assert resp.status_code == 200

    approvals = core_db.get_pending_approvals(status="pending")
    report_runs = core_db.get_report_runs(report_type="daily")

    assert core_db.get_recommendations(report_type="daily") == []
    assert core_db.get_action_items(status="open") == []
    assert len([a for a in approvals if a["entity_type"] == "recommendation"]) == 1
    assert len([a for a in approvals if a["entity_type"] == "action_item"]) == 1
    assert len([a for a in approvals if a["entity_type"] == "watch_trigger"]) == 1
    assert [a for a in approvals if a["entity_type"] == "watch_trigger"][0]["action_id"] == "create_watch_trigger"
    assert len(report_runs) == 1

    recommendation_approval = [a for a in approvals if a["entity_type"] == "recommendation"][0]
    core_db.resolve_approval(recommendation_approval["id"], "approved", "Apply recommendation")
    recommendations = core_db.get_recommendations(report_type="daily")
    assert len(recommendations) == 1

    latest = auth_client.get("/api/v1/recommendations/latest")
    assert latest.status_code == 200
    assert latest.json()["daily"]["id"] == recommendations[0]["id"]


def test_weekly_report_sync_persists_operating_artifacts(auth_client, monkeypatch, temp_investing_dbs):
    monkeypatch.setenv("REPORT_SYNC_SECRET", "sync-secret")
    weekly_recs = _recommendations_payload("do_nothing")
    weekly_recs["report_type"] = "weekly"
    weekly_recs["as_of"] = "2026-05-02"
    payload = {
        "as_of": "2026-05-02",
        "report_md": "# Weekly",
        "summary": {
            "watchlist_triggers": ["Liquidity score > 70"],
            "thesis_monitoring": {
                "thesis_evaluations": [
                    {
                        "ticker": "CRWD",
                        "thesis_status": "weaken",
                        "technical_read": "deteriorating",
                        "fundamental_read": "mixed",
                        "action": "reassess",
                        "confidence": "medium",
                        "key_developments": ["Margins under pressure"],
                        "earnings_note": None,
                        "risk_flag": "margin compression",
                    }
                ],
                "positions_reviewed": ["CRWD"],
                "positions_needing_reassessment": ["CRWD"],
                "material_developments": [
                    {"ticker": "CRWD", "type": "new_risk", "summary": "Margin pressure emerged."}
                ],
            },
            "data_quality": {},
        },
        "recommendations": weekly_recs,
        "bundle": {},
        "thesis_claims": [
            {
                "ticker": "CRWD",
                "claim": "CRWD can sustain premium growth.",
                "expected_evidence": "ARR growth remains elevated.",
                "disconfirming_evidence": "ARR growth decelerates sharply.",
                "source_requirements": ["earnings"],
                "cadence": "weekly",
                "confidence": 0.6,
                "linked_catalyst_ids": [],
                "linked_kill_condition_ids": [],
            }
        ],
    }

    resp = auth_client.post(
        "/api/v1/report-sync/weekly",
        headers={"X-Report-Sync-Secret": "sync-secret"},
        json=payload,
    )

    assert resp.status_code == 200
    assert thesis_db.get_evaluations("CRWD", limit=1) == []
    assert core_db.get_research_notes(ticker="CRWD") == []
    assert core_db.get_action_items(ticker="CRWD", status="open") == []
    approvals = core_db.get_pending_approvals(status="pending")
    assert {approval["entity_type"] for approval in approvals} >= {
        "evaluation",
        "research_note",
        "action_item",
        "watch_trigger",
        "thesis_claim",
        "recommendation",
    }

    claim_approval = [approval for approval in approvals if approval["entity_type"] == "thesis_claim"][0]
    core_db.resolve_approval(claim_approval["id"], "approved", "Apply claim")
    claim = core_db.get_thesis_claims(ticker="CRWD")[0]
    assert claim["claim"] == "CRWD can sustain premium growth."


def test_watch_trigger_monitor_fires_price_technical_and_macro(monkeypatch, temp_investing_dbs):
    from api import watch_trigger_monitor

    monkeypatch.setattr(watch_trigger_monitor, "_latest_price", lambda _ticker: {"value": 151.0, "as_of": "2026-05-02"})
    monkeypatch.setattr(
        "portfolio.technical_analysis.technical_analysis.get_data",
        lambda *_args, **_kwargs: {
            "summary": [{"Indicator": "Price vs 200D SMA", "Signal": "Above", "Bias": "Bullish"}],
            "timestamp": "2026-05-02",
        },
    )
    monkeypatch.setattr(
        "api.signal_snapshot.get_signal_aggregator_snapshot_or_module_response",
        lambda **_kwargs: {"regime": {"score": 72}, "_meta": {"snapshot": {"as_of": "2026-05-02"}}},
    )

    core_db.create_watch_trigger(
        "MU >= 150",
        "price_level",
        ticker="MU",
        source_type="workflow",
        definition={"type": "price_level", "ticker": "MU", "operator": ">=", "threshold": 150},
    )
    core_db.create_watch_trigger(
        "MU above 200D",
        "technical",
        ticker="MU",
        source_type="workflow",
        definition={
            "type": "technical",
            "ticker": "MU",
            "indicator_contains": "200D",
            "field": "Signal",
            "expected": "Above",
        },
    )
    core_db.create_watch_trigger(
        "Regime score high",
        "macro",
        source_type="workflow",
        definition={"type": "macro", "field": "regime.score", "operator": ">=", "threshold": 70},
    )

    result = watch_trigger_monitor.run_watch_trigger_monitor()

    assert result["fired"] == 3
    assert len(core_db.get_watch_triggers(status="fired")) == 0
    approvals = core_db.get_pending_approvals(status="pending")
    assert len([approval for approval in approvals if approval["action_id"] == "fire_watch_trigger"]) == 3
    assert len([approval for approval in approvals if approval["action_id"] == "create_action_item"]) == 3


def test_recommendation_postmortem_stores_process_attribution(monkeypatch, temp_investing_dbs):
    from auto_report import recommendations as recs

    as_of = (datetime.now(UTC).date() - timedelta(days=10)).isoformat()
    created = core_db.create_recommendation(
        {
            "report_type": "daily",
            "as_of": as_of,
            "stance": "Neutral / Watchful",
            "recommendation_status": "clear",
            "critical_data_quality": "ok",
            "action": "buy",
            "ticker": "MU",
            "instrument": "MU",
            "horizon": "1 trading day",
            "target_change": "start one-third size",
            "rationale": "Validated setup.",
            "confidence": 0.8,
            "source_quality": "ok",
            "evidence": ["breakout"],
            "disconfirming_evidence": [],
            "catalyst": "earnings",
            "expected_onset_window": "1 week",
        }
    )

    def fake_close(ticker, *_args):
        if ticker == "SPY":
            return pd.Series([100.0, 101.0])
        return pd.Series([100.0, 110.0])

    monkeypatch.setattr(recs, "_download_close_series", fake_close)

    result = recs.evaluate_due_recommendations()

    assert result["updated"] == 1
    updated = core_db.get_recommendation(created["id"])
    outcome = updated["outcome_json"]
    assert outcome["benchmark_relative_return_pct"] == 9.0
    assert outcome["max_favorable_move_pct"] == 10.0
    assert outcome["process_label"] == "good_process_good_outcome"


def test_github_actions_keep_artifacts_and_sync_steps():
    daily = open(".github/workflows/daily_report.yml", encoding="utf-8").read()
    weekly = open(".github/workflows/weekly_report.yml", encoding="utf-8").read()

    for workflow, report_type in ((daily, "daily"), (weekly, "weekly")):
        assert "actions/upload-artifact" in workflow
        assert "Sync report state to app" in workflow
        assert f"python -m auto_report.sync_report_state {report_type}" in workflow
        assert "REPORT_SYNC_SECRET" in workflow
