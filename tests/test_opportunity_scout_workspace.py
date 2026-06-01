from __future__ import annotations

import pytest

import api.routers.workspace as workspace_router


def test_workspace_includes_ranked_opportunity_candidates(monkeypatch):
    class _Reads:
        def workspace_bundle(self):
            return {
                "latest_evaluations": [],
                "theses": [],
                "pending_approvals": [],
                "latest_daily_recommendation": None,
                "latest_weekly_recommendation": None,
                "pending_actionable_recommendations": [],
                "open_action_items": [],
                "optimizer_alerts": [],
                "active_monitor_definitions": [],
                "active_mission_definitions": [],
                "active_watch_triggers": [],
                "recent_monitor_hits": [],
                "open_opportunity_candidates": [
                    {
                        "candidate_id": "candidate:mu:monitor",
                        "object_uid": "opportunity_candidate:candidate-mu-monitor",
                        "ticker": "MU",
                        "source_kind": "monitor_hit",
                        "trigger": "Kill condition monitor hit",
                        "opportunity_type": "unsustainable_process",
                        "consensus": "Automated",
                        "variant_view": "Threshold breach",
                        "why_now": "High-severity monitor signal",
                        "price_confirmation": "Not verified",
                        "missing_inputs": ["decision_quality pressure-test"],
                        "next_action": "research",
                        "summary": "Review",
                        "status": "open",
                        "decision_state": "generated",
                        "severity": "high",
                        "rank_signals": {"severity": "high", "hit_type": "triggered"},
                        "opportunity_candidate_gate": {
                            "status": "pass",
                            "final_action": "research",
                            "should_graduate": False,
                        },
                        "updated_at": "2026-06-01T12:00:00Z",
                    }
                ],
                "recent_workflow_runs": [],
                "recent_report_runs": [],
                "challenged_claims": [],
                "disconfirmed_claims": [],
                "pending_draft_decision_outcomes": [],
                "recent_finalized_decision_outcomes": [],
            }

    monkeypatch.setattr(workspace_router, "OntologyRuntimeReadService", _Reads)
    monkeypatch.setattr(workspace_router, "get_setting", lambda key: None)
    monkeypatch.setattr("api.agent_tools.execute_tool", lambda name, args: {"positions": []})
    monkeypatch.setattr(
        "api.signal_snapshot.get_signal_aggregator_snapshot_or_module_response",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        workspace_router,
        "build_workspace_source_health",
        lambda **kwargs: None,
    )

    payload = workspace_router.get_workspace()
    assert payload["opportunity_candidates"]["count"] == 1
    item = payload["opportunity_candidates"]["items"][0]
    assert item["candidate_id"] == "candidate:mu:monitor"
    assert item["trigger"]
    assert item["why_now"]
    assert item["missing_inputs"]
    assert item["next_action"] == "research"
    assert item["rank_score"] > 0


def test_dismiss_opportunity_candidate_stages_status_proposal(monkeypatch):
    class _Reads:
        def opportunity_candidates(self, *, ticker=None, status=None, limit=50):
            return [
                {
                    "candidate_id": "candidate:mu:monitor",
                    "object_uid": "opportunity_candidate:candidate-mu-monitor",
                    "trigger": "Monitor hit",
                    "status": "open",
                }
            ]

    monkeypatch.setattr(workspace_router, "OntologyRuntimeReadService", _Reads)
    monkeypatch.setattr(
        workspace_router,
        "stage_api_action",
        lambda action_id, payload, **kwargs: {"id": "approval:1", "action_id": action_id, **payload},
    )

    result = workspace_router.dismiss_opportunity_candidate(
        workspace_router.OpportunityCandidateFeedbackRequest(candidate_id="candidate:mu:monitor")
    )
    assert result["status"] == "proposal_created"
    assert result["approval"]["action_id"] == "update_opportunity_candidate_status"
    assert result["approval"]["status"] == "dismissed"
