from __future__ import annotations

import json


def test_workspace_router_uses_runtime_bundle(monkeypatch):
    import api.agent_tools as agent_tools
    import api.routers.workspace as workspace_router

    class _Reads:
        def workspace_bundle(self):
            return {
                "latest_evaluations": [
                    {
                        "ticker": "MU",
                        "action": "trim",
                        "confidence": 0.8,
                        "risk_flag": None,
                        "evaluated_at": "2026-05-05",
                    },
                    {
                        "ticker": "CRWD",
                        "action": "trim",
                        "confidence": 0.7,
                        "risk_flag": "No longer owned",
                        "evaluated_at": "2026-05-05",
                    },
                ],
                "theses": [{"ticker": "MU", "status": "active"}, {"ticker": "CRWD", "status": "active"}],
                "pending_approvals": [{"id": 7, "status": "pending", "proposed_change": {"recommendation_id": 3}}],
                "latest_daily_recommendation": {
                    "id": 3,
                    "report_type": "daily",
                    "as_of": "2026-05-05",
                    "recommendation_status": "blocked",
                    "critical_data_quality": "failed",
                    "blocked_reasons_json": ["liquidity stale"],
                },
                "latest_weekly_recommendation": None,
                "pending_actionable_recommendations": [{"id": 4, "approval_status": "pending", "action": "buy"}],
                "open_action_items": [{"id": action_id, "status": "open", "ticker": "MU"} for action_id in range(1, 8)],
                "optimizer_alerts": [{"id": 6, "status": "open", "ticker": "MU"}],
                "active_watch_triggers": [
                    {"id": trigger_id, "status": "active", "ticker": "MU"} for trigger_id in range(1, 8)
                ],
                "recent_workflow_runs": [{"id": "workflow_run:1"}],
                "recent_report_runs": [{"id": "report_run:1"}],
                "challenged_claims": [{"id": 9, "status": "challenged"}],
                "disconfirmed_claims": [{"id": 10, "status": "disconfirmed"}],
            }

    monkeypatch.setattr(workspace_router, "OntologyRuntimeReadService", _Reads)
    monkeypatch.setattr(workspace_router, "get_setting", lambda key: None)
    monkeypatch.setattr(agent_tools, "execute_tool", lambda name, args: {"positions": [{"ticker": "MU"}]})
    monkeypatch.setattr(
        "api.signal_snapshot.get_signal_aggregator_snapshot_or_module_response",
        lambda **kwargs: {"regime": {"label": "neutral", "score": 0}},
    )
    monkeypatch.setattr(
        workspace_router,
        "build_workspace_source_health",
        lambda **kwargs: {
            "generated_at": "2026-05-14T00:00:00",
            "overall_quality": "ok",
            "counts": {"total": 1, "ok": 1},
            "domains": [{"domain": "market", "label": "Market", "sources": []}],
        },
    )

    payload = workspace_router.get_workspace()

    assert payload["source_health"]["overall_quality"] == "ok"
    assert payload["what_changed"]["counts"]["total"] == 0
    assert [row["ticker"] for row in payload["thesis_pressure"]] == ["MU"]
    assert payload["thesis_pressure"][0]["pressure_key"].startswith("MU:")
    assert payload["pending_approvals"]["count"] == 1
    assert payload["recommendations"]["latest_daily"]["id"] == 3
    assert payload["recommendations"]["pending_actionable"]["count"] == 1
    assert payload["open_actions"]["count"] == 7
    assert len(payload["open_actions"]["items"]) == 7
    assert payload["continuous_optimization"]["open_alert_count"] == 1
    assert payload["active_triggers"]["count"] == 7
    assert len(payload["active_triggers"]["items"]) == 7
    assert payload["thesis_claims"]["challenged_count"] == 2


def test_workspace_thesis_pressure_dismissal_filters_until_signal_changes(monkeypatch):
    import api.agent_tools as agent_tools
    import api.routers.workspace as workspace_router

    dismissed_row = {
        "ticker": "MU",
        "action": "trim",
        "confidence": "medium",
        "risk_flag": "liquidity pressure",
        "evaluated_at": "2026-05-05",
    }
    dismissed_key = workspace_router._pressure_key(dismissed_row)
    evaluation = dict(dismissed_row)

    class _Reads:
        def workspace_bundle(self):
            return {
                "latest_evaluations": [dict(evaluation)],
                "theses": [{"ticker": "MU", "status": "active"}],
                "pending_approvals": [],
                "latest_daily_recommendation": None,
                "latest_weekly_recommendation": None,
                "pending_actionable_recommendations": [],
                "open_action_items": [],
                "optimizer_alerts": [],
                "active_watch_triggers": [],
                "recent_workflow_runs": [],
                "recent_report_runs": [],
                "challenged_claims": [],
                "disconfirmed_claims": [],
            }

    monkeypatch.setattr(workspace_router, "OntologyRuntimeReadService", _Reads)
    monkeypatch.setattr(
        workspace_router,
        "get_setting",
        lambda key: {
            "key": key,
            "value": json.dumps(
                {
                    dismissed_key: {
                        "ticker": "MU",
                        "pressure_key": dismissed_key,
                        "dismissed_at": "2026-05-06T00:00:00Z",
                    }
                }
            ),
            "updated_at": "2026-05-06T00:00:00Z",
        },
    )
    monkeypatch.setattr(agent_tools, "execute_tool", lambda name, args: {"positions": [{"ticker": "MU"}]})
    monkeypatch.setattr(
        "api.signal_snapshot.get_signal_aggregator_snapshot_or_module_response",
        lambda **kwargs: {"regime": {"label": "neutral", "score": 0}},
    )
    monkeypatch.setattr(workspace_router, "build_workspace_source_health", lambda **kwargs: None)

    payload = workspace_router.get_workspace()
    assert payload["thesis_pressure"] == []

    evaluation["evaluated_at"] = "2026-05-07"
    payload = workspace_router.get_workspace()

    assert [row["ticker"] for row in payload["thesis_pressure"]] == ["MU"]
    assert payload["thesis_pressure"][0]["pressure_key"] != dismissed_key


def test_workspace_thesis_pressure_dismiss_endpoint_persists_ack(monkeypatch):
    import api.routers.workspace as workspace_router

    saved: dict[str, str] = {}
    monkeypatch.setattr(workspace_router, "get_setting", lambda key: None)
    monkeypatch.setattr(workspace_router, "set_setting", lambda key, value: saved.update({key: value}))

    response = workspace_router.dismiss_thesis_pressure(
        workspace_router.DismissThesisPressureRequest(ticker="mu", pressure_key="MU:abc", note="done")
    )

    assert response == {"status": "dismissed", "ticker": "MU", "pressure_key": "MU:abc"}
    stored = json.loads(saved[workspace_router.THESIS_PRESSURE_DISMISSALS_KEY])
    assert stored["MU:abc"]["ticker"] == "MU"
    assert stored["MU:abc"]["note"] == "done"


def test_dossier_router_uses_bundle_without_position_scan(monkeypatch):
    import api.routers.dossier as dossier_router
    import api.state_storage as state_storage
    import portfolio.management_quality_content as management_quality_content

    class _Reads:
        def dossier_bundle(self, ticker: str):
            assert ticker == "MU"
            return {
                "position": {"ticker": "MU"},
                "thesis_meta": {"ticker": "MU", "status": "active"},
                "management_quality_assessment": None,
                "evaluations": [{"ticker": "MU"}],
                "catalysts": [{"ticker": "MU"}],
                "kill_conditions": [{"ticker": "MU"}],
                "thesis_claims": [{"ticker": "MU"}],
                "workflow_runs": [{"ticker": "MU"}],
                "action_items": [{"ticker": "MU", "status": "open"}],
                "watch_triggers": [{"ticker": "MU"}],
                "pending_approvals": [{"id": 11, "ticker": "MU", "status": "pending"}],
            }

        def positions(self):
            raise AssertionError("dossier route should not scan all positions")

    monkeypatch.setattr(dossier_router, "OntologyRuntimeReadService", _Reads)
    monkeypatch.setattr(state_storage, "exists_text", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(management_quality_content, "management_quality_exists", lambda *_args, **_kwargs: False)

    payload = dossier_router.get_dossier("mu")

    assert payload["ticker"] == "MU"
    assert payload["what_changed"]["counts"]["total"] == 0
    assert payload["position"]["ticker"] == "MU"
    assert payload["thesis"]["meta"]["ticker"] == "MU"
    assert payload["evaluations"][0]["ticker"] == "MU"
    assert payload["action_items"][0]["decision_state"] == "open"
    assert payload["pending_approvals"][0]["decision_state"] == "pending_approval"
