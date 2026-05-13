from __future__ import annotations


def test_workflow_run_writes_refresh_operational_read_model(monkeypatch):
    import ontology.domain_write_service as domain_write_service
    import ontology.read_model as read_model
    from api import workflows

    refresh_calls: list[str] = []
    store: dict[str, dict] = {}

    class _Repo:
        def refresh(self):
            refresh_calls.append("refresh")

    class _Objects:
        def write_object(self, _object_type, business_key, props, *_args, **_kwargs):
            store[str(business_key)] = dict(props)
            return {"properties": dict(props)}

        def get_object(self, object_uid):
            run_id = str(object_uid).removeprefix("workflow_run:")
            props = store.get(str(object_uid)) or store.get(run_id)
            return {"properties": dict(props)} if props else None

    monkeypatch.setattr(read_model, "TemporalReadModelRepository", _Repo)
    monkeypatch.setattr(workflows, "OntologyObjectService", _Objects)

    run = workflows.create_workflow_run("thesis_review", "NVDA")
    completed = workflows.complete_workflow_run(str(run["run_id"]), "synthesis")

    assert completed["status"] == "succeeded"
    assert completed["ticker"] == "NVDA"
    assert completed["updated_at"]
    assert refresh_calls == ["refresh", "refresh"]


def test_runtime_bundles_merge_fresh_workflow_runs_when_read_model_is_stale(monkeypatch):
    import ontology.runtime_read_service as runtime_read_service
    from ontology.runtime_read_service import OntologyRuntimeReadService

    stale_run = {
        "object_type": "WorkflowRun",
        "object_uid": "workflow_run:workflow:thesis_review:old",
        "properties_json": {
            "run_id": "workflow:thesis_review:old",
            "workflow_name": "thesis_review",
            "ticker": "NVDA",
            "status": "succeeded",
            "updated_at": "2026-05-09T10:00:00+00:00",
        },
    }
    fresh_run = {
        "object_type": "WorkflowRun",
        "object_uid": "workflow_run:workflow:thesis_review:new",
        "properties_json": {
            "run_id": "workflow:thesis_review:new",
            "workflow_name": "thesis_review",
            "ticker": "NVDA",
            "status": "succeeded",
            "updated_at": "2026-05-13T14:00:00+00:00",
            "started_at": "2026-05-13T13:59:00+00:00",
        },
        "tx_from": "2026-05-13T14:00:00+00:00",
    }

    class _Repo:
        def fetch_workspace_bundle(self):
            return {
                "latest_evaluations": [],
                "theses": [],
                "pending_approvals": [],
                "latest_daily_recommendation": None,
                "latest_weekly_recommendation": None,
                "pending_actionable_recommendations": [],
                "open_action_items": [],
                "optimizer_alerts": [],
                "active_watch_triggers": [],
                "recent_workflow_runs": [stale_run],
                "recent_report_runs": [],
                "challenged_claims": [],
                "disconfirmed_claims": [],
            }

        def fetch_dossier_bundle(self, _ticker):
            return {
                "position": None,
                "thesis_meta": None,
                "management_quality_assessment": None,
                "evaluations": [],
                "catalysts": [],
                "kill_conditions": [],
                "thesis_claims": [],
                "workflow_runs": [stale_run],
                "action_items": [],
                "watch_triggers": [],
                "pending_approvals": [],
            }

    class _Objects:
        def query_objects(self, object_type, filters=None, **_kwargs):
            if object_type != "WorkflowRun":
                return []
            ticker = (filters or {}).get("ticker")
            if ticker and ticker != "NVDA":
                return []
            return [fresh_run]

    service = OntologyRuntimeReadService(object_service=_Objects(), read_model_repository=_Repo())

    workspace = service.workspace_bundle()
    dossier = service.dossier_bundle("NVDA")

    assert [run["run_id"] for run in workspace["recent_workflow_runs"]] == [
        "workflow:thesis_review:new",
        "workflow:thesis_review:old",
    ]
    assert [run["run_id"] for run in dossier["workflow_runs"]] == [
        "workflow:thesis_review:new",
        "workflow:thesis_review:old",
    ]
