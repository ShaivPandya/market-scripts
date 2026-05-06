from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.async_job_runner import enqueue_registered_job, enqueue_response
from ontology.object_service import OntologyObjectService
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()


class OptimizationRunRequest(BaseModel):
    source: str = "manual"
    force: bool = False


class DismissOptimizationAlertRequest(BaseModel):
    note: str | None = None


@router.get("/optimization/missions")
def list_optimization_missions(status: str | None = None):
    missions = OntologyRuntimeReadService().list_objects(
        "OptimizationMission",
        filters={"status": status} if status else None,
        limit=100,
    )
    return {"missions": missions, "count": len(missions)}


@router.post("/optimization/missions/{mission_id}/run")
def run_optimization_mission(mission_id: str, req: OptimizationRunRequest | None = None):
    mission = _get_optimization_object("OptimizationMission", mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Unknown optimization mission")
    body = req or OptimizationRunRequest()
    payload: dict[str, Any] = {
        "mission_id": mission.get("id") or mission_id,
        "source": body.source or "manual",
        "force": body.force,
    }
    row, _disposition = enqueue_registered_job(
        "continuous_optimizer",
        payload,
        cache_key=None,
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/v1/admin/jobs/{job_id}")


@router.get("/optimization/runs")
def list_optimization_runs(mission_id: str | None = None, limit: int = 20):
    filters = {"mission_id": str(mission_id)} if mission_id is not None else None
    runs = OntologyRuntimeReadService().list_objects("OptimizationRun", filters=filters, limit=limit)
    return {"runs": runs, "count": len(runs)}


@router.get("/optimization/runs/{run_id}")
def get_optimization_run_detail(run_id: str):
    run = _get_optimization_object("OptimizationRun", run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Unknown optimization run")
    return run


@router.get("/optimization/alerts")
def list_optimization_alerts(status: str | None = "open", mission_id: str | None = None, limit: int = 50):
    filters: dict[str, str] = {}
    if status:
        filters["status"] = status
    if mission_id is not None:
        filters["mission_id"] = str(mission_id)
    alerts = OntologyRuntimeReadService().list_objects("OptimizationAlert", filters=filters, limit=limit)
    return {"alerts": alerts, "count": len(alerts)}


@router.put("/optimization/alerts/{alert_id}/dismiss")
def dismiss_optimization_alert(alert_id: str, req: DismissOptimizationAlertRequest | None = None):
    alert = _get_optimization_object("OptimizationAlert", alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Unknown optimization alert")
    props = {**alert, "status": "dismissed", "dismissal_note": req.note if req else None}
    props.pop("_meta", None)
    props.pop("id", None)
    props.pop("object_uid", None)
    row = OntologyObjectService().write_object(
        "OptimizationAlert",
        alert.get("object_uid") or alert.get("id") or alert_id,
        props,
        props.get("created_at") or props.get("as_of") or datetime.now(UTC).isoformat(),
        provenance=f"pv:optimization_alert:{alert_id}:dismiss",
    )
    return dict(row.get("properties") or props)


def _get_optimization_object(object_type: str, object_id: str) -> dict[str, Any] | None:
    reads = OntologyRuntimeReadService()
    return reads.get(object_id) or reads.get(f"{object_type.lower()}:{object_id}")
