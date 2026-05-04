from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.async_job_runner import enqueue_registered_job, enqueue_response

router = APIRouter()


class OptimizationRunRequest(BaseModel):
    source: str = "manual"
    force: bool = False


class DismissOptimizationAlertRequest(BaseModel):
    note: str | None = None


@router.get("/optimization/missions")
def list_optimization_missions(status: str | None = None):
    from portfolio.core_db import get_optimization_missions

    missions = get_optimization_missions(status=status)
    return {"missions": missions, "count": len(missions)}


@router.post("/optimization/missions/{mission_id}/run")
def run_optimization_mission(mission_id: int, req: OptimizationRunRequest | None = None):
    from portfolio.core_db import get_optimization_mission

    if not get_optimization_mission(mission_id):
        raise HTTPException(status_code=404, detail="Unknown optimization mission")
    body = req or OptimizationRunRequest()
    payload: dict[str, Any] = {"mission_id": mission_id, "source": body.source or "manual", "force": body.force}
    row, _disposition = enqueue_registered_job(
        "continuous_optimizer",
        payload,
        cache_key=None,
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/v1/admin/jobs/{job_id}")


@router.get("/optimization/runs")
def list_optimization_runs(mission_id: int | None = None, limit: int = 20):
    from portfolio.core_db import get_optimization_runs

    runs = get_optimization_runs(mission_id=mission_id, limit=limit)
    return {"runs": runs, "count": len(runs)}


@router.get("/optimization/runs/{run_id}")
def get_optimization_run_detail(run_id: str):
    from portfolio.core_db import get_optimization_run

    run = get_optimization_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Unknown optimization run")
    return run


@router.get("/optimization/alerts")
def list_optimization_alerts(status: str | None = "open", mission_id: int | None = None, limit: int = 50):
    from portfolio.core_db import get_optimization_alerts

    alerts = get_optimization_alerts(status=status, mission_id=mission_id, limit=limit)
    return {"alerts": alerts, "count": len(alerts)}


@router.put("/optimization/alerts/{alert_id}/dismiss")
def dismiss_optimization_alert(alert_id: int, req: DismissOptimizationAlertRequest | None = None):
    from portfolio.core_db import dismiss_optimization_alert

    try:
        return dismiss_optimization_alert(alert_id, note=(req.note if req else None))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
