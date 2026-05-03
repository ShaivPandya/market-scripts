"""Workflow run list/detail API endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api.exceptions import NotFoundError

router = APIRouter()


@router.get("/workflow-runs")
def list_workflow_runs(
    workflow_name: str | None = None,
    ticker: str | None = None,
    limit: int = 20,
):
    from portfolio.core_db import get_workflow_runs

    safe_limit = max(1, min(int(limit), 100))
    runs = get_workflow_runs(workflow_name=workflow_name, ticker=ticker, limit=safe_limit)
    return {"runs": runs, "count": len(runs)}


@router.get("/workflow-runs/{run_id}")
def get_workflow_run_detail(run_id: str):
    from portfolio.core_db import get_workflow_run, provenance_summary

    run = get_workflow_run(run_id)
    if not run:
        raise NotFoundError("Workflow run", run_id)
    run["provenance_summary"] = provenance_summary(workflow_run_id=run_id)
    return run
