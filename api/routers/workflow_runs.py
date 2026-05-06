"""Workflow run list/detail API endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api.exceptions import NotFoundError
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()


@router.get("/workflow-runs")
def list_workflow_runs(
    workflow_name: str | None = None,
    ticker: str | None = None,
    limit: int = 20,
):
    safe_limit = max(1, min(int(limit), 100))
    reads = OntologyRuntimeReadService()
    runs = reads.workflow_runs(ticker=ticker, limit=safe_limit)
    if workflow_name:
        runs = [run for run in runs if run.get("workflow_name") == workflow_name]
    return {"runs": runs, "count": len(runs)}


@router.get("/workflow-runs/{run_id}")
def get_workflow_run_detail(run_id: str):
    run = OntologyRuntimeReadService().get(run_id if run_id.startswith("workflow_run:") else f"workflow_run:{run_id}")
    if not run:
        raise NotFoundError("Workflow run", run_id)
    run["provenance_summary"] = {"selector": {"workflow_run_id": run_id}, "lineage_state": "ontology"}
    return run
