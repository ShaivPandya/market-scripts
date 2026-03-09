from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Literal, TypedDict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.exceptions import DataFetchError
from ontology.service import OntologyQueryService, OntologyRunNotFoundError

router = APIRouter()
_service = OntologyQueryService()


class OntologyFilters(BaseModel):
    tickers: list[str] | None = None
    sectors: list[str] | None = None
    assets: list[str] | None = None
    max_results: int | None = None
    min_risk_score: float | None = None


class OntologyQueryRequest(BaseModel):
    query: str | None = None
    intent: (
        Literal[
            "portfolio_risk_exposure",
            "positions_in_deteriorating_macro",
            "entity_context",
        ]
        | None
    ) = None
    filters: OntologyFilters | None = None
    timeframe: str = "Daily"
    include_graph: bool = False
    run_id: str | None = None
    refresh_snapshot: bool = False


def _extract_filters(req: OntologyQueryRequest) -> dict[str, Any]:
    return req.filters.model_dump(exclude_none=True) if req.filters else {}


def _execute_query(req: OntologyQueryRequest) -> dict[str, Any]:
    filters = _extract_filters(req)
    return _service.query(
        query=req.query,
        intent=req.intent,
        filters=filters,
        timeframe=req.timeframe,
        include_graph=req.include_graph,
        run_id=req.run_id,
        refresh_snapshot=req.refresh_snapshot,
    )


@router.get("/ontology/runs")
def list_ontology_runs(limit: int = 100):
    safe_limit = max(1, min(int(limit), 500))
    try:
        runs = _service.list_runs(limit=safe_limit)
        return {"runs": runs}
    except HTTPException:
        raise
    except Exception as exc:
        raise DataFetchError(source="ontology", detail=str(exc)) from exc


@router.post("/ontology/query")
def query_ontology(req: OntologyQueryRequest):
    try:
        return _execute_query(req)
    except OntologyRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise DataFetchError(source="ontology", detail=str(exc)) from exc


class _OntologyJob(TypedDict, total=False):
    status: Literal["queued", "running", "done", "error"]
    created_at: float
    updated_at: float
    cache_key: str
    params: dict[str, Any]
    result: dict[str, Any]
    error: str


_jobs: dict[str, _OntologyJob] = {}
_jobs_lock = threading.Lock()
_JOB_TTL_S = 60 * 30


def _job_cleanup_locked(now: float) -> None:
    to_delete: list[str] = []
    for job_id, job in _jobs.items():
        updated_at = float(job.get("updated_at") or job.get("created_at") or 0.0)
        if updated_at and (now - updated_at) > _JOB_TTL_S:
            to_delete.append(job_id)
    for job_id in to_delete:
        _jobs.pop(job_id, None)


def _job_cache_key(req: OntologyQueryRequest) -> str:
    payload = req.model_dump(exclude_none=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _spawn_ontology_job(job_id: str, req: OntologyQueryRequest) -> None:
    req_copy = req.model_copy(deep=True)

    def _run() -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["updated_at"] = time.time()
        try:
            result = _execute_query(req_copy)
            with _jobs_lock:
                job = _jobs.get(job_id)
                if not job:
                    return
                job["status"] = "done"
                job["result"] = result
                job["updated_at"] = time.time()
        except Exception as exc:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if not job:
                    return
                job["status"] = "error"
                job["error"] = str(exc) or "Ontology query failed"
                job["updated_at"] = time.time()

    t = threading.Thread(target=_run, name=f"ontology-query-job-{job_id}", daemon=True)
    t.start()


@router.post("/ontology/query/async")
def start_query_ontology_async(req: OntologyQueryRequest):
    key = _job_cache_key(req)
    now = time.time()
    with _jobs_lock:
        _job_cleanup_locked(now)
        for existing_id, job in _jobs.items():
            if job.get("cache_key") != key:
                continue
            status = str(job.get("status") or "queued")
            if status in ("queued", "running"):
                return {"job_id": existing_id, "status": status}
            if status == "done":
                return {"job_id": existing_id, "status": "done", "result": job.get("result")}

        job_id = uuid.uuid4().hex
        _jobs[job_id] = {
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "cache_key": key,
            "params": req.model_dump(exclude_none=True),
        }

    _spawn_ontology_job(job_id, req)
    return {"job_id": job_id, "status": "queued"}


@router.get("/ontology/query/async/{job_id}")
def get_query_ontology_async(job_id: str):
    now = time.time()
    with _jobs_lock:
        _job_cleanup_locked(now)
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        status = str(job.get("status") or "queued")
        if status == "done":
            return {"job_id": job_id, "status": "done", "result": job.get("result")}
        if status == "error":
            return {"job_id": job_id, "status": "error", "error": job.get("error") or "Ontology query failed"}
        return {"job_id": job_id, "status": status}
