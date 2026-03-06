from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Literal, TypedDict

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


class AnalyzerRequest(BaseModel):
    # Legacy optimizer fields are accepted for backward compatibility and ignored by analyzer logic.
    book: float | None = None
    target_leverage: float | None = None
    beta_neutral: bool | None = None


def _cache_key(_req: AnalyzerRequest) -> str:
    strategy_version = "v1_signal_factor_table"
    return f"portfolio_analyzer:{strategy_version}"


class _Job(TypedDict, total=False):
    status: Literal["queued", "running", "done", "error"]
    created_at: float
    updated_at: float
    cache_key: str
    params: dict[str, Any]
    result: dict[str, Any]
    error: str


_jobs: dict[str, _Job] = {}
_jobs_lock = threading.Lock()
_JOB_TTL_S = 60 * 30


def _compute_analyzer_result(req: AnalyzerRequest) -> dict[str, Any]:
    try:
        from portfolio_optimizer.portfolio_analyzer import get_data

        data = get_data(
            book=req.book,
            target_leverage=req.target_leverage,
            beta_neutral=req.beta_neutral,
        )
    except Exception as e:
        raise RuntimeError(str(e)) from e

    if "error" in data and data["error"]:
        raise RuntimeError(str(data["error"]))

    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    return result


def _job_cleanup_locked(now: float) -> None:
    to_delete: list[str] = []
    for job_id, job in _jobs.items():
        updated_at = float(job.get("updated_at") or job.get("created_at") or 0.0)
        if updated_at and (now - updated_at) > _JOB_TTL_S:
            to_delete.append(job_id)
    for job_id in to_delete:
        _jobs.pop(job_id, None)


def _spawn_analyzer_job(job_id: str, req: AnalyzerRequest, cache_key: str) -> None:
    def _run() -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["updated_at"] = time.time()
        try:
            result = _compute_analyzer_result(req)
            set_cached(short_cache, cache_key, result)
            with _jobs_lock:
                job = _jobs.get(job_id)
                if not job:
                    return
                job["status"] = "done"
                job["result"] = result
                job["updated_at"] = time.time()
        except Exception as e:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if not job:
                    return
                job["status"] = "error"
                job["error"] = str(e) or "Portfolio analyzer failed"
                job["updated_at"] = time.time()

    t = threading.Thread(target=_run, name=f"analyzer-job-{job_id}", daemon=True)
    t.start()


@router.post("/portfolio-analyzer")
@router.post("/portfolio-optimizer")
def run_analyzer(req: AnalyzerRequest = Body(default_factory=AnalyzerRequest)):
    key = _cache_key(req)
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached

    try:
        result = _compute_analyzer_result(req)
    except Exception as e:
        raise DataFetchError(source="portfolio_analyzer", detail=str(e)) from e

    set_cached(short_cache, key, result)
    return result


@router.post("/portfolio-analyzer/async")
@router.post("/portfolio-optimizer/async")
def start_analyzer(req: AnalyzerRequest = Body(default_factory=AnalyzerRequest)):
    """
    Start an analyzer job and return a job_id quickly.
    """
    key = _cache_key(req)
    cached = get_cached(short_cache, key)
    if cached is not None:
        job_id = f"cached:{uuid.uuid4().hex}"
        return {"job_id": job_id, "status": "done", "result": cached}

    now = time.time()
    with _jobs_lock:
        _job_cleanup_locked(now)
        for existing_id, job in _jobs.items():
            if job.get("cache_key") == key and job.get("status") in ("queued", "running"):
                return {"job_id": existing_id, "status": job.get("status")}

        job_id = uuid.uuid4().hex
        _jobs[job_id] = {
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "cache_key": key,
            "params": {
                "book": req.book,
                "target_leverage": req.target_leverage,
                "beta_neutral": req.beta_neutral,
            },
        }

    _spawn_analyzer_job(job_id, req, key)
    return {"job_id": job_id, "status": "queued"}


@router.get("/portfolio-analyzer/async/{job_id}")
@router.get("/portfolio-optimizer/async/{job_id}")
def get_analyzer_job(job_id: str):
    now = time.time()

    if job_id.startswith("cached:"):
        return {"job_id": job_id, "status": "done"}

    with _jobs_lock:
        _job_cleanup_locked(now)
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")

        status = job.get("status")
        if status == "done":
            return {"job_id": job_id, "status": "done", "result": job.get("result")}
        if status == "error":
            return {"job_id": job_id, "status": "error", "error": job.get("error") or "Portfolio analyzer failed"}
        return {"job_id": job_id, "status": status}
