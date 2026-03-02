from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()

import threading
import time
import uuid
from typing import Any, Literal, TypedDict


class OptimizerRequest(BaseModel):
    book: int = 100_000
    target_leverage: float = 2.0
    beta_neutral: bool = True


def _cache_key(req: OptimizerRequest) -> str:
    strategy_version = "v2_anchor_abs_long"
    return (
        f"portfolio_optimizer:{strategy_version}:"
        f"book={int(req.book)}:lev={float(req.target_leverage):.4f}:bn={req.beta_neutral}"
    )

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
_JOB_TTL_S = 60 * 30  # best-effort cleanup window; results are also cached separately


def _compute_optimizer_result(req: OptimizerRequest) -> dict[str, Any]:
    try:
        from portfolio_optimizer.portfolio_optimizer import get_data
        data = get_data(book=req.book, target_leverage=req.target_leverage, beta_neutral=req.beta_neutral)
    except Exception as e:
        raise RuntimeError(str(e)) from e

    if "error" in data and data["error"]:
        raise RuntimeError(str(data["error"]))

    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if k == "max_scaled" and isinstance(v, dict):
            inner: dict[str, Any] = {}
            for ik, iv in v.items():
                if isinstance(iv, pd.DataFrame):
                    inner[ik] = serialize_dataframe(iv.reset_index())
                else:
                    inner[ik] = serialize_value(iv)
            result[k] = inner
        elif isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    return result


def _job_cleanup_locked(now: float) -> None:
    # Best-effort: drop old jobs to avoid unbounded memory growth.
    to_delete: list[str] = []
    for job_id, job in _jobs.items():
        updated_at = float(job.get("updated_at") or job.get("created_at") or 0.0)
        if updated_at and (now - updated_at) > _JOB_TTL_S:
            to_delete.append(job_id)
    for job_id in to_delete:
        _jobs.pop(job_id, None)


def _spawn_optimizer_job(job_id: str, req: OptimizerRequest, cache_key: str) -> None:
    def _run():
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["updated_at"] = time.time()
        try:
            result = _compute_optimizer_result(req)
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
                job["error"] = str(e) or "Optimizer failed"
                job["updated_at"] = time.time()

    t = threading.Thread(target=_run, name=f"optimizer-job-{job_id}", daemon=True)
    t.start()


@router.post("/portfolio-optimizer")
def run_optimizer(req: OptimizerRequest):
    key = _cache_key(req)
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached

    try:
        result = _compute_optimizer_result(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    set_cached(short_cache, key, result)
    return result


@router.post("/portfolio-optimizer/async")
def start_optimizer(req: OptimizerRequest):
    """
    Start an optimizer job and return a job_id quickly.

    This is mainly to avoid edge proxy timeouts (e.g., Cloudflare) for the long-running
    first run when yfinance / cvxpy are slow.
    """
    key = _cache_key(req)
    cached = get_cached(short_cache, key)
    if cached is not None:
        # Return a "done" job immediately (no need to create background work).
        job_id = f"cached:{uuid.uuid4().hex}"
        return {"job_id": job_id, "status": "done", "result": cached}

    now = time.time()
    with _jobs_lock:
        _job_cleanup_locked(now)
        # Reuse an existing running job for identical params.
        for existing_id, job in _jobs.items():
            if job.get("cache_key") == key and job.get("status") in ("queued", "running"):
                return {"job_id": existing_id, "status": job.get("status")}

        job_id = uuid.uuid4().hex
        _jobs[job_id] = {
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "cache_key": key,
            "params": {"book": req.book, "target_leverage": req.target_leverage, "beta_neutral": req.beta_neutral},
        }

    _spawn_optimizer_job(job_id, req, key)
    return {"job_id": job_id, "status": "queued"}


@router.get("/portfolio-optimizer/async/{job_id}")
def get_optimizer_job(job_id: str):
    now = time.time()

    if job_id.startswith("cached:"):
        # Client already has the payload from start_optimizer; avoid storing it server-side.
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
            return {"job_id": job_id, "status": "error", "error": job.get("error") or "Optimizer failed"}
        return {"job_id": job_id, "status": status}
