from __future__ import annotations

import math
import threading
import time
import uuid
from typing import Any, Literal, TypedDict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


class HedgingPosition(BaseModel):
    ticker: str = ""
    weight: float


class HedgingRequest(BaseModel):
    book: float = 100_000
    positions: list[HedgingPosition] = Field(default_factory=list)


def _canonical_positions(req: HedgingRequest) -> list[tuple[str, float]]:
    aggregated: dict[str, float] = {}
    for idx, row in enumerate(req.positions):
        ticker = str(row.ticker).strip().upper()
        weight = float(row.weight)
        if not math.isfinite(weight):
            raise ValueError(f"Position '{ticker}' has a non-finite weight.")
        if not ticker:
            raise ValueError(f"Position at index {idx} has an empty ticker.")
        aggregated[ticker] = aggregated.get(ticker, 0.0) + weight
    if not aggregated:
        raise ValueError("No valid positions provided.")
    return sorted(aggregated.items(), key=lambda x: x[0])


def _cache_key(req: HedgingRequest) -> str:
    strategy_version = "v1_signed_all_positions"
    canonical = _canonical_positions(req)
    token = "|".join(f"{ticker}:{weight:.12g}" for ticker, weight in canonical) or "none"
    return f"hedging_tool:{strategy_version}:book={float(req.book):.4f}:positions={token}"


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


def _compute_hedging_result(req: HedgingRequest) -> dict[str, Any]:
    try:
        from portfolio.portfolio_optimizer.hedging_tool import get_data

        payload = [row.model_dump() for row in req.positions]
        data = get_data(positions=payload, book=float(req.book))
    except ValueError:
        raise
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


def _spawn_hedging_job(job_id: str, req: HedgingRequest, cache_key: str) -> None:
    def _run():
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["updated_at"] = time.time()
        try:
            result = _compute_hedging_result(req)
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
                job["error"] = str(e) or "Hedging tool failed"
                job["updated_at"] = time.time()

    t = threading.Thread(target=_run, name=f"hedging-job-{job_id}", daemon=True)
    t.start()


@router.post("/hedging-tool")
def run_hedging_tool(req: HedgingRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached

    try:
        result = _compute_hedging_result(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904
    except Exception as e:
        raise DataFetchError(source="hedging_tool", detail=str(e)) from e

    set_cached(short_cache, key, result)
    return result


@router.post("/hedging-tool/async")
def start_hedging_tool(req: HedgingRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

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
                "positions": [row.model_dump() for row in req.positions],
            },
        }

    _spawn_hedging_job(job_id, req, key)
    return {"job_id": job_id, "status": "queued"}


@router.get("/hedging-tool/async/{job_id}")
def get_hedging_tool_job(job_id: str):
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
            return {"job_id": job_id, "status": "error", "error": job.get("error") or "Hedging tool failed"}
        return {"job_id": job_id, "status": status}


@router.get("/hedging-tool/prefill")
def get_hedging_tool_prefill():
    try:
        from portfolio.portfolio_db import get_positions_df

        df = get_positions_df()
        if "ticker" not in df.columns:
            raise ValueError("Portfolio database is missing required 'ticker' column.")

        tickers = df["ticker"].astype(str).str.strip().str.upper()
        tickers = [t for t in tickers.tolist() if t]
        # Preserve order while deduplicating.
        deduped = list(dict.fromkeys(tickers))

        return {
            "positions": [{"ticker": t, "weight": 0.0} for t in deduped],
            "source": "portfolio.db",
            "count": len(deduped),
        }
    except Exception as e:
        raise DataFetchError(source="hedging_tool", detail=str(e)) from e
