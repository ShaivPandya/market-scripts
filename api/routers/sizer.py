from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Literal, TypedDict

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


class SizerPosition(BaseModel):
    ticker: str = ""
    conviction: int = 3


class SizerRequest(BaseModel):
    book: float = 100_000
    target_leverage: float = 2.0
    positions: list[SizerPosition] = Field(default_factory=list)


def _canonical_positions(req: SizerRequest) -> list[tuple[str, int]]:
    aggregated: dict[str, int] = {}
    for idx, row in enumerate(req.positions):  # noqa: B007
        ticker = str(row.ticker).strip().upper()
        conviction = int(row.conviction)
        if not ticker:
            continue
        if conviction < 1 or conviction > 5:
            raise ValueError(f"Position '{ticker}' conviction must be 1-5, got {conviction}.")
        # Take the max conviction for duplicate tickers
        aggregated[ticker] = max(aggregated.get(ticker, 0), conviction)
    if not aggregated:
        raise ValueError("No valid positions provided.")
    return sorted(aggregated.items(), key=lambda x: x[0])


def _cache_key(req: SizerRequest) -> str:
    strategy_version = "v1_conviction_sizing"
    canonical = _canonical_positions(req)
    token = "|".join(f"{ticker}:{conviction}" for ticker, conviction in canonical) or "none"
    return f"portfolio_sizer:{strategy_version}:book={float(req.book):.4f}:lev={float(req.target_leverage):.4f}:positions={token}"


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


def _compute_sizer_result(req: SizerRequest) -> dict[str, Any]:
    try:
        from portfolio_optimizer.portfolio_sizer import get_data

        payload = [row.model_dump() for row in req.positions]
        data = get_data(
            positions=payload,
            book=float(req.book),
            target_leverage=float(req.target_leverage),
        )
    except ValueError:
        raise
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
    to_delete: list[str] = []
    for job_id, job in _jobs.items():
        updated_at = float(job.get("updated_at") or job.get("created_at") or 0.0)
        if updated_at and (now - updated_at) > _JOB_TTL_S:
            to_delete.append(job_id)
    for job_id in to_delete:
        _jobs.pop(job_id, None)


def _spawn_sizer_job(job_id: str, req: SizerRequest, cache_key: str) -> None:
    def _run():
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["updated_at"] = time.time()
        try:
            result = _compute_sizer_result(req)
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
                job["error"] = str(e) or "Portfolio sizer failed"
                job["updated_at"] = time.time()

    t = threading.Thread(target=_run, name=f"sizer-job-{job_id}", daemon=True)
    t.start()


@router.post("/portfolio-sizer")
def run_portfolio_sizer(req: SizerRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached

    try:
        result = _compute_sizer_result(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904
    except Exception as e:
        raise DataFetchError(source="portfolio_sizer", detail=str(e)) from e

    set_cached(short_cache, key, result)
    return result


@router.post("/portfolio-sizer/async")
def start_portfolio_sizer(req: SizerRequest):
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
                "target_leverage": req.target_leverage,
                "positions": [row.model_dump() for row in req.positions],
            },
        }

    _spawn_sizer_job(job_id, req, key)
    return {"job_id": job_id, "status": "queued"}


@router.get("/portfolio-sizer/async/{job_id}")
def get_portfolio_sizer_job(job_id: str):
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
            return {"job_id": job_id, "status": "error", "error": job.get("error") or "Portfolio sizer failed"}
        return {"job_id": job_id, "status": status}


@router.get("/portfolio-sizer/prefill")
def get_sizer_prefill():
    try:
        from portfolio_db import get_positions_df

        df = get_positions_df()
        if "ticker" not in df.columns:
            raise ValueError("Portfolio database is missing required 'ticker' column.")

        tickers = df["ticker"].astype(str).str.strip().str.upper()
        directions = (
            df["direction"].astype(str).str.strip().str.lower()
            if "direction" in df.columns
            else pd.Series([""] * len(df))
        )
        convictions = (
            pd.to_numeric(df["conviction"], errors="coerce").fillna(3).astype(int).clip(1, 5)
            if "conviction" in df.columns
            else pd.Series([3] * len(df))
        )

        deduped_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ticker, direction, conviction in zip(tickers.tolist(), directions.tolist(), convictions.tolist()):  # noqa: B905
            if ticker and ticker not in seen:
                seen.add(ticker)
                deduped_rows.append(
                    {
                        "ticker": ticker,
                        "conviction": conviction,
                        "direction": direction,
                    }
                )

        return {
            "positions": deduped_rows,
            "source": "portfolio.db",
            "count": len(deduped_rows),
        }
    except Exception as e:
        raise DataFetchError(source="portfolio_sizer", detail=str(e)) from e
