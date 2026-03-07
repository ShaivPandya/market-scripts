import threading
import time
import uuid
from typing import Any, Literal, TypedDict, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()

_UNIVERSE_MAP = {
    "S&P 500": "sp500",
    "Russell 2000": "russell2000",
    "S&P 400": "sp400",
}

_SECTOR_PREFIX_MAP = {
    "XLB — Materials": "XLB",
    "XLC — Communication Services": "XLC",
    "XLE — Energy": "XLE",
    "XLF — Financials": "XLF",
    "XLI — Industrials": "XLI",
    "XLK — Technology": "XLK",
    "XLP — Consumer Staples": "XLP",
    "XLRE — Real Estate": "XLRE",
    "XLU — Utilities": "XLU",
    "XLV — Health Care": "XLV",
    "XLY — Consumer Discretionary": "XLY",
}


class FMRequest(BaseModel):
    screen_type: str = "Both"  # "EPS" | "Revenue" | "Both"
    universe: str = "S&P 500"
    tickers: str = ""
    benchmark: str = "S&P 500"
    input_mode: str = "Universe"


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


def _resolve_tickers(req: FMRequest) -> list[str]:
    if req.input_mode == "Custom Tickers":
        return [t.strip().upper() for t in req.tickers.split(",") if t.strip()]
    from common import get_universe_tickers

    key = _UNIVERSE_MAP.get(req.universe) or _SECTOR_PREFIX_MAP.get(req.universe, req.universe)
    try:
        return cast(list[str], get_universe_tickers(key))
    except Exception:
        return []


def _resolve_benchmark(req: FMRequest) -> str:
    if req.benchmark == "S&P 500":
        return "sp500"
    if req.benchmark == "Same as Input":
        return "self"
    return req.benchmark


def _cache_key(req: FMRequest) -> str:
    if req.screen_type not in ("EPS", "Revenue", "Both"):
        raise ValueError("Invalid screen_type. Use one of: EPS, Revenue, Both.")
    if req.input_mode not in ("Universe", "Custom Tickers"):
        raise ValueError("Invalid input_mode. Use one of: Universe, Custom Tickers.")

    canonical_tickers = ""
    if req.input_mode == "Custom Tickers":
        items = [t.strip().upper() for t in req.tickers.split(",") if t.strip()]
        canonical_tickers = ",".join(sorted(dict.fromkeys(items)))

    return (
        "fundamental_momentum:v1:"
        f"screen_type={req.screen_type}:"
        f"input_mode={req.input_mode}:"
        f"universe={req.universe.strip()}:"
        f"tickers={canonical_tickers}:"
        f"benchmark={_resolve_benchmark(req)}"
    )


def _serialize_fm(data: dict) -> dict:
    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    return result


def _compute_fundamental_momentum(req: FMRequest) -> dict[str, Any]:
    if req.screen_type not in ("EPS", "Revenue", "Both"):
        raise ValueError("Invalid screen_type. Use one of: EPS, Revenue, Both.")

    tickers = _resolve_tickers(req)
    if not tickers:
        raise ValueError("No tickers resolved.")

    benchmark = _resolve_benchmark(req)
    result: dict[str, Any] = {"screen_type": req.screen_type}

    if req.screen_type in ("EPS", "Both"):
        from eps_screen import get_data as get_eps

        eps_data = get_eps(tickers=tickers, benchmark=benchmark)
        result["eps"] = _serialize_fm(eps_data)

    if req.screen_type in ("Revenue", "Both"):
        from revenue_screen import get_data as get_rev

        rev_data = get_rev(tickers=tickers, benchmark=benchmark)
        result["rev"] = _serialize_fm(rev_data)

    return result


def _job_cleanup_locked(now: float) -> None:
    to_delete: list[str] = []
    for job_id, job in _jobs.items():
        updated_at = float(job.get("updated_at") or job.get("created_at") or 0.0)
        if updated_at and (now - updated_at) > _JOB_TTL_S:
            to_delete.append(job_id)
    for job_id in to_delete:
        _jobs.pop(job_id, None)


def _spawn_fm_job(job_id: str, req: FMRequest, cache_key: str) -> None:
    def _run():
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["updated_at"] = time.time()
        try:
            result = _compute_fundamental_momentum(req)
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
                job["error"] = str(e) or "Fundamental momentum failed"
                job["updated_at"] = time.time()

    t = threading.Thread(target=_run, name=f"fundamental-momentum-job-{job_id}", daemon=True)
    t.start()


@router.post("/fundamental-momentum")
def run_fundamental_momentum(req: FMRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached

    try:
        result = _compute_fundamental_momentum(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904
    except Exception as e:
        raise DataFetchError(source="fundamental_momentum", detail=str(e)) from e

    set_cached(short_cache, key, result)
    return result


@router.post("/fundamental-momentum/async")
def start_fundamental_momentum(req: FMRequest):
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
            "params": req.model_dump(),
        }

    _spawn_fm_job(job_id, req, key)
    return {"job_id": job_id, "status": "queued"}


@router.get("/fundamental-momentum/async/{job_id}")
def get_fundamental_momentum_job(job_id: str):
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
            return {"job_id": job_id, "status": "error", "error": job.get("error") or "Fundamental momentum failed"}
        return {"job_id": job_id, "status": status}
