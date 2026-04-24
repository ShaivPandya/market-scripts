import json
import threading
import time
import uuid
from typing import Any, Literal, TypedDict, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import delete_cached, get_cached, long_cache, set_cached
from api.exceptions import DataFetchError
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()

_UNIVERSE_MAP = {
    "S&P 500": "sp500",
    "Russell 2000": "russell2000",
    "S&P 400": "sp400",
}

_SECTOR_PREFIX_MAP = {
    "VAW — Materials": "VAW",
    "VOX — Communication Services": "VOX",
    "VDE — Energy": "VDE",
    "VFH — Financials": "VFH",
    "VIS — Industrials": "VIS",
    "VGT — Technology": "VGT",
    "VDC — Consumer Staples": "VDC",
    "VNQ — Real Estate": "VNQ",
    "VPU — Utilities": "VPU",
    "VHT — Health Care": "VHT",
    "VCR — Consumer Discretionary": "VCR",
}

_UNIVERSE_TO_ETF = {
    "S&P 500": "SPY",
    "Russell 2000": "IWM",
    "S&P 400": "MDY",
}


class LongScreenRequest(BaseModel):
    input_mode: str = "Universe"
    universe: str = "Russell 2000"
    tickers: str = ""
    pb_threshold: float | None = 1.5
    profit_type: str | None = "Gross Profit"
    check_issuance: bool = False
    check_revenue: bool = False
    min_revenue_growth: float = 5.0
    check_eps: bool = False
    min_eps_growth: float = 5.0
    check_52w_positive: bool = False
    check_min_drawdown: bool = False
    min_drawdown_pct: float = 25.0
    check_max_drawdown: bool = False
    max_drawdown_pct: float = 60.0
    check_3m_pos_momentum: bool = False
    check_2m_pos_rel_momentum: bool = False
    rel_momentum_benchmark: str = "IWM"


class _Job(TypedDict, total=False):
    status: Literal["queued", "running", "done", "error"]
    created_at: float
    updated_at: float
    cache_key: str
    params: dict[str, Any]
    progress: dict[str, Any]
    result: dict[str, Any]
    error: str


_JOB_STORE = long_cache
_RESULT_CACHE = long_cache
_JOB_NS = "long_screen:job:"
_ACTIVE_NS = "long_screen:active:"


def _resolve_tickers(req: LongScreenRequest) -> list[str]:
    if req.input_mode == "Custom Tickers":
        return [t.strip().upper() for t in req.tickers.split(",") if t.strip()]
    from equities.common import get_universe_tickers

    key = _UNIVERSE_MAP.get(req.universe) or _SECTOR_PREFIX_MAP.get(req.universe, req.universe)
    try:
        return cast(list[str], get_universe_tickers(key))
    except Exception:
        return []


def _resolve_benchmark_ticker(label: str, universe_label: str) -> str:
    if label == "Same as Input":
        etf = _UNIVERSE_TO_ETF.get(universe_label)
        if etf:
            return etf
        sector_etf = _SECTOR_PREFIX_MAP.get(universe_label)
        if sector_etf:
            return sector_etf
        return "SPY"
    return label


def _canonical_custom_tickers(tickers: str) -> str:
    items = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return ",".join(sorted(dict.fromkeys(items)))


def _cache_key(req: LongScreenRequest) -> str:
    if req.input_mode not in ("Universe", "Custom Tickers"):
        raise ValueError("Invalid input_mode. Use one of: Universe, Custom Tickers.")

    payload = req.model_dump()
    if req.input_mode == "Custom Tickers":
        payload["tickers"] = _canonical_custom_tickers(req.tickers)
    else:
        payload["tickers"] = ""
    payload["benchmark_ticker"] = _resolve_benchmark_ticker(req.rel_momentum_benchmark, req.universe)
    return "long_screen:v2:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _serialize_screen(data: dict) -> dict[str, Any]:
    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index(drop=True))
        else:
            result[k] = serialize_value(v)
    return result


def _job_store_key(job_id: str) -> str:
    return f"{_JOB_NS}{job_id}"


def _active_job_key(cache_key: str) -> str:
    return f"{_ACTIVE_NS}{cache_key}"


def _read_job(job_id: str) -> _Job | None:
    raw = get_cached(_JOB_STORE, _job_store_key(job_id))
    if isinstance(raw, dict):
        return cast(_Job, raw)
    return None


def _write_job(job_id: str, job: _Job) -> None:
    set_cached(_JOB_STORE, _job_store_key(job_id), job)


def _compute_long_screen(req: LongScreenRequest, progress_callback=None) -> dict[str, Any]:
    tickers = _resolve_tickers(req)
    if not tickers:
        raise ValueError("No tickers resolved for the requested universe/input.")

    benchmark = _resolve_benchmark_ticker(req.rel_momentum_benchmark, req.universe)

    from equities.long_screen.long_screen import get_data

    data = get_data(
        tickers=tickers,
        pb_threshold=req.pb_threshold,
        profit_type=req.profit_type,
        check_issuance=req.check_issuance,
        check_revenue=req.check_revenue,
        min_revenue_growth=req.min_revenue_growth,
        check_eps=req.check_eps,
        min_eps_growth=req.min_eps_growth,
        check_52w_positive=req.check_52w_positive,
        check_min_drawdown=req.check_min_drawdown,
        min_drawdown_pct=req.min_drawdown_pct,
        check_max_drawdown=req.check_max_drawdown,
        max_drawdown_pct=req.max_drawdown_pct,
        check_3m_pos_momentum=req.check_3m_pos_momentum,
        check_2m_pos_rel_momentum=req.check_2m_pos_rel_momentum,
        benchmark_ticker=benchmark,
        progress_callback=progress_callback,
    )

    if data.get("error"):
        raise RuntimeError(str(data["error"]))

    return _serialize_screen(data)


def _spawn_long_screen_job(job_id: str, req: LongScreenRequest, cache_key: str) -> None:
    def _run() -> None:
        job = _read_job(job_id)
        if not job:
            return
        job["status"] = "running"
        job["progress"] = {"phase": "fundamentals", "done": 0, "total": 0}
        job["updated_at"] = time.time()
        _write_job(job_id, job)

        def _progress(phase: str, done: int, total: int) -> None:
            current = _read_job(job_id)
            if not current:
                return
            current["progress"] = {"phase": phase, "done": done, "total": total}
            current["updated_at"] = time.time()
            _write_job(job_id, current)

        try:
            result = _compute_long_screen(req, progress_callback=_progress)
            _progress("finalizing", 0, 0)
            set_cached(_RESULT_CACHE, cache_key, result)
            job = _read_job(job_id)
            if not job:
                return
            job["status"] = "done"
            job["progress"] = {
                "phase": "done",
                "done": result.get("final_count", 0),
                "total": result.get("final_count", 0),
            }
            job["result"] = result
            job["updated_at"] = time.time()
            _write_job(job_id, job)
            delete_cached(_JOB_STORE, _active_job_key(cache_key))
        except Exception as e:
            job = _read_job(job_id)
            if not job:
                return
            job["status"] = "error"
            job["error"] = str(e) or "Long screen failed"
            job["updated_at"] = time.time()
            _write_job(job_id, job)
            delete_cached(_JOB_STORE, _active_job_key(cache_key))

    thread = threading.Thread(target=_run, name=f"long-screen-job-{job_id}", daemon=True)
    thread.start()


@router.post("/long-screen")
def run_long_screen(req: LongScreenRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    cached = get_cached(_RESULT_CACHE, key)
    if cached is not None:
        return cached

    try:
        result = _compute_long_screen(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904
    except Exception as e:
        raise DataFetchError(source="long_screen", detail=str(e)) from e

    set_cached(_RESULT_CACHE, key, result)
    return result


@router.post("/long-screen/async")
def start_long_screen(req: LongScreenRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    cached = get_cached(_RESULT_CACHE, key)
    if cached is not None:
        return {"job_id": f"cached:{uuid.uuid4().hex}", "status": "done", "result": cached}

    active_raw = get_cached(_JOB_STORE, _active_job_key(key))
    if isinstance(active_raw, dict):
        existing_id = str(active_raw.get("job_id") or "")
        if existing_id:
            existing_job = _read_job(existing_id)
            if existing_job and existing_job.get("status") in ("queued", "running"):
                return {
                    "job_id": existing_id,
                    "status": existing_job.get("status"),
                    "progress": existing_job.get("progress"),
                }
            delete_cached(_JOB_STORE, _active_job_key(key))

    now = time.time()
    job_id = uuid.uuid4().hex
    _write_job(
        job_id,
        {
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "cache_key": key,
            "params": req.model_dump(),
            "progress": {"phase": "queued", "done": 0, "total": 0},
        },
    )
    set_cached(_JOB_STORE, _active_job_key(key), {"job_id": job_id, "updated_at": now})
    _spawn_long_screen_job(job_id, req, key)
    return {"job_id": job_id, "status": "queued", "progress": {"phase": "queued", "done": 0, "total": 0}}


@router.get("/long-screen/async/{job_id}")
def get_long_screen_job(job_id: str):
    if job_id.startswith("cached:"):
        return {"job_id": job_id, "status": "done"}

    job = _read_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    status = job.get("status")
    if status == "done":
        return {
            "job_id": job_id,
            "status": "done",
            "progress": job.get("progress"),
            "result": job.get("result"),
        }
    if status == "error":
        return {
            "job_id": job_id,
            "status": "error",
            "progress": job.get("progress"),
            "error": job.get("error") or "Long screen failed",
        }
    return {"job_id": job_id, "status": status, "progress": job.get("progress")}
