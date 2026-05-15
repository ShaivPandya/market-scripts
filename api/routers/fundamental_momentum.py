from typing import Any, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.serializers import serialize_dataframe, serialize_value
from ontology.sources.source_registry import attach_source_registry_metadata

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


class FMRequest(BaseModel):
    screen_type: str = "Both"  # "EPS" | "Revenue" | "Both"
    universe: str = "S&P 500"
    tickers: str = ""
    benchmark: str = "S&P 500"
    input_mode: str = "Universe"


def _resolve_tickers(req: FMRequest) -> list[str]:
    if req.input_mode == "Custom Tickers":
        return [t.strip().upper() for t in req.tickers.split(",") if t.strip()]
    from equities.common import get_universe_tickers

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
    if req.benchmark in _UNIVERSE_MAP:
        return _UNIVERSE_MAP[req.benchmark]
    if req.benchmark in _SECTOR_PREFIX_MAP:
        return _SECTOR_PREFIX_MAP[req.benchmark]
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
        from portfolio.momentum.fundamental_momentum.eps_screen import get_data as get_eps

        eps_data = get_eps(tickers=tickers, benchmark=benchmark)
        result["eps"] = _serialize_fm(eps_data)

    if req.screen_type in ("Revenue", "Both"):
        from portfolio.momentum.fundamental_momentum.revenue_screen import get_data as get_rev

        rev_data = get_rev(tickers=tickers, benchmark=benchmark)
        result["rev"] = _serialize_fm(rev_data)

    return attach_source_registry_metadata(result, source_id="fundamental_momentum")


@router.post("/fundamental-momentum")
def run_fundamental_momentum(req: FMRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    row, _disposition = enqueue_registered_job("fundamental_momentum", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/fundamental-momentum/async/{job_id}")


@router.post("/fundamental-momentum/async")
def start_fundamental_momentum(req: FMRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    row, _disposition = enqueue_registered_job("fundamental_momentum", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/fundamental-momentum/async/{job_id}")


@router.get("/fundamental-momentum/async/{job_id}")
def get_fundamental_momentum_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")  # noqa: B904
