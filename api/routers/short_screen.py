import json
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
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


class ShortScreenRequest(BaseModel):
    input_mode: str = "Universe"
    universe: str = "Russell 2000"
    tickers: str = ""
    pb_threshold: float | None = 3.0
    loss_type: str | None = "Gross Loss"
    check_issuance: bool = False
    check_revenue: bool = False
    max_revenue_growth: float = 0.0
    check_eps: bool = False
    max_eps_growth: float = 0.0
    check_52w_positive: bool = False
    check_min_drawdown: bool = False
    min_drawdown_pct: float = 25.0
    check_max_drawdown: bool = False
    max_drawdown_pct: float = 60.0
    check_3m_neg_momentum: bool = False
    check_2m_neg_rel_momentum: bool = False
    rel_momentum_benchmark: str = "IWM"


def _resolve_tickers(req: ShortScreenRequest) -> list[str]:
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


def _cache_key(req: ShortScreenRequest) -> str:
    if req.input_mode not in ("Universe", "Custom Tickers"):
        raise ValueError("Invalid input_mode. Use one of: Universe, Custom Tickers.")

    payload = req.model_dump()
    if req.input_mode == "Custom Tickers":
        payload["tickers"] = _canonical_custom_tickers(req.tickers)
    else:
        payload["tickers"] = ""
    payload["benchmark_ticker"] = _resolve_benchmark_ticker(req.rel_momentum_benchmark, req.universe)
    return "short_screen:v2:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _serialize_screen(data: dict) -> dict[str, Any]:
    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index(drop=True))
        else:
            result[k] = serialize_value(v)
    return result


def _compute_short_screen(req: ShortScreenRequest, progress_callback=None) -> dict[str, Any]:
    tickers = _resolve_tickers(req)
    if not tickers:
        raise ValueError("No tickers resolved for the requested universe/input.")

    benchmark = _resolve_benchmark_ticker(req.rel_momentum_benchmark, req.universe)

    from equities.short_screen.short_screen import get_data

    data = get_data(
        tickers=tickers,
        pb_threshold=req.pb_threshold,
        loss_type=req.loss_type,
        check_issuance=req.check_issuance,
        check_revenue=req.check_revenue,
        max_revenue_growth=req.max_revenue_growth,
        check_eps=req.check_eps,
        max_eps_growth=req.max_eps_growth,
        check_52w_positive=req.check_52w_positive,
        check_min_drawdown=req.check_min_drawdown,
        min_drawdown_pct=req.min_drawdown_pct,
        check_max_drawdown=req.check_max_drawdown,
        max_drawdown_pct=req.max_drawdown_pct,
        check_3m_neg_momentum=req.check_3m_neg_momentum,
        check_2m_neg_rel_momentum=req.check_2m_neg_rel_momentum,
        benchmark_ticker=benchmark,
        progress_callback=progress_callback,
    )

    if data.get("error"):
        raise RuntimeError(str(data["error"]))

    return _serialize_screen(data)


@router.post("/short-screen")
def run_short_screen(req: ShortScreenRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    row, _disposition = enqueue_registered_job("short_screen", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/short-screen/async/{job_id}")


@router.post("/short-screen/async")
def start_short_screen(req: ShortScreenRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    row, _disposition = enqueue_registered_job("short_screen", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/short-screen/async/{job_id}")


@router.get("/short-screen/async/{job_id}")
def get_short_screen_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")  # noqa: B904
