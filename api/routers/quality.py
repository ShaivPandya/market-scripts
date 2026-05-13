import json
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
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

# Resolves display labels to backend keys; sector/universe labels fall through
# to _SECTOR_PREFIX_MAP and _UNIVERSE_MAP in the endpoint handler.
_BENCHMARK_MAP = {
    "S&P 500": "sp500",
}


class QualityRequest(BaseModel):
    universe: str = "S&P 500"  # display label from dropdown
    tickers: str = ""  # comma-separated, used when universe == "custom"
    benchmark: str = "S&P 500"
    input_mode: str = "Universe"  # "Universe" | "Custom Tickers"


def _resolve_universe_tickers(universe_label: str) -> list[str]:
    """Return list of tickers for a named universe label."""
    from equities.common import get_universe_tickers

    ticker_key = _UNIVERSE_MAP.get(universe_label)
    if ticker_key:
        return cast(list[str], get_universe_tickers(ticker_key))
    sector_etf = _SECTOR_PREFIX_MAP.get(universe_label)
    if sector_etf:
        return cast(list[str], get_universe_tickers(sector_etf))
    # Try as a raw key
    try:
        return cast(list[str], get_universe_tickers(universe_label))
    except Exception:
        return []


def _canonical_custom_tickers(tickers: str) -> str:
    items = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return ",".join(sorted(dict.fromkeys(items)))


def _resolve_tickers(req: QualityRequest) -> list[str]:
    if req.input_mode == "Custom Tickers":
        items = [t.strip().upper() for t in req.tickers.split(",") if t.strip()]
        return list(dict.fromkeys(items))
    return _resolve_universe_tickers(req.universe)


def _resolve_benchmark(label: str) -> str:
    if label == "Same as Input":
        return "self"
    return _BENCHMARK_MAP.get(label) or _SECTOR_PREFIX_MAP.get(label) or _UNIVERSE_MAP.get(label) or label


def _cache_key(req: QualityRequest) -> str:
    if req.input_mode not in ("Universe", "Custom Tickers"):
        raise ValueError("Invalid input_mode. Use one of: Universe, Custom Tickers.")

    payload = req.model_dump()
    if req.input_mode == "Custom Tickers":
        payload["tickers"] = _canonical_custom_tickers(req.tickers)
    else:
        payload["tickers"] = ""
    payload["benchmark_key"] = _resolve_benchmark(req.benchmark)
    return "quality_screen:v1:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _serialize_quality_screen(data: dict) -> dict[str, Any]:
    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    return result


def _compute_quality_screen(req: QualityRequest, progress_callback=None) -> dict[str, Any]:
    tickers = _resolve_tickers(req)
    if not tickers:
        raise ValueError("No tickers resolved for the requested universe/input.")

    benchmark = _resolve_benchmark(req.benchmark)

    from equities.quality.quality import get_data

    def progress(current: int, total: int) -> None:
        if progress_callback:
            progress_callback("quality", current, total)

    try:
        data = get_data(tickers=tickers, benchmark=benchmark, progress_callback=progress)
    except Exception as e:
        raise DataFetchError(source="quality", detail=str(e)) from e

    if data.get("error"):
        failed = data.get("failed") or []
        message = str(data["error"])
        if failed:
            message = f"{message} (failed: {', '.join(str(t) for t in failed)})"
        raise RuntimeError(message)

    result = _serialize_quality_screen(data)
    result["final_count"] = int(result.get("scored_count") or 0)
    return result


@router.post("/quality-screen")
def run_quality_screen(req: QualityRequest):
    return start_quality_screen(req)


@router.post("/quality-screen/async")
def start_quality_screen(req: QualityRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    row, _disposition = enqueue_registered_job("quality_screen", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/quality-screen/async/{job_id}")


@router.get("/quality-screen/async/{job_id}")
def get_quality_screen_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id") from None
