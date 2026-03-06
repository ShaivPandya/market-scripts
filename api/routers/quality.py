from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import get_cached, long_cache, set_cached
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

_BENCHMARK_MAP = {
    "S&P 500": "sp500",
    "Same as Input": None,  # resolved in handler
}


class QualityRequest(BaseModel):
    universe: str = "S&P 500"  # display label from dropdown
    tickers: str = ""  # comma-separated, used when universe == "custom"
    benchmark: str = "S&P 500"
    input_mode: str = "Universe"  # "Universe" | "Custom Tickers"


def _resolve_universe_tickers(universe_label: str) -> list[str]:
    """Return list of tickers for a named universe label."""
    from common import get_universe_tickers, list_universes

    ticker_key = _UNIVERSE_MAP.get(universe_label)
    if ticker_key:
        return get_universe_tickers(ticker_key)
    sector_etf = _SECTOR_PREFIX_MAP.get(universe_label)
    if sector_etf:
        return get_universe_tickers(sector_etf)
    # Try as a raw key
    try:
        return get_universe_tickers(universe_label)
    except Exception:
        return []


@router.post("/quality-screen")
def run_quality_screen(req: QualityRequest):
    key = f"quality:{req.input_mode}:{req.universe}:{req.tickers}:{req.benchmark}"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached
    try:
        if req.input_mode == "Custom Tickers":
            tickers = [t.strip().upper() for t in req.tickers.split(",") if t.strip()]
        else:
            tickers = _resolve_universe_tickers(req.universe)

        if not tickers:
            raise HTTPException(status_code=400, detail="No tickers resolved for the requested universe/input.")

        benchmark_label = req.benchmark
        if benchmark_label == "Same as Input":
            benchmark = "self"
        else:
            benchmark = _BENCHMARK_MAP.get(benchmark_label, benchmark_label)

        from quality import get_data

        data = get_data(tickers=tickers, benchmark=benchmark)
    except HTTPException:
        raise
    except Exception as e:
        raise DataFetchError(source="quality", detail=str(e)) from e

    if data.get("error"):
        raise HTTPException(
            status_code=422,
            detail={
                "message": data["error"],
                "failed": data.get("failed") or [],
            },
        )

    import pandas as pd

    result = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    set_cached(long_cache, key, result)
    return result
