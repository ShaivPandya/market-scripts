from typing import Any, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


def _resolve_tickers(req: FMRequest) -> list[str]:
    if req.input_mode == "Custom Tickers":
        return [t.strip().upper() for t in req.tickers.split(",") if t.strip()]
    from common import get_universe_tickers

    key = _UNIVERSE_MAP.get(req.universe) or _SECTOR_PREFIX_MAP.get(req.universe, req.universe)
    try:
        return cast(list[str], get_universe_tickers(key))
    except Exception:
        return []


def _serialize_fm(data: dict) -> dict:
    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    return result


@router.post("/fundamental-momentum")
def run_fundamental_momentum(req: FMRequest):
    try:
        tickers = _resolve_tickers(req)
        if not tickers:
            raise HTTPException(status_code=400, detail="No tickers resolved.")

        if req.benchmark == "S&P 500":
            benchmark = "sp500"
        elif req.benchmark == "Same as Input":
            benchmark = "self"
        else:
            benchmark = req.benchmark

        result: dict[str, Any] = {"screen_type": req.screen_type}

        if req.screen_type in ("EPS", "Both"):
            from eps_screen import get_data as get_eps

            eps_data = get_eps(tickers=tickers, benchmark=benchmark)
            result["eps"] = _serialize_fm(eps_data)

        if req.screen_type in ("Revenue", "Both"):
            from revenue_screen import get_data as get_rev

            rev_data = get_rev(tickers=tickers, benchmark=benchmark)
            result["rev"] = _serialize_fm(rev_data)

    except HTTPException:
        raise
    except Exception as e:
        raise DataFetchError(source="fundamental_momentum", detail=str(e)) from e

    return result
