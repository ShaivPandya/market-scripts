from typing import cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import stamp_fresh
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


@router.post("/short-screen")
def run_short_screen(req: ShortScreenRequest):
    tickers = _resolve_tickers(req)
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers resolved for the requested universe/input.")

    benchmark = _resolve_benchmark_ticker(req.rel_momentum_benchmark, req.universe)

    try:
        from equities.short_screen.short_screen import get_data

        data = get_data(
            tickers=tickers,
            pb_threshold=req.pb_threshold,
            loss_type=req.loss_type,
            check_issuance=req.check_issuance,
            check_52w_positive=req.check_52w_positive,
            check_min_drawdown=req.check_min_drawdown,
            min_drawdown_pct=req.min_drawdown_pct,
            check_max_drawdown=req.check_max_drawdown,
            max_drawdown_pct=req.max_drawdown_pct,
            check_3m_neg_momentum=req.check_3m_neg_momentum,
            check_2m_neg_rel_momentum=req.check_2m_neg_rel_momentum,
            benchmark_ticker=benchmark,
        )
    except Exception as e:
        raise DataFetchError(source="short_screen", detail=str(e)) from e

    if data.get("error"):
        raise DataFetchError(source="short_screen", detail=data["error"])

    import pandas as pd

    result = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index(drop=True))
        else:
            result[k] = serialize_value(v)
    return stamp_fresh(result)
