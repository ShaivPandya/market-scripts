import json
from typing import Any, cast

import pandas as pd
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

_BENCHMARK_MAP = {
    **_UNIVERSE_MAP,
    **_SECTOR_PREFIX_MAP,
    "S&P 500": "SPY",
    "Russell 2000": "IWM",
    "S&P 400": "MDY",
}

_UNIVERSE_TO_ETF = {
    "S&P 500": "SPY",
    "Russell 2000": "IWM",
    "S&P 400": "MDY",
}


class PriceMomentumRequest(BaseModel):
    input_mode: str = "Universe"
    universe: str = "S&P 500"
    tickers: str = ""
    benchmark: str = "Same as Input"


def _canonical_custom_tickers(tickers: str) -> str:
    items = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return ",".join(sorted(dict.fromkeys(items)))


def _resolve_tickers(req: PriceMomentumRequest) -> list[str]:
    if req.input_mode == "Custom Tickers":
        items = [t.strip().upper() for t in req.tickers.split(",") if t.strip()]
        return list(dict.fromkeys(items))

    from equities.common import get_universe_tickers

    key = _UNIVERSE_MAP.get(req.universe) or _SECTOR_PREFIX_MAP.get(req.universe, req.universe)
    try:
        return cast(list[str], get_universe_tickers(key))
    except Exception:
        return []


def _resolve_benchmark_ticker(label: str, universe_label: str, input_mode: str) -> str:
    if label == "Same as Input":
        if input_mode == "Custom Tickers":
            return "SPY"
        etf = _UNIVERSE_TO_ETF.get(universe_label)
        if etf:
            return etf
        sector_etf = _SECTOR_PREFIX_MAP.get(universe_label)
        if sector_etf:
            return sector_etf
        return "SPY"

    mapped = _BENCHMARK_MAP.get(label)
    if mapped:
        return mapped.upper()
    return label.strip().upper() or "SPY"


def _cache_key(req: PriceMomentumRequest) -> str:
    if req.input_mode not in ("Universe", "Custom Tickers"):
        raise ValueError("Invalid input_mode. Use one of: Universe, Custom Tickers.")

    payload = req.model_dump()
    if req.input_mode == "Custom Tickers":
        payload["tickers"] = _canonical_custom_tickers(req.tickers)
    else:
        payload["tickers"] = ""
    payload["benchmark_ticker"] = _resolve_benchmark_ticker(req.benchmark, req.universe, req.input_mode)
    return "price_momentum:v1:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _serialize_price_momentum(data: dict) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index(drop=True))
        else:
            result[k] = serialize_value(v)
    return result


def _compute_price_momentum(req: PriceMomentumRequest, progress_callback=None) -> dict[str, Any]:
    tickers = _resolve_tickers(req)
    if not tickers:
        raise ValueError("No tickers resolved for the requested universe/input.")

    benchmark_ticker = _resolve_benchmark_ticker(req.benchmark, req.universe, req.input_mode)

    from portfolio.momentum.price_momentum.momentum import analyze_ticker, fetch_prices_batch

    total = len(tickers)
    if progress_callback:
        progress_callback("prices", 0, total)

    symbols = list(dict.fromkeys([*tickers, benchmark_ticker]))
    prices_map, volumes_map = fetch_prices_batch(symbols, years=5)
    benchmark_prices = prices_map.get(benchmark_ticker)

    rows: list[dict[str, Any]] = []
    failed_tickers: list[str] = []
    latest_date = None

    for idx, ticker in enumerate(tickers, start=1):
        ticker_prices = prices_map.get(ticker)
        if ticker_prices is None:
            failed_tickers.append(f"{ticker}: no price data")
        elif benchmark_prices is None:
            failed_tickers.append(f"{ticker}: benchmark {benchmark_ticker} unavailable")
        else:
            result = analyze_ticker(
                ticker,
                benchmark_prices,
                years=5,
                ticker_prices=ticker_prices,
                ticker_volume=volumes_map.get(ticker),
            )
            if result is None:
                failed_tickers.append(f"{ticker}: insufficient data")
            else:
                latest_date = max(latest_date, result["date"]) if latest_date is not None else result["date"]
                rows.append(
                    {
                        "ticker": result["ticker"],
                        "close": result["close"],
                        "avg20_roc63": result["avg20_roc63"],
                        "roc63": result["roc63"],
                        "rel_roc42": result["rel_roc42"],
                        "avg10_rel_roc": result["avg10_rel_roc"],
                        "benchmark": benchmark_ticker,
                    }
                )

        if progress_callback:
            progress_callback("prices", idx, total)

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values("avg20_roc63", ascending=False).reset_index(drop=True)

    return _serialize_price_momentum(
        {
            "results_df": results_df,
            "failed_tickers": failed_tickers,
            "input_count": total,
            "scored_count": len(results_df),
            "benchmark_name": benchmark_ticker,
            "date": latest_date.date() if latest_date is not None else None,
            "final_count": len(results_df),
        }
    )


@router.post("/price-momentum/async")
def start_price_momentum(req: PriceMomentumRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    row, _disposition = enqueue_registered_job("price_momentum", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/v1/price-momentum/async/{job_id}")


@router.get("/price-momentum/async/{job_id}")
def get_price_momentum_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")  # noqa: B904
