from typing import Optional
from fastapi import APIRouter, HTTPException
from api.cache import long_cache, get_cached, set_cached
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


@router.get("/positioning/summary")
def get_positioning_summary(
    instruments: str = "SP500,NASDAQ,RUSSELL,US10Y,EUR",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    groups: Optional[str] = None,
    z_window: int = 0,
    force_threshold: float = 2.0,
    app_token: Optional[str] = None,
):
    key = f"positioning_summary:{instruments}:{start}:{end}:{groups}:{z_window}:{force_threshold}"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached
    try:
        from positioning import (
            fetch_multiple_instruments,
            DATASETS,
            DEFAULT_DOMAIN,
            INSTRUMENTS,
        )
        instrument_list = [i.strip() for i in instruments.split(",") if i.strip()]
        results = fetch_multiple_instruments(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=app_token or None,
            instruments=instrument_list,
            start=start,
            end=end,
            groups=groups or None,
            z_window=z_window,
            force_threshold=force_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = serialize_value(results)
    set_cached(long_cache, key, result)
    return result


@router.get("/positioning/timeseries")
def get_positioning_timeseries(
    market: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    groups: Optional[str] = None,
    z_window: int = 0,
    force_threshold: float = 2.0,
    app_token: Optional[str] = None,
):
    key = f"positioning_ts:{market}:{start}:{end}:{groups}:{z_window}:{force_threshold}"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached
    try:
        from positioning import fetch_market_timeseries, DATASETS, DEFAULT_DOMAIN
        df = fetch_market_timeseries(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=app_token or None,
            market_exact=market,
            start=start,
            end=end,
            groups=groups or None,
            z_window=z_window,
            force_threshold=force_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = serialize_dataframe(df.reset_index(drop=True))
    set_cached(long_cache, key, result)
    return result


@router.get("/positioning/instruments")
def get_positioning_instruments():
    """Return available instrument aliases."""
    try:
        from positioning import INSTRUMENTS
        return {"instruments": INSTRUMENTS}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
