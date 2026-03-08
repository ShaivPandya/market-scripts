from fastapi import APIRouter, Query

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_value
from api.signal_aggregator import (
    DEFAULT_LOOKBACK_WEEKS,
    DEFAULT_POSITIONING_INSTRUMENTS,
    build_signal_aggregator,
)

router = APIRouter()


@router.get("/signal-aggregator")
def get_signal_aggregator(
    lookback_weeks: int = Query(DEFAULT_LOOKBACK_WEEKS, ge=26, le=520),
    positioning_instruments: str = Query(DEFAULT_POSITIONING_INSTRUMENTS),
    include_raw_modules: bool = Query(False),
):
    key = f"signal_aggregator:{lookback_weeks}:{positioning_instruments}:{include_raw_modules}"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached

    try:
        data = build_signal_aggregator(
            lookback_weeks=lookback_weeks,
            positioning_instruments=positioning_instruments,
            include_raw_modules=include_raw_modules,
        )
    except Exception as e:
        raise DataFetchError(source="signal_aggregator", detail=str(e)) from e

    result = serialize_value(data)
    set_cached(short_cache, key, result)
    return result
