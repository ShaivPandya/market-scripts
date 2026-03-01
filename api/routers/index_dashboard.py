from fastapi import APIRouter, HTTPException
from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_series, serialize_value

router = APIRouter()

VALID_TIMEFRAMES = {"This Week", "Daily", "Weekly", "Monthly"}


@router.get("/index-dashboard")
def get_index_dashboard(timeframe: str = "Daily"):
    if timeframe not in VALID_TIMEFRAMES:
        timeframe = "Daily"
    key = f"index_dashboard:{timeframe}"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from index_dashboard import get_data
        data = get_data(timeframe=timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in data and data["error"]:
        raise HTTPException(status_code=500, detail=data["error"])

    indices_raw = data.get("indices", {})
    result = {
        "indices": {
            name: serialize_series(series)
            for name, series in indices_raw.items()
        },
        "timeframe": timeframe,
        "timestamp": serialize_value(data.get("timestamp")),
    }

    try:
        from index_dashboard import INDEX_ORDER
        result["index_order"] = INDEX_ORDER
    except ImportError:
        result["index_order"] = list(indices_raw.keys())

    set_cached(short_cache, key, result)
    return result
