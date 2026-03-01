from fastapi import APIRouter, HTTPException
from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_series, serialize_value

router = APIRouter()

VALID_TIMEFRAMES = {"This Week", "Daily", "Weekly", "Monthly"}


@router.get("/fx-dashboard")
def get_fx_dashboard(timeframe: str = "Daily"):
    if timeframe not in VALID_TIMEFRAMES:
        timeframe = "Daily"
    key = f"fx_dashboard:{timeframe}"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from fx_dashboard import get_data
        data = get_data(timeframe=timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in data and data["error"]:
        raise HTTPException(status_code=500, detail=data["error"])

    pairs_raw = data.get("pairs", {})
    result = {
        "pairs": {
            name: serialize_series(series)
            for name, series in pairs_raw.items()
        },
        "timeframe": timeframe,
        "timestamp": serialize_value(data.get("timestamp")),
    }

    try:
        from fx_dashboard import PAIR_ORDER
        result["pair_order"] = PAIR_ORDER
    except ImportError:
        result["pair_order"] = list(pairs_raw.keys())

    set_cached(short_cache, key, result)
    return result
