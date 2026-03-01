from fastapi import APIRouter, HTTPException
from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_series, serialize_value

router = APIRouter()

VALID_TIMEFRAMES = {"This Week", "Daily", "Weekly", "Monthly"}


@router.get("/commodities")
def get_commodities(timeframe: str = "Daily"):
    if timeframe not in VALID_TIMEFRAMES:
        timeframe = "Daily"
    key = f"commodities:{timeframe}"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from commodities_dashboard import get_data
        data = get_data(timeframe=timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in data and data["error"]:
        raise HTTPException(status_code=500, detail=data["error"])

    commodities_raw = data.get("commodities", {})
    result = {
        "commodities": {
            name: serialize_series(series)
            for name, series in commodities_raw.items()
        },
        "timeframe": timeframe,
        "timestamp": serialize_value(data.get("timestamp")),
    }

    try:
        from commodities_dashboard import COMMODITY_ORDER
        result["commodity_order"] = COMMODITY_ORDER
    except ImportError:
        result["commodity_order"] = list(commodities_raw.keys())

    set_cached(short_cache, key, result)
    return result
