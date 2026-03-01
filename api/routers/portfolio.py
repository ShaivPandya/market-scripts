from fastapi import APIRouter, HTTPException
from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_series, serialize_value

router = APIRouter()

VALID_TIMEFRAMES = {"This Week", "Daily", "Weekly", "Monthly"}


@router.get("/portfolio")
def get_portfolio(timeframe: str = "Daily"):
    if timeframe not in VALID_TIMEFRAMES:
        timeframe = "Daily"
    key = f"portfolio:{timeframe}"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from portfolio_dashboard import get_data
        data = get_data(timeframe=timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in data and data["error"]:
        raise HTTPException(status_code=500, detail=data["error"])

    positions_raw = data.get("positions", {})
    result = {
        "positions": {
            ticker: serialize_series(series)
            for ticker, series in positions_raw.items()
        },
        "metadata": serialize_value(data.get("metadata", {})),
        "timeframe": timeframe,
        "timestamp": data["timestamp"].isoformat() if data.get("timestamp") else None,
    }

    # Include ordering / display name lists if the module exposes them
    try:
        from portfolio_dashboard import POSITION_ORDER
        result["position_order"] = POSITION_ORDER
    except ImportError:
        result["position_order"] = list(positions_raw.keys())

    set_cached(short_cache, key, result)
    return result
