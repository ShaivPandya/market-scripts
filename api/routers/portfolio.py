from typing import Any

from fastapi import APIRouter

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_series, serialize_value

router = APIRouter()

VALID_TIMEFRAMES = {"This Week", "Daily", "Weekly", "Monthly"}


@router.get("/portfolio")
def get_portfolio(timeframe: str = "Daily", all_timeframes: bool = False):
    if all_timeframes:
        key = "portfolio:all_timeframes"
    else:
        if timeframe not in VALID_TIMEFRAMES:
            timeframe = "Daily"
        key = f"portfolio:{timeframe}"

    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached

    try:
        from portfolio.portfolio_dashboard import get_data

        data = get_data(timeframe=timeframe, all_timeframes=all_timeframes)
    except Exception as e:
        raise DataFetchError(source="portfolio", detail=str(e)) from e

    if "error" in data and data["error"]:
        raise DataFetchError(source="portfolio", detail=data["error"])

    # Include ordering / display name lists if the module exposes them
    try:
        from portfolio.portfolio_dashboard import POSITION_ORDER

        position_order = POSITION_ORDER
    except ImportError:
        position_order = []

    if all_timeframes:
        timeframes = data.get("timeframes", {})
        result: dict[str, Any] = {
            "timeframes": {},
            "timestamp": data["timestamp"].isoformat() if data.get("timestamp") else None,
        }
        if data.get("warning"):
            result["warning"] = data["warning"]

        result["analytics"] = serialize_value(data.get("analytics", {}))

        for tf_name, tf_data in timeframes.items():
            positions_raw = tf_data.get("positions", {})
            tf_result = {
                "positions": {ticker: serialize_series(series) for ticker, series in positions_raw.items()},
                "metadata": serialize_value(tf_data.get("metadata", {})),
                "timeframe": tf_name,
                "timestamp": tf_data["timestamp"].isoformat() if tf_data.get("timestamp") else None,
                "position_order": position_order or list(positions_raw.keys()),
            }
            if tf_data.get("warning"):
                tf_result["warning"] = tf_data["warning"]
            result["timeframes"][tf_name] = tf_result
    else:
        positions_raw = data.get("positions", {})
        result = {
            "positions": {ticker: serialize_series(series) for ticker, series in positions_raw.items()},
            "metadata": serialize_value(data.get("metadata", {})),
            "timeframe": timeframe,
            "timestamp": data["timestamp"].isoformat() if data.get("timestamp") else None,
            "position_order": position_order or list(positions_raw.keys()),
            "analytics": serialize_value(data.get("analytics", {})),
        }
        if data.get("warning"):
            result["warning"] = data["warning"]

    set_cached(short_cache, key, result)
    return result
