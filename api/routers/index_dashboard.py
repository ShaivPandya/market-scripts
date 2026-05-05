from fastapi import APIRouter

from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_series, serialize_value

router = APIRouter()

VALID_TIMEFRAMES = {"This Week", "Daily", "Weekly", "Monthly"}


@router.get("/index-dashboard")
def get_index_dashboard(timeframe: str = "Daily"):
    if timeframe not in VALID_TIMEFRAMES:
        timeframe = "Daily"
    key = f"index_dashboard:{timeframe}"

    def loader():
        try:
            from equities.index_dashboard.index_dashboard import get_data

            data = get_data(timeframe=timeframe)
        except Exception as e:
            raise DataFetchError(source="index_dashboard", detail=str(e)) from e

        if "error" in data and data["error"]:
            raise DataFetchError(source="index_dashboard", detail=data["error"])

        indices_raw = data.get("indices", {})
        result = {
            "indices": {name: serialize_series(series) for name, series in indices_raw.items()},
            "timeframe": timeframe,
            "timestamp": serialize_value(data.get("timestamp")),
        }

        try:
            from equities.index_dashboard.index_dashboard import INDEX_ORDER

            result["index_order"] = INDEX_ORDER
        except ImportError:
            result["index_order"] = list(indices_raw.keys())

        return result

    return get_or_set_cached(short_cache, key, loader)
