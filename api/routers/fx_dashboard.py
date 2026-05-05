from fastapi import APIRouter

from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_series, serialize_value

router = APIRouter()

VALID_TIMEFRAMES = {"This Week", "Daily", "Weekly", "Monthly"}


@router.get("/fx-dashboard")
def get_fx_dashboard(timeframe: str = "Daily"):
    if timeframe not in VALID_TIMEFRAMES:
        timeframe = "Daily"
    key = f"fx_dashboard:{timeframe}"

    def loader():
        try:
            from fx.fx_dashboard.fx_dashboard import get_data

            data = get_data(timeframe=timeframe)
        except Exception as e:
            raise DataFetchError(source="fx_dashboard", detail=str(e)) from e

        if "error" in data and data["error"]:
            raise DataFetchError(source="fx_dashboard", detail=data["error"])

        pairs_raw = data.get("pairs", {})
        result = {
            "pairs": {name: serialize_series(series) for name, series in pairs_raw.items()},
            "timeframe": timeframe,
            "timestamp": serialize_value(data.get("timestamp")),
        }

        try:
            from fx.fx_dashboard.fx_dashboard import PAIR_ORDER

            result["pair_order"] = PAIR_ORDER
        except ImportError:
            result["pair_order"] = list(pairs_raw.keys())

        return result

    return get_or_set_cached(short_cache, key, loader)
