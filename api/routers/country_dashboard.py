from fastapi import APIRouter

from api.cache import get_or_set_cached, long_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_series, serialize_value

router = APIRouter()

VALID_METRICS = {"Inflation", "Unemployment", "GDP"}


@router.get("/country-dashboard")
def get_country_dashboard(metric: str = "Inflation"):
    if metric not in VALID_METRICS:
        metric = "Inflation"
    key = f"country_dashboard:{metric}"

    def loader():
        try:
            from macro.country_dashboard.country_dashboard import COUNTRY_ORDER, get_data

            data = get_data(metric=metric)
        except Exception as e:
            raise DataFetchError(source="country_dashboard", detail=str(e)) from e

        countries_raw = data.get("countries", {})
        return {
            "countries": {
                name: (serialize_series(series) if series is not None and not series.empty else None)
                for name, series in countries_raw.items()
            },
            "country_order": COUNTRY_ORDER,
            "errors": data.get("errors", {}),
            "series_used": data.get("series_used", {}),
            "latest_observation_dates": {
                k: (v.isoformat() if hasattr(v, "isoformat") else str(v))
                for k, v in data.get("latest_observation_dates", {}).items()
            },
            "timestamp": data["timestamp"].isoformat() if data.get("timestamp") else None,
            "metric": metric,
        }

    return get_or_set_cached(long_cache, key, loader)
