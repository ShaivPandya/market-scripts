from fastapi import APIRouter, HTTPException
from api.cache import long_cache, get_cached, set_cached
from api.serializers import serialize_series, serialize_value

router = APIRouter()

VALID_METRICS = {"Inflation", "Unemployment", "GDP"}


@router.get("/country-dashboard")
def get_country_dashboard(metric: str = "Inflation"):
    if metric not in VALID_METRICS:
        metric = "Inflation"
    key = f"country_dashboard:{metric}"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached
    try:
        from country_dashboard import get_data, COUNTRY_ORDER
        data = get_data(metric=metric)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    countries_raw = data.get("countries", {})
    result = {
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
    set_cached(long_cache, key, result)
    return result
