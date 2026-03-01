from fastapi import APIRouter, HTTPException
from api.cache import long_cache, get_cached, set_cached
from api.serializers import serialize_response

router = APIRouter()


@router.get("/industry-monitor")
def get_industry_monitor(refresh: bool = False):
    key = f"industry_monitor:{refresh}"
    if not refresh:
        cached = get_cached(long_cache, key)
        if cached is not None:
            return cached
    try:
        from industry_monitor import get_data
        data = get_data(refresh=refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = serialize_response(data)
    set_cached(long_cache, key, result)
    return result
