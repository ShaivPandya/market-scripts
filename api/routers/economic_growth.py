from fastapi import APIRouter, HTTPException
from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_response

router = APIRouter()


@router.get("/economic-growth")
def get_economic_growth():
    key = "economic_growth"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from economic_growth import get_data
        data = get_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result
