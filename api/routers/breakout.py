from fastapi import APIRouter, HTTPException
from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_response

router = APIRouter()


@router.get("/breakout")
def get_breakout():
    key = "breakout"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from breakout import get_data
        data = get_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result
