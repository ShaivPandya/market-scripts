from fastapi import APIRouter

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
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
        raise DataFetchError(source="breakout", detail=str(e)) from e
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result
