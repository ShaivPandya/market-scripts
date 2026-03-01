from fastapi import APIRouter, HTTPException
from api.cache import long_cache, get_cached, set_cached
from api.serializers import serialize_response

router = APIRouter()


@router.get("/central-banks")
def get_central_banks(refresh: bool = False):
    key = f"central_banks:{refresh}"
    if not refresh:
        cached = get_cached(long_cache, key)
        if cached is not None:
            return cached
    try:
        from central_bank import get_data
        data = get_data(refresh=refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = serialize_response(data)
    set_cached(long_cache, key, result)
    return result
