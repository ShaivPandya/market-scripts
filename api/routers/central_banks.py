from fastapi import APIRouter

from api.cache import get_cached, long_cache, set_cached
from api.exceptions import DataFetchError
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
        from macro.central_banks.central_bank import get_data

        data = get_data(refresh=refresh)
    except Exception as e:
        raise DataFetchError(source="central_banks", detail=str(e)) from e

    result = serialize_response(data)
    set_cached(long_cache, key, result)
    return result
