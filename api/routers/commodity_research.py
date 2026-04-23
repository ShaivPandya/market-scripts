from fastapi import APIRouter

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_value

router = APIRouter()


@router.get("/commodity-research")
def get_commodity_research():
    key = "commodity_research:v3"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached

    try:
        from commodities.commodity_research import build_commodity_research

        data = build_commodity_research()
    except Exception as e:
        raise DataFetchError(source="commodity_research", detail=str(e)) from e

    result = serialize_value(data)
    set_cached(short_cache, key, result)
    return result
