from fastapi import APIRouter

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


@router.get("/momentum")
def get_momentum():
    key = "momentum"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from portfolio.momentum.price_momentum.momentum import get_data

        data = get_data()
    except Exception as e:
        raise DataFetchError(source="momentum", detail=str(e)) from e

    result = {}
    for k, v in data.items():
        import pandas as pd

        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)

    set_cached(short_cache, key, result)
    return result
