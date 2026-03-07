from fastapi import APIRouter

from api.cache import get_cached, long_cache, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_value

router = APIRouter()


@router.get("/sentiment/put-call")
def get_put_call(lookback_days: int = 180):
    key = f"sentiment_put_call:{lookback_days}"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from sentiment import get_put_call

        data = get_put_call(lookback_days=lookback_days)
    except Exception as e:
        raise DataFetchError(source="sentiment_put_call", detail=str(e)) from e
    result = serialize_value(data)
    set_cached(short_cache, key, result)
    return result


@router.get("/sentiment/surveys")
def get_surveys():
    key = "sentiment_surveys"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached
    try:
        from sentiment import get_surveys

        data = get_surveys()
    except Exception as e:
        raise DataFetchError(source="sentiment_surveys", detail=str(e)) from e
    result = serialize_value(data)
    set_cached(long_cache, key, result)
    return result


@router.get("/sentiment/volatility")
def get_volatility(lookback_days: int = 365):
    key = f"sentiment_volatility:{lookback_days}"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from sentiment import get_volatility

        data = get_volatility(lookback_days=lookback_days)
    except Exception as e:
        raise DataFetchError(source="sentiment_volatility", detail=str(e)) from e
    result = serialize_value(data)
    set_cached(short_cache, key, result)
    return result
