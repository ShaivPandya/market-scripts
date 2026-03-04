from fastapi import APIRouter, HTTPException
from api.cache import long_cache, get_cached, set_cached
from api.serializers import serialize_response

router = APIRouter()


@router.get("/portfolio-news")
def get_portfolio_news(refresh: bool = False):
    key = f"portfolio_news:{refresh}"
    if not refresh:
        cached = get_cached(long_cache, key)
        if cached is not None:
            return cached
    try:
        from portfolio.portfolio_news import get_data
        data = get_data(refresh=refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = serialize_response(data)
    set_cached(long_cache, key, result)
    return result
