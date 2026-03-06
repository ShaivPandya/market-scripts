from fastapi import APIRouter

from api.cache import get_cached, long_cache, set_cached
from api.exceptions import DataFetchError
from api.serializers import serialize_response

router = APIRouter()


@router.get("/portfolio-news")
def get_portfolio_news(refresh: bool = False):
    key = "portfolio_news"
    if not refresh:
        cached = get_cached(long_cache, key)
        if cached is not None:
            return cached
    try:
        from portfolio.portfolio_news import get_data

        data = get_data(refresh=refresh)
    except Exception as e:
        raise DataFetchError(source="portfolio_news", detail=str(e)) from e

    result = serialize_response(data)
    set_cached(long_cache, key, result)
    return result
