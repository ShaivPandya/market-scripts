from fastapi import APIRouter, HTTPException
from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_response

router = APIRouter()


@router.get("/market-breadth")
def get_market_breadth():
    key = "market_breadth"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from market_breadth import get_data
        data = get_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result


@router.get("/top50-breadth")
def get_top50_breadth():
    key = "top50_breadth"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from top50_breadth import get_data
        data = get_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result


@router.get("/price-volume-signals")
def get_price_volume_signals():
    key = "price_volume_signals"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from price_volume_signals import get_data
        data = get_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result


@router.get("/vix-term-structure")
def get_vix_term_structure():
    key = "vix_term_structure"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from vix_term_structure import get_data
        data = get_data(tail=252, signals_count=20)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result
