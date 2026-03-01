from fastapi import APIRouter, HTTPException
from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_response

router = APIRouter()


@router.get("/liquidity")
def get_liquidity(skip_ecb: bool = False):
    key = f"liquidity:{skip_ecb}"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from liquidity import get_snapshot
        data = get_snapshot(skip_ecb=skip_ecb)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Drop large DataFrame/Series objects that React doesn't need
    # (composite_series and df_weekly are internal computation artifacts)
    filtered = {
        k: v for k, v in data.items()
        if k not in ("df_weekly", "composite_series")
    }
    result = serialize_response(filtered)
    set_cached(short_cache, key, result)
    return result
