from fastapi import APIRouter, HTTPException

from api.cache import get_cached, set_cached, short_cache
from api.serializers import serialize_response

router = APIRouter()

VALID_COMMODITIES = {"CL", "BZ", "NG"}


@router.get("/commodities-curve")
def get_commodities_curve(commodity: str = "CL", lookback_days: int = 30):
    if commodity not in VALID_COMMODITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid commodity. Must be one of: {', '.join(sorted(VALID_COMMODITIES))}",
        )
    if lookback_days < 1 or lookback_days > 365:
        raise HTTPException(
            status_code=400,
            detail="lookback_days must be between 1 and 365",
        )

    key = f"commodities_curve:{commodity}:{lookback_days}"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached

    try:
        from commodities_curve import get_data

        data = get_data(commodity=commodity, lookback_days=lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result
