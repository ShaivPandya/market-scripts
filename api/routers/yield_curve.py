from fastapi import APIRouter, HTTPException

from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_response

router = APIRouter()


@router.get("/yield-curve")
def get_yield_curve(lookback_days: int = 90):
    if lookback_days < 1 or lookback_days > 3650:
        raise HTTPException(status_code=400, detail="lookback_days must be between 1 and 3650")

    key = f"yield_curve:{lookback_days}"

    def loader():
        try:
            from government_bonds.yield_curve import get_data

            data = get_data(lookback_days=lookback_days)
        except Exception as exc:
            raise DataFetchError(source="yield_curve", detail=str(exc)) from exc

        return serialize_response(data)

    return get_or_set_cached(short_cache, key, loader)
