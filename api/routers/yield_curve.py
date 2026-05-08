from fastapi import APIRouter, HTTPException

from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_response

router = APIRouter()
SUPPORTED_COUNTRIES = {"US", "UK", "DE", "JP"}


@router.get("/yield-curve")
def get_yield_curve(lookback_days: int = 90, country: str | None = None):
    if lookback_days < 1 or lookback_days > 3650:
        raise HTTPException(status_code=400, detail="lookback_days must be between 1 and 3650")

    country_code = country.strip().upper() if country else None
    if country_code and country_code not in SUPPORTED_COUNTRIES:
        supported = ", ".join(sorted(SUPPORTED_COUNTRIES))
        raise HTTPException(status_code=400, detail=f"country must be one of: {supported}")

    key = f"yield_curve:{lookback_days}:{country_code or 'ALL'}"

    def loader():
        try:
            from government_bonds.yield_curve import get_data

            data = get_data(lookback_days=lookback_days, country=country_code)
        except Exception as exc:
            raise DataFetchError(source="yield_curve", detail=str(exc)) from exc

        return serialize_response(data)

    return get_or_set_cached(short_cache, key, loader)
