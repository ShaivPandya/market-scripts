from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import get_cached, long_cache, set_cached
from api.exceptions import DataFetchError
from api.serializers import serialize_value

router = APIRouter()


class FinancialsRequest(BaseModel):
    ticker: str


@router.post("/financials")
def run_financials(req: FinancialsRequest):
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    key = f"financials:v8:{ticker}"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached

    legacy_keys = [
        f"financials:v7:{ticker}",
        f"financials:v6:{ticker}",
        f"financials:v5:{ticker}",
        f"financials:v4:{ticker}",
        f"financials:v3:{ticker}",
    ]
    legacy_cached = None
    for old_key in legacy_keys:
        old = get_cached(long_cache, old_key)
        if old is not None:
            legacy_cached = old
            break

    try:
        from portfolio.momentum.fundamental_momentum.financials_single import get_data

        data = get_data(ticker)
    except ValueError as e:
        if legacy_cached is not None:
            return legacy_cached
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904
    except HTTPException:
        if legacy_cached is not None:
            return legacy_cached
        raise
    except Exception as e:
        if legacy_cached is not None:
            return legacy_cached
        raise DataFetchError(source="financials", detail=str(e)) from e

    result = serialize_value(data)
    set_cached(long_cache, key, result)
    return result
