from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import long_cache, get_cached, set_cached
from api.serializers import serialize_value

router = APIRouter()


class FinancialsRequest(BaseModel):
    ticker: str


@router.post("/financials")
def run_financials(req: FinancialsRequest):
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    key = f"financials:v3:{ticker}"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached

    try:
        from financials_single import get_data

        data = get_data(ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = serialize_value(data)
    set_cached(long_cache, key, result)
    return result
