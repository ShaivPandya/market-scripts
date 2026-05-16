from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import get_or_set_cached, long_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_value
from ontology.sources.source_registry import attach_source_registry_metadata

router = APIRouter()


class FinancialsRequest(BaseModel):
    ticker: str


@router.post("/financials")
def run_financials(req: FinancialsRequest):
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    key = f"financials:v13:{ticker}"

    def loader():
        try:
            from portfolio.momentum.fundamental_momentum.financials_single import get_data

            data = get_data(ticker)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))  # noqa: B904
        except Exception as e:
            raise DataFetchError(source="financials", detail=str(e)) from e

        return attach_source_registry_metadata(serialize_value(data), source_id="financials")

    return get_or_set_cached(long_cache, key, loader)
