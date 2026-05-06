from fastapi import APIRouter

from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_value
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()

VALID_TIMEFRAMES = {"This Week", "Daily", "Weekly", "Monthly"}


@router.get("/portfolio")
def get_portfolio(timeframe: str = "Daily", all_timeframes: bool = False):
    if all_timeframes:
        key = "portfolio:all_timeframes:v2"
    else:
        if timeframe not in VALID_TIMEFRAMES:
            timeframe = "Daily"
        key = f"portfolio:v2:{timeframe}"

    def loader():
        try:
            from portfolio import portfolio_dashboard

            data = portfolio_dashboard.get_data(timeframe=timeframe, all_timeframes=all_timeframes)
        except Exception as e:
            raise DataFetchError(source="portfolio", detail=str(e)) from e

        if "error" in data and data["error"]:
            raise DataFetchError(source="portfolio", detail=data["error"])

        result = serialize_value(data)

        try:
            holdings = OntologyRuntimeReadService().positions(include_hedges=True)
        except Exception:
            holdings = []
        if isinstance(result, dict) and holdings:
            result["holdings"] = serialize_value(holdings)

        return result

    return get_or_set_cached(short_cache, key, loader)
