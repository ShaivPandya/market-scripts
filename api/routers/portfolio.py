import hashlib
import json
from typing import Any

from fastapi import APIRouter

from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_value
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()

VALID_TIMEFRAMES = {"This Week", "Daily", "Weekly", "Monthly"}
CACHE_VERSION = "v3"


def _current_holdings() -> list[Any]:
    try:
        return serialize_value(OntologyRuntimeReadService().positions(include_hedges=True))
    except Exception:
        return []


def _holdings_cache_token(holdings: list[Any]) -> str:
    try:
        encoded = json.dumps(holdings, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        encoded = repr(holdings)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@router.get("/portfolio")
def get_portfolio(timeframe: str = "Daily", all_timeframes: bool = False):
    holdings = _current_holdings()
    holdings_token = _holdings_cache_token(holdings)

    if all_timeframes:
        key = f"portfolio:all_timeframes:{CACHE_VERSION}:{holdings_token}"
    else:
        if timeframe not in VALID_TIMEFRAMES:
            timeframe = "Daily"
        key = f"portfolio:{CACHE_VERSION}:{timeframe}:{holdings_token}"

    def loader():
        try:
            from portfolio import portfolio_dashboard

            data = portfolio_dashboard.get_data(timeframe=timeframe, all_timeframes=all_timeframes)
        except Exception as e:
            raise DataFetchError(source="portfolio", detail=str(e)) from e

        if "error" in data and data["error"]:
            raise DataFetchError(source="portfolio", detail=data["error"])

        result = serialize_value(data)

        if isinstance(result, dict) and holdings:
            result["holdings"] = holdings

        return result

    return get_or_set_cached(short_cache, key, loader)
