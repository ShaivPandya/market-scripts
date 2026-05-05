from fastapi import APIRouter

from api.cache import get_or_set_cached, long_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_response

router = APIRouter()

CACHE_KEY = "central_banks"


def _error_detail(data) -> str | None:
    if not isinstance(data, dict):
        return None
    detail = data.get("error")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return None


@router.get("/central-banks")
def get_central_banks(refresh: bool = False):
    def loader():
        try:
            from macro.central_banks.central_bank import get_data

            data = get_data(refresh=refresh)
        except Exception as e:
            raise DataFetchError(source="central_banks", detail=str(e)) from e

        detail = _error_detail(data)
        if detail:
            raise DataFetchError(source="central_banks", detail=detail)

        return serialize_response(data)

    return get_or_set_cached(long_cache, CACHE_KEY, loader, force_refresh=refresh)
