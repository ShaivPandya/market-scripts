from fastapi import APIRouter

from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_response

router = APIRouter()


@router.get("/breakout")
def get_breakout():
    key = "breakout"

    def loader():
        try:
            from macro.breakout.breakout import get_data

            data = get_data()
        except Exception as e:
            raise DataFetchError(source="breakout", detail=str(e)) from e
        return serialize_response(data)

    return get_or_set_cached(short_cache, key, loader)
