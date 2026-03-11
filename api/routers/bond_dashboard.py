from fastapi import APIRouter

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_response

router = APIRouter()


@router.get("/bond-dashboard")
def get_bond_dashboard():
    key = "bond_dashboard"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached

    try:
        from government_bonds.bond_dashboard import get_data

        data = get_data()
    except Exception as exc:
        raise DataFetchError(source="bond_dashboard", detail=str(exc)) from exc

    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result
