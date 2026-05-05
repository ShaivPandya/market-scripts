from fastapi import APIRouter

from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_response

router = APIRouter()


@router.get("/bond-dashboard")
def get_bond_dashboard():
    key = "bond_dashboard"

    def loader():
        try:
            from government_bonds.bond_dashboard import get_data

            data = get_data()
        except Exception as exc:
            raise DataFetchError(source="bond_dashboard", detail=str(exc)) from exc

        return serialize_response(data)

    return get_or_set_cached(short_cache, key, loader)
