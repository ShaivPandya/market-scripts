from typing import Any

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
        key = "portfolio:all_timeframes"
    else:
        if timeframe not in VALID_TIMEFRAMES:
            timeframe = "Daily"
        key = f"portfolio:{timeframe}"

    def loader():
        try:
            positions = OntologyRuntimeReadService().positions(include_hedges=True)
        except Exception as e:
            raise DataFetchError(source="portfolio", detail=str(e)) from e

        position_order = [str(row.get("ticker") or row.get("id") or "") for row in positions if row.get("ticker")]
        analytics = {
            "portfolio": {"position_count": len(positions)},
            "per_position": {
                str(row.get("ticker") or row.get("id")): {
                    "weight": row.get("weight"),
                    "current_notional": row.get("notional_base"),
                    "cost_notional": row.get("cost_basis_base"),
                }
                for row in positions
            },
        }

        if all_timeframes:
            result: dict[str, Any] = {
                "timeframes": {
                    "Current": {
                        "positions": {},
                        "metadata": {"source": "ontology"},
                        "timeframe": "Current",
                        "timestamp": None,
                        "position_order": position_order,
                    }
                },
                "timestamp": None,
            }
            result["analytics"] = serialize_value(analytics)
        else:
            result = {
                "positions": {},
                "metadata": {"source": "ontology", "position_count": len(positions)},
                "timeframe": timeframe,
                "timestamp": None,
                "position_order": position_order,
                "analytics": serialize_value(analytics),
                "holdings": positions,
            }

        return result

    return get_or_set_cached(short_cache, key, loader)
