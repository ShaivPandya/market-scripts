"""Position valuation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import delete_cached, get_or_set_cached, long_cache, stamp_fresh
from api.exceptions import DataFetchError
from api.serializers import serialize_value

router = APIRouter()


class ValuationProfileOverrideRequest(BaseModel):
    profile_id: str | None = None


class ValuationValueRangeScenarioRequest(BaseModel):
    multiple: float
    denominator: float


class ValuationValueRangeRequest(BaseModel):
    metric: str
    scenarios: dict[str, ValuationValueRangeScenarioRequest]


def valuation_cache_key(ticker: str, profile_override: str | None = None) -> str:
    profile = profile_override or "auto"
    return f"position_valuation:v2:{ticker.strip().upper()}:profile={profile}"


def delete_valuation_cache_variants(ticker: str, profile_overrides: list[str | None]) -> None:
    for profile_override in {None, *profile_overrides}:
        delete_cached(long_cache, valuation_cache_key(ticker, profile_override))


@router.get("/valuation/{ticker}")
def get_position_valuation_endpoint(ticker: str):
    normalized = ticker.strip().upper()
    if not normalized:
        raise HTTPException(status_code=400, detail="Ticker is required")

    try:
        from equities.valuation.multiples import get_position_valuation, read_profile_override

        override = read_profile_override(normalized)
        key = valuation_cache_key(normalized, override)

        def loader():
            return serialize_value(get_position_valuation(normalized))

        return get_or_set_cached(long_cache, key, loader)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise DataFetchError(source="position_valuation", detail=str(exc)) from exc


@router.put("/valuation/{ticker}/profile-override")
def update_position_valuation_profile_override(ticker: str, req: ValuationProfileOverrideRequest):
    normalized = ticker.strip().upper()
    if not normalized:
        raise HTTPException(status_code=400, detail="Ticker is required")

    try:
        from equities.valuation.multiples import read_profile_override, write_profile_override

        previous = read_profile_override(normalized)
        result = write_profile_override(normalized, req.profile_id)
        delete_valuation_cache_variants(normalized, [previous, result.get("profile_override")])
        return stamp_fresh(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise DataFetchError(source="position_valuation_profile_override", detail=str(exc)) from exc


@router.put("/valuation/{ticker}/value-range")
def update_position_valuation_value_range(ticker: str, req: ValuationValueRangeRequest):
    normalized = ticker.strip().upper()
    if not normalized:
        raise HTTPException(status_code=400, detail="Ticker is required")

    try:
        from equities.valuation.multiples import profile_options, read_profile_override, write_value_range_assumption

        profile_override = read_profile_override(normalized)
        result = write_value_range_assumption(normalized, req.model_dump(mode="python"))
        profile_ids = [option["id"] for option in profile_options()]
        delete_valuation_cache_variants(normalized, [profile_override, *profile_ids])
        return stamp_fresh(serialize_value(result))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise DataFetchError(source="position_valuation_value_range", detail=str(exc)) from exc
