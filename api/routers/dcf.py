"""DCF valuation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.cache import get_or_set_cached, long_cache, stamp_fresh
from api.exceptions import DataFetchError
from api.serializers import serialize_value

router = APIRouter()


# ---------------------------------------------------------------------------
# GET — historical data for the Historical tab
# ---------------------------------------------------------------------------


@router.get("/dcf/historical/{ticker}")
def get_dcf_historical(ticker: str):
    """Fetch historical financials + multiples for DCF Historical tab."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    key = f"dcf_historical:v1:{ticker}"

    def loader():
        try:
            from equities.valuation.dcf import get_historical_data

            data = get_historical_data(ticker)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise DataFetchError(source="dcf_historical", detail=str(e)) from e

        return serialize_value(data)

    return get_or_set_cached(long_cache, key, loader)


# ---------------------------------------------------------------------------
# POST — run DCF valuation with user assumptions
# ---------------------------------------------------------------------------


class ScenarioMultiples(BaseModel):
    bear: float
    base: float
    bull: float


class TerminalGrowthRates(BaseModel):
    bear: float = 0.02
    base: float = 0.03
    bull: float = 0.04


class DCFValuationRequest(BaseModel):
    ticker: str
    revenue_growth_rates: list[float] = Field(..., min_length=5, max_length=5)
    ebitda_margin: float = Field(..., gt=0, lt=1)
    tax_rate: float = Field(0.21, ge=0, lt=1)
    da_pct_revenue: float = Field(..., ge=0, lt=1)
    nwc_pct_revenue: float = Field(..., ge=-1, le=1)
    capex_pct_revenue: float = Field(..., ge=0, lt=1)
    wacc: float = Field(..., gt=0, lt=1)
    terminal_growth_rates: TerminalGrowthRates = TerminalGrowthRates()
    exit_ev_ebitda: ScenarioMultiples
    exit_ev_revenue: ScenarioMultiples


@router.post("/dcf/valuation")
def run_dcf_valuation(req: DCFValuationRequest):
    """Run full DCF valuation with user-provided assumptions."""
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    try:
        from equities.valuation.dcf import run_valuation

        data = run_valuation(ticker, req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise DataFetchError(source="dcf_valuation", detail=str(e)) from e

    return stamp_fresh(serialize_value(data))
