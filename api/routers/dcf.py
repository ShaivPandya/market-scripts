"""DCF valuation endpoints."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, model_validator

from api.cache import get_or_set_cached, long_cache, stamp_fresh
from api.exceptions import DataFetchError
from api.serializers import serialize_value
from ontology.sources.source_registry import attach_source_registry_metadata

router = APIRouter()
XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _xlsx_download_filename(ticker: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in ticker.strip().upper()).strip("_")
    return f"{clean or 'ticker'}_dcf_model.xlsx"


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

        return attach_source_registry_metadata(serialize_value(data), source_id="dcf_historical")

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
    revenue_growth_rates: list[float] = Field(..., min_length=5, max_length=8)
    ebitda_margin: float | list[float]
    tax_rate: float | list[float] = 0.21
    da_pct_revenue: float | list[float]
    nwc_pct_revenue: float | list[float]
    capex_pct_revenue: float | list[float]
    wacc: float = Field(..., gt=0, lt=1)
    terminal_growth_rates: TerminalGrowthRates = TerminalGrowthRates()
    exit_ev_ebitda: ScenarioMultiples
    exit_ev_revenue: ScenarioMultiples

    @staticmethod
    def _normalize_yearly(
        field_name: str,
        value: float | list[float],
        years: int,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
    ) -> list[float]:
        values = value if isinstance(value, list) else [value] * years
        if len(values) != years:
            raise ValueError(f"{field_name} must have {years} values")

        normalized: list[float] = []
        for v in values:
            if not math.isfinite(v):
                raise ValueError(f"{field_name} must contain only finite numbers")
            if min_value is not None:
                if min_inclusive and v < min_value:
                    raise ValueError(f"{field_name} values must be >= {min_value}")
                if not min_inclusive and v <= min_value:
                    raise ValueError(f"{field_name} values must be > {min_value}")
            if max_value is not None:
                if max_inclusive and v > max_value:
                    raise ValueError(f"{field_name} values must be <= {max_value}")
                if not max_inclusive and v >= max_value:
                    raise ValueError(f"{field_name} values must be < {max_value}")
            normalized.append(v)
        return normalized

    @model_validator(mode="after")
    def normalize_yearly_assumptions(self):
        years = len(self.revenue_growth_rates)
        if any(not math.isfinite(v) for v in self.revenue_growth_rates):
            raise ValueError("revenue_growth_rates must contain only finite numbers")

        self.ebitda_margin = self._normalize_yearly(
            "ebitda_margin",
            self.ebitda_margin,
            years,
            min_value=0,
            max_value=1,
            min_inclusive=False,
            max_inclusive=False,
        )
        self.tax_rate = self._normalize_yearly(
            "tax_rate",
            self.tax_rate,
            years,
            min_value=0,
            max_value=1,
            max_inclusive=False,
        )
        self.da_pct_revenue = self._normalize_yearly(
            "da_pct_revenue",
            self.da_pct_revenue,
            years,
            min_value=0,
            max_value=1,
            max_inclusive=False,
        )
        self.nwc_pct_revenue = self._normalize_yearly(
            "nwc_pct_revenue",
            self.nwc_pct_revenue,
            years,
            min_value=-1,
            max_value=1,
        )
        self.capex_pct_revenue = self._normalize_yearly(
            "capex_pct_revenue",
            self.capex_pct_revenue,
            years,
            min_value=0,
            max_value=1,
            max_inclusive=False,
        )
        return self


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

    return attach_source_registry_metadata(stamp_fresh(serialize_value(data)), source_id="dcf_valuation")


@router.post("/dcf/valuation/excel")
def download_dcf_valuation_excel(req: DCFValuationRequest):
    """Download a formula-driven Excel workbook for a DCF valuation."""
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    try:
        from equities.valuation.dcf import get_historical_data, run_valuation
        from equities.valuation.dcf_excel import build_dcf_workbook_bytes

        assumptions = req.model_dump()
        assumptions["ticker"] = ticker
        valuation = run_valuation(ticker, assumptions)
        historical = get_historical_data(ticker)
        workbook_bytes = build_dcf_workbook_bytes(valuation, historical)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise DataFetchError(source="dcf_excel", detail=str(e)) from e

    return Response(
        content=workbook_bytes,
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{_xlsx_download_filename(ticker)}"'},
    )
