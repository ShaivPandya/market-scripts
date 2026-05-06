from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from api.action_execution import stage_api_action
from api.exceptions import AppError, DataFetchError
from ontology.runtime_read_service import OntologyRuntimeReadService
from portfolio.instruments import (
    default_contract_multiplier,
    is_continuous_future_symbol,
    normalize_asset,
    normalize_instrument_type,
    normalize_quantity,
    normalize_symbol,
)

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.=-]{0,31}$")


class PortfolioPosition(BaseModel):
    ticker: str
    asset: Literal["equity", "commodity", "fx", "bond"] | None = None
    direction: Literal["long", "short"]
    contrarian: bool = False
    conviction: int = Field(default=3, ge=1, le=5)
    cost_basis: float | None = None
    shares: float | None = None
    quantity: float | None = None
    instrument_type: Literal["security", "future"] | None = None
    price_symbol: str | None = None
    contract_multiplier: float | None = None
    currency: str | None = None
    country: str | None = None
    exchange: str | None = None
    base_currency: str | None = None
    fx_rate_to_base: float | None = None
    fx_rate_as_of: str | None = None
    cost_basis_base: float | None = None
    notional_base: float | None = None
    valuation_status: str | None = None

    @model_validator(mode="after")
    def _normalize_instrument(self) -> PortfolioPosition:
        self.ticker = normalize_symbol(self.ticker)
        self.price_symbol = normalize_symbol(self.price_symbol or self.ticker, field_name="price_symbol")
        self.instrument_type = normalize_instrument_type(
            self.instrument_type,
            ticker=self.ticker,
            price_symbol=self.price_symbol,
        )
        if self.instrument_type == "future" and not is_continuous_future_symbol(self.price_symbol):
            raise ValueError("Futures positions require a continuous '=F' price_symbol.")
        self.asset = normalize_asset(self.asset, instrument_type=self.instrument_type, symbol=self.price_symbol)
        self.contract_multiplier = default_contract_multiplier(
            instrument_type=self.instrument_type,
            symbol=self.price_symbol,
            override=self.contract_multiplier,
        )
        self.quantity = normalize_quantity(quantity=self.quantity, shares=self.shares)
        self.shares = self.quantity
        return self


class PortfolioUpdateRequest(BaseModel):
    positions: list[PortfolioPosition]
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.get("/portfolio-positions")
def get_portfolio_positions(include_hedges: bool = False):
    try:
        return {"positions": OntologyRuntimeReadService().positions(include_hedges=include_hedges)}
    except Exception as e:
        raise DataFetchError(source="portfolio_positions", detail=str(e)) from e


@router.put("/portfolio-positions")
def update_portfolio_positions(req: PortfolioUpdateRequest):
    try:
        result = stage_api_action(
            "update_portfolio_positions",
            {"positions": [position.model_dump() for position in req.positions]},
            source_id="portfolio_edit.update_portfolio_positions",
            reason=req.reason,
            apply=req.apply,
            approval_note=req.approval_note,
            validation_status_code=400,
        )
    except (HTTPException, AppError):
        raise
    except Exception as e:
        raise DataFetchError(source="portfolio_positions", detail=str(e)) from e

    return result


def _flatten_object(row: dict) -> dict:
    props = dict(row.get("properties") or row.get("properties_json") or {})
    props["id"] = str(row.get("object_uid") or props.get("id") or "")
    props["object_uid"] = props["id"]
    return props


# ---------------------------------------------------------------------------
# Hedge positions
# ---------------------------------------------------------------------------


class HedgePosition(BaseModel):
    ticker: str
    direction: Literal["long", "short"]
    cost_basis: float | None = None
    shares: float | None = None
    quantity: float | None = None
    asset: Literal["equity", "commodity", "fx", "bond"] | None = None
    instrument_type: Literal["security", "future"] | None = None
    price_symbol: str | None = None
    contract_multiplier: float | None = None
    currency: str | None = None
    country: str | None = None
    exchange: str | None = None
    base_currency: str | None = None
    fx_rate_to_base: float | None = None
    fx_rate_as_of: str | None = None
    cost_basis_base: float | None = None
    notional_base: float | None = None
    valuation_status: str | None = None

    @model_validator(mode="after")
    def _normalize_instrument(self) -> HedgePosition:
        self.ticker = normalize_symbol(self.ticker)
        self.price_symbol = normalize_symbol(self.price_symbol or self.ticker, field_name="price_symbol")
        self.instrument_type = normalize_instrument_type(
            self.instrument_type,
            ticker=self.ticker,
            price_symbol=self.price_symbol,
        )
        if self.instrument_type == "future" and not is_continuous_future_symbol(self.price_symbol):
            raise ValueError("Futures hedge positions require a continuous '=F' price_symbol.")
        self.asset = normalize_asset(self.asset, instrument_type=self.instrument_type, symbol=self.price_symbol)
        self.contract_multiplier = default_contract_multiplier(
            instrument_type=self.instrument_type,
            symbol=self.price_symbol,
            override=self.contract_multiplier,
        )
        self.quantity = normalize_quantity(quantity=self.quantity, shares=self.shares)
        self.shares = self.quantity
        return self


class HedgeUpdateRequest(BaseModel):
    positions: list[HedgePosition]
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.get("/hedge-positions")
def get_hedge_positions_endpoint():
    try:
        rows = OntologyRuntimeReadService().list_objects("HedgePosition", limit=1000)
        return {"positions": rows}
    except Exception as e:
        raise DataFetchError(source="hedge_positions", detail=str(e)) from e


@router.put("/hedge-positions")
def update_hedge_positions(req: HedgeUpdateRequest):
    try:
        result = stage_api_action(
            "update_hedge_positions",
            {"positions": [position.model_dump() for position in req.positions]},
            source_id="portfolio_edit.update_hedge_positions",
            reason=req.reason,
            apply=req.apply,
            approval_note=req.approval_note,
            validation_status_code=400,
        )
    except (HTTPException, AppError):
        raise
    except Exception as e:
        raise DataFetchError(source="hedge_positions", detail=str(e)) from e

    return result
