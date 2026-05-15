from __future__ import annotations

import re
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from api.action_execution import stage_api_action
from api.audit import emit_audit_event
from api.exceptions import AppError, DataFetchError
from api.portfolio_settings import (
    DEFAULT_PORTFOLIO_BOOK_SIZE,
    MAX_PORTFOLIO_BOOK_SIZE,
    MIN_PORTFOLIO_BOOK_SIZE,
    get_configured_portfolio_book_size,
    get_portfolio_book_size,
    normalize_portfolio_book_size,
    set_portfolio_book_size,
)
from api.routers.auth import require_actor
from ontology.policy import Actor
from ontology.runtime_read_service import OntologyRuntimeReadService
from portfolio.instruments import (
    default_contract_multiplier,
    is_continuous_future_symbol,
    normalize_asset,
    normalize_instrument_type,
    normalize_quantity,
    normalize_spot_fx_symbol,
    normalize_symbol,
    spot_fx_currencies,
)
from portfolio.position_groups import (
    canonicalize_position_group_rows,
    normalize_position_group_fields,
    validate_position_groups,
)

router = APIRouter()
ActorDep = Annotated[Actor, Depends(require_actor)]

_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.=-]{0,31}$")


class PortfolioSettingsRequest(BaseModel):
    book_size: float = Field(ge=MIN_PORTFOLIO_BOOK_SIZE, le=MAX_PORTFOLIO_BOOK_SIZE)

    @model_validator(mode="after")
    def _normalize_book_size(self) -> PortfolioSettingsRequest:
        self.book_size = normalize_portfolio_book_size(self.book_size)
        return self


def _portfolio_settings_response() -> dict:
    configured = get_configured_portfolio_book_size()
    return {
        "book_size": configured or get_portfolio_book_size(),
        "default_book_size": DEFAULT_PORTFOLIO_BOOK_SIZE,
        "configured": configured is not None,
        "min_book_size": MIN_PORTFOLIO_BOOK_SIZE,
        "max_book_size": MAX_PORTFOLIO_BOOK_SIZE,
    }


@router.get("/portfolio-settings")
def get_portfolio_settings():
    return _portfolio_settings_response()


@router.put("/portfolio-settings")
def update_portfolio_settings(req: PortfolioSettingsRequest, actor: ActorDep):
    try:
        before = _portfolio_settings_response()
        set_portfolio_book_size(req.book_size)
        after = _portfolio_settings_response()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    emit_audit_event(
        "portfolio.settings.updated",
        "permission",
        "succeeded",
        actor=actor,
        before_summary={
            "book_size": before.get("book_size"),
            "configured": before.get("configured"),
        },
        after_summary={
            "book_size": after.get("book_size"),
            "configured": after.get("configured"),
        },
    )
    return after


class PortfolioPosition(BaseModel):
    ticker: str
    asset: Literal["equity", "commodity", "fx", "bond"] | None = None
    direction: Literal["long", "short"]
    contrarian: bool = False
    conviction: int = Field(default=3, ge=1, le=5)
    cost_basis: float | None = None
    shares: float | None = None
    quantity: float | None = None
    instrument_type: Literal["security", "future", "spot_fx"] | None = None
    price_symbol: str | None = None
    contract_multiplier: float | None = None
    fx_base_currency: str | None = None
    fx_quote_currency: str | None = None
    currency: str | None = None
    country: str | None = None
    exchange: str | None = None
    base_currency: str | None = None
    fx_rate_to_base: float | None = None
    fx_rate_as_of: str | None = None
    cost_basis_base: float | None = None
    notional_base: float | None = None
    valuation_status: str | None = None
    group_name: str | None = None
    group_conviction: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def _normalize_instrument(self) -> PortfolioPosition:
        self.instrument_type = normalize_instrument_type(
            self.instrument_type,
            ticker=str(self.ticker),
            price_symbol=str(self.price_symbol or self.ticker),
        )
        if self.instrument_type == "spot_fx":
            self.price_symbol = normalize_spot_fx_symbol(self.price_symbol or self.ticker, field_name="price_symbol")
            self.ticker = self.price_symbol
            self.fx_base_currency, self.fx_quote_currency = spot_fx_currencies(self.price_symbol)
            self.asset = "fx"
            self.currency = self.fx_quote_currency
            self.exchange = self.exchange or "FX"
        else:
            self.ticker = normalize_symbol(self.ticker)
            self.price_symbol = normalize_symbol(self.price_symbol or self.ticker, field_name="price_symbol")
        if self.instrument_type == "future" and not is_continuous_future_symbol(self.price_symbol):
            raise ValueError("Futures positions require a continuous '=F' price_symbol.")
        self.asset = normalize_asset(self.asset, instrument_type=self.instrument_type, symbol=self.price_symbol)
        self.contract_multiplier = default_contract_multiplier(
            instrument_type=self.instrument_type,
            symbol=self.price_symbol,
            override=self.contract_multiplier,
        )
        self.quantity = normalize_quantity(quantity=self.quantity, shares=self.shares, allow_negative=True)
        self.shares = self.quantity
        self.group_name, self.group_conviction = normalize_position_group_fields(self.model_dump())
        return self


class PortfolioUpdateRequest(BaseModel):
    positions: list[PortfolioPosition]
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None

    @model_validator(mode="after")
    def _validate_groups(self) -> PortfolioUpdateRequest:
        rows = canonicalize_position_group_rows([position.model_dump() for position in self.positions])
        validate_position_groups(rows)
        for position, row in zip(self.positions, rows, strict=False):
            position.group_name = row.get("group_name")
            position.group_conviction = row.get("group_conviction")
        return self


@router.get("/portfolio-positions")
def get_portfolio_positions(include_hedges: bool = False):
    try:
        return {"positions": OntologyRuntimeReadService().positions(include_hedges=include_hedges)}
    except Exception as e:
        raise DataFetchError(source="portfolio_positions", detail=str(e)) from e


@router.put("/portfolio-positions")
def update_portfolio_positions(req: PortfolioUpdateRequest, actor: ActorDep):
    try:
        result = stage_api_action(
            "update_portfolio_positions",
            {"positions": [position.model_dump() for position in req.positions]},
            source_id="portfolio_edit.update_portfolio_positions",
            actor=actor,
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
    instrument_type: Literal["security", "future", "spot_fx"] | None = None
    price_symbol: str | None = None
    contract_multiplier: float | None = None
    fx_base_currency: str | None = None
    fx_quote_currency: str | None = None
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
        self.instrument_type = normalize_instrument_type(
            self.instrument_type,
            ticker=str(self.ticker),
            price_symbol=str(self.price_symbol or self.ticker),
        )
        if self.instrument_type == "spot_fx":
            self.price_symbol = normalize_spot_fx_symbol(self.price_symbol or self.ticker, field_name="price_symbol")
            self.ticker = self.price_symbol
            self.fx_base_currency, self.fx_quote_currency = spot_fx_currencies(self.price_symbol)
            self.asset = "fx"
            self.currency = self.fx_quote_currency
            self.exchange = self.exchange or "FX"
        else:
            self.ticker = normalize_symbol(self.ticker)
            self.price_symbol = normalize_symbol(self.price_symbol or self.ticker, field_name="price_symbol")
        if self.instrument_type == "future" and not is_continuous_future_symbol(self.price_symbol):
            raise ValueError("Futures hedge positions require a continuous '=F' price_symbol.")
        self.asset = normalize_asset(self.asset, instrument_type=self.instrument_type, symbol=self.price_symbol)
        self.contract_multiplier = default_contract_multiplier(
            instrument_type=self.instrument_type,
            symbol=self.price_symbol,
            override=self.contract_multiplier,
        )
        self.quantity = normalize_quantity(quantity=self.quantity, shares=self.shares, allow_negative=True)
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
def update_hedge_positions(req: HedgeUpdateRequest, actor: ActorDep):
    try:
        result = stage_api_action(
            "update_hedge_positions",
            {"positions": [position.model_dump() for position in req.positions]},
            source_id="portfolio_edit.update_hedge_positions",
            actor=actor,
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
