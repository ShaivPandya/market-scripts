from __future__ import annotations

import re
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
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
from api.request_limits import read_upload_file_bytes
from api.routers.auth import require_actor
from ontology.policy import Actor
from ontology.runtime_read_service import OntologyRuntimeReadService
from portfolio.ibkr_flex_import import (
    merge_ibkr_flex_hedge_replacement,
    merge_preserved_portfolio_metadata,
    parse_ibkr_flex_open_positions_xml,
    split_ibkr_flex_import_rows,
)
from portfolio.instruments import (
    normalize_portfolio_instrument_row,
    position_row_id,
)
from portfolio.position_groups import (
    canonicalize_position_group_rows,
    normalize_position_group_fields,
    validate_position_groups,
)

router = APIRouter()
ActorDep = Annotated[Actor, Depends(require_actor)]

MAX_IBKR_FLEX_UPLOAD_BYTES = 10 * 1024 * 1024

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
    instrument_type: Literal["security", "future", "spot_fx", "option"] | None = None
    price_symbol: str | None = None
    contract_multiplier: float | None = None
    position_id: str | None = None
    underlying_ticker: str | None = None
    option_contract_symbol: str | None = None
    option_expiration: str | None = None
    option_strike: float | None = None
    option_type: Literal["call", "put"] | None = None
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
        normalized = normalize_portfolio_instrument_row(self.model_dump())
        for key, value in normalized.items():
            if key in self.model_fields:
                object.__setattr__(self, key, value)
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


def _is_ibkr_flex_xml_upload(file: UploadFile) -> bool:
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    filename = (file.filename or "").lower()
    return filename.endswith(".xml") or content_type in {"application/xml", "text/xml"}


@router.post("/portfolio-positions/import/ibkr-flex")
async def import_ibkr_flex_portfolio_positions(
    actor: ActorDep,
    file: UploadFile = File(...),  # noqa: B008 - FastAPI parameter declaration
    reason: str | None = Form(default=None),
):
    """Parse IBKR Flex Open Positions XML and stage a replacement portfolio proposal."""
    if not _is_ibkr_flex_xml_upload(file):
        raise HTTPException(status_code=400, detail="File must be an XML (.xml) Flex export.")

    payload = await read_upload_file_bytes(file, limit_bytes=MAX_IBKR_FLEX_UPLOAD_BYTES, limit_label="10 MiB")
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        read_service = OntologyRuntimeReadService()
        imported_rows = parse_ibkr_flex_open_positions_xml(payload)
        portfolio_imported, hedge_imported = split_ibkr_flex_import_rows(imported_rows)
        existing_rows = read_service.positions(include_hedges=False)
        merged_portfolio = merge_preserved_portfolio_metadata(portfolio_imported, existing_rows)
        portfolio_positions = [PortfolioPosition(**row) for row in merged_portfolio]

        import_reason = reason or f"Import IBKR Flex open positions from {file.filename or 'upload.xml'}"
        staged_proposals: list[dict] = []
        primary_result: dict | None = None

        portfolio_result = stage_api_action(
            "update_portfolio_positions",
            {"positions": [position.model_dump() for position in portfolio_positions]},
            source_id="portfolio_edit.import_ibkr_flex_portfolio_positions",
            actor=actor,
            reason=import_reason,
            apply=False,
            validation_status_code=400,
        )
        staged_proposals.append(portfolio_result)
        primary_result = portfolio_result

        hedge_result: dict | None = None
        if hedge_imported:
            existing_hedges = read_service.list_objects("HedgePosition", limit=1000)
            merged_hedges = merge_ibkr_flex_hedge_replacement(hedge_imported, existing_hedges)
            hedge_positions = [HedgePosition(**row) for row in merged_hedges]
            hedge_result = stage_api_action(
                "update_hedge_positions",
                {"positions": [position.model_dump() for position in hedge_positions]},
                source_id="portfolio_edit.import_ibkr_flex_hedge_positions",
                actor=actor,
                reason=import_reason,
                apply=False,
                validation_status_code=400,
            )
            staged_proposals.append(hedge_result)

        preserved_metadata_count = sum(
            1
            for imported, merged in zip(portfolio_imported, merged_portfolio, strict=False)
            if any(
                merged.get(field) != imported.get(field)
                for field in ("conviction", "contrarian", "group_name", "group_conviction")
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (HTTPException, AppError):
        raise
    except Exception as e:
        raise DataFetchError(source="portfolio_positions", detail=str(e)) from e

    hedge_tickers = sorted({str(row.get("ticker") or "").upper() for row in hedge_imported if row.get("ticker")})
    return {
        **(primary_result or {}),
        "staged_proposals": staged_proposals,
        "import_summary": {
            "source": "ibkr_flex",
            "filename": file.filename,
            "imported_count": len(imported_rows),
            "portfolio_imported_count": len(portfolio_imported),
            "hedge_imported_count": len(hedge_imported),
            "hedge_tickers": hedge_tickers,
            "preserved_metadata_count": preserved_metadata_count,
        },
    }


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
    instrument_type: Literal["security", "future", "spot_fx", "option"] | None = None
    price_symbol: str | None = None
    contract_multiplier: float | None = None
    position_id: str | None = None
    underlying_ticker: str | None = None
    option_contract_symbol: str | None = None
    option_expiration: str | None = None
    option_strike: float | None = None
    option_type: Literal["call", "put"] | None = None
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
        normalized = normalize_portfolio_instrument_row(self.model_dump())
        for key, value in normalized.items():
            if key in self.model_fields:
                object.__setattr__(self, key, value)
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
