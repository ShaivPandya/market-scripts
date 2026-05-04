from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.action_execution import stage_api_action
from api.exceptions import AppError, DataFetchError
from portfolio.action_registry import ActionConflictError, ActionValidationError

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9.]{1,20}$")


class PortfolioPosition(BaseModel):
    ticker: str
    asset: Literal["equity", "commodity", "fx", "bond"]
    direction: Literal["long", "short"]
    contrarian: bool = False
    conviction: int = Field(default=3, ge=1, le=5)
    cost_basis: float | None = None
    shares: float | None = None


class PortfolioUpdateRequest(BaseModel):
    positions: list[PortfolioPosition]
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.get("/portfolio-positions")
def get_portfolio_positions(include_hedges: bool = False):
    try:
        from portfolio.portfolio_db import get_positions

        return {"positions": get_positions(include_hedges=include_hedges)}
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
    except ActionValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except ActionConflictError as e:
        raise HTTPException(status_code=409, detail=e.message) from e
    except Exception as e:
        raise DataFetchError(source="portfolio_positions", detail=str(e)) from e

    return result


# ---------------------------------------------------------------------------
# Hedge positions
# ---------------------------------------------------------------------------


class HedgePosition(BaseModel):
    ticker: str
    direction: Literal["long", "short"]
    cost_basis: float | None = None
    shares: float | None = None


class HedgeUpdateRequest(BaseModel):
    positions: list[HedgePosition]
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.get("/hedge-positions")
def get_hedge_positions_endpoint():
    try:
        from portfolio.portfolio_db import get_hedge_positions

        return {"positions": get_hedge_positions()}
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
    except ActionValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except ActionConflictError as e:
        raise HTTPException(status_code=409, detail=e.message) from e
    except Exception as e:
        raise DataFetchError(source="hedge_positions", detail=str(e)) from e

    return result
