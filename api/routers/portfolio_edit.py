from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.exceptions import DataFetchError
from portfolio.action_registry import ActionConflictError, ActionContext, ActionValidationError, execute_action

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
        result = execute_action(
            "update_portfolio_positions",
            req.model_dump(),
            ActionContext(actor_type="user", source_type="api", source_id="portfolio_edit.update_portfolio_positions"),
        ).output
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
        result = execute_action(
            "update_hedge_positions",
            req.model_dump(),
            ActionContext(actor_type="user", source_type="api", source_id="portfolio_edit.update_hedge_positions"),
        ).output
    except ActionValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except ActionConflictError as e:
        raise HTTPException(status_code=409, detail=e.message) from e
    except Exception as e:
        raise DataFetchError(source="hedge_positions", detail=str(e)) from e

    return result
