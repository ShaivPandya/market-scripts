from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.cache import invalidate_all
from api.exceptions import DataFetchError

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9.]{1,20}$")


class PortfolioPosition(BaseModel):
    ticker: str
    asset: Literal["equity", "commodity", "fx", "bond"]
    direction: Literal["long", "short"]
    distressed: bool = False
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
    if not req.positions:
        raise HTTPException(status_code=400, detail="At least one position is required.")

    tickers_seen: set[str] = set()
    for pos in req.positions:
        ticker = pos.ticker.strip().upper()
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker cannot be empty.")
        if not _TICKER_RE.match(ticker):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ticker format: '{ticker}'. Only letters, digits, and dots are allowed.",
            )
        if ticker in tickers_seen:
            raise HTTPException(status_code=400, detail=f"Duplicate ticker: '{ticker}'.")
        tickers_seen.add(ticker)

    try:
        from portfolio.portfolio_db import save_positions

        rows = [
            {
                "ticker": pos.ticker.strip().upper(),
                "asset": pos.asset,
                "direction": pos.direction,
                "distressed": pos.distressed,
                "conviction": pos.conviction,
                "cost_basis": pos.cost_basis,
                "shares": pos.shares,
            }
            for pos in req.positions
        ]
        save_positions(rows)
    except Exception as e:
        raise DataFetchError(source="portfolio_positions", detail=str(e)) from e

    try:
        from portfolio.portfolio_dashboard import reload_portfolio

        reload_portfolio()
    except Exception:
        pass

    invalidate_all()

    return {"status": "ok", "count": len(req.positions)}


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
    if not req.positions:
        raise HTTPException(status_code=400, detail="At least one hedge position is required.")

    # Validate tickers
    tickers_seen: set[str] = set()
    for pos in req.positions:
        ticker = pos.ticker.strip().upper()
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker cannot be empty.")
        if not _TICKER_RE.match(ticker):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ticker format: '{ticker}'.",
            )
        if ticker in tickers_seen:
            raise HTTPException(status_code=400, detail=f"Duplicate ticker: '{ticker}'.")
        tickers_seen.add(ticker)

    # Ensure hedge tickers don't collide with existing position tickers
    try:
        from portfolio.portfolio_db import get_positions

        existing_position_tickers = {p["ticker"] for p in get_positions(include_hedges=False)}
        collisions = tickers_seen & existing_position_tickers
        if collisions:
            raise HTTPException(
                status_code=409,
                detail=f"Ticker(s) already exist as portfolio positions: {sorted(collisions)}. "
                f"A ticker cannot be both a position and a hedge.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise DataFetchError(source="hedge_positions", detail=str(e)) from e

    try:
        from portfolio.portfolio_db import save_positions

        rows = [
            {
                "ticker": pos.ticker.strip().upper(),
                "asset": "equity",
                "direction": pos.direction,
                "distressed": False,
                "conviction": 3,
                "cost_basis": pos.cost_basis,
                "shares": pos.shares,
            }
            for pos in req.positions
        ]
        save_positions(rows, role="hedge")
    except Exception as e:
        raise DataFetchError(source="hedge_positions", detail=str(e)) from e

    return {"status": "ok", "count": len(req.positions)}
