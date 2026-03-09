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
def get_portfolio_positions():
    try:
        from portfolio.portfolio_db import get_positions

        return {"positions": get_positions()}
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
