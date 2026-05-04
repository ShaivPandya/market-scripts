from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.audit import emit_audit_event
from api.exceptions import DataFetchError, NotFoundError
from api.position_risk import (
    get_latest_portfolio_risk,
    get_latest_position_risk,
    refresh_portfolio_risk,
    refresh_position_risk,
)
from api.routers.auth import require_actor
from ontology.policy import Actor

router = APIRouter()
ActorDep = Annotated[Actor, Depends(require_actor)]


@router.get("/risk/positions/{ticker}/latest")
def latest_position_risk(ticker: str, actor: ActorDep):
    ticker_norm = ticker.strip().upper()
    snapshot = get_latest_position_risk(ticker_norm)
    emit_audit_event(
        "risk.position.latest",
        "position_risk",
        "read",
        actor=actor,
        metadata={"ticker": ticker_norm, "found": snapshot is not None},
    )
    if snapshot is None:
        raise NotFoundError("Position risk snapshot", ticker_norm)
    return snapshot


@router.post("/risk/positions/{ticker}/refresh")
def refresh_position_risk_endpoint(ticker: str, actor: ActorDep):
    ticker_norm = ticker.strip().upper()
    try:
        snapshot = refresh_position_risk(ticker_norm)
    except NotFoundError:
        raise
    except Exception as exc:
        raise DataFetchError(source="position_risk", detail=str(exc)) from exc
    emit_audit_event(
        "risk.position.refresh",
        "position_risk",
        "write",
        actor=actor,
        metadata={
            "ticker": ticker_norm,
            "result_id": snapshot.get("result_id"),
            "quality": snapshot.get("quality"),
            "confidence": snapshot.get("confidence"),
        },
    )
    return snapshot


@router.get("/risk/portfolio/latest")
def latest_portfolio_risk(actor: ActorDep):
    snapshot = get_latest_portfolio_risk()
    emit_audit_event(
        "risk.portfolio.latest",
        "portfolio_risk",
        "read",
        actor=actor,
        metadata={"found": snapshot is not None},
    )
    if snapshot is None:
        raise NotFoundError("Portfolio risk snapshot", "latest")
    return snapshot


@router.post("/risk/portfolio/refresh")
def refresh_portfolio_risk_endpoint(actor: ActorDep):
    try:
        snapshot = refresh_portfolio_risk()
    except Exception as exc:
        raise DataFetchError(source="portfolio_risk", detail=str(exc)) from exc
    emit_audit_event(
        "risk.portfolio.refresh",
        "portfolio_risk",
        "write",
        actor=actor,
        metadata={
            "result_id": snapshot.get("result_id"),
            "quality": snapshot.get("quality"),
            "confidence": snapshot.get("confidence"),
            "position_count": snapshot.get("position_count"),
        },
    )
    return snapshot
