from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.audit import emit_audit_event
from api.exceptions import DataFetchError, NotFoundError
from api.position_risk import get_latest_position_risk, refresh_position_risk
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
