"""Catalysts and Kill Conditions CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.exceptions import NotFoundError

router = APIRouter()


# ---------------------------------------------------------------------------
# Catalysts
# ---------------------------------------------------------------------------


class CreateCatalystRequest(BaseModel):
    ticker: str
    description: str
    category: str = "fundamental"
    target_date: str | None = None


class UpdateCatalystStatusRequest(BaseModel):
    status: str
    evidence: str | None = None


@router.get("/catalysts")
def list_catalysts(ticker: str):
    from portfolio.core_db import get_catalysts

    catalysts = get_catalysts(ticker)
    return {"catalysts": catalysts, "count": len(catalysts)}


@router.post("/catalysts")
def create_catalyst(body: CreateCatalystRequest):
    from portfolio.core_db import create_catalyst

    result = create_catalyst(
        ticker=body.ticker,
        description=body.description,
        category=body.category,
        target_date=body.target_date,
        created_by="user",
    )
    try:
        from portfolio.thesis_sync import sync_markdown_from_entities

        sync_markdown_from_entities(body.ticker)
    except Exception:
        pass
    return result


@router.put("/catalysts/{catalyst_id}/status")
def update_catalyst_status(catalyst_id: int, body: UpdateCatalystStatusRequest):
    from portfolio.core_db import update_catalyst_status

    try:
        result = update_catalyst_status(catalyst_id, body.status, body.evidence)
    except ValueError as e:
        raise NotFoundError("Catalyst", str(catalyst_id)) from e
    try:
        from portfolio.thesis_sync import sync_markdown_from_entities

        sync_markdown_from_entities(result["ticker"])
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Kill Conditions
# ---------------------------------------------------------------------------


class CreateKillConditionRequest(BaseModel):
    ticker: str
    condition: str
    metric: str | None = None
    threshold: str | None = None


class UpdateKillConditionStatusRequest(BaseModel):
    status: str


@router.get("/kill-conditions")
def list_kill_conditions(ticker: str):
    from portfolio.core_db import get_kill_conditions

    kcs = get_kill_conditions(ticker)
    return {"kill_conditions": kcs, "count": len(kcs)}


@router.post("/kill-conditions")
def create_kill_condition(body: CreateKillConditionRequest):
    from portfolio.core_db import create_kill_condition

    result = create_kill_condition(
        ticker=body.ticker,
        condition=body.condition,
        metric=body.metric,
        threshold=body.threshold,
        created_by="user",
    )
    try:
        from portfolio.thesis_sync import sync_markdown_from_entities

        sync_markdown_from_entities(body.ticker)
    except Exception:
        pass
    return result


@router.put("/kill-conditions/{kc_id}/status")
def update_kill_condition_status(kc_id: int, body: UpdateKillConditionStatusRequest):
    from portfolio.core_db import update_kill_condition_status

    try:
        result = update_kill_condition_status(kc_id, body.status)
    except ValueError as e:
        raise NotFoundError("Kill condition", str(kc_id)) from e
    try:
        from portfolio.thesis_sync import sync_markdown_from_entities

        sync_markdown_from_entities(result["ticker"])
    except Exception:
        pass
    return result
