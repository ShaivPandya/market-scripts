"""Recommendation ledger API endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api.exceptions import NotFoundError

router = APIRouter()


@router.get("/recommendations")
def list_recommendations(
    report_type: str | None = None,
    status: str | None = None,
    ticker: str | None = None,
    approval_status: str | None = None,
    outcome_status: str | None = None,
    limit: int = 50,
):
    from portfolio.core_db import get_recommendations

    items = get_recommendations(
        report_type=report_type,
        status=status,
        ticker=ticker,
        approval_status=approval_status,
        outcome_status=outcome_status,
        limit=limit,
    )
    return {"recommendations": items, "count": len(items)}


@router.get("/recommendations/latest")
def latest_recommendations():
    from portfolio.core_db import get_latest_recommendation

    return {
        "daily": get_latest_recommendation("daily"),
        "weekly": get_latest_recommendation("weekly"),
    }


@router.get("/recommendations/{recommendation_id}")
def get_recommendation_detail(recommendation_id: int):
    from portfolio.core_db import get_recommendation

    item = get_recommendation(recommendation_id)
    if not item:
        raise NotFoundError("Recommendation", str(recommendation_id))
    return item
