from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.exceptions import DataFetchError
from ontology.service import OntologyQueryService, OntologyRunNotFoundError

router = APIRouter()
_service = OntologyQueryService()


class OntologyFilters(BaseModel):
    tickers: list[str] | None = None
    sectors: list[str] | None = None
    assets: list[str] | None = None
    max_results: int | None = None
    min_risk_score: float | None = None


class OntologyQueryRequest(BaseModel):
    query: str | None = None
    intent: (
        Literal[
            "portfolio_risk_exposure",
            "positions_in_deteriorating_macro",
            "entity_context",
        ]
        | None
    ) = None
    filters: OntologyFilters | None = None
    timeframe: str = "Daily"
    include_graph: bool = False
    run_id: str | None = None


@router.post("/ontology/query")
def query_ontology(req: OntologyQueryRequest):
    try:
        filters = req.filters.model_dump(exclude_none=True) if req.filters else {}
        return _service.query(
            query=req.query,
            intent=req.intent,
            filters=filters,
            timeframe=req.timeframe,
            include_graph=req.include_graph,
            run_id=req.run_id,
        )
    except OntologyRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise DataFetchError(source="ontology", detail=str(exc)) from exc
