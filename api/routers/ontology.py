from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.exceptions import DataFetchError, NotFoundError
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
    refresh_snapshot: bool = False


def _extract_filters(req: OntologyQueryRequest) -> dict[str, Any]:
    return req.filters.model_dump(exclude_none=True) if req.filters else {}


def _execute_query(req: OntologyQueryRequest) -> dict[str, Any]:
    filters = _extract_filters(req)
    try:
        return _service.query(
            query=req.query,
            intent=req.intent,
            filters=filters,
            timeframe=req.timeframe,
            include_graph=req.include_graph,
            run_id=req.run_id,
            refresh_snapshot=req.refresh_snapshot,
        )
    except OntologyRunNotFoundError as exc:
        raise NotFoundError("Ontology run", str(exc)) from exc


@router.get("/ontology/runs")
def list_ontology_runs(limit: int = 100):
    safe_limit = max(1, min(int(limit), 500))
    try:
        runs = _service.list_runs(limit=safe_limit)
        return {"runs": runs}
    except Exception as exc:
        raise DataFetchError(source="ontology", detail=str(exc)) from exc


@router.post("/ontology/query")
def query_ontology(req: OntologyQueryRequest):
    row, _disposition = enqueue_registered_job(
        "ontology",
        req.model_dump(exclude_none=True),
        cache_key=_job_cache_key(req),
    )
    return enqueue_response(row, "/api/v1/ontology/query/async/{job_id}")


def _job_cache_key(req: OntologyQueryRequest) -> str:
    payload = req.model_dump(exclude_none=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@router.post("/ontology/query/async")
def start_query_ontology_async(req: OntologyQueryRequest):
    key = _job_cache_key(req)
    row, _disposition = enqueue_registered_job("ontology", req.model_dump(exclude_none=True), cache_key=key)
    return enqueue_response(row, "/api/v1/ontology/query/async/{job_id}")


@router.get("/ontology/query/async/{job_id}")
def get_query_ontology_async(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise NotFoundError("Ontology job", job_id)  # noqa: B904
