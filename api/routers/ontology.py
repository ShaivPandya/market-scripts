from __future__ import annotations

import json
from inspect import Parameter, signature
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.audit import emit_audit_event
from api.exceptions import DataFetchError, NotFoundError
from api.job_queue import get_job
from api.routers.auth import require_actor
from ontology.policy import (
    Actor,
    OntologyAction,
    PolicyDenied,
    actor_from_dict,
    actor_to_dict,
    require_allowed,
)
from ontology.service import OntologyQueryService, OntologyRunNotFoundError

router = APIRouter()
_service = OntologyQueryService()
ActorDep = Annotated[Actor, Depends(require_actor)]


class OntologyFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[str] | None = None
    sectors: list[str] | None = None
    assets: list[str] | None = None
    min_risk_score: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_max_results(cls, value: Any):
        if isinstance(value, dict) and "max_results" in value:
            raise ValueError("filters.max_results has been removed; use top-level page_size instead")
        return value


class OntologyQueryRequest(BaseModel):
    query: str | None = None
    intent: (
        Literal[
            "portfolio_risk_exposure",
            "positions_in_deteriorating_macro",
            "entity_context",
            "thesis_review",
            "temporal_comparison",
        ]
        | None
    ) = None
    filters: OntologyFilters | None = None
    timeframe: str = "Daily"
    include_graph: bool = False
    run_id: str | None = None
    refresh_snapshot: bool = False
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)
    schema_mode: Literal["stored", "upgraded"]


class OntologyQueryJobRequest(OntologyQueryRequest):
    actor: dict[str, Any] = Field(default_factory=dict)


def _extract_filters(req: OntologyQueryRequest) -> dict[str, Any]:
    return req.filters.model_dump(exclude_none=True) if req.filters else {}


def _call_with_optional_actor(func, *, actor: Actor, **kwargs):
    params = signature(func).parameters.values()
    supports_actor = any(p.kind == Parameter.VAR_KEYWORD or p.name == "actor" for p in params)
    if supports_actor:
        return func(**kwargs, actor=actor)
    return func(**kwargs)


def _execute_query(req: OntologyQueryJobRequest | OntologyQueryRequest) -> dict[str, Any]:
    filters = _extract_filters(req)
    actor = actor_from_dict(getattr(req, "actor", None))
    try:
        return _call_with_optional_actor(
            _service.query,
            actor=actor,
            query=req.query,
            intent=req.intent,
            filters=filters,
            timeframe=req.timeframe,
            include_graph=req.include_graph,
            run_id=req.run_id,
            refresh_snapshot=req.refresh_snapshot,
            page=req.page,
            page_size=req.page_size,
            schema_mode=req.schema_mode,
        )
    except OntologyRunNotFoundError as exc:
        raise NotFoundError("Ontology run", str(exc)) from exc


@router.get("/ontology/runs")
def list_ontology_runs(actor: ActorDep, limit: int = 100):
    safe_limit = max(1, min(int(limit), 500))
    try:
        runs = _call_with_optional_actor(_service.list_runs, actor=actor, limit=safe_limit)
        return {"runs": runs}
    except PolicyDenied:
        raise
    except Exception as exc:
        raise DataFetchError(source="ontology", detail=str(exc)) from exc


@router.post("/ontology/query")
def query_ontology(req: OntologyQueryRequest, actor: ActorDep):
    _preflight_query_policy(req, actor)
    job_req = _job_request(req, actor)
    row, _disposition = enqueue_registered_job(
        "ontology",
        job_req.model_dump(exclude_none=True),
        cache_key=_job_cache_key(job_req),
    )
    return enqueue_response(row, "/api/v1/ontology/query/async/{job_id}")


def _job_cache_key(req: OntologyQueryRequest | OntologyQueryJobRequest) -> str:
    payload = req.model_dump(exclude_none=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@router.post("/ontology/query/async")
def start_query_ontology_async(req: OntologyQueryRequest, actor: ActorDep):
    _preflight_query_policy(req, actor)
    job_req = _job_request(req, actor)
    key = _job_cache_key(job_req)
    row, _disposition = enqueue_registered_job("ontology", job_req.model_dump(exclude_none=True), cache_key=key)
    return enqueue_response(row, "/api/v1/ontology/query/async/{job_id}")


@router.get("/ontology/query/async/{job_id}")
def get_query_ontology_async(job_id: str, actor: ActorDep):
    try:
        _preflight_job_read(job_id, actor)
        result = poll_registered_job(job_id)
        emit_audit_event(
            "ontology.job.read",
            "ontology_read",
            "succeeded",
            actor=actor,
            object_refs=[{"type": "async_job", "id": job_id}],
            after_summary={"job_id": job_id, "status": result.get("status")},
        )
        return result
    except KeyError:
        raise NotFoundError("Ontology job", job_id)  # noqa: B904


def _job_request(req: OntologyQueryRequest, actor: Actor) -> OntologyQueryJobRequest:
    payload = req.model_dump(exclude_none=True)
    payload["actor"] = actor_to_dict(actor)
    return OntologyQueryJobRequest.model_validate(payload)


def _preflight_query_policy(req: OntologyQueryRequest, actor: Actor) -> None:
    policy = getattr(_service, "policy", None)
    if policy is None:
        return
    try:
        require_allowed(policy.check_action(actor, OntologyAction.QUERY, {"intent": req.intent, "run_id": req.run_id}))
        if req.include_graph:
            require_allowed(policy.check_action(actor, OntologyAction.GRAPH_READ, {"run_id": req.run_id}))
        if req.refresh_snapshot:
            require_allowed(policy.check_action(actor, OntologyAction.SNAPSHOT_REFRESH, {"run_id": req.run_id}))
    except PolicyDenied as exc:
        emit_audit_event(
            "ontology.query.preflight",
            "ontology_read",
            "denied",
            actor=actor,
            metadata={"intent": req.intent, "run_id": req.run_id, "include_graph": req.include_graph},
            error=exc.reason,
        )
        raise


def _preflight_job_read(job_id: str, actor: Actor) -> None:
    policy = getattr(_service, "policy", None)
    try:
        if policy is not None:
            require_allowed(policy.check_action(actor, OntologyAction.JOB_READ, {"job_id": job_id}))

        row = get_job(job_id)
        if row is None:
            raise KeyError(job_id)
        payload = row.get("payload_json")
        payload_actor = actor_from_dict(payload.get("actor") if isinstance(payload, dict) else None)
        roles = {role.lower() for role in actor.roles}
        if actor.actor_type != "system" and "admin" not in roles and actor.actor_id != payload_actor.actor_id:
            raise PolicyDenied("Actor is not allowed to read this ontology job")
    except PolicyDenied as exc:
        emit_audit_event(
            "ontology.job.read",
            "ontology_read",
            "denied",
            actor=actor,
            object_refs=[{"type": "async_job", "id": job_id}],
            error=exc.reason,
        )
        raise
