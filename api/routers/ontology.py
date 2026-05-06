from __future__ import annotations

import json
import os
from inspect import Parameter, signature
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.audit import emit_audit_event
from api.exceptions import DataFetchError, NotFoundError, ValidationError
from api.job_queue import get_job
from api.routers.auth import require_actor
from ontology.action_registry import get_tool_exposure
from ontology.domain_write_service import ontology_read_model_enabled
from ontology.object_service import OntologyObjectService
from ontology.policy import (
    Actor,
    OntologyAction,
    PolicyDenied,
    actor_from_dict,
    actor_to_dict,
    require_allowed,
)
from ontology.service import OntologyQueryService, OntologyRunNotFoundError
from ontology.temporal_repository import TemporalOntologyRepository

router = APIRouter()
_service = OntologyQueryService()
ActorDep = Annotated[Actor, Depends(require_actor)]


def _env_bool(name: str, *, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return True
    if raw in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _success_job_read_audit_enabled() -> bool:
    return _env_bool("ONTOLOGY_JOB_SUCCESS_READ_AUDIT_ENABLED", default=False)


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
    as_of: str | None = None
    tx_as_of: str | None = None
    include_history: bool = False


class OntologyQueryJobRequest(OntologyQueryRequest):
    actor: dict[str, Any] = Field(default_factory=dict)


def _extract_filters(req: OntologyQueryRequest) -> dict[str, Any]:
    return req.filters.model_dump(exclude_none=True) if req.filters else {}


def _call_with_optional_actor(func, *, actor: Actor, **kwargs):
    params = signature(func).parameters
    values = params.values()
    supports_var_kwargs = any(p.kind == Parameter.VAR_KEYWORD for p in values)
    supports_actor = supports_var_kwargs or "actor" in params
    call_kwargs = (
        dict(kwargs) if supports_var_kwargs else {key: value for key, value in kwargs.items() if key in params}
    )
    if supports_actor:
        call_kwargs["actor"] = actor
    return func(**call_kwargs)


def _execute_query(req: OntologyQueryJobRequest | OntologyQueryRequest) -> dict[str, Any]:
    filters = _extract_filters(req)
    actor = actor_from_dict(getattr(req, "actor", None))
    try:
        return cast(
            dict[str, Any],
            _call_with_optional_actor(
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
                as_of=req.as_of,
                tx_as_of=req.tx_as_of,
                include_history=req.include_history,
            ),
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


@router.get("/ontology/runs/{run_id}")
def get_ontology_run(run_id: str, actor: ActorDep):
    policy = getattr(_service, "policy", None)
    if policy is not None:
        require_allowed(policy.check_action(actor, OntologyAction.RUNS_LIST, {"run_id": run_id}))
    run = _service.repo.get_run(run_id)
    if run is None:
        raise NotFoundError("Ontology run", run_id)
    run["provenance_summary"] = {"selector": {"ontology_run_id": run_id}, "lineage_state": "ontology"}
    return run


@router.get("/ontology/objects")
def list_ontology_objects(
    actor: ActorDep,
    object_type: str | None = None,
    business_key: str | None = None,
    object_uid: str | None = None,
    as_of: str | None = None,
    tx_as_of: str | None = None,
    include_history: bool = False,
    limit: int = 100,
    offset: int = 0,
):
    _require_temporal_read(actor)
    filters: dict[str, Any] = {}
    if business_key:
        filters["business_key"] = business_key
    if object_uid:
        filters["object_uid"] = object_uid
    try:
        objects = OntologyObjectService().query_objects(
            object_type=object_type,
            filters=filters,
            as_of=as_of,
            tx_as_of=tx_as_of,
            include_history=include_history,
            limit=limit,
            offset=offset,
        )
        return {"objects": objects}
    except Exception as exc:
        raise DataFetchError(source="ontology_objects", detail=str(exc)) from exc


@router.get("/ontology/objects/{object_uid}")
def get_ontology_object(
    object_uid: str,
    actor: ActorDep,
    as_of: str | None = None,
    tx_as_of: str | None = None,
):
    _require_temporal_read(actor)
    try:
        obj = OntologyObjectService().get_object(object_uid, as_of=as_of, tx_as_of=tx_as_of)
    except Exception as exc:
        raise DataFetchError(source="ontology_objects", detail=str(exc)) from exc
    if obj is None:
        raise NotFoundError("Ontology object", object_uid)
    return obj


@router.get("/ontology/relations")
def list_ontology_relations(
    actor: ActorDep,
    relation_type: str | None = None,
    source_object_uid: str | None = None,
    target_object_uid: str | None = None,
    as_of: str | None = None,
    tx_as_of: str | None = None,
    include_history: bool = False,
    limit: int = 100,
    offset: int = 0,
):
    _require_temporal_read(actor)
    try:
        relations = OntologyObjectService().query_relations(
            relation_type=relation_type,
            source_object_uid=source_object_uid,
            target_object_uid=target_object_uid,
            as_of=as_of,
            tx_as_of=tx_as_of,
            include_history=include_history,
            limit=limit,
            offset=offset,
        )
        return {"relations": relations}
    except Exception as exc:
        raise DataFetchError(source="ontology_relations", detail=str(exc)) from exc


@router.get("/ontology/source-records")
def list_ontology_source_records(
    actor: ActorDep,
    vendor: str | None = None,
    source_name: str | None = None,
    record_kind: str | None = None,
    as_of: str | None = None,
    tx_as_of: str | None = None,
    include_history: bool = False,
    limit: int = 100,
    offset: int = 0,
):
    _require_temporal_read(actor)
    try:
        rows = TemporalOntologyRepository().query_source_records(
            vendor=vendor,
            source_name=source_name,
            record_kind=record_kind,
            as_of=as_of,
            tx_as_of=tx_as_of,
            include_history=include_history,
            limit=limit,
            offset=offset,
        )
        return {"source_records": rows}
    except Exception as exc:
        raise DataFetchError(source="ontology_source_records", detail=str(exc)) from exc


@router.post("/ontology/query")
def query_ontology(req: OntologyQueryRequest, actor: ActorDep):
    _preflight_read_model_deprecations(req)
    _preflight_query_policy(req, actor)
    job_req = _job_request(req, actor)
    row, _disposition = enqueue_registered_job(
        "ontology",
        job_req.model_dump(exclude_none=True),
        cache_key=_job_cache_key(job_req),
        reuse_completed=_reuse_completed_job(req),
    )
    return enqueue_response(row, "/api/v1/ontology/query/async/{job_id}")


def _job_cache_key(req: OntologyQueryRequest | OntologyQueryJobRequest) -> str:
    payload = req.model_dump(exclude_none=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _reuse_completed_job(req: OntologyQueryRequest | OntologyQueryJobRequest) -> bool:
    return not bool(req.refresh_snapshot)


@router.post("/ontology/query/async")
def start_query_ontology_async(req: OntologyQueryRequest, actor: ActorDep):
    _preflight_read_model_deprecations(req)
    _preflight_query_policy(req, actor)
    job_req = _job_request(req, actor)
    key = _job_cache_key(job_req)
    row, _disposition = enqueue_registered_job(
        "ontology",
        job_req.model_dump(exclude_none=True),
        cache_key=key,
        reuse_completed=_reuse_completed_job(req),
    )
    return enqueue_response(row, "/api/v1/ontology/query/async/{job_id}")


@router.get("/ontology/query/async/{job_id}")
def get_query_ontology_async(job_id: str, actor: ActorDep):
    try:
        row = _preflight_job_read(job_id, actor)
        result = poll_registered_job(job_id, row=row)
        if _success_job_read_audit_enabled():
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


def _preflight_read_model_deprecations(req: OntologyQueryRequest) -> None:
    if ontology_read_model_enabled() and req.refresh_snapshot and not req.run_id:
        raise ValidationError(
            "refresh_snapshot is deprecated when ONTOLOGY_READ_MODEL=true; "
            "omit refresh_snapshot for temporal read-model queries or provide run_id for snapshot compatibility."
        )


def _preflight_query_policy(req: OntologyQueryRequest, actor: Actor) -> None:
    policy = getattr(_service, "policy", None)
    if policy is None:
        return
    query_tool = get_tool_exposure("query_ontology")
    query_policy = query_tool.policy_spec
    try:
        required_actions = query_policy.ontology_actions if query_policy else (OntologyAction.QUERY,)
        for action_name in required_actions:
            require_allowed(policy.check_action(actor, action_name, {"intent": req.intent, "run_id": req.run_id}))
        dynamic_actions = (
            query_policy.dynamic_ontology_actions(
                {"include_graph": req.include_graph, "refresh_snapshot": req.refresh_snapshot}
            )
            if query_policy and query_policy.dynamic_ontology_actions
            else ()
        )
        for action_name in dynamic_actions:
            require_allowed(policy.check_action(actor, action_name, {"run_id": req.run_id}))
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


def _require_temporal_read(actor: Actor) -> None:
    policy = getattr(_service, "policy", None)
    if policy is None:
        return
    require_allowed(policy.check_action(actor, OntologyAction.QUERY, {"surface": "temporal_ontology"}))


def _preflight_job_read(job_id: str, actor: Actor) -> dict[str, Any]:
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
        return row
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
