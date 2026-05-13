"""Admin provenance trace API endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.exceptions import ValidationError
from api.provenance_graph import (
    DIRECTIONS,
    ProvenanceGraphService,
    build_legacy_decision_lineage_graph,
    build_legacy_trace_graph,
)
from api.routers.auth import require_actor
from ontology.domain_write_service import ontology_primary_writes_enabled
from ontology.policy import Actor, PolicyDenied
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()
ActorDep = Annotated[Actor, Depends(require_actor)]


def _require_admin(actor: Actor) -> None:
    roles = {role.lower() for role in actor.roles}
    if actor.actor_type != "system" and "admin" not in roles:
        raise PolicyDenied("Admin access is required for provenance traces.")


@router.get("/provenance/trace")
def get_provenance_trace(
    actor: ActorDep,
    recommendation_id: str | None = None,
    workflow_run_id: str | None = None,
    ontology_run_id: str | None = None,
    object_version_id: str | None = None,
    relation_version_id: str | None = None,
    source_record_id: str | None = None,
    snapshot_id: str | None = None,
    approval_id: str | None = None,
    action_run_id: str | None = None,
    agent_session_id: str | None = None,
    event_id: str | None = None,
    ref_type: str | None = None,
    ref_id: str | None = None,
    max_depth: int = 3,
    direction: str = "both",
    max_nodes: int = 250,
    max_edges: int = 500,
):
    _require_admin(actor)
    selector = _validate_selector(
        recommendation_id=recommendation_id,
        workflow_run_id=workflow_run_id,
        ontology_run_id=ontology_run_id,
        object_version_id=object_version_id,
        relation_version_id=relation_version_id,
        source_record_id=source_record_id,
        snapshot_id=snapshot_id,
        approval_id=approval_id,
        action_run_id=action_run_id,
        agent_session_id=agent_session_id,
        event_id=event_id,
        ref_type=ref_type,
        ref_id=ref_id,
    )
    _validate_graph_params(max_depth=max_depth, direction=direction, max_nodes=max_nodes, max_edges=max_edges)
    direction = _normalize_direction(direction)

    return _ontology_trace(selector, max_depth=max_depth, direction=direction, max_nodes=max_nodes, max_edges=max_edges)


@router.get("/provenance/entity/{ref_type}/{ref_id}")
def get_entity_provenance(
    ref_type: str,
    ref_id: str,
    actor: ActorDep,
    max_depth: int = 3,
    direction: str = "both",
    max_nodes: int = 250,
    max_edges: int = 500,
):
    _require_admin(actor)
    if not ref_type.strip() or not ref_id.strip():
        raise ValidationError("ref_type and ref_id are required.")
    _validate_graph_params(max_depth=max_depth, direction=direction, max_nodes=max_nodes, max_edges=max_edges)
    direction = _normalize_direction(direction)

    return _ontology_trace(
        {"ref_type": ref_type, "ref_id": ref_id},
        max_depth=max_depth,
        direction=direction,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


@router.get("/governance/lineage")
def get_governance_lineage_report(
    actor: ActorDep,
    recommendation_id: str | None = None,
    ontology_run_id: str | None = None,
    source_record_id: str | None = None,
    snapshot_id: str | None = None,
    approval_id: str | None = None,
    action_run_id: str | None = None,
    workflow_run_id: str | None = None,
    agent_session_id: str | None = None,
    event_id: str | None = None,
    object_version_id: str | None = None,
    relation_version_id: str | None = None,
    ref_type: str | None = None,
    ref_id: str | None = None,
    max_depth: int = 5,
    direction: str = "both",
    max_nodes: int = 250,
    max_edges: int = 500,
):
    _require_admin(actor)
    selector = _validate_selector(
        recommendation_id=recommendation_id,
        workflow_run_id=workflow_run_id,
        ontology_run_id=ontology_run_id,
        object_version_id=object_version_id,
        relation_version_id=relation_version_id,
        source_record_id=source_record_id,
        snapshot_id=snapshot_id,
        approval_id=approval_id,
        action_run_id=action_run_id,
        agent_session_id=agent_session_id,
        event_id=event_id,
        ref_type=ref_type,
        ref_id=ref_id,
        message="Provide exactly one governance lineage selector.",
    )
    _validate_graph_params(max_depth=max_depth, direction=direction, max_nodes=max_nodes, max_edges=max_edges)
    direction = _normalize_direction(direction)

    if not ontology_primary_writes_enabled():
        legacy_decision_keys = {
            "recommendation_id",
            "approval_id",
            "action_run_id",
            "workflow_run_id",
            "object_version_id",
            "relation_version_id",
        }
        from portfolio import core_db

        if set(selector).issubset(legacy_decision_keys):
            report = core_db.get_decision_lineage_report(
                recommendation_id=_legacy_numeric_id(recommendation_id),
                approval_id=_legacy_numeric_id(approval_id),
                action_run_id=_legacy_numeric_id(action_run_id),
                workflow_run_id=workflow_run_id,
                object_version_id=object_version_id,
                relation_version_id=relation_version_id,
                max_depth=max_depth,
            )
            return build_legacy_decision_lineage_graph(
                report,
                selector=selector,
                direction=direction,
                max_depth=max_depth,
            )
        return _legacy_trace(selector, max_depth=max_depth, direction=direction)

    return _ontology_trace(selector, max_depth=max_depth, direction=direction, max_nodes=max_nodes, max_edges=max_edges)


def _legacy_numeric_id(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    try:
        return int(text)
    except (TypeError, ValueError):
        raise ValidationError(f"Legacy provenance selector must be numeric: {value}") from None


def _validate_selector(
    message: str = "Provide exactly one provenance trace selector.", **selector: str | None
) -> dict[str, str]:
    ref_type = selector.get("ref_type")
    ref_id = selector.get("ref_id")
    if (ref_type is None) != (ref_id is None):
        raise ValidationError("ref_type and ref_id must be provided together.")
    clean_selector = {key: str(value) for key, value in selector.items() if value is not None}
    if ref_type is not None and ref_id is not None:
        clean_selector["ref_type"] = str(ref_type)
        clean_selector["ref_id"] = str(ref_id)
    selector_count = sum(1 for key, value in clean_selector.items() if value and key not in {"ref_id"})
    if selector_count != 1:
        raise ValidationError(message)
    return clean_selector


def _validate_graph_params(*, max_depth: int, direction: str, max_nodes: int, max_edges: int) -> None:
    if max_depth < 1 or max_depth > 8:
        raise ValidationError("max_depth must be between 1 and 8.")
    if str(direction or "both").strip().lower() not in DIRECTIONS:
        raise ValidationError("direction must be one of: both, upstream, downstream.")
    if max_nodes < 1 or max_nodes > 1000:
        raise ValidationError("max_nodes must be between 1 and 1000.")
    if max_edges < 1 or max_edges > 2500:
        raise ValidationError("max_edges must be between 1 and 2500.")


def _normalize_direction(direction: str) -> str:
    return str(direction or "both").strip().lower()


def _ontology_trace(
    selector: dict[str, str],
    *,
    max_depth: int = 3,
    direction: str = "both",
    max_nodes: int = 250,
    max_edges: int = 500,
) -> dict:
    if not ontology_primary_writes_enabled():
        return _legacy_trace(selector, max_depth=max_depth, direction=direction)
    return ProvenanceGraphService(reads=OntologyRuntimeReadService()).trace(
        selector=selector,
        direction=direction,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


def _legacy_trace(selector: dict[str, str], *, max_depth: int, direction: str) -> dict:
    from portfolio import core_db

    trace = core_db.get_provenance_trace(
        workflow_run_id=selector.get("workflow_run_id"),
        ontology_run_id=selector.get("ontology_run_id"),
        approval_id=_legacy_numeric_id(selector.get("approval_id")),
        action_run_id=_legacy_numeric_id(selector.get("action_run_id")),
        agent_session_id=selector.get("agent_session_id"),
        event_id=selector.get("event_id"),
        ref_type=selector.get("ref_type") or _selector_ref_type(selector),
        ref_id=selector.get("ref_id") or _selector_ref_id(selector),
        max_depth=max_depth,
    )
    return build_legacy_trace_graph(trace, selector=selector, direction=direction, max_depth=max_depth)


def _selector_ref_type(selector: dict[str, str]) -> str | None:
    mapping = {
        "recommendation_id": "recommendation",
        "object_version_id": "ontology_object_version",
        "relation_version_id": "relation_version",
        "source_record_id": "source_record",
        "snapshot_id": "computed_snapshot_version",
    }
    for key, ref_type in mapping.items():
        if key in selector:
            return ref_type
    return None


def _selector_ref_id(selector: dict[str, str]) -> str | None:
    for key in ("recommendation_id", "object_version_id", "relation_version_id", "source_record_id", "snapshot_id"):
        if key in selector:
            return selector[key]
    return None
