"""Admin provenance trace API endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.exceptions import ValidationError
from api.routers.auth import require_actor
from ontology.policy import Actor, PolicyDenied

router = APIRouter()
ActorDep = Annotated[Actor, Depends(require_actor)]


def _require_admin(actor: Actor) -> None:
    roles = {role.lower() for role in actor.roles}
    if actor.actor_type != "system" and "admin" not in roles:
        raise PolicyDenied("Admin access is required for provenance traces.")


@router.get("/provenance/trace")
def get_provenance_trace(
    actor: ActorDep,
    workflow_run_id: str | None = None,
    ontology_run_id: str | None = None,
    approval_id: int | None = None,
    action_run_id: int | None = None,
    agent_session_id: str | None = None,
    event_id: str | None = None,
):
    _require_admin(actor)
    selectors = [
        workflow_run_id,
        ontology_run_id,
        approval_id,
        action_run_id,
        agent_session_id,
        event_id,
    ]
    if sum(value is not None for value in selectors) != 1:
        raise ValidationError("Provide exactly one provenance trace selector.")

    from portfolio.core_db import get_provenance_trace

    return get_provenance_trace(
        workflow_run_id=workflow_run_id,
        ontology_run_id=ontology_run_id,
        approval_id=approval_id,
        action_run_id=action_run_id,
        agent_session_id=agent_session_id,
        event_id=event_id,
    )


@router.get("/provenance/entity/{ref_type}/{ref_id}")
def get_entity_provenance(ref_type: str, ref_id: str, actor: ActorDep):
    _require_admin(actor)
    if not ref_type.strip() or not ref_id.strip():
        raise ValidationError("ref_type and ref_id are required.")

    from portfolio.core_db import get_provenance_trace

    return get_provenance_trace(ref_type=ref_type, ref_id=ref_id)
