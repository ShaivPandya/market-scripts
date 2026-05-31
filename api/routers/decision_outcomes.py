"""Decision outcome and post-mortem review API endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.decision_state import normalize_decision_outcome
from api.exceptions import NotFoundError
from api.routers.auth import ActorDep
from ontology.decision_outcome_service import finalize_decision_outcome
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()


class FinalizeDecisionOutcomeRequest(BaseModel):
    decision: Literal["confirm", "correct", "reject"]
    note: str | None = None
    corrected_postmortem: str | None = None
    lessons_learned: str | None = None


@router.get("/decision-outcomes")
def list_decision_outcomes(
    ticker: str | None = None,
    outcome_status: str | None = None,
    final_label_status: str | None = None,
    limit: int = 50,
):
    items = OntologyRuntimeReadService().decision_outcomes(
        ticker=ticker,
        outcome_status=outcome_status,
        final_label_status=final_label_status,
        limit=limit,
    )
    return {
        "decision_outcomes": [normalize_decision_outcome(item) for item in items],
        "count": len(items),
    }


@router.get("/decision-outcomes/{decision_outcome_id}")
def get_decision_outcome(decision_outcome_id: str):
    reads = OntologyRuntimeReadService()
    item = reads.get(
        decision_outcome_id
        if decision_outcome_id.startswith("decision_outcome:")
        else f"decision_outcome:{decision_outcome_id}"
    )
    if not item:
        raise NotFoundError("DecisionOutcome", decision_outcome_id)
    return normalize_decision_outcome(item)


@router.post("/decision-outcomes/{decision_outcome_id}/finalize")
def finalize_decision_outcome_endpoint(
    decision_outcome_id: str,
    body: FinalizeDecisionOutcomeRequest,
    actor: ActorDep,
):
    try:
        updated = finalize_decision_outcome(
            decision_outcome_id,
            decision=body.decision,
            note=body.note,
            corrected_postmortem=body.corrected_postmortem,
            lessons_learned=body.lessons_learned,
            actor_id=actor.actor_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return normalize_decision_outcome(updated)
