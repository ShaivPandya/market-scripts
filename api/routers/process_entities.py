"""Catalysts and Kill Conditions CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.action_execution import stage_api_action
from api.routers.auth import ActorDep
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()


def _canonical_route_id(value: str, prefix: str) -> str:
    text = str(value or "").strip()
    marker = f"{prefix}:"
    if text.startswith(marker):
        suffix = text.removeprefix(marker)
        if suffix.isdigit():
            return suffix
    return text


# ---------------------------------------------------------------------------
# Catalysts
# ---------------------------------------------------------------------------


class CreateCatalystRequest(BaseModel):
    ticker: str
    description: str
    category: str = "fundamental"
    target_date: str | None = None
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


class UpdateCatalystStatusRequest(BaseModel):
    status: str
    evidence: str | None = None
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.get("/catalysts")
def list_catalysts(ticker: str):
    catalysts = OntologyRuntimeReadService().catalysts(ticker)
    return {"catalysts": catalysts, "count": len(catalysts)}


@router.post("/catalysts")
def create_catalyst(body: CreateCatalystRequest, actor: ActorDep):
    return stage_api_action(
        "create_catalyst",
        body.model_dump(exclude={"reason", "apply", "approval_note"}),
        source_id="process_entities.create_catalyst",
        actor=actor,
        reason=body.reason or f"Create catalyst for {body.ticker}",
        apply=body.apply,
        approval_note=body.approval_note,
    )


@router.put("/catalysts/{catalyst_id}/status")
def update_catalyst_status(catalyst_id: str, body: UpdateCatalystStatusRequest, actor: ActorDep):
    resolved_catalyst_id = _canonical_route_id(catalyst_id, "catalyst")
    return stage_api_action(
        "update_catalyst_status",
        {"catalyst_id": resolved_catalyst_id, "status": body.status, "evidence": body.evidence},
        source_id="process_entities.update_catalyst_status",
        actor=actor,
        reason=body.reason or f"Update catalyst {resolved_catalyst_id} status",
        apply=body.apply,
        approval_note=body.approval_note,
        entity_id=resolved_catalyst_id,
    )


# ---------------------------------------------------------------------------
# Kill Conditions
# ---------------------------------------------------------------------------


class CreateKillConditionRequest(BaseModel):
    ticker: str
    condition: str
    metric: str | None = None
    threshold: str | None = None
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


class UpdateKillConditionStatusRequest(BaseModel):
    status: str
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.get("/kill-conditions")
def list_kill_conditions(ticker: str):
    kcs = OntologyRuntimeReadService().kill_conditions(ticker)
    return {"kill_conditions": kcs, "count": len(kcs)}


@router.post("/kill-conditions")
def create_kill_condition(body: CreateKillConditionRequest, actor: ActorDep):
    return stage_api_action(
        "create_kill_condition",
        body.model_dump(exclude={"reason", "apply", "approval_note"}),
        source_id="process_entities.create_kill_condition",
        actor=actor,
        reason=body.reason or f"Create kill condition for {body.ticker}",
        apply=body.apply,
        approval_note=body.approval_note,
    )


# ---------------------------------------------------------------------------
# Thesis Claims
# ---------------------------------------------------------------------------


class SourceRequirementRequest(BaseModel):
    type: str = "custom"
    description: str
    required: bool = True
    freshness_days: int | None = Field(default=None, ge=0)


SourceRequirementInput = str | SourceRequirementRequest


class CreateThesisClaimRequest(BaseModel):
    ticker: str
    claim: str
    expected_evidence: str | None = None
    disconfirming_evidence: str | None = None
    source_requirements: list[SourceRequirementInput] = Field(default_factory=list)
    cadence: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: str = "active"
    linked_catalyst_ids: list[int] = Field(default_factory=list)
    linked_kill_condition_ids: list[int] = Field(default_factory=list)
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


class UpdateThesisClaimRequest(BaseModel):
    claim: str | None = None
    expected_evidence: str | None = None
    disconfirming_evidence: str | None = None
    source_requirements: list[SourceRequirementInput] | None = None
    cadence: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: str | None = None
    linked_catalyst_ids: list[int] | None = None
    linked_kill_condition_ids: list[int] | None = None
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


def _source_requirements_payload(values: list[SourceRequirementInput] | None) -> list:
    result: list = []
    for value in values or []:
        if isinstance(value, str):
            result.append(value)
        else:
            result.append(value.model_dump())
    return result


@router.get("/thesis-claims")
def list_thesis_claims(
    ticker: str | None = None,
    status: str | None = None,
    limit: int = 100,
):
    claims = OntologyRuntimeReadService().thesis_claims(ticker=ticker, status=status, limit=limit)
    return {"claims": claims, "count": len(claims)}


@router.post("/thesis-claims")
def create_thesis_claim(body: CreateThesisClaimRequest, actor: ActorDep):
    return stage_api_action(
        "create_thesis_claim",
        {
            "ticker": body.ticker,
            "claim": body.claim,
            "expected_evidence": body.expected_evidence,
            "disconfirming_evidence": body.disconfirming_evidence,
            "source_requirements": _source_requirements_payload(body.source_requirements),
            "cadence": body.cadence,
            "confidence": body.confidence,
            "status": body.status,
            "linked_catalyst_ids": body.linked_catalyst_ids,
            "linked_kill_condition_ids": body.linked_kill_condition_ids,
            "source_type": "user",
        },
        source_id="process_entities.create_thesis_claim",
        actor=actor,
        reason=body.reason or f"Create thesis claim for {body.ticker}",
        apply=body.apply,
        approval_note=body.approval_note,
    )


@router.put("/thesis-claims/{claim_id}")
def update_thesis_claim(claim_id: str, body: UpdateThesisClaimRequest, actor: ActorDep):
    updates = body.model_dump(exclude_unset=True, exclude={"reason", "apply", "approval_note"})
    if "source_requirements" in updates:
        updates["source_requirements"] = _source_requirements_payload(body.source_requirements)
    return stage_api_action(
        "update_thesis_claim",
        {"claim_id": claim_id, **updates},
        source_id="process_entities.update_thesis_claim",
        actor=actor,
        reason=body.reason or f"Update thesis claim {claim_id}",
        apply=body.apply,
        approval_note=body.approval_note,
        entity_id=claim_id,
    )


@router.put("/kill-conditions/{kc_id}/status")
def update_kill_condition_status(kc_id: str, body: UpdateKillConditionStatusRequest, actor: ActorDep):
    resolved_kc_id = _canonical_route_id(kc_id, "kill_condition")
    return stage_api_action(
        "update_kill_condition_status",
        {"kill_condition_id": resolved_kc_id, "status": body.status},
        source_id="process_entities.update_kill_condition_status",
        actor=actor,
        reason=body.reason or f"Update kill condition {resolved_kc_id} status",
        apply=body.apply,
        approval_note=body.approval_note,
        entity_id=resolved_kc_id,
    )
