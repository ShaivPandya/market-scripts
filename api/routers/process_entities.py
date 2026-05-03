"""Catalysts and Kill Conditions CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.exceptions import NotFoundError, ValidationError

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


def _source_requirements_payload(values: list[SourceRequirementInput] | None) -> list:
    result: list = []
    for value in values or []:
        if isinstance(value, str):
            result.append(value)
        else:
            result.append(value.model_dump())
    return result


def _sync_claim_markdown(ticker: str | None) -> None:
    if not ticker:
        return
    try:
        from portfolio.thesis_sync import sync_markdown_from_entities

        sync_markdown_from_entities(ticker)
    except Exception:
        pass


def _raise_claim_error(exc: ValueError, claim_id: int | None = None) -> None:
    message = str(exc)
    if message.startswith("No thesis claim") and claim_id is not None:
        raise NotFoundError("Thesis claim", str(claim_id)) from exc
    raise ValidationError(message) from exc


@router.get("/thesis-claims")
def list_thesis_claims(
    ticker: str | None = None,
    status: str | None = None,
    limit: int = 100,
):
    from portfolio.core_db import get_thesis_claims

    claims = get_thesis_claims(ticker=ticker, status=status, limit=limit)
    return {"claims": claims, "count": len(claims)}


@router.post("/thesis-claims")
def create_thesis_claim(body: CreateThesisClaimRequest):
    from portfolio.core_db import create_thesis_claim

    try:
        result = create_thesis_claim(
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
            }
        )
    except ValueError as e:
        _raise_claim_error(e)
    _sync_claim_markdown(result.get("ticker"))
    return result


@router.put("/thesis-claims/{claim_id}")
def update_thesis_claim(claim_id: int, body: UpdateThesisClaimRequest):
    from portfolio.core_db import update_thesis_claim

    updates = body.model_dump(exclude_unset=True)
    if "source_requirements" in updates:
        updates["source_requirements"] = _source_requirements_payload(body.source_requirements)
    try:
        result = update_thesis_claim(claim_id, updates)
    except ValueError as e:
        _raise_claim_error(e, claim_id)
    _sync_claim_markdown(result.get("ticker"))
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
