"""Financial policy gate API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from api.exceptions import NotFoundError

router = APIRouter()


class PolicyGateEvaluateRequest(BaseModel):
    action_id: str
    payload: dict[str, Any]
    context: dict[str, Any] | None = None


@router.get("/investment-policy/default-account")
def get_default_investment_policy():
    from portfolio.policy_gate import default_policy_snapshot

    return default_policy_snapshot()


@router.post("/policy-gate/evaluate")
def evaluate_policy_gate(body: PolicyGateEvaluateRequest):
    from portfolio.policy_gate import evaluate_policy_gate

    return evaluate_policy_gate(body.action_id, body.payload, context=body.context)


@router.get("/policy-gate-results")
def list_policy_gate_results(
    decision: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    action_id: str | None = None,
    limit: int = 50,
):
    from portfolio.core_db import list_policy_gate_results

    items = list_policy_gate_results(
        decision=decision,
        target_type=target_type,
        target_id=target_id,
        action_id=action_id,
        limit=limit,
    )
    return {"policy_gate_results": items, "count": len(items)}


@router.get("/policy-gate-results/{result_id}")
def get_policy_gate_result(result_id: int):
    from portfolio.core_db import get_policy_gate_result

    item = get_policy_gate_result(result_id)
    if not item:
        raise NotFoundError("PolicyGateResult", str(result_id))
    return item
