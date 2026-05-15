"""Scenario simulator API for non-executing investment options."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from api.exceptions import ValidationError
from api.routers.auth import ActorDep
from ontology.decision_writeback import DecisionOntologyWriteback
from ontology.policy import actor_to_dict
from portfolio.scenario_simulator import (
    ScenarioSimulatorValidationError,
    attach_persistence_artifacts,
    simulate_investment_options,
)

router = APIRouter()


class ScenarioSimulatorPortfolio(BaseModel):
    model_config = ConfigDict(extra="allow")

    portfolio_id: str | None = None
    account_id: str | None = None
    base_currency: str = "USD"
    book_value: float | None = None
    cash: float | None = None
    positions: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioSimulatorPosition(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticker: str
    direction: str = "long"
    quantity: float | None = None
    shares: float | None = None
    current_price: float | None = None
    cost_basis: float | None = None
    notional_base: float | None = None
    currency: str | None = None
    instrument_type: str | None = None
    instrument_id: str | None = None
    position_uid: str | None = None


class ScenarioSimulatorCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str
    candidate_id: str | None = None
    delta: dict[str, Any] | None = None
    rationale: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class ScenarioSimulatorScenario(BaseModel):
    model_config = ConfigDict(extra="allow")

    scenario_id: str | None = None
    name: str | None = None
    scenario_type: str = "stress"
    price_move_pct: float | None = None
    probability: float | None = None
    stress_loss_pct: float | None = None
    drawdown_pct: float | None = None
    daily_volatility_pct: float | None = None
    thesis_pressure: float | dict[str, Any] | None = None
    source_refs: list[str] = Field(default_factory=list)


class ScenarioSimulatorAssumption(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    value: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    unit: str | None = None
    confidence: float | None = None
    source_refs: list[str] = Field(default_factory=list)


class ScenarioSimulatorEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    portfolio: ScenarioSimulatorPortfolio
    position: ScenarioSimulatorPosition
    candidates: list[ScenarioSimulatorCandidate] = Field(min_length=1)
    scenarios: list[ScenarioSimulatorScenario] = Field(min_length=1)
    assumptions: list[ScenarioSimulatorAssumption] = Field(default_factory=list)
    persist: bool = False


@router.post("/scenario-simulator/evaluate")
def evaluate_scenario_simulator(body: ScenarioSimulatorEvaluateRequest, actor: ActorDep):
    payload = body.model_dump(mode="json")
    persist = bool(payload.pop("persist", False))
    try:
        result = simulate_investment_options(
            payload,
            context={
                "actor_id": actor.actor_id,
                "actor_type": actor.actor_type,
                "actor_roles": list(actor.roles),
            },
        )
    except ScenarioSimulatorValidationError as exc:
        raise ValidationError(str(exc)) from exc

    if not persist:
        return result

    artifacts = DecisionOntologyWriteback().record_scenario_simulation(
        simulation=result,
        request_payload=payload,
        actor=actor_to_dict(actor),
        provenance=f"pv:{result['simulation_id']}",
    )
    return attach_persistence_artifacts(result, artifacts)
