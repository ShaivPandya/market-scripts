"""Scenario simulator API for non-executing investment options."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from api.exceptions import ValidationError
from api.position_risk import get_latest_portfolio_risk, get_latest_position_risk
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
    average_daily_volume_notional: float | None = None


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
    execution_assumptions: dict[str, Any] | None = None
    enrich_from_risk_snapshot: bool = False
    position_risk_snapshot_id: str | None = None
    portfolio_risk_snapshot_id: str | None = None
    persist: bool = False


def _ratio_from_pct(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) > 1.0:
        number = number / 100.0
    return number


def _enrich_payload_from_risk_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill missing scenario and position inputs from persisted risk snapshots."""
    ticker = str((payload.get("position") or {}).get("ticker") or "").strip().upper()
    if not ticker:
        return payload

    enriched = dict(payload)
    risk_provenance: dict[str, Any] = dict(enriched.get("risk_provenance") or {})
    position_snapshot_id = str(enriched.get("position_risk_snapshot_id") or "").strip()
    portfolio_snapshot_id = str(enriched.get("portfolio_risk_snapshot_id") or "").strip()

    position_snapshot = None
    if position_snapshot_id:
        latest = get_latest_position_risk(ticker)
        if latest and str(latest.get("result_id") or "") == position_snapshot_id:
            position_snapshot = latest
    elif bool(enriched.get("enrich_from_risk_snapshot")):
        position_snapshot = get_latest_position_risk(ticker)

    portfolio_snapshot = None
    if portfolio_snapshot_id:
        latest_portfolio = get_latest_portfolio_risk()
        if latest_portfolio and str(latest_portfolio.get("result_id") or "") == portfolio_snapshot_id:
            portfolio_snapshot = latest_portfolio
    elif bool(enriched.get("enrich_from_risk_snapshot")) and not portfolio_snapshot_id:
        portfolio_snapshot = get_latest_portfolio_risk()

    if position_snapshot:
        position_snapshot_id = str(position_snapshot.get("result_id") or position_snapshot_id or "")
        risk_provenance["position_risk_snapshot_id"] = position_snapshot_id
        position = dict(enriched.get("position") or {})
        if position.get("average_daily_volume_notional") is None:
            position["average_daily_volume_notional"] = position_snapshot.get("average_daily_volume_notional") or (
                position_snapshot.get("position") or {}
            ).get("average_daily_volume_notional")
        enriched["position"] = position
        _apply_risk_snapshot_to_scenarios(enriched, position_snapshot)

    if portfolio_snapshot:
        portfolio_snapshot_id = str(portfolio_snapshot.get("result_id") or portfolio_snapshot_id or "")
        risk_provenance["portfolio_risk_snapshot_id"] = portfolio_snapshot_id

    if risk_provenance:
        enriched["risk_provenance"] = risk_provenance
        enriched["position_risk_snapshot_id"] = risk_provenance.get("position_risk_snapshot_id")
        enriched["portfolio_risk_snapshot_id"] = risk_provenance.get("portfolio_risk_snapshot_id")
    return enriched


def _apply_risk_snapshot_to_scenarios(payload: dict[str, Any], snapshot: dict[str, Any]) -> None:
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return
    risk_score = _ratio_from_pct(snapshot.get("risk_score"))
    components = snapshot.get("component_scores") if isinstance(snapshot.get("component_scores"), dict) else {}
    volatility = _ratio_from_pct(components.get("volatility_cluster"))
    breadth = _ratio_from_pct(components.get("breadth_stress"))
    sector = _ratio_from_pct(components.get("sector_stress"))
    macro = _ratio_from_pct(components.get("macro_regime"))
    drawdown = (
        max(value for value in (breadth, sector, macro) if value is not None)
        if any(value is not None for value in (breadth, sector, macro))
        else None
    )

    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        if scenario.get("stress_loss_pct") is None and risk_score is not None:
            scenario["stress_loss_pct"] = round(risk_score * 100.0, 4)
        if scenario.get("daily_volatility_pct") is None and volatility is not None:
            scenario["daily_volatility_pct"] = round(volatility * 100.0, 4)
        if scenario.get("drawdown_pct") is None and drawdown is not None:
            scenario["drawdown_pct"] = round(drawdown * 100.0, 4)
        if scenario.get("thesis_pressure") is None and risk_score is not None:
            scenario["thesis_pressure"] = round(risk_score * 100.0, 4)
        refs = list(scenario.get("source_refs") or [])
        snapshot_ref = f"position_risk_snapshot:{snapshot.get('result_id')}"
        if snapshot_ref not in refs:
            refs.append(snapshot_ref)
        scenario["source_refs"] = refs


@router.post("/scenario-simulator/evaluate")
def evaluate_scenario_simulator(body: ScenarioSimulatorEvaluateRequest, actor: ActorDep):
    payload = body.model_dump(mode="json")
    persist = bool(payload.pop("persist", False))
    if (
        payload.get("enrich_from_risk_snapshot")
        or payload.get("position_risk_snapshot_id")
        or payload.get("portfolio_risk_snapshot_id")
    ):
        payload = _enrich_payload_from_risk_snapshot(payload)
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

    if payload.get("risk_provenance"):
        result["risk_provenance"] = dict(payload.get("risk_provenance") or {})

    if not persist:
        return result

    artifacts = DecisionOntologyWriteback().record_scenario_simulation(
        simulation=result,
        request_payload=payload,
        actor=actor_to_dict(actor),
        provenance=f"pv:{result['simulation_id']}",
    )
    return attach_persistence_artifacts(result, artifacts)
