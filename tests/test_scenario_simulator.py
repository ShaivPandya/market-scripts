from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ontology.decision_writeback import DecisionOntologyWriteback
from ontology.object_service import OntologyObjectService
from ontology.temporal_repository import ObjectVersionWrite, RelationVersionWrite
from portfolio.scenario_simulator import ScenarioSimulatorValidationError, simulate_investment_options


def _base_payload(direction: str = "long") -> dict:
    return {
        "portfolio": {
            "portfolio_id": "default-portfolio",
            "account_id": "default-account",
            "base_currency": "USD",
            "book_value": 10000,
            "positions": [
                {
                    "ticker": "MU",
                    "asset": "equity",
                    "direction": direction,
                    "quantity": 10,
                    "current_price": 100,
                    "notional_base": 1000,
                }
            ],
        },
        "position": {
            "ticker": "MU",
            "asset": "equity",
            "direction": direction,
            "quantity": 10,
            "current_price": 100,
            "notional_base": 1000,
            "average_daily_volume_notional": 250,
            "position_uid": "position:MU",
        },
        "candidates": [{"action": "hold"}],
        "scenarios": [{"scenario_id": "base", "name": "Base", "price_move_pct": 10, "probability": 1}],
    }


def _outcome_by_action(result: dict, action: str) -> dict:
    return next(item for item in result["outcomes"] if item["action"] == action)


def test_position_mechanics_for_hold_add_trim_exit_long():
    payload = _base_payload("long")
    payload["candidates"] = [
        {"action": "hold"},
        {"action": "add", "delta": {"notional_base": 500}},
        {"action": "trim", "delta": {"pct_position": 0.25}},
        {"action": "exit"},
    ]

    result = simulate_investment_options(payload)

    assert _outcome_by_action(result, "hold")["target_position"]["notional_base"] == 1000
    assert _outcome_by_action(result, "add")["target_position"]["notional_base"] == 1500
    assert _outcome_by_action(result, "trim")["target_position"]["notional_base"] == 750
    assert _outcome_by_action(result, "exit")["target_position"]["notional_base"] == 0
    assert _outcome_by_action(result, "add")["exposure"]["delta_notional_base"] == 500


def test_position_mechanics_for_add_and_trim_short():
    payload = _base_payload("short")
    payload["candidates"] = [
        {"action": "add", "delta": {"notional_base": 500}},
        {"action": "trim", "delta": {"notional_base": 500}},
    ]

    result = simulate_investment_options(payload)

    add = _outcome_by_action(result, "add")
    trim = _outcome_by_action(result, "trim")
    assert add["target_position"]["notional_base"] == 1500
    assert add["exposure"]["delta_notional_base"] == -500
    assert trim["target_position"]["notional_base"] == 500
    assert trim["exposure"]["delta_notional_base"] == 500


def test_scenario_math_liquidity_uncertainty_and_policy_gate_payload(monkeypatch):
    captured: dict = {}

    def fake_gate(action_id, payload, *, context=None, source_quality=None):
        captured["action_id"] = action_id
        captured["payload"] = payload
        captured["context"] = context
        return {"decision": "pass", "approval_required": True, "review_required": False}

    import portfolio.policy_gate as policy_gate

    monkeypatch.setattr(policy_gate, "evaluate_policy_gate", fake_gate)
    payload = _base_payload("long")
    payload["candidates"] = [{"action": "add", "delta": {"notional_base": 500}}]
    payload["scenarios"] = [
        {
            "scenario_id": "bull",
            "name": "Bull",
            "price_move_pct": 10,
            "probability": 1,
            "stress_loss_pct": 15,
            "drawdown_pct": 12,
            "daily_volatility_pct": 2,
            "thesis_pressure": 20,
        }
    ]

    result = simulate_investment_options(payload, context={"actor_id": "unit"})
    outcome = result["outcomes"][0]

    assert outcome["scenario_outcomes"][0]["target_pnl_base"] == 150
    assert outcome["scenario_outcomes"][0]["target_return_pct_of_book"] == 1.5
    assert outcome["liquidity"]["estimated_exit_days"] == 2
    assert outcome["uncertainty"]["level"] == "low"
    assert captured["action_id"] == "update_portfolio_positions"
    assert captured["payload"]["positions"][0]["notional_base"] == 1500
    assert captured["payload"]["stress_loss_pct"] == 0.15
    assert captured["context"]["request_mode"] == "simulation"


def test_hedge_is_rejected():
    payload = _base_payload("long")
    payload["candidates"] = [{"action": "hedge"}]

    with pytest.raises(ScenarioSimulatorValidationError, match="Hedging is out of scope"):
        simulate_investment_options(payload)


class _FakeTemporalRepo:
    def __init__(self):
        self.object_writes: list[ObjectVersionWrite] = []
        self.relation_writes: list[RelationVersionWrite] = []

    def write_object_version(self, write: ObjectVersionWrite):
        self.object_writes.append(write)
        return {
            "version_id": "version-1",
            "object_uid": write.object_uid,
            "object_type": write.object_type,
            "business_key": write.business_key,
            "schema_name": write.schema_name,
            "schema_version": write.schema_version,
            "properties_json": write.properties,
            "valid_from": datetime(2026, 5, 1, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 1, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }

    def write_relation_version(self, write: RelationVersionWrite):
        self.relation_writes.append(write)
        return {
            "version_id": "relation-1",
            "relation_uid": write.relation_uid,
            "source_object_uid": write.source_object_uid,
            "target_object_uid": write.target_object_uid,
            "relation_type": write.relation_type,
            "relation_schema_name": write.relation_schema_name,
            "relation_schema_version": write.relation_schema_version,
            "properties_json": write.properties,
            "valid_from": datetime(2026, 5, 1, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 1, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }


def test_scenario_simulation_persistence_writes_coa_artifacts():
    repo = _FakeTemporalRepo()
    service = OntologyObjectService(repository=repo)
    payload = _base_payload("long")
    payload["candidates"] = [{"action": "add", "delta": {"notional_base": 500}, "evidence_refs": ["evidence:mu_hbm"]}]
    payload["assumptions"] = [{"name": "Liquidity", "value": "ADV supports two-day resize", "confidence": 0.7}]
    simulation = simulate_investment_options(payload)

    artifacts = DecisionOntologyWriteback(service).record_scenario_simulation(
        simulation=simulation,
        request_payload=payload,
        actor={"actor_type": "system", "actor_id": "unit"},
        provenance="pv:unit_scenario_simulation",
    )

    object_types = {write.object_type for write in repo.object_writes}
    relation_types = {write.relation_type for write in repo.relation_writes}
    assert {
        "CourseOfActionComparison",
        "CourseOfAction",
        "Scenario",
        "ScenarioAssumption",
        "SimulatedOutcome",
        "PolicyGateResult",
    }.issubset(object_types)
    assert {
        "comparison_includes_course_of_action",
        "course_of_action_has_simulated_outcome",
        "course_of_action_uses_scenario",
        "scenario_has_assumption",
    }.issubset(relation_types)
    candidate_artifacts = artifacts["outcome_artifact_ids"]["candidate:1"]
    assert candidate_artifacts["course_of_action_id"].startswith("course_of_action:")
    assert candidate_artifacts["simulated_outcome_ids"][0].startswith("simulated_outcome:")
