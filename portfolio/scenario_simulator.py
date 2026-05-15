"""Deterministic scenario simulator for investment action options."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from math import isfinite
from typing import Any

CALCULATION_VERSION = "scenario_simulator_v1"
SUPPORTED_ACTIONS = {"hold", "add", "trim", "exit"}


class ScenarioSimulatorValidationError(ValueError):
    """Raised when a simulator request cannot be evaluated."""


def simulate_investment_options(
    payload: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Generate comparable, non-executing investment action outcomes."""
    portfolio = _as_dict(payload.get("portfolio"))
    position = _as_dict(payload.get("position"))
    candidates = _dicts(payload.get("candidates"))
    scenarios = _dicts(payload.get("scenarios"))
    assumptions = _dicts(payload.get("assumptions"))

    if not portfolio:
        raise ScenarioSimulatorValidationError("portfolio is required.")
    if not position:
        raise ScenarioSimulatorValidationError("position is required.")
    if not candidates:
        raise ScenarioSimulatorValidationError("At least one candidate action is required.")
    if not scenarios:
        raise ScenarioSimulatorValidationError("At least one scenario is required.")

    normalized_candidates = [_normalize_candidate(candidate, index) for index, candidate in enumerate(candidates)]
    normalized_scenarios, scenario_notes = _normalize_scenarios(scenarios)
    base_position = _normalize_position(position)
    generated_at = datetime.now(UTC).isoformat()
    input_hash = _hash_value(payload, length=32)
    simulation_id = f"scenario_simulation:{input_hash[:16]}"
    portfolio_book = _positive_float(
        portfolio.get("book_value")
        or portfolio.get("gross_asset_value")
        or portfolio.get("nav")
        or portfolio.get("portfolio_value")
    )
    base_source_refs = _source_refs([portfolio, position, *scenarios, *assumptions])
    outcomes: list[dict[str, Any]] = []

    for candidate in normalized_candidates:
        target = _target_position(base_position, candidate, portfolio_book=portfolio_book)
        scenario_outcomes = [
            _scenario_outcome(base_position, target, scenario, portfolio_book=portfolio_book)
            for scenario in normalized_scenarios
        ]
        risk = _risk_summary(scenario_outcomes, normalized_scenarios, portfolio_book=portfolio_book)
        liquidity = _liquidity_summary(base_position, target, candidate, position)
        thesis_pressure = _thesis_pressure_summary(normalized_scenarios)
        uncertainty = _uncertainty_summary(
            base_position=base_position,
            target=target,
            portfolio_book=portfolio_book,
            liquidity=liquidity,
            scenarios=normalized_scenarios,
            scenario_notes=scenario_notes,
            candidate=candidate,
        )
        policy_gate_payload = _policy_gate_payload(
            portfolio=portfolio,
            base_position=base_position,
            target=target,
            risk=risk,
            liquidity=liquidity,
        )
        policy_gate = _evaluate_policy_gate(policy_gate_payload, context=context)
        source_refs = _unique([*base_source_refs, *_source_refs([candidate["raw"]])])
        candidate_hash = _hash_value({"candidate": candidate["raw"], "target": target}, length=16)
        outcome = {
            "candidate_id": candidate["candidate_id"],
            "action": candidate["action"],
            "rationale": candidate.get("rationale"),
            "target_position": target,
            "exposure": _exposure_summary(base_position, target, portfolio_book=portfolio_book),
            "scenario_outcomes": scenario_outcomes,
            "risk": risk,
            "liquidity": liquidity,
            "thesis_pressure": thesis_pressure,
            "uncertainty": uncertainty,
            "policy_gate": policy_gate,
            "provenance": {
                "calculation_version": CALCULATION_VERSION,
                "simulation_id": simulation_id,
                "input_hash": input_hash,
                "candidate_hash": candidate_hash,
                "source_refs": source_refs,
                "evidence_refs": [ref for ref in source_refs if ref.startswith("evidence:")],
                "source_record_refs": [ref for ref in source_refs if ref.startswith("source_record:")],
            },
            "artifact_ids": {},
        }
        outcome["ranking_score"] = _ranking_score(outcome)
        outcomes.append(outcome)

    ranked = sorted(outcomes, key=lambda item: (-float(item["ranking_score"]), str(item["candidate_id"])))
    comparison = {
        "ranking": [
            {
                "rank": index + 1,
                "candidate_id": item["candidate_id"],
                "action": item["action"],
                "ranking_score": item["ranking_score"],
                "policy_gate_decision": (item.get("policy_gate") or {}).get("decision"),
                "uncertainty_level": (item.get("uncertainty") or {}).get("level"),
            }
            for index, item in enumerate(ranked)
        ],
        "summary_metrics": {
            "candidate_count": len(outcomes),
            "scenario_count": len(normalized_scenarios),
            "best_ranking_score": ranked[0]["ranking_score"] if ranked else None,
            "worst_ranking_score": ranked[-1]["ranking_score"] if ranked else None,
        },
        "selection": None,
        "selection_policy": "No automatic trade recommendation or execution is produced by the simulator.",
    }

    return {
        "simulation_id": simulation_id,
        "input_hash": input_hash,
        "generated_at": generated_at,
        "persisted": False,
        "calculation_version": CALCULATION_VERSION,
        "portfolio": {
            "portfolio_id": portfolio.get("portfolio_id"),
            "account_id": portfolio.get("account_id"),
            "base_currency": portfolio.get("base_currency") or "USD",
            "book_value": portfolio_book,
        },
        "comparison": comparison,
        "outcomes": outcomes,
    }


def attach_persistence_artifacts(result: dict[str, Any], artifacts: Mapping[str, Any]) -> dict[str, Any]:
    """Return a result copy annotated with ontology artifact IDs."""
    updated = dict(result)
    updated["persisted"] = True
    updated["artifact_ids"] = dict(artifacts.get("artifact_ids") or {})
    by_candidate = {
        str(key): value
        for key, value in (artifacts.get("outcome_artifact_ids") or {}).items()
        if isinstance(value, Mapping)
    }
    next_outcomes: list[dict[str, Any]] = []
    for outcome in result.get("outcomes") or []:
        if not isinstance(outcome, Mapping):
            continue
        item = dict(outcome)
        item["artifact_ids"] = dict(by_candidate.get(str(item.get("candidate_id")), {}))
        next_outcomes.append(item)
    updated["outcomes"] = next_outcomes
    return updated


def _normalize_candidate(candidate: Mapping[str, Any], index: int) -> dict[str, Any]:
    action = str(candidate.get("action") or "").strip().lower()
    if action == "hedge":
        raise ScenarioSimulatorValidationError("Hedging is out of scope for scenario simulator v1.")
    if action not in SUPPORTED_ACTIONS:
        raise ScenarioSimulatorValidationError(
            f"Unsupported candidate action '{action or '<empty>'}'. Supported actions: add, exit, hold, trim."
        )
    return {
        "candidate_id": str(candidate.get("candidate_id") or candidate.get("id") or f"candidate:{index + 1}"),
        "action": action,
        "delta": _as_dict(candidate.get("delta")),
        "rationale": _optional_text(candidate.get("rationale")),
        "raw": dict(candidate),
    }


def _normalize_scenarios(scenarios: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    missing_probability = any(_probability_value(scenario) is None for scenario in scenarios)
    total_probability = sum(_probability_value(scenario) or 0.0 for scenario in scenarios)
    equal_probability = 1.0 / len(scenarios)
    notes: list[str] = []
    if missing_probability:
        notes.append("One or more scenario probabilities were missing; equal weights were used.")
    elif total_probability <= 0:
        notes.append("Scenario probabilities summed to zero; equal weights were used.")

    normalized: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        raw_probability = _probability_value(scenario)
        probability = (
            equal_probability if missing_probability or total_probability <= 0 else raw_probability / total_probability
        )
        price_move = _ratio_value(
            scenario.get("price_move_pct")
            or scenario.get("price_change_pct")
            or scenario.get("price_return_pct")
            or scenario.get("price_move")
            or scenario.get("return")
        )
        stress_loss = _ratio_value(scenario.get("stress_loss_pct") or scenario.get("scenario_stress_loss_pct"))
        drawdown = _ratio_value(scenario.get("drawdown_pct"))
        volatility = _ratio_value(scenario.get("daily_volatility_pct") or scenario.get("volatility_pct"))
        thesis_pressure = _thesis_pressure_value(scenario.get("thesis_pressure"))
        normalized.append(
            {
                "scenario_id": str(scenario.get("scenario_id") or scenario.get("id") or f"scenario:{index + 1}"),
                "name": str(scenario.get("name") or scenario.get("label") or f"Scenario {index + 1}"),
                "scenario_type": str(scenario.get("scenario_type") or scenario.get("type") or "stress"),
                "probability": round(probability, 6),
                "raw_probability": raw_probability,
                "price_move_ratio": price_move,
                "price_move_pct": _pct(price_move),
                "stress_loss_ratio": stress_loss,
                "drawdown_ratio": drawdown,
                "daily_volatility_ratio": volatility,
                "thesis_pressure": thesis_pressure,
                "source_refs": _source_refs([scenario]),
                "raw": dict(scenario),
            }
        )
    return normalized, notes


def _normalize_position(position: Mapping[str, Any]) -> dict[str, Any]:
    ticker = str(position.get("ticker") or position.get("symbol") or "").strip().upper()
    if not ticker:
        raise ScenarioSimulatorValidationError("position.ticker is required.")
    direction = str(position.get("direction") or "long").strip().lower()
    if direction not in {"long", "short"}:
        raise ScenarioSimulatorValidationError("position.direction must be long or short.")
    quantity = _positive_float(
        position.get("quantity") if position.get("quantity") is not None else position.get("shares")
    )
    price = _positive_float(position.get("current_price") or position.get("price") or position.get("last_price"))
    multiplier = _positive_float(position.get("contract_multiplier")) or 1.0
    notional = _positive_float(position.get("notional_base") or position.get("notional"))
    if notional is None and quantity is not None and price is not None:
        notional = abs(quantity * price * multiplier)
    signed_notional = _signed(notional, direction)
    return {
        "ticker": ticker,
        "asset": str(position.get("asset") or "equity").strip().lower(),
        "direction": direction,
        "quantity": quantity,
        "shares": quantity,
        "current_price": price,
        "cost_basis": _positive_float(position.get("cost_basis")),
        "contract_multiplier": multiplier,
        "notional_base": notional,
        "signed_notional_base": signed_notional,
        "currency": position.get("currency"),
        "base_currency": position.get("base_currency"),
        "instrument_type": position.get("instrument_type"),
        "instrument_id": position.get("instrument_id"),
        "position_uid": position.get("position_uid") or position.get("object_uid") or position.get("id"),
        "estimated_exit_days": _positive_float(position.get("estimated_exit_days")),
        "average_daily_volume_notional": _positive_float(
            position.get("average_daily_volume_notional") or position.get("adv_notional")
        ),
    }


def _target_position(
    base: Mapping[str, Any], candidate: Mapping[str, Any], *, portfolio_book: float | None
) -> dict[str, Any]:
    action = str(candidate["action"])
    delta = _as_dict(candidate.get("delta"))
    current_qty = _positive_float(base.get("quantity")) or 0.0
    current_notional = _positive_float(base.get("notional_base"))
    target_qty = current_qty
    target_notional = current_notional
    delta_notes: list[str] = []

    if action == "exit":
        target_qty = 0.0
        target_notional = 0.0
    elif action in {"add", "trim"}:
        target_qty, target_notional, delta_notes = _apply_delta(
            action,
            current_qty=current_qty,
            current_notional=current_notional,
            delta=delta,
            portfolio_book=portfolio_book,
        )

    direction = str(base.get("direction") or "long")
    return {
        "ticker": base.get("ticker"),
        "asset": base.get("asset"),
        "direction": direction,
        "quantity": _round(target_qty),
        "shares": _round(target_qty),
        "current_price": base.get("current_price"),
        "contract_multiplier": base.get("contract_multiplier"),
        "notional_base": _round(target_notional),
        "signed_notional_base": _round(_signed(target_notional, direction)),
        "currency": base.get("currency"),
        "base_currency": base.get("base_currency"),
        "instrument_type": base.get("instrument_type"),
        "instrument_id": base.get("instrument_id"),
        "position_uid": base.get("position_uid"),
        "delta_notes": delta_notes,
    }


def _apply_delta(
    action: str,
    *,
    current_qty: float,
    current_notional: float | None,
    delta: Mapping[str, Any],
    portfolio_book: float | None,
) -> tuple[float, float | None, list[str]]:
    sign = 1.0 if action == "add" else -1.0
    notes: list[str] = []
    target_qty = current_qty
    target_notional = current_notional
    target_qty_input = _positive_float(delta.get("target_quantity") or delta.get("target_shares"))
    target_notional_input = _positive_float(delta.get("target_notional") or delta.get("target_notional_base"))
    quantity_delta = _positive_float(
        delta.get("quantity") if delta.get("quantity") is not None else delta.get("shares")
    )
    notional_delta = _positive_float(delta.get("notional") or delta.get("notional_base"))
    pct_position = _ratio_value(delta.get("pct_position") or delta.get("percent_of_position"))
    pct_book = _ratio_value(delta.get("pct_book") or delta.get("percent_of_book"))

    if target_qty_input is not None:
        target_qty = target_qty_input
    elif quantity_delta is not None:
        target_qty = max(0.0, current_qty + sign * quantity_delta)
    elif pct_position is not None:
        target_qty = max(0.0, current_qty * (1.0 + sign * abs(pct_position)))

    if target_notional_input is not None:
        target_notional = target_notional_input
    elif notional_delta is not None and target_notional is not None:
        target_notional = max(0.0, target_notional + sign * notional_delta)
    elif pct_book is not None and portfolio_book is not None and target_notional is not None:
        target_notional = max(0.0, target_notional + sign * abs(pct_book) * portfolio_book)
    elif pct_position is not None and target_notional is not None:
        target_notional = max(0.0, target_notional * (1.0 + sign * abs(pct_position)))
    elif not delta:
        notes.append(f"{action} candidate did not include delta sizing; target remains unchanged.")

    return target_qty, target_notional, notes


def _scenario_outcome(
    base: Mapping[str, Any],
    target: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    portfolio_book: float | None,
) -> dict[str, Any]:
    price_move = _float_or_none(scenario.get("price_move_ratio")) or 0.0
    direction = str(base.get("direction") or "long")
    current_notional = _positive_float(base.get("notional_base")) or 0.0
    target_notional = _positive_float(target.get("notional_base")) or 0.0
    current_pnl = _signed(current_notional * price_move, direction) or 0.0
    target_pnl = _signed(target_notional * price_move, direction) or 0.0
    incremental_pnl = target_pnl - current_pnl
    return {
        "scenario_id": scenario.get("scenario_id"),
        "name": scenario.get("name"),
        "probability": scenario.get("probability"),
        "price_move_pct": scenario.get("price_move_pct"),
        "current_pnl_base": _round(current_pnl),
        "target_pnl_base": _round(target_pnl),
        "incremental_pnl_base": _round(incremental_pnl),
        "target_return_pct_of_book": _pct_ratio(target_pnl, portfolio_book),
        "incremental_return_pct_of_book": _pct_ratio(incremental_pnl, portfolio_book),
        "loss_pct_of_book": _pct_ratio(max(0.0, -target_pnl), portfolio_book),
        "thesis_pressure": scenario.get("thesis_pressure"),
        "source_refs": list(scenario.get("source_refs") or []),
    }


def _risk_summary(
    scenario_outcomes: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    *,
    portfolio_book: float | None,
) -> dict[str, Any]:
    weighted_pnl = sum(
        (_float_or_none(outcome.get("target_pnl_base")) or 0.0) * (_float_or_none(outcome.get("probability")) or 0.0)
        for outcome in scenario_outcomes
    )
    worst_pnl = min((_float_or_none(outcome.get("target_pnl_base")) or 0.0) for outcome in scenario_outcomes)
    worst_loss = max(0.0, -worst_pnl)
    scenario_stress = max((_float_or_none(scenario.get("stress_loss_ratio")) or 0.0) for scenario in scenarios)
    computed_stress = (worst_loss / portfolio_book) if portfolio_book and portfolio_book > 0 else None
    stress_loss_ratio = max(value for value in (scenario_stress, computed_stress or 0.0) if value is not None)
    drawdown = max((_float_or_none(scenario.get("drawdown_ratio")) or 0.0) for scenario in scenarios)
    volatility = max((_float_or_none(scenario.get("daily_volatility_ratio")) or 0.0) for scenario in scenarios)
    return {
        "expected_pnl_base": _round(weighted_pnl),
        "expected_return_pct": _pct_ratio(weighted_pnl, portfolio_book),
        "worst_case_pnl_base": _round(worst_pnl),
        "worst_loss_base": _round(worst_loss),
        "worst_loss_pct": _pct_ratio(worst_loss, portfolio_book),
        "stress_loss_ratio": _round(stress_loss_ratio),
        "stress_loss_pct": _pct(stress_loss_ratio),
        "drawdown_ratio": _round(drawdown),
        "drawdown_pct": _pct(drawdown),
        "daily_volatility_ratio": _round(volatility),
        "daily_volatility_pct": _pct(volatility),
        "scenario_count": len(scenario_outcomes),
    }


def _liquidity_summary(
    base: Mapping[str, Any],
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
    raw_position: Mapping[str, Any],
) -> dict[str, Any]:
    delta = _as_dict(candidate.get("delta"))
    explicit_exit_days = _positive_float(
        delta.get("estimated_exit_days")
        or candidate.get("estimated_exit_days")
        or raw_position.get("estimated_exit_days")
    )
    adv = _positive_float(
        delta.get("average_daily_volume_notional")
        or raw_position.get("average_daily_volume_notional")
        or raw_position.get("adv_notional")
    )
    current_notional = _positive_float(base.get("notional_base")) or 0.0
    target_notional = _positive_float(target.get("notional_base")) or 0.0
    action = str(candidate.get("action") or "")
    traded_notional = current_notional if action == "exit" else abs(target_notional - current_notional)
    estimated_exit_days = explicit_exit_days
    if estimated_exit_days is None and adv and adv > 0:
        estimated_exit_days = traded_notional / adv
    status = "estimated" if estimated_exit_days is not None else "missing"
    return {
        "traded_notional_base": _round(traded_notional),
        "average_daily_volume_notional": _round(adv),
        "estimated_exit_days": _round(estimated_exit_days),
        "status": status,
        "notes": [] if status != "missing" else ["Liquidity inputs missing; policy gate may understate exit risk."],
    }


def _thesis_pressure_summary(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        ((_float_or_none(scenario.get("thesis_pressure")) or 0.0), (_float_or_none(scenario.get("probability")) or 0.0))
        for scenario in scenarios
        if scenario.get("thesis_pressure") is not None
    ]
    if not values:
        return {"weighted_pressure": None, "max_pressure": None, "status": "missing"}
    weighted = sum(value * probability for value, probability in values)
    return {
        "weighted_pressure": _round(weighted),
        "max_pressure": _round(max(value for value, _probability in values)),
        "status": "estimated",
    }


def _uncertainty_summary(
    *,
    base_position: Mapping[str, Any],
    target: Mapping[str, Any],
    portfolio_book: float | None,
    liquidity: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    scenario_notes: Sequence[str],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    notes: list[str] = list(scenario_notes)
    if base_position.get("notional_base") is None:
        notes.append("Current notional could not be determined from position inputs.")
    if target.get("notional_base") is None:
        notes.append("Target notional could not be determined from candidate sizing.")
    if portfolio_book is None:
        notes.append("Portfolio book value missing; percent-of-book metrics are unavailable.")
    if liquidity.get("status") == "missing":
        notes.extend(str(note) for note in liquidity.get("notes") or [])
    if not any(scenario.get("thesis_pressure") is not None for scenario in scenarios):
        notes.append("Thesis pressure inputs missing across scenarios.")
    if candidate.get("action") in {"add", "trim"} and not _as_dict(candidate.get("delta")):
        notes.append("Sizing delta missing for non-hold candidate.")
    missing_count = len(notes)
    level = "high" if missing_count >= 3 else "medium" if missing_count else "low"
    return {"level": level, "missing_input_count": missing_count, "notes": _unique(notes)}


def _policy_gate_payload(
    *,
    portfolio: Mapping[str, Any],
    base_position: Mapping[str, Any],
    target: Mapping[str, Any],
    risk: Mapping[str, Any],
    liquidity: Mapping[str, Any],
) -> dict[str, Any]:
    positions = _candidate_policy_positions(portfolio, base_position, target)
    payload = {
        "positions": positions,
        "estimated_exit_days": liquidity.get("estimated_exit_days"),
        "stress_loss_pct": risk.get("stress_loss_ratio"),
        "drawdown_pct": risk.get("drawdown_ratio"),
        "daily_volatility_pct": risk.get("daily_volatility_ratio"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _candidate_policy_positions(
    portfolio: Mapping[str, Any],
    base_position: Mapping[str, Any],
    target: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_positions = _dicts(portfolio.get("positions"))
    ticker = str(base_position.get("ticker") or "").upper()
    target_row = {
        "ticker": ticker,
        "asset": target.get("asset") or base_position.get("asset") or "equity",
        "direction": target.get("direction") or base_position.get("direction") or "long",
        "quantity": target.get("quantity"),
        "shares": target.get("shares"),
        "notional_base": target.get("notional_base"),
        "currency": target.get("currency") or base_position.get("currency"),
        "base_currency": target.get("base_currency") or portfolio.get("base_currency") or "USD",
        "instrument_type": target.get("instrument_type") or base_position.get("instrument_type"),
        "contract_multiplier": target.get("contract_multiplier") or base_position.get("contract_multiplier") or 1.0,
        "valuation_status": "ok" if target.get("notional_base") is not None else "missing_position_inputs",
    }
    rows: list[dict[str, Any]] = []
    replaced = False
    for row in raw_positions:
        row_ticker = str(row.get("ticker") or row.get("symbol") or "").upper()
        if row_ticker == ticker:
            rows.append(target_row)
            replaced = True
        else:
            rows.append(dict(row))
    if not replaced:
        rows.append(target_row)
    return rows


def _evaluate_policy_gate(payload: Mapping[str, Any], *, context: Mapping[str, Any] | None) -> dict[str, Any]:
    try:
        from portfolio.policy_gate import evaluate_policy_gate

        return evaluate_policy_gate(
            "update_portfolio_positions",
            payload,
            context={
                "source_type": "api",
                "source_id": "scenario_simulator.evaluate",
                "request_mode": "simulation",
                **dict(context or {}),
            },
        )
    except Exception as exc:
        return {
            "decision": "error",
            "failure_reasons": [{"code": "policy_gate_error", "message": str(exc)}],
            "warnings": [],
            "check_results": [],
            "review_required": True,
            "approval_required": True,
        }


def _exposure_summary(
    base: Mapping[str, Any], target: Mapping[str, Any], *, portfolio_book: float | None
) -> dict[str, Any]:
    current_notional = _positive_float(base.get("notional_base")) or 0.0
    target_notional = _positive_float(target.get("notional_base")) or 0.0
    direction = str(base.get("direction") or "long")
    current_signed = _signed(current_notional, direction) or 0.0
    target_signed = _signed(target_notional, direction) or 0.0
    delta = target_signed - current_signed
    return {
        "current_notional_base": _round(current_notional),
        "target_notional_base": _round(target_notional),
        "delta_notional_base": _round(delta),
        "gross_delta_notional_base": _round(target_notional - current_notional),
        "current_weight_pct": _pct_ratio(current_notional, portfolio_book),
        "target_weight_pct": _pct_ratio(target_notional, portfolio_book),
        "delta_weight_pct": _pct_ratio(delta, portfolio_book),
    }


def _ranking_score(outcome: Mapping[str, Any]) -> float:
    risk = _as_dict(outcome.get("risk"))
    uncertainty = _as_dict(outcome.get("uncertainty"))
    policy_gate = _as_dict(outcome.get("policy_gate"))
    thesis = _as_dict(outcome.get("thesis_pressure"))
    expected_return = (_float_or_none(risk.get("expected_return_pct")) or 0.0) / 100.0
    worst_loss = (_float_or_none(risk.get("worst_loss_pct")) or 0.0) / 100.0
    pressure = _float_or_none(thesis.get("weighted_pressure")) or 0.0
    uncertainty_penalty = {"low": 0.0, "medium": 0.015, "high": 0.04}.get(str(uncertainty.get("level")), 0.02)
    gate_penalty = {
        "pass": 0.0,
        "warn": 0.01,
        "review_required": 0.03,
        "blocked": 0.08,
        "error": 0.05,
    }.get(str(policy_gate.get("decision") or ""), 0.02)
    return round(expected_return - worst_loss - pressure * 0.03 - uncertainty_penalty - gate_penalty, 6)


def _probability_value(value: Mapping[str, Any]) -> float | None:
    raw = value.get("probability")
    if raw is None:
        raw = value.get("probability_pct")
    return _ratio_value(raw)


def _thesis_pressure_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("score") or value.get("level") or value.get("value")
    return _ratio_value(value)


def _ratio_value(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    if abs(number) > 1.0:
        number = number / 100.0
    return number


def _pct_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return _round((numerator / denominator) * 100.0)


def _pct(value: float | None) -> float | None:
    if value is None:
        return None
    return _round(value * 100.0)


def _signed(value: float | None, direction: str) -> float | None:
    if value is None:
        return None
    return -abs(value) if direction == "short" else abs(value)


def _source_refs(values: Sequence[Mapping[str, Any]]) -> list[str]:
    refs: list[str] = []
    for value in values:
        for key in ("source_refs", "evidence_refs", "source_record_refs"):
            raw = value.get(key)
            if isinstance(raw, list):
                refs.extend(str(item) for item in raw if str(item or "").strip())
            elif raw:
                refs.append(str(raw))
    return _unique(refs)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dicts(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, Mapping)]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _positive_float(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return abs(number)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if isfinite(out) else None


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _unique(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _hash_value(value: Any, *, length: int = 16) -> str:
    raw = json.dumps(_jsonable(value), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
