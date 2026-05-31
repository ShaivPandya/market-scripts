"""Deterministic financial policy gate for recommendations and proposals.

The gate is intentionally deterministic and conservative. It does not decide
whether an investment idea is good; it decides whether the idea has enough
risk, freshness, and disclosure context to be staged for human review.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from portfolio.policy_matrix import FinancialPolicyFacts, evaluate_financial_policy_matrix

POLICY_GATE_DECISIONS = ("pass", "warn", "review_required", "blocked", "error")
ACTIONABLE_RECOMMENDATION_ACTIONS = {"buy", "add", "short", "sell", "trim", "reduce", "exit", "hedge", "rebalance"}
FINANCIAL_ACTION_ITEM_TYPES = {"enter", "exit", "resize", "hedge"}
FINANCIAL_ACTION_IDS = {
    "create_course_of_action",
    "create_recommendation",
    "update_portfolio_positions",
    "update_hedge_positions",
}

FAILURE_REASON_CODES = {
    "data_missing",
    "review_warning",
    "liquidity_shortfall",
    "concentration_limit",
    "leverage_limit",
    "volatility_limit",
    "drawdown_limit",
    "benchmark_mismatch",
    "scenario_stress_fail",
    "stale_data",
    "insufficient_history",
    "unknown_instrument",
    "required_disclosure_missing",
}

DEFAULT_POLICY: dict[str, Any] = {
    "investor": {
        "investor_id": "default-investor",
        "name": "Default Investor",
    },
    "account": {
        "account_id": "default-account",
    },
    "portfolio": {
        "portfolio_id": "default-portfolio",
        "base_currency": "USD",
        "cash": None,
        "benchmark": "SPY",
    },
    "policy": {
        "policy_id": "default-investment-policy",
        "max_position_weight_pct": 0.20,
        "max_issuer_weight_pct": 0.25,
        "max_asset_class_weight_pct": 0.85,
        "max_gross_leverage": 4.0,
        "max_net_leverage": 3.0,
        "max_daily_volatility_pct": 0.05,
        "max_drawdown_pct": 0.30,
        "max_stress_loss_pct": 0.20,
        "max_exit_days": 3,
    },
}

DECISION_SUPPORT_DISCLOSURES = [
    "Decision support only; human approval required.",
    "Automated checks cover data quality, concentration, leverage, liquidity, scenario stress, and required disclosures.",
]


def default_policy_snapshot() -> dict[str, Any]:
    return deepcopy(DEFAULT_POLICY)


def is_financial_action(action_id: str, payload: Mapping[str, Any] | None = None) -> bool:
    action = str(action_id or "").strip()
    if action in {"update_portfolio_positions", "update_hedge_positions"}:
        return True
    if action in {"create_course_of_action", "create_recommendation"}:
        record = _recommendation_record(payload)
        return str(record.get("action") or "").lower() in ACTIONABLE_RECOMMENDATION_ACTIONS
    if action == "create_action_item":
        return str((payload or {}).get("action_type") or "").lower() in FINANCIAL_ACTION_ITEM_TYPES
    return action in FINANCIAL_ACTION_IDS


def ensure_policy_gate_for_action(
    action_id: str,
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    object_service: Any | None = None,
    raise_on_blocked: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach or reuse a policy gate result for a financial action payload."""
    mutable = deepcopy(dict(payload))
    if not is_financial_action(action_id, mutable):
        return mutable, None

    existing = _existing_gate_result(action_id, mutable)
    if existing:
        gate = normalize_policy_gate_result(existing)
        if gate["decision"] == "blocked" and raise_on_blocked:
            raise PolicyGateBlockedError(_gate_summary(gate))
        return mutable, gate

    gate = evaluate_policy_gate(action_id, mutable, context=context)
    from api.provenance import stable_hash
    from ontology.schemas.identity import policy_gate_result_id

    target_id = stable_hash({"action_id": action_id, "payload": mutable})
    gate_key = f"{action_id}:action_payload:{target_id}"
    gate_uid = policy_gate_result_id(gate_key)
    from ontology.object_service import OntologyObjectService
    from ontology.policy import actor_to_dict, system_actor

    actor = system_actor("policy_gate")
    objects = object_service or OntologyObjectService()
    objects.write_object(
        "PolicyGateResult",
        gate_uid,
        {
            "gate_result_id": gate_key,
            "decision": gate.get("decision") or "review_required",
            "review_required": bool(gate.get("review_required")),
            "approval_required": bool(gate.get("approval_required", True)),
            "approval_mode": gate.get("approval_mode"),
            "approval_requirements": gate.get("approval_requirements", []),
            "rule_id": gate.get("rule_id"),
            "reason": gate.get("reason"),
            "remediation": gate.get("remediation"),
            "matched_rules": gate.get("matched_rules", []),
            "limit_overrides": gate.get("limit_overrides", {}),
            "failure_reasons": gate.get("failure_reasons", []),
            "warnings": gate.get("warnings", []),
            "account_id": gate.get("account_id"),
            "portfolio_id": gate.get("portfolio_id"),
            "policy_id": gate.get("policy_id"),
            "policy_matrix_id": gate.get("policy_matrix_id"),
            "evaluated_at": datetime.now(UTC).isoformat(),
            "ontology_run_id": "operational",
        },
        datetime.now(UTC).isoformat(),
        actor=actor_to_dict(actor),
        provenance=f"pv:policy_gate:{target_id}",
        input_hash=target_id,
    )
    gate["policy_gate_result_id"] = gate_uid
    if gate["decision"] == "blocked" and raise_on_blocked:
        raise PolicyGateBlockedError(_gate_summary(gate))
    return _attach_gate_to_payload(action_id, mutable, gate), gate


def attach_policy_gate_to_recommendation(
    record: Mapping[str, Any],
    *,
    source_quality: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach a policy gate result to one recommendation record."""
    mutable = deepcopy(dict(record))
    if not is_financial_action("create_recommendation", {"record": mutable}):
        return mutable, None
    gate = evaluate_policy_gate(
        "create_recommendation",
        {"record": mutable},
        context=context,
        source_quality=source_quality,
    )
    _apply_gate_fields(mutable, gate)
    return mutable, gate


def evaluate_policy_gate(
    action_id: str,
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    source_quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic risk, freshness, and data checks."""
    if not isinstance(payload, Mapping):
        return _result(
            action_id=action_id,
            decision="blocked",
            check_results=[
                _check(
                    "payload",
                    "fail",
                    "required_disclosure_missing",
                    "Policy gate requires a structured proposal payload.",
                    severity="block",
                )
            ],
            context=context,
        )

    from api.financial_policy_settings import get_financial_policy_matrix_setting

    matrix = get_financial_policy_matrix_setting()
    facts = _financial_policy_facts(action_id, payload, context=context, source_quality=source_quality)
    matrix_decision = evaluate_financial_policy_matrix(matrix, facts)
    policy = default_policy_snapshot()
    _apply_policy_limit_overrides(policy, matrix_decision)
    check_results = _policy_gate_check_results(action_id, payload, policy, source_quality=source_quality)

    return _result(
        action_id=action_id,
        decision=None,
        check_results=check_results,
        context=context,
        policy=policy,
        matrix_decision=matrix_decision,
    )


def normalize_policy_gate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    gate = dict(result)
    decision = str(gate.get("decision") or "error").lower()
    if decision not in POLICY_GATE_DECISIONS:
        decision = "error"
    gate["decision"] = decision
    gate["failure_reasons"] = _normalize_reason_entries(gate.get("failure_reasons"))
    gate["warnings"] = _normalize_reason_entries(gate.get("warnings"))
    gate["check_results"] = [dict(item) for item in _as_list(gate.get("check_results")) if isinstance(item, Mapping)]
    gate["review_required"] = bool(gate.get("review_required") or decision == "review_required")
    gate["override_acknowledged"] = bool(gate.get("override_acknowledged"))
    gate["approval_required"] = bool(gate.get("approval_required", True))
    gate["approval_mode"] = str(gate.get("approval_mode") or "approval_required")
    gate["approval_requirements"] = [
        dict(item) for item in _as_list(gate.get("approval_requirements")) if isinstance(item, Mapping)
    ]
    gate["rule_id"] = str(gate.get("rule_id") or "") or None
    gate["reason"] = str(gate.get("reason") or "")
    gate["remediation"] = str(gate.get("remediation") or "")
    gate["matched_rules"] = [dict(item) for item in _as_list(gate.get("matched_rules")) if isinstance(item, Mapping)]
    gate["limit_overrides"] = (
        dict(gate.get("limit_overrides") or {}) if isinstance(gate.get("limit_overrides"), Mapping) else {}
    )
    gate.setdefault("disclosures", list(DECISION_SUPPORT_DISCLOSURES))
    gate.setdefault("assumptions", [])
    gate.setdefault("uncertainty", {})
    return gate


class PolicyGateBlockedError(ValueError):
    """Raised when the policy gate cannot produce a reviewable result."""


def _result(
    *,
    action_id: str,
    decision: str | None,
    check_results: list[dict[str, Any]],
    context: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
    matrix_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy_snapshot = deepcopy(dict(policy or default_policy_snapshot()))
    if decision is None:
        decision = _decision_from_checks(check_results)
    failures = [_reason_from_check(c) for c in check_results if c.get("status") == "fail"]
    warnings = [_reason_from_check(c) for c in check_results if c.get("status") == "warn"]
    scenario_results = _scenario_results_from_checks(check_results)
    metrics_snapshot = _metrics_from_checks(check_results)
    ctx = dict(context or {})
    gate = {
        "decision": decision,
        "failure_reasons": failures,
        "warnings": warnings,
        "check_results": check_results,
        "constraints_snapshot": policy_snapshot,
        "metrics_snapshot": metrics_snapshot,
        "scenario_results": scenario_results,
        "assumptions": _assumptions(policy_snapshot),
        "uncertainty": _uncertainty(check_results),
        "disclosures": list(DECISION_SUPPORT_DISCLOSURES),
        "review_required": decision == "review_required",
        "override_acknowledged": False,
        "account_id": policy_snapshot["account"]["account_id"],
        "portfolio_id": policy_snapshot["portfolio"]["portfolio_id"],
        "policy_id": policy_snapshot["policy"]["policy_id"],
        "evaluated_at": datetime.now(UTC).isoformat(),
        "action_id": action_id,
    }
    if ctx:
        gate["context"] = ctx
    if matrix_decision:
        _apply_matrix_decision(gate, matrix_decision)
    return normalize_policy_gate_result(gate)


def _policy_gate_check_results(
    action_id: str,
    payload: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    source_quality: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    check_results: list[dict[str, Any]] = []
    check_results.extend(_required_disclosure_checks(payload))
    check_results.extend(_data_freshness_checks(payload, source_quality=source_quality))
    check_results.extend(_portfolio_constraint_checks(action_id, payload, policy))
    check_results.extend(_liquidity_checks(payload, policy))
    check_results.extend(_scenario_checks(action_id, payload, policy))
    return check_results


def _apply_policy_limit_overrides(policy: dict[str, Any], matrix_decision: Mapping[str, Any]) -> None:
    overrides = matrix_decision.get("limit_overrides")
    if not isinstance(overrides, Mapping):
        return
    policy_limits = policy.setdefault("policy", {})
    if not isinstance(policy_limits, dict):
        return
    for key, item in overrides.items():
        if not isinstance(item, Mapping) or "value" not in item:
            continue
        policy_limits[str(key)] = item.get("value")


def _apply_matrix_decision(gate: dict[str, Any], matrix_decision: Mapping[str, Any]) -> None:
    outcome = str(matrix_decision.get("outcome") or "use_checks").strip().lower()
    rule_id = str(matrix_decision.get("rule_id") or "").strip()
    reason = str(matrix_decision.get("reason") or "").strip()
    remediation = str(matrix_decision.get("remediation") or "").strip()
    gate["policy_matrix_id"] = matrix_decision.get("policy_id")
    gate["policy_matrix_schema_version"] = matrix_decision.get("schema_version")
    gate["rule_id"] = rule_id or None
    gate["reason"] = reason
    gate["remediation"] = remediation
    gate["approval_required"] = bool(matrix_decision.get("approval_required", True))
    gate["approval_mode"] = str(matrix_decision.get("approval_mode") or "approval_required")
    gate["approval_requirements"] = [
        dict(item) for item in _as_list(matrix_decision.get("approval_requirements")) if isinstance(item, Mapping)
    ]
    gate["matched_rules"] = [
        dict(item) for item in _as_list(matrix_decision.get("matched_rules")) if isinstance(item, Mapping)
    ]
    gate["limit_overrides"] = (
        dict(matrix_decision.get("limit_overrides") or {})
        if isinstance(matrix_decision.get("limit_overrides"), Mapping)
        else {}
    )
    if outcome == "use_checks" or outcome not in POLICY_GATE_DECISIONS:
        return

    gate["decision"] = outcome
    gate["review_required"] = outcome == "review_required"
    entry = {
        "code": "policy_matrix_rule",
        "check": "policy_matrix",
        "rule_id": rule_id or None,
        "message": reason or f"Policy matrix rule set decision to {outcome}.",
        "remediation": remediation or None,
    }
    if outcome == "pass":
        gate["failure_reasons"] = []
        gate["warnings"] = []
    elif outcome == "warn":
        gate["failure_reasons"] = []
        gate["warnings"] = [entry]
    else:
        gate["failure_reasons"] = [entry]


def _decision_from_checks(checks: list[dict[str, Any]]) -> str:
    if any(c.get("severity") == "block" and c.get("status") == "fail" for c in checks):
        return "blocked"
    if any(c.get("status") == "fail" for c in checks):
        return "review_required"
    if any(c.get("status") == "warn" for c in checks):
        return "warn"
    return "pass"


def _check(
    check_name: str,
    status: str,
    reason_code: str | None,
    message: str,
    *,
    severity: str = "info",
    observed: Any = None,
    limit: Any = None,
) -> dict[str, Any]:
    if reason_code and reason_code not in FAILURE_REASON_CODES:
        reason_code = "review_warning"
    return {
        "check": check_name,
        "status": status,
        "severity": severity,
        "reason_code": reason_code,
        "message": message,
        "observed": observed,
        "limit": limit,
    }


def _reason_from_check(check: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "code": check.get("reason_code") or "review_warning",
        "check": check.get("check"),
        "message": check.get("message"),
        "observed": check.get("observed"),
        "limit": check.get("limit"),
    }


def _required_disclosure_checks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    record = _recommendation_record(payload)
    if not record:
        return []
    checks: list[dict[str, Any]] = []
    if str(record.get("action") or "").lower() in ACTIONABLE_RECOMMENDATION_ACTIONS:
        if not _as_list(record.get("disconfirming_evidence")):
            checks.append(
                _check(
                    "disconfirming_evidence",
                    "warn",
                    "required_disclosure_missing",
                    "Actionable recommendation lacks explicit disconfirming evidence.",
                    severity="warn",
                )
            )
        if not str(record.get("invalidation") or "").strip():
            checks.append(
                _check(
                    "invalidation",
                    "warn",
                    "required_disclosure_missing",
                    "Actionable recommendation lacks an invalidation condition.",
                    severity="warn",
                )
            )
    return checks


def _data_freshness_checks(
    payload: Mapping[str, Any],
    *,
    source_quality: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    record = _recommendation_record(payload)
    action = str(record.get("action") or "").lower()
    qualities = [
        record.get("critical_data_quality"),
        record.get("source_quality"),
        (source_quality or {}).get("critical_data_quality"),
        (source_quality or {}).get("overall_status"),
    ]
    if any(str(q or "").lower() in {"stale", "failed"} for q in qualities):
        checks.append(
            _check(
                "data_freshness",
                "fail",
                "stale_data",
                "Critical source data is stale or failed; explicit review is required.",
                severity="fail",
                observed=[q for q in qualities if q],
            )
        )
    elif any(str(q or "").lower() == "degraded" for q in qualities):
        checks.append(
            _check(
                "data_freshness",
                "warn",
                "insufficient_history",
                "One or more sources are degraded.",
                severity="warn",
                observed=[q for q in qualities if q],
            )
        )
    if action in ACTIONABLE_RECOMMENDATION_ACTIONS:
        try:
            from api.position_risk import risk_recommendation_gate_enabled
        except Exception:
            risk_gate_enabled = False
        else:
            risk_gate_enabled = risk_recommendation_gate_enabled()

        if risk_gate_enabled:
            risk_quality = str(record.get("risk_quality") or "missing").lower()
            risk_snapshot_id = record.get("risk_snapshot_id")
            portfolio_risk_snapshot_id = record.get("portfolio_risk_snapshot_id")
            risk_score = _risk_score_from_recommendation(record)
            if risk_score is None:
                checks.append(
                    _check(
                        "risk.first_class_snapshot",
                        "fail",
                        "stale_data",
                        "Actionable recommendation requires a risk score.",
                        severity="block",
                        observed={
                            "risk_quality": risk_quality,
                            "risk_snapshot_id": risk_snapshot_id,
                            "portfolio_risk_snapshot_id": portfolio_risk_snapshot_id,
                        },
                    )
                )
            elif risk_quality != "ok":
                checks.append(
                    _check(
                        "risk.first_class_snapshot",
                        "fail",
                        "insufficient_history",
                        "Risk is degraded; human review is required before action.",
                        severity="fail",
                        observed={
                            "risk_quality": risk_quality,
                            "risk_score": risk_score,
                            "risk_snapshot_id": risk_snapshot_id,
                            "portfolio_risk_snapshot_id": portfolio_risk_snapshot_id,
                        },
                    )
                )
            elif not risk_snapshot_id and not portfolio_risk_snapshot_id:
                checks.append(
                    _check(
                        "risk.first_class_snapshot",
                        "fail",
                        "stale_data",
                        "Actionable recommendation requires linked risk snapshots.",
                        severity="block",
                        observed={"risk_quality": risk_quality, "risk_score": risk_score},
                    )
                )
    return checks


def _portfolio_constraint_checks(
    action_id: str,
    payload: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    positions = _candidate_positions(action_id, payload)
    if positions is None:
        record = _recommendation_record(payload)
        if str(record.get("action") or "").lower() in ACTIONABLE_RECOMMENDATION_ACTIONS and not (
            record.get("ticker") or record.get("instrument")
        ):
            checks.append(
                _check(
                    "instrument",
                    "fail",
                    "unknown_instrument",
                    "Actionable recommendation has no identifiable instrument.",
                    severity="fail",
                )
            )
        return checks

    exposures = _position_exposures(positions, hedge_action=action_id == "update_hedge_positions")
    for row in exposures:
        if row["notional"] is not None:
            continue
        status = str(row.get("valuation_status") or "missing_position_inputs")
        is_fx_missing = status in {"missing_currency", "missing_fx_rate"}
        checks.append(
            _check(
                "portfolio.valuation.fx_missing" if is_fx_missing else "portfolio.valuation.inputs_missing",
                "fail" if is_fx_missing else "warn",
                "data_missing",
                f"{row['ticker']} cannot be valued in base currency: {status.replace('_', ' ')}.",
                severity="fail" if is_fx_missing else "warn",
                observed={
                    "ticker": row["ticker"],
                    "currency": row.get("currency"),
                    "base_currency": row.get("base_currency"),
                    "valuation_status": status,
                },
            )
        )

    valued_exposures = [row for row in exposures if row["notional"] is not None]
    total_abs = sum(abs(row["notional"]) for row in valued_exposures) or 0.0
    total_net = sum(row["notional"] for row in valued_exposures)
    if total_abs <= 0:
        checks.append(
            _check(
                "portfolio.notional",
                "warn",
                "data_missing",
                "Portfolio notionals are unavailable; using incomplete sizing context.",
                severity="warn",
            )
        )
        return checks

    book_size = _book_size_for_concentration(payload, fallback=total_abs)

    max_position = float(_deep_get(policy, ("policy", "max_position_weight_pct")) or 0)
    for row in valued_exposures:
        if row.get("is_hedge"):
            continue
        weight = abs(row["notional"]) / book_size
        if max_position and weight > max_position:
            checks.append(
                _check(
                    "concentration.position",
                    "fail",
                    "concentration_limit",
                    f"{row['ticker']} exceeds max position concentration.",
                    severity="fail",
                    observed=round(weight, 4),
                    limit=max_position,
                )
            )

    max_asset_class = float(_deep_get(policy, ("policy", "max_asset_class_weight_pct")) or 0)
    by_asset: dict[str, float] = {}
    for row in valued_exposures:
        by_asset[row["asset"]] = by_asset.get(row["asset"], 0.0) + abs(row["notional"])
    for asset, notional in by_asset.items():
        weight = notional / book_size
        if max_asset_class and weight > max_asset_class:
            checks.append(
                _check(
                    "concentration.asset_class",
                    "fail",
                    "concentration_limit",
                    f"{asset} exposure exceeds asset-class concentration limit.",
                    severity="fail",
                    observed=round(weight, 4),
                    limit=max_asset_class,
                )
            )

    gross_leverage = _payload_number(payload, "gross_leverage")
    if gross_leverage is None:
        gross_leverage = _payload_number(payload, "target_leverage")
    if gross_leverage is not None:
        max_gross = float(_deep_get(policy, ("policy", "max_gross_leverage")) or 0)
        if max_gross and gross_leverage > max_gross:
            checks.append(
                _check(
                    "leverage.gross",
                    "fail",
                    "leverage_limit",
                    "Gross leverage exceeds policy limit.",
                    severity="fail",
                    observed=gross_leverage,
                    limit=max_gross,
                )
            )
    net_leverage = _payload_number(payload, "net_leverage")
    if net_leverage is None and total_abs:
        net_leverage = abs(total_net) / total_abs
    max_net = float(_deep_get(policy, ("policy", "max_net_leverage")) or 0)
    if net_leverage is not None and max_net and net_leverage > max_net:
        checks.append(
            _check(
                "leverage.net",
                "fail",
                "leverage_limit",
                "Net leverage exceeds policy limit.",
                severity="fail",
                observed=round(net_leverage, 4),
                limit=max_net,
            )
        )

    return checks


def _liquidity_checks(
    payload: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    exit_days = _payload_number(payload, "estimated_exit_days")
    max_exit = _deep_get(policy, ("policy", "max_exit_days"))
    if exit_days is not None and max_exit is not None and exit_days > float(max_exit):
        checks.append(
            _check(
                "liquidity.exit_days",
                "fail",
                "liquidity_shortfall",
                "Estimated exit time exceeds policy liquidity window.",
                severity="fail",
                observed=exit_days,
                limit=max_exit,
            )
        )
    return checks


def _scenario_checks(
    action_id: str,
    payload: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    stress_loss = _payload_number(payload, "stress_loss_pct")
    if stress_loss is None:
        stress_loss = _payload_number(payload, "scenario_stress_loss_pct")
    max_stress = float(_deep_get(policy, ("policy", "max_stress_loss_pct")) or 0)
    if stress_loss is not None and max_stress and abs(stress_loss) > max_stress:
        checks.append(
            _check(
                "scenario.stress_loss",
                "fail",
                "scenario_stress_fail",
                "Scenario stress loss exceeds policy tolerance.",
                severity="fail",
                observed=stress_loss,
                limit=max_stress,
            )
        )
    drawdown = _payload_number(payload, "drawdown_pct")
    max_drawdown = float(_deep_get(policy, ("policy", "max_drawdown_pct")) or 0)
    if drawdown is not None and max_drawdown and abs(drawdown) > max_drawdown:
        checks.append(
            _check(
                "risk.drawdown",
                "fail",
                "drawdown_limit",
                "Drawdown exceeds policy tolerance.",
                severity="fail",
                observed=drawdown,
                limit=max_drawdown,
            )
        )
    daily_vol = _payload_number(payload, "daily_volatility_pct")
    max_vol = float(_deep_get(policy, ("policy", "max_daily_volatility_pct")) or 0)
    if daily_vol is not None and max_vol and daily_vol > max_vol:
        checks.append(
            _check(
                "risk.volatility",
                "fail",
                "volatility_limit",
                "Daily volatility exceeds policy tolerance.",
                severity="fail",
                observed=daily_vol,
                limit=max_vol,
            )
        )
    return checks


def _candidate_positions(action_id: str, payload: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    if action_id in {"update_portfolio_positions", "update_hedge_positions"}:
        positions = payload.get("positions")
        return positions if isinstance(positions, list) else []
    return None


def _position_exposures(positions: list[Mapping[str, Any]], *, hedge_action: bool = False) -> list[dict[str, Any]]:
    exposures: list[dict[str, Any]] = []
    for raw in positions:
        if not isinstance(raw, Mapping):
            continue
        ticker = str(raw.get("ticker") or "").upper()
        asset = str(raw.get("asset") or "equity").lower()
        direction = str(raw.get("direction") or "long").lower()
        role = str(raw.get("role") or "").strip().lower()
        position_type = str(raw.get("type") or raw.get("position_type") or "").strip().lower()
        is_hedge = hedge_action or role == "hedge" or position_type == "hedge"
        currency = str(raw.get("currency") or "").strip() or None
        base_currency = str(raw.get("base_currency") or "USD").strip().upper() or "USD"
        valuation_status = str(raw.get("valuation_status") or "").strip()
        notional = _to_float(raw.get("notional_base"))
        if notional is None:
            if valuation_status in {"missing_currency", "missing_fx_rate"}:
                pass
            elif _position_needs_fx_conversion(ticker, currency, base_currency):
                valuation_status = "missing_fx_rate" if currency else "missing_currency"
            else:
                cost_basis = _to_float(raw.get("cost_basis"))
                quantity = _to_float(raw.get("quantity") if raw.get("quantity") is not None else raw.get("shares"))
                multiplier = _to_float(raw.get("contract_multiplier")) or 1.0
                if cost_basis is not None and quantity is not None and multiplier > 0:
                    notional = abs(cost_basis * quantity * multiplier)
                    valuation_status = "ok"
                elif not valuation_status:
                    valuation_status = "missing_position_inputs"
        if notional is not None and direction == "short":
            notional *= -1
        exposures.append(
            {
                "ticker": ticker or "UNKNOWN",
                "asset": asset,
                "direction": direction,
                "is_hedge": is_hedge,
                "notional": notional,
                "currency": currency,
                "base_currency": base_currency,
                "valuation_status": valuation_status or ("ok" if notional is not None else "missing_position_inputs"),
            }
        )
    return exposures


def _position_needs_fx_conversion(ticker: str, currency: str | None, base_currency: str) -> bool:
    if currency:
        return currency.upper() != base_currency.upper()
    try:
        from portfolio.valuation import fallback_market_metadata

        fallback_currency = str(fallback_market_metadata(ticker).get("currency") or "").strip().upper()
    except Exception:
        fallback_currency = ""
    return bool(fallback_currency and fallback_currency != base_currency.upper())


def _recommendation_record(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    record = payload.get("record")
    if isinstance(record, Mapping):
        return dict(record)
    return dict(payload)


def _financial_policy_facts(
    action_id: str,
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None,
    source_quality: Mapping[str, Any] | None,
) -> FinancialPolicyFacts:
    record = _recommendation_record(payload)
    ctx = dict(context or {})
    actor_roles = ctx.get("actor_roles")
    if not isinstance(actor_roles, (list, tuple, set)):
        actor_roles = []
    request_mode = str(ctx.get("request_mode") or "").strip().lower()
    source_id = str(ctx.get("source_id") or "").strip().lower()
    if not request_mode:
        if source_id.startswith("break_glass."):
            request_mode = "break_glass"
        elif source_id.endswith(".self_apply") or ".self_apply" in source_id:
            request_mode = "self_apply"
        else:
            request_mode = "proposal"
    return FinancialPolicyFacts(
        action_id=str(action_id or ""),
        action_kind=_financial_action_kind(action_id, payload),
        request_mode=request_mode,
        actor_id=str(ctx.get("actor_id") or ""),
        actor_roles=tuple(str(role) for role in actor_roles if str(role).strip()),
        account_id=str(
            record.get("account_id") or payload.get("account_id") or DEFAULT_POLICY["account"]["account_id"]
        ),
        portfolio_id=str(
            record.get("portfolio_id") or payload.get("portfolio_id") or DEFAULT_POLICY["portfolio"]["portfolio_id"]
        ),
        risk_level=_risk_level_for_policy(record, payload),
        data_freshness=_data_freshness_for_policy(record, source_quality=source_quality),
    )


def _financial_action_kind(action_id: str, payload: Mapping[str, Any]) -> str:
    action = str(action_id or "").strip()
    if action == "update_portfolio_positions":
        return "portfolio_positions"
    if action == "update_hedge_positions":
        return "hedge_positions"
    if action == "create_action_item":
        return str(payload.get("action_type") or "action_item").strip().lower() or "action_item"
    if action in {"create_course_of_action", "create_recommendation"}:
        record = _recommendation_record(payload)
        return str(record.get("action") or "recommendation").strip().lower() or "recommendation"
    return action


def _data_freshness_for_policy(
    record: Mapping[str, Any],
    *,
    source_quality: Mapping[str, Any] | None,
) -> str:
    qualities = [
        record.get("critical_data_quality"),
        record.get("source_quality"),
        record.get("risk_quality"),
        (source_quality or {}).get("critical_data_quality"),
        (source_quality or {}).get("overall_status"),
        (source_quality or {}).get("quality"),
    ]
    normalized = {str(item or "").strip().lower() for item in qualities if str(item or "").strip()}
    if "failed" in normalized:
        return "failed"
    if "stale" in normalized:
        return "stale"
    if "degraded" in normalized or "insufficient" in normalized:
        return "degraded"
    if "ok" in normalized or "fresh" in normalized:
        return "ok"
    return "missing"


def _risk_level_for_policy(record: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    for value in (record.get("risk_level"), payload.get("risk_level")):
        normalized = str(value or "").strip().lower()
        if normalized in {"low", "medium", "high", "unknown"}:
            return normalized
    risk_score = _risk_score_from_recommendation(record)
    if risk_score is None:
        risk_score = _payload_number(payload, "risk_score")
    if risk_score is None:
        return "unknown"
    if risk_score >= 0.75:
        return "high"
    if risk_score >= 0.5:
        return "medium"
    return "low"


def _existing_gate_result(action_id: str, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    if action_id in {"create_course_of_action", "create_recommendation"}:
        record = _recommendation_record(payload)
        existing = record.get("policy_gate_result")
        return dict(existing) if isinstance(existing, Mapping) else None
    existing = payload.get("policy_gate_result")
    return dict(existing) if isinstance(existing, Mapping) else None


def _attach_gate_to_payload(action_id: str, payload: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    if action_id in {"create_course_of_action", "create_recommendation"}:
        record = _recommendation_record(payload)
        _apply_gate_fields(record, gate)
        payload["record"] = record
        return payload
    payload["policy_gate_result"] = gate
    payload["policy_gate_decision"] = gate["decision"]
    payload["policy_gate_review_required"] = gate["review_required"]
    payload["policy_gate_rule_id"] = gate.get("rule_id")
    payload["approval_mode"] = gate.get("approval_mode")
    payload["approval_required"] = gate.get("approval_required")
    return payload


def _apply_gate_fields(record: dict[str, Any], gate: dict[str, Any]) -> None:
    record["policy_gate_result"] = gate
    record["policy_gate_result_id"] = gate.get("policy_gate_result_id")
    record["policy_gate_status"] = gate["decision"]
    record["policy_gate_decision"] = gate["decision"]
    record["policy_gate_review_required"] = bool(gate.get("review_required"))
    record["policy_gate_failures"] = gate.get("failure_reasons", [])
    record["policy_gate_warnings"] = gate.get("warnings", [])
    record["policy_gate_disclosures"] = gate.get("disclosures", [])
    record["policy_gate_rule_id"] = gate.get("rule_id")
    record["approval_mode"] = gate.get("approval_mode")
    record["approval_required"] = gate.get("approval_required")
    record["account_id"] = gate.get("account_id")
    record["portfolio_id"] = gate.get("portfolio_id")
    record["policy_id"] = gate.get("policy_id")


def _gate_summary(gate: Mapping[str, Any]) -> str:
    if str(gate.get("reason") or "").strip() and str(gate.get("decision") or "").strip().lower() == "blocked":
        return str(gate.get("reason"))
    reasons = gate.get("failure_reasons") or gate.get("warnings") or []
    if isinstance(reasons, list) and reasons:
        first = reasons[0]
        if isinstance(first, Mapping):
            return str(first.get("message") or first.get("code") or "Policy gate blocked the action")
        return str(first)
    return "Policy gate blocked the action"


def _asset_classes_from_payload(payload: Mapping[str, Any]) -> set[str]:
    positions = payload.get("positions")
    if isinstance(positions, list):
        return {
            str(row.get("asset") or "equity").lower()
            for row in positions
            if isinstance(row, Mapping) and str(row.get("asset") or "").strip()
        }
    record = _recommendation_record(payload)
    instrument = str(record.get("instrument") or "")
    if instrument.lower() in {"equity", "commodity", "fx", "bond"}:
        return {instrument.lower()}
    return set()


def _payload_number(payload: Mapping[str, Any], key: str) -> float | None:
    if key in payload:
        return _to_float(payload.get(key))
    record = _recommendation_record(payload)
    if key in record:
        return _to_float(record.get(key))
    return None


def _book_size_for_concentration(payload: Mapping[str, Any], *, fallback: float) -> float:
    for key in ("book_size", "book"):
        value = _payload_number(payload, key)
        if value is not None and value > 0:
            return value
    try:
        from api.portfolio_settings import get_portfolio_book_size

        configured = _to_float(get_portfolio_book_size())
    except Exception:
        configured = None
    if configured is not None and configured > 0:
        return configured
    return fallback


def _risk_score_from_recommendation(record: Mapping[str, Any]) -> float | None:
    direct = _to_float(record.get("risk_score"))
    if direct is not None:
        return direct
    bindings = record.get("risk_bindings")
    if isinstance(bindings, Mapping):
        for key in ("risk_score", "position_risk_score", "portfolio_risk_score", "average_risk_score"):
            value = _to_float(bindings.get(key))
            if value is not None:
                return value
        position = bindings.get("position")
        if isinstance(position, Mapping):
            value = _to_float(position.get("risk_score"))
            if value is not None:
                return value
        portfolio = bindings.get("portfolio")
        if isinstance(portfolio, Mapping):
            for key in ("risk_score", "average_risk_score"):
                value = _to_float(portfolio.get(key))
                if value is not None:
                    return value
    return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1]
        try:
            return float(text) / 100.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _deep_get(mapping: Mapping[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_reason_entries(value: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            entries.append(dict(item))
        elif str(item).strip():
            entries.append({"code": "review_warning", "message": str(item)})
    return entries


def _scenario_results_from_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scenario": check["check"],
            "status": check["status"],
            "observed": check.get("observed"),
            "limit": check.get("limit"),
        }
        for check in checks
        if str(check.get("check") or "").startswith("scenario.")
    ]


def _metrics_from_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for check in checks:
        observed = check.get("observed")
        if observed is not None:
            metrics[str(check.get("check"))] = observed
    return metrics


def _assumptions(policy: Mapping[str, Any]) -> list[str]:
    return [
        "Current prices and account cash may be incomplete unless supplied by the caller.",
    ]


def _uncertainty(checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "level": "medium",
        "notes": [],
    }
