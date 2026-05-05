"""Deterministic financial policy gate for recommendations and proposals.

The gate is intentionally deterministic and conservative. It does not decide
whether an investment idea is good; it decides whether the idea has enough
account, mandate, risk, freshness, and disclosure context to be staged for
human review.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

POLICY_GATE_DECISIONS = ("pass", "warn", "review_required", "blocked", "error")
ACTIONABLE_RECOMMENDATION_ACTIONS = {"buy", "sell", "reduce", "exit", "rebalance", "hedge"}
FINANCIAL_ACTION_ITEM_TYPES = {"enter", "exit", "resize", "hedge"}
FINANCIAL_ACTION_IDS = {"create_recommendation", "update_portfolio_positions", "update_hedge_positions"}

FAILURE_REASON_CODES = {
    "data_missing",
    "missing_constraint",
    "mandate_violation",
    "suitability_warning",
    "horizon_mismatch",
    "liquidity_shortfall",
    "concentration_limit",
    "leverage_limit",
    "volatility_limit",
    "drawdown_limit",
    "cash_shortfall",
    "tax_flag",
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
        "suitability_profile": "unspecified",
    },
    "account": {
        "account_id": "default-account",
        "account_type": "unspecified",
        "tax_status": "unknown",
    },
    "portfolio": {
        "portfolio_id": "default-portfolio",
        "base_currency": "USD",
        "cash": None,
        "benchmark": "SPY",
    },
    "mandate": {
        "mandate_id": "default-mandate",
        "permitted_asset_classes": ["equity", "commodity", "fx", "bond"],
        "permitted_actions": sorted(ACTIONABLE_RECOMMENDATION_ACTIONS | {"hold", "watch", "avoid", "do_nothing"}),
        "benchmark": "SPY",
        "time_horizon_days_min": None,
        "time_horizon_days_max": None,
        "liquidity_needs": None,
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
        "min_cash_reserve_pct": None,
        "max_exit_days": 3,
        "taxable_account_rules": None,
    },
}

MISSING_CONSTRAINT_PATHS: tuple[tuple[str, ...], ...] = (
    ("investor", "suitability_profile"),
    ("account", "account_type"),
    ("account", "tax_status"),
    ("mandate", "time_horizon_days_min"),
    ("mandate", "time_horizon_days_max"),
    ("mandate", "liquidity_needs"),
    ("policy", "min_cash_reserve_pct"),
    ("policy", "taxable_account_rules"),
)

DECISION_SUPPORT_DISCLOSURES = [
    "Decision support only; human approval required.",
    "Policy gate checks deterministic constraints and data quality; it does not certify suitability.",
    "Missing investor, account, tax, or mandate constraints are surfaced as warnings in v1.",
]


def default_policy_snapshot() -> dict[str, Any]:
    return deepcopy(DEFAULT_POLICY)


def is_financial_action(action_id: str, payload: Mapping[str, Any] | None = None) -> bool:
    action = str(action_id or "").strip()
    if action in {"update_portfolio_positions", "update_hedge_positions"}:
        return True
    if action == "create_recommendation":
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
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Attach or reuse a policy gate result for a financial action payload."""
    mutable = deepcopy(dict(payload))
    if not is_financial_action(action_id, mutable):
        return mutable, None

    existing = _existing_gate_result(action_id, mutable)
    if existing:
        gate = normalize_policy_gate_result(existing)
        if gate["decision"] == "blocked" and action_id != "create_recommendation":
            raise PolicyGateBlockedError(_gate_summary(gate))
        return mutable, gate

    gate = evaluate_policy_gate(action_id, mutable, context=context)
    from api.provenance import stable_hash
    from portfolio import core_db

    target_id = stable_hash({"action_id": action_id, "payload": mutable})
    existing_rows = core_db.list_policy_gate_results(
        action_id=action_id,
        target_type="action_payload",
        target_id=target_id,
        limit=1,
    )
    if existing_rows:
        persisted_gate = existing_rows[0].get("result_json")
        if isinstance(persisted_gate, Mapping):
            gate = normalize_policy_gate_result(persisted_gate)
        gate["policy_gate_result_id"] = existing_rows[0]["id"]
    else:
        persisted = core_db.create_policy_gate_result(
            gate,
            action_id=action_id,
            source_type=str((context or {}).get("source_type") or "policy_gate"),
            source_id=str((context or {}).get("source_id") or (context or {}).get("proposal_action_run_id") or ""),
            target_type="action_payload",
            target_id=target_id,
            payload=mutable,
        )
        gate["policy_gate_result_id"] = persisted["id"]
    if gate["decision"] == "blocked":
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
    """Evaluate deterministic suitability, mandate, risk, and data checks."""
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

    policy = default_policy_snapshot()
    check_results: list[dict[str, Any]] = []
    check_results.extend(_missing_constraint_checks(policy))
    check_results.extend(_required_disclosure_checks(payload))
    check_results.extend(_data_freshness_checks(payload, source_quality=source_quality))
    check_results.extend(_mandate_checks(action_id, payload, policy))
    check_results.extend(_portfolio_constraint_checks(action_id, payload, policy))
    check_results.extend(_horizon_liquidity_checks(action_id, payload, policy))
    check_results.extend(_tax_checks(payload, policy))
    check_results.extend(_scenario_checks(action_id, payload, policy))

    return _result(action_id=action_id, decision=None, check_results=check_results, context=context, policy=policy)


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
        "mandate_id": policy_snapshot["mandate"]["mandate_id"],
        "evaluated_at": datetime.now(UTC).isoformat(),
        "action_id": action_id,
    }
    if ctx:
        gate["context"] = ctx
    return normalize_policy_gate_result(gate)


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
        reason_code = "suitability_warning"
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
        "code": check.get("reason_code") or "suitability_warning",
        "check": check.get("check"),
        "message": check.get("message"),
        "observed": check.get("observed"),
        "limit": check.get("limit"),
    }


def _missing_constraint_checks(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for path in MISSING_CONSTRAINT_PATHS:
        value = _deep_get(policy, path)
        if value in (None, "", [], {}):
            label = ".".join(path)
            checks.append(
                _check(
                    label,
                    "warn",
                    "missing_constraint",
                    f"Missing investor/account constraint: {label}.",
                    severity="warn",
                )
            )
    return checks


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
                        "Actionable recommendation requires a first-class risk score.",
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
                        "First-class risk is degraded; human review is required before action.",
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
                        "Actionable recommendation requires linked first-class risk snapshots.",
                        severity="block",
                        observed={"risk_quality": risk_quality, "risk_score": risk_score},
                    )
                )
    return checks


def _mandate_checks(action_id: str, payload: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    record = _recommendation_record(payload)
    action = str(record.get("action") or payload.get("action_type") or action_id).lower()
    permitted = {str(item).lower() for item in _as_list(_deep_get(policy, ("mandate", "permitted_actions")))}
    if action in ACTIONABLE_RECOMMENDATION_ACTIONS and action not in permitted:
        checks.append(
            _check(
                "mandate.permitted_actions",
                "fail",
                "mandate_violation",
                f"Action {action!r} is outside the mandate.",
                severity="fail",
                observed=action,
                limit=sorted(permitted),
            )
        )
    asset_classes = _asset_classes_from_payload(payload)
    permitted_assets = {
        str(item).lower() for item in _as_list(_deep_get(policy, ("mandate", "permitted_asset_classes")))
    }
    for asset_class in sorted(asset_classes - permitted_assets):
        checks.append(
            _check(
                "mandate.permitted_asset_classes",
                "fail",
                "mandate_violation",
                f"Asset class {asset_class!r} is outside the mandate.",
                severity="fail",
                observed=asset_class,
                limit=sorted(permitted_assets),
            )
        )
    benchmark = str(_deep_get(policy, ("mandate", "benchmark")) or "")
    portfolio_benchmark = str(_deep_get(policy, ("portfolio", "benchmark")) or "")
    if benchmark and portfolio_benchmark and benchmark != portfolio_benchmark:
        checks.append(
            _check(
                "benchmark",
                "fail",
                "benchmark_mismatch",
                "Portfolio benchmark does not match the governing mandate.",
                severity="fail",
                observed=portfolio_benchmark,
                limit=benchmark,
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

    exposures = _position_exposures(positions)
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
                "missing_constraint",
                "Portfolio notionals are unavailable; using incomplete sizing context.",
                severity="warn",
            )
        )
        return checks

    max_position = float(_deep_get(policy, ("policy", "max_position_weight_pct")) or 0)
    for row in valued_exposures:
        weight = abs(row["notional"]) / total_abs
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
        weight = notional / total_abs
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

    cash = _payload_number(payload, "cash")
    min_cash = _deep_get(policy, ("policy", "min_cash_reserve_pct"))
    if cash is not None and min_cash is not None and total_abs and cash / total_abs < float(min_cash):
        checks.append(
            _check(
                "cash.reserve",
                "fail",
                "cash_shortfall",
                "Cash reserve is below the policy minimum.",
                severity="fail",
                observed=round(cash / total_abs, 4),
                limit=float(min_cash),
            )
        )
    return checks


def _horizon_liquidity_checks(
    action_id: str,
    payload: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    record = _recommendation_record(payload)
    horizon_days = _horizon_days(record.get("horizon"))
    min_days = _deep_get(policy, ("mandate", "time_horizon_days_min"))
    max_days = _deep_get(policy, ("mandate", "time_horizon_days_max"))
    if horizon_days is not None and min_days is not None and horizon_days < float(min_days):
        checks.append(
            _check(
                "horizon.minimum",
                "fail",
                "horizon_mismatch",
                "Recommendation horizon is shorter than mandate minimum.",
                severity="fail",
                observed=horizon_days,
                limit=min_days,
            )
        )
    if horizon_days is not None and max_days is not None and horizon_days > float(max_days):
        checks.append(
            _check(
                "horizon.maximum",
                "fail",
                "horizon_mismatch",
                "Recommendation horizon is longer than mandate maximum.",
                severity="fail",
                observed=horizon_days,
                limit=max_days,
            )
        )
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


def _tax_checks(payload: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    account = _deep_get(policy, ("account",), default={}) or {}
    checks: list[dict[str, Any]] = []
    tax_status = str(account.get("tax_status") or "unknown").lower()
    record = _recommendation_record(payload)
    if tax_status in {"taxable", "unknown"} and str(record.get("action") or "").lower() in {"sell", "reduce", "exit"}:
        checks.append(
            _check(
                "tax.taxable_account",
                "warn",
                "tax_flag",
                "Tax impact must be reviewed before reducing or exiting taxable/unknown-tax-status accounts.",
                severity="warn",
                observed=tax_status,
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


def _position_exposures(positions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    exposures: list[dict[str, Any]] = []
    for raw in positions:
        if not isinstance(raw, Mapping):
            continue
        ticker = str(raw.get("ticker") or "").upper()
        asset = str(raw.get("asset") or "equity").lower()
        direction = str(raw.get("direction") or "long").lower()
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


def _existing_gate_result(action_id: str, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    if action_id == "create_recommendation":
        record = _recommendation_record(payload)
        existing = record.get("policy_gate_result")
        return dict(existing) if isinstance(existing, Mapping) else None
    existing = payload.get("policy_gate_result")
    return dict(existing) if isinstance(existing, Mapping) else None


def _attach_gate_to_payload(action_id: str, payload: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    if action_id == "create_recommendation":
        record = _recommendation_record(payload)
        _apply_gate_fields(record, gate)
        payload["record"] = record
        return payload
    payload["policy_gate_result"] = gate
    payload["policy_gate_decision"] = gate["decision"]
    payload["policy_gate_review_required"] = gate["review_required"]
    return payload


def _apply_gate_fields(record: dict[str, Any], gate: dict[str, Any]) -> None:
    record["policy_gate_result"] = gate
    record["policy_gate_status"] = gate["decision"]
    record["policy_gate_decision"] = gate["decision"]
    record["policy_gate_review_required"] = bool(gate.get("review_required"))
    record["policy_gate_failures"] = gate.get("failure_reasons", [])
    record["policy_gate_warnings"] = gate.get("warnings", [])
    record["policy_gate_disclosures"] = gate.get("disclosures", [])
    record["account_id"] = gate.get("account_id")
    record["portfolio_id"] = gate.get("portfolio_id")
    record["policy_id"] = gate.get("policy_id")


def _gate_summary(gate: Mapping[str, Any]) -> str:
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


def _horizon_days(value: Any) -> float | None:
    text = str(value or "").lower()
    number_match = re.search(r"(\d+(?:\.\d+)?)", text)
    number = float(number_match.group(1)) if number_match else 1.0
    if "month" in text:
        return number * 30
    if "year" in text:
        return number * 365
    if "week" in text:
        return number * 7
    if "day" in text:
        return number
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
            entries.append({"code": "suitability_warning", "message": str(item)})
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
        f"Default account {policy['account']['account_id']} governs v1 recommendations.",
        "Current prices, account cash, and liquidity needs may be incomplete unless supplied by the caller.",
    ]


def _uncertainty(checks: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [c for c in checks if c.get("reason_code") == "missing_constraint"]
    return {
        "level": "high" if missing else "medium",
        "missing_constraint_count": len(missing),
        "notes": ["Missing constraints are warnings in v1, not hard blocks."] if missing else [],
    }
