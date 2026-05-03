"""Deterministic MVP watch-trigger evaluator."""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Any

OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op not in OPS:
        raise ValueError(f"Unsupported trigger operator: {op}")
    try:
        actual_f = float(actual)
        expected_f = float(expected)
        return bool(OPS[op](actual_f, expected_f))
    except (TypeError, ValueError):
        return bool(OPS[op](str(actual), str(expected)))


def _nested_get(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _latest_price(ticker: str) -> dict[str, Any]:
    import yfinance as yf

    hist = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist:
        raise RuntimeError(f"No close price history for {ticker}")
    close = hist["Close"]
    if getattr(close, "ndim", 1) > 1:
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise RuntimeError(f"Empty close price history for {ticker}")
    latest_idx = close.index[-1]
    return {"value": float(close.iloc[-1]), "as_of": str(getattr(latest_idx, "date", lambda: latest_idx)())}


def _evaluate_price_level(definition: dict[str, Any], fallback_ticker: str | None) -> dict[str, Any]:
    ticker = str(definition.get("ticker") or fallback_ticker or "").upper()
    if not ticker:
        raise ValueError("price_level trigger requires ticker")
    price = _latest_price(ticker)
    op = str(definition.get("operator") or definition.get("op") or ">=")
    threshold = definition.get("threshold", definition.get("value"))
    fired = _compare(price["value"], op, threshold)
    return {
        "fired": fired,
        "actual": price["value"],
        "operator": op,
        "expected": threshold,
        "evidence": f"{ticker} close {price['value']:.2f} {op} {threshold}",
        "as_of": price["as_of"],
    }


def _evaluate_technical(definition: dict[str, Any], fallback_ticker: str | None) -> dict[str, Any]:
    ticker = str(definition.get("ticker") or fallback_ticker or "").upper()
    if not ticker:
        raise ValueError("technical trigger requires ticker")
    from portfolio.technical_analysis.technical_analysis import get_data

    data = get_data(ticker, lookback=str(definition.get("lookback") or "2Y"))
    summary = data.get("summary") if isinstance(data, dict) else []
    indicator_contains = str(definition.get("indicator_contains") or definition.get("indicator") or "").lower()
    field = str(definition.get("field") or "Signal")
    op = str(definition.get("operator") or definition.get("op") or "==")
    expected = definition.get("expected", definition.get("value"))
    matches = []
    for row in summary if isinstance(summary, list) else []:
        if not isinstance(row, dict):
            continue
        indicator = str(row.get("Indicator") or "")
        if indicator_contains and indicator_contains not in indicator.lower():
            continue
        actual = row.get(field)
        if _compare(actual, op, expected):
            matches.append({"indicator": indicator, "field": field, "actual": actual})
    fired = bool(matches)
    return {
        "fired": fired,
        "actual": matches,
        "operator": op,
        "expected": expected,
        "evidence": f"{ticker} technical trigger matched {len(matches)} row(s)",
        "as_of": str(data.get("timestamp")) if isinstance(data, dict) else None,
    }


def _evaluate_macro(definition: dict[str, Any], _fallback_ticker: str | None) -> dict[str, Any]:
    from api.signal_snapshot import get_signal_aggregator_snapshot_or_module_response

    data = get_signal_aggregator_snapshot_or_module_response(lookback_weeks=156, include_raw_modules=False)
    if data is None:
        from api.signal_aggregator import build_signal_aggregator

        data = build_signal_aggregator(include_history=False)
    field_path = str(definition.get("field") or "regime.score")
    actual = _nested_get(data, field_path) if isinstance(data, dict) else None
    op = str(definition.get("operator") or definition.get("op") or ">=")
    expected = definition.get("threshold", definition.get("value"))
    fired = _compare(actual, op, expected)
    return {
        "fired": fired,
        "actual": actual,
        "operator": op,
        "expected": expected,
        "evidence": f"macro {field_path}={actual} {op} {expected}",
        "as_of": str(_nested_get(data, "_meta.snapshot.as_of")) if isinstance(data, dict) else None,
    }


def evaluate_trigger(trigger: dict[str, Any]) -> dict[str, Any]:
    definition = trigger.get("definition_json")
    if not isinstance(definition, dict) or not definition:
        return {
            "fired": False,
            "skipped": True,
            "evidence": "Trigger has no machine-readable definition.",
        }
    trigger_type = str(definition.get("type") or trigger.get("trigger_type") or "").lower()
    fallback_ticker = trigger.get("ticker")
    if trigger_type == "price_level":
        return _evaluate_price_level(definition, fallback_ticker)
    if trigger_type == "technical":
        return _evaluate_technical(definition, fallback_ticker)
    if trigger_type == "macro":
        return _evaluate_macro(definition, fallback_ticker)
    return {"fired": False, "skipped": True, "evidence": f"Unsupported trigger type: {trigger_type}"}


def run_watch_trigger_monitor(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    from portfolio.core_db import (
        create_action_item_once,
        fire_watch_trigger,
        get_watch_triggers,
        update_watch_trigger_check,
    )

    checked = 0
    fired = 0
    skipped = 0
    errors = 0
    for trigger in get_watch_triggers(status="active"):
        checked += 1
        trigger_id = int(trigger["id"])
        try:
            result = evaluate_trigger(trigger)
            evidence = str(result.get("evidence") or "")
            if result.get("skipped"):
                skipped += 1
                update_watch_trigger_check(trigger_id, result=result, evidence=evidence)
                continue
            if result.get("fired"):
                fired += 1
                updated = fire_watch_trigger(trigger_id, result=result, evidence=evidence)
                create_action_item_once(
                    description=f"Review fired watch trigger: {updated.get('condition')}",
                    action_type="review",
                    ticker=updated.get("ticker"),
                    urgency="high",
                    source_type="workflow",
                    source_id=f"watch_trigger:{trigger_id}:{datetime.now(UTC).date().isoformat()}",
                )
            else:
                update_watch_trigger_check(trigger_id, result=result, evidence=evidence)
        except Exception as exc:
            errors += 1
            update_watch_trigger_check(
                trigger_id,
                result={"error": str(exc), "fired": False},
                evidence=str(exc),
            )
    return {"checked": checked, "fired": fired, "skipped": skipped, "errors": errors}
