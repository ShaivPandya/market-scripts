"""Continuous catalyst and kill-condition monitor with MonitorHit ontology events."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any

from api.generated_approval_filters import should_suppress_generated_review_approval
from api.watch_trigger_monitor import _canonical_hash, _compare, _latest_price

APPROACHING_DAYS = 14
STATUS_PROPOSAL_CONFIDENCE = 0.75
APPROACHING_THRESHOLD_RATIO = 0.05


def _monitor_entity_source_id(entity_type: str, entity_id: Any) -> str:
    text = str(entity_id or "").strip()
    prefix = f"{entity_type}:"
    if text.startswith(prefix):
        return text
    return f"{prefix}{text}"


def _result_fingerprint(result: dict[str, Any]) -> str:
    return _canonical_hash(
        {
            "hit_type": result.get("hit_type"),
            "entity_type": result.get("entity_type"),
            "entity_id": result.get("entity_id"),
            "actual": result.get("actual"),
            "expected": result.get("expected"),
            "operator": result.get("operator"),
            "target_date": result.get("target_date"),
        }
    )


def _parse_target_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_metric_threshold(metric: Any, threshold: Any) -> tuple[str | None, str | None, Any]:
    metric_text = str(metric or "").strip().lower()
    threshold_text = str(threshold or "").strip()
    if not metric_text or not threshold_text:
        return None, None, None
    match = re.match(r"^(>=|<=|>|<|==|!=)\s*(-?\d+(?:\.\d+)?)$", threshold_text.replace(" ", ""))
    if match:
        return metric_text, match.group(1), float(match.group(2))
    try:
        return metric_text, ">=", float(threshold_text)
    except ValueError:
        return None, None, None


def _metric_value(metric: str, ticker: str) -> float | None:
    metric_key = metric.strip().lower()
    if metric_key in {"price", "close", "last_price"}:
        try:
            return float(_latest_price(ticker)["value"])
        except Exception:
            return None
    return None


def evaluate_catalyst(catalyst: dict[str, Any]) -> dict[str, Any]:
    ticker = str(catalyst.get("ticker") or "").strip().upper()
    entity_id = str(catalyst.get("object_uid") or catalyst.get("id") or "").strip()
    label = str(catalyst.get("description") or catalyst.get("name") or "Catalyst").strip()
    status = str(catalyst.get("status") or "pending").strip().lower()
    if status != "pending":
        return {
            "entity_type": "catalyst",
            "entity_id": entity_id,
            "entity_label": label,
            "ticker": ticker,
            "hit_type": "skipped",
            "severity": "low",
            "confidence": 0.0,
            "evidence": f"Catalyst status is {status}; monitor only tracks pending catalysts.",
            "skipped": True,
        }

    target = _parse_target_date(catalyst.get("target_date"))
    today = datetime.now(UTC).date()
    if target is None:
        return {
            "entity_type": "catalyst",
            "entity_id": entity_id,
            "entity_label": label,
            "ticker": ticker,
            "hit_type": "needs_review",
            "severity": "medium",
            "confidence": 0.55,
            "evidence": "Pending catalyst has no target date to evaluate proximity.",
            "result": {"reason": "missing_target_date"},
        }

    days_until = (target - today).days
    if days_until < 0:
        return {
            "entity_type": "catalyst",
            "entity_id": entity_id,
            "entity_label": label,
            "ticker": ticker,
            "hit_type": "needs_review",
            "severity": "high",
            "confidence": 0.8,
            "evidence": f"Catalyst target date {target.isoformat()} has passed ({abs(days_until)} days ago).",
            "result": {"target_date": target.isoformat(), "days_until": days_until},
            "suggested_status": "played_out",
        }
    if days_until <= APPROACHING_DAYS:
        return {
            "entity_type": "catalyst",
            "entity_id": entity_id,
            "entity_label": label,
            "ticker": ticker,
            "hit_type": "approaching",
            "severity": "medium",
            "confidence": 0.7,
            "evidence": f"Catalyst target date {target.isoformat()} is {days_until} days away.",
            "result": {"target_date": target.isoformat(), "days_until": days_until},
        }
    return {
        "entity_type": "catalyst",
        "entity_id": entity_id,
        "entity_label": label,
        "ticker": ticker,
        "hit_type": "ok",
        "severity": "low",
        "confidence": 0.4,
        "evidence": f"Catalyst target date {target.isoformat()} is {days_until} days away.",
        "result": {"target_date": target.isoformat(), "days_until": days_until},
        "skipped": True,
    }


def evaluate_kill_condition(kill_condition: dict[str, Any]) -> dict[str, Any]:
    ticker = str(kill_condition.get("ticker") or "").strip().upper()
    entity_id = str(kill_condition.get("object_uid") or kill_condition.get("id") or "").strip()
    label = str(kill_condition.get("condition") or "Kill condition").strip()
    status = str(kill_condition.get("status") or "active").strip().lower()
    if status != "active":
        return {
            "entity_type": "kill_condition",
            "entity_id": entity_id,
            "entity_label": label,
            "ticker": ticker,
            "hit_type": "skipped",
            "severity": "low",
            "confidence": 0.0,
            "evidence": f"Kill condition status is {status}; monitor only tracks active conditions.",
            "skipped": True,
        }

    metric, operator, expected = _parse_metric_threshold(
        kill_condition.get("metric"),
        kill_condition.get("threshold"),
    )
    if metric is None or operator is None or expected is None or not ticker:
        return {
            "entity_type": "kill_condition",
            "entity_id": entity_id,
            "entity_label": label,
            "ticker": ticker,
            "hit_type": "skipped",
            "severity": "low",
            "confidence": 0.35,
            "evidence": "Kill condition metric/threshold is not machine-readable; skipped deterministic check.",
            "result": {"reason": "unparseable_metric_threshold"},
            "skipped": True,
        }

    actual = _metric_value(metric, ticker)
    if actual is None:
        return {
            "entity_type": "kill_condition",
            "entity_id": entity_id,
            "entity_label": label,
            "ticker": ticker,
            "hit_type": "needs_review",
            "severity": "medium",
            "confidence": 0.5,
            "evidence": f"Unable to fetch metric '{metric}' for {ticker}.",
            "result": {"metric": metric, "blocked": True},
        }

    fired = _compare(actual, operator, expected)
    expected_f = float(expected)
    distance = abs(actual - expected_f)
    approaching = not fired and expected_f != 0 and (distance / abs(expected_f)) <= APPROACHING_THRESHOLD_RATIO
    if fired:
        return {
            "entity_type": "kill_condition",
            "entity_id": entity_id,
            "entity_label": label,
            "ticker": ticker,
            "hit_type": "triggered",
            "severity": "high",
            "confidence": 0.85,
            "evidence": f"{ticker} {metric} {actual:.2f} {operator} {expected_f}",
            "result": {
                "metric": metric,
                "actual": actual,
                "operator": operator,
                "expected": expected_f,
                "fired": True,
            },
            "suggested_status": "triggered",
        }
    if approaching:
        return {
            "entity_type": "kill_condition",
            "entity_id": entity_id,
            "entity_label": label,
            "ticker": ticker,
            "hit_type": "approaching",
            "severity": "medium",
            "confidence": 0.65,
            "evidence": f"{ticker} {metric} {actual:.2f} is within {APPROACHING_THRESHOLD_RATIO:.0%} of {expected_f}",
            "result": {
                "metric": metric,
                "actual": actual,
                "operator": operator,
                "expected": expected_f,
                "approaching": True,
            },
        }
    return {
        "entity_type": "kill_condition",
        "entity_id": entity_id,
        "entity_label": label,
        "ticker": ticker,
        "hit_type": "ok",
        "severity": "low",
        "confidence": 0.4,
        "evidence": f"{ticker} {metric} {actual:.2f} is not near threshold {expected_f}",
        "result": {
            "metric": metric,
            "actual": actual,
            "operator": operator,
            "expected": expected_f,
        },
        "skipped": True,
    }


def _should_record_hit(result: dict[str, Any]) -> bool:
    if result.get("skipped"):
        return False
    hit_type = str(result.get("hit_type") or "").strip().lower()
    return hit_type not in {"ok"}


def _should_propose_status(result: dict[str, Any]) -> bool:
    if result.get("skipped"):
        return False
    confidence = float(result.get("confidence") or 0.0)
    if confidence < STATUS_PROPOSAL_CONFIDENCE:
        return False
    if result.get("result", {}).get("blocked"):
        return False
    return bool(result.get("suggested_status"))


def _existing_hit_fingerprints(reads: Any, entity_id: str) -> set[str]:
    fingerprints: set[str] = set()
    for row in reads.monitor_hits(entity_id=entity_id, limit=100):
        fingerprint = str(row.get("fingerprint") or "").strip()
        if fingerprint:
            fingerprints.add(fingerprint)
    return fingerprints


def run_catalyst_kill_monitor(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    from api.action_execution import execute_api_action
    from ontology.command_service import OntologyCommandContext, OntologyCommandService
    from ontology.policy import system_actor
    from ontology.runtime_read_service import OntologyRuntimeReadService

    command_service = OntologyCommandService()
    reads = OntologyRuntimeReadService()
    actor = system_actor("catalyst_kill_monitor")

    def propose_status(action_id: str, payload: dict[str, Any], *, source_id: str, reason: str) -> dict[str, Any]:
        return command_service.propose_action(
            action_id,
            payload,
            OntologyCommandContext(
                actor=actor,
                source_type="workflow",
                source_id=source_id,
            ),
            reason=reason,
        )

    def record_monitor_hit(payload: dict[str, Any], *, source_id: str) -> dict[str, Any]:
        return execute_api_action(
            "create_monitor_hit",
            payload,
            source_id=source_id,
            actor=actor,
            request_mode="self_apply",
        )

    checked = 0
    hits = 0
    skipped = 0
    proposals = 0
    errors = 0

    entities: list[tuple[str, dict[str, Any], Any]] = []
    for catalyst in reads.catalysts(status="pending"):
        entities.append(("catalyst", catalyst, evaluate_catalyst))
    for kill_condition in reads.kill_conditions(status="active"):
        entities.append(("kill_condition", kill_condition, evaluate_kill_condition))

    for entity_type, entity, evaluator in entities:
        checked += 1
        entity_id = str(entity.get("object_uid") or entity.get("id") or "").strip()
        if not entity_id:
            errors += 1
            continue
        entity_source_id = _monitor_entity_source_id(entity_type, entity_id)
        try:
            result = evaluator(entity)
            if not _should_record_hit(result):
                skipped += 1
                continue

            fingerprint = _result_fingerprint(result)
            if fingerprint in _existing_hit_fingerprints(reads, entity_id):
                skipped += 1
                continue

            hit_source_id = f"{entity_source_id}:{fingerprint}"
            hit_payload = {
                "ticker": result.get("ticker") or entity.get("ticker"),
                "entity_type": entity_type,
                "entity_id": entity_id,
                "entity_label": result.get("entity_label"),
                "hit_type": result.get("hit_type"),
                "severity": result.get("severity"),
                "confidence": result.get("confidence"),
                "evidence": result.get("evidence"),
                "result": result.get("result"),
                "fingerprint": fingerprint,
            }
            record_monitor_hit(hit_payload, source_id=hit_source_id)
            hits += 1

            if not _should_propose_status(result):
                continue

            suggested_status = str(result.get("suggested_status") or "").strip()
            status_action = "update_catalyst_status" if entity_type == "catalyst" else "update_kill_condition_status"
            status_payload: dict[str, Any] = {
                "ticker": result.get("ticker") or entity.get("ticker"),
                "status": suggested_status,
                "evidence": result.get("evidence"),
                "monitor_result": result.get("result"),
            }
            if entity_type == "catalyst":
                status_payload["catalyst_id"] = entity_id
            else:
                status_payload["kill_condition_id"] = entity_id

            approval = propose_status(
                status_action,
                status_payload,
                source_id=hit_source_id,
                reason=f"Monitor proposes {suggested_status} for {entity_type} {entity_id}",
            )
            proposals += 1

            action_item_payload = {
                "description": f"Review monitor hit for {entity_type.replace('_', ' ')}: {result.get('entity_label')}",
                "action_type": "review",
                "ticker": result.get("ticker") or entity.get("ticker"),
                "urgency": "high" if result.get("severity") == "high" else "normal",
                "alert_context": {
                    "change_summary": result.get("entity_label") or entity_id,
                    "source": "monitor_hit",
                    "ticker": result.get("ticker") or entity.get("ticker"),
                },
            }
            from decision_quality.proactive_alert_gate import apply_proactive_alert_gate

            action_item_payload, _gate_result = apply_proactive_alert_gate(
                "create_action_item",
                action_item_payload,
                source_type="workflow",
                alert_context=action_item_payload.get("alert_context"),
            )
            if not should_suppress_generated_review_approval(
                "create_action_item",
                action_item_payload,
                source_type="workflow",
            ):
                propose_status(
                    "create_action_item",
                    action_item_payload,
                    source_id=hit_source_id,
                    reason=f"Create review action item for monitor hit on {entity_id}",
                )
            _ = approval
        except Exception as exc:
            errors += 1
            try:
                error_fingerprint = _canonical_hash({"error": str(exc), "entity_id": entity_id})
                record_monitor_hit(
                    {
                        "ticker": entity.get("ticker"),
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "entity_label": entity.get("description") or entity.get("condition"),
                        "hit_type": "needs_review",
                        "severity": "medium",
                        "confidence": 0.3,
                        "evidence": str(exc),
                        "result": {"error": str(exc)},
                        "fingerprint": error_fingerprint,
                    },
                    source_id=f"{entity_source_id}:error:{error_fingerprint}",
                )
            except Exception:
                pass

    return {
        "checked": checked,
        "hits": hits,
        "skipped": skipped,
        "proposals": proposals,
        "errors": errors,
    }
