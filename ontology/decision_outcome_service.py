"""Decision learning loop: retrospective outcomes and post-mortems."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from ontology.object_service import OntologyObjectService
from ontology.policy import actor_to_dict, system_actor
from ontology.runtime_read_service import OntologyRuntimeReadService
from ontology.schemas.identity import course_of_action_id, decision_outcome_id, recommendation_id

OPERATIONAL_ONTOLOGY_RUN_ID = "operational"


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _horizon_days(horizon: str | None) -> int:
    text = (horizon or "").lower()
    if "trading day" in text or "1 day" in text:
        return 1
    if "week" in text:
        return 7
    if "month" in text:
        return 30
    return 7


def _expected_direction(action: str) -> str | None:
    if action in {"buy", "add"}:
        return "up"
    if action in {"short", "sell", "trim", "reduce", "exit", "avoid"}:
        return "down"
    return None


def _download_close_series(ticker: str, start: date, end: date):
    import yfinance as yf

    hist = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if hist is None or hist.empty or "Close" not in hist:
        raise RuntimeError(f"no close price history for {ticker}")
    close = hist["Close"]
    if hasattr(close, "iloc") and getattr(close, "ndim", 1) > 1:
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise RuntimeError(f"empty close series for {ticker}")
    return close


def _series_return_pct(close) -> float:
    start = float(close.iloc[0])
    end = float(close.iloc[-1])
    if start == 0:
        return 0.0
    return (end / start - 1.0) * 100.0


def _excursions_pct(close, expected_direction: str) -> tuple[float, float]:
    start = float(close.iloc[0])
    running = (close / start - 1.0) * 100.0
    if expected_direction == "up":
        return float(running.min()), float(running.max())
    return float(-running.max()), float(-running.min())


def _timing_label(as_of: date, today: date, horizon: str | None, expected_onset_window: str | None) -> str:
    elapsed_days = (today - as_of).days
    horizon_days = _horizon_days(horizon)
    onset_days = _horizon_days(expected_onset_window)
    if elapsed_days < max(1, min(onset_days, horizon_days)):
        return "too_early"
    if elapsed_days <= max(onset_days, horizon_days) * 2:
        return "on_time"
    return "late"


def _process_label(process_quality: str, outcome_quality: str) -> str:
    if process_quality == "inconclusive" or outcome_quality == "inconclusive":
        return "inconclusive"
    return f"{process_quality}_process_{outcome_quality}_outcome"


def _thesis_and_kill_context(ticker: str | None) -> dict[str, Any]:
    if not ticker:
        return {"thesis_validation": None, "kill_condition_status": None}
    context: dict[str, Any] = {"thesis_validation": None, "kill_condition_status": None}
    try:
        latest = OntologyRuntimeReadService().evaluations(ticker, limit=1)
        if latest:
            ev = latest[0]
            context["thesis_validation"] = {
                "evaluated_at": ev.get("evaluated_at"),
                "thesis_status": ev.get("thesis_status"),
                "action": ev.get("action"),
                "risk_flag": ev.get("risk_flag"),
            }
    except Exception:
        context["thesis_validation"] = {"status": "unavailable"}
    try:
        conditions = OntologyRuntimeReadService().kill_conditions(ticker)
        context["kill_condition_status"] = {
            "active": sum(1 for row in conditions if row.get("status") == "active"),
            "triggered": sum(1 for row in conditions if row.get("status") == "triggered"),
            "retired": sum(1 for row in conditions if row.get("status") == "retired"),
        }
    except Exception:
        context["kill_condition_status"] = {"status": "unavailable"}
    return context


def _outcome_key_for_recommendation(rec: Mapping[str, Any]) -> str:
    rec_uid = str(rec.get("object_uid") or rec.get("id") or rec.get("recommendation_id") or "")
    return f"rec:{rec_uid}"


def _outcome_key_for_course_of_action(coa: Mapping[str, Any]) -> str:
    coa_uid = str(coa.get("object_uid") or coa.get("course_of_action_id") or "")
    return f"coa:{coa_uid}"


def _write_decision_outcome(
    *,
    objects: OntologyObjectService,
    actor: Mapping[str, Any],
    outcome_key: str,
    source_kind: str,
    parent_uid: str,
    relation_type: str,
    rec_or_coa: Mapping[str, Any],
    outcome_status: str,
    outcome_payload: dict[str, Any],
    decision_quality_snapshot: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(UTC).isoformat()
    outcome_uid = decision_outcome_id(outcome_key)
    props = {
        "decision_outcome_id": outcome_key,
        "source_kind": source_kind,
        "recommendation_id": rec_or_coa.get("recommendation_id") if source_kind == "recommendation" else None,
        "course_of_action_id": rec_or_coa.get("course_of_action_id") if source_kind == "course_of_action" else None,
        "action_run_id": str(rec_or_coa.get("action_run_id")) if rec_or_coa.get("action_run_id") is not None else None,
        "ticker": rec_or_coa.get("ticker"),
        "as_of": rec_or_coa.get("as_of"),
        "horizon": rec_or_coa.get("horizon"),
        "outcome_status": outcome_status,
        "final_label_status": outcome_payload.get("final_label_status") or "draft",
        "evaluation_authority": outcome_payload.get("evaluation_authority") or "ai_draft_user_final",
        "process_label": outcome_payload.get("process_label"),
        "draft_postmortem": outcome_payload.get("draft_postmortem"),
        "metrics": {
            key: value
            for key, value in outcome_payload.items()
            if key
            not in {
                "evaluation_authority",
                "final_label_status",
                "process_label",
                "draft_postmortem",
                "final_postmortem",
                "lessons_learned",
            }
        },
        "decision_quality_snapshot": decision_quality_snapshot or rec_or_coa.get("decision_quality"),
        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
    }
    objects.write_object(
        "DecisionOutcome",
        outcome_uid,
        props,
        now,
        actor=actor,
        provenance=f"pv:decision_outcome:{outcome_key}",
    )
    objects.write_relation(
        parent_uid,
        outcome_uid,
        relation_type,
        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
        now,
        actor=actor,
        provenance=f"pv:decision_outcome:{outcome_key}",
    )
    return outcome_uid


def record_recommendation_outcome(
    rec: Mapping[str, Any],
    status: str,
    outcome: dict[str, Any],
    *,
    objects: OntologyObjectService | None = None,
    actor: Mapping[str, Any] | None = None,
) -> str:
    """Persist recommendation outcome on legacy payload and first-class DecisionOutcome."""
    objects = objects or OntologyObjectService()
    actor = actor or actor_to_dict(system_actor("recommendation_evaluator"))
    rec_uid = str(rec.get("object_uid") or rec.get("id") or rec.get("recommendation_id") or "")
    if not rec_uid.startswith("recommendation:"):
        rec_uid = recommendation_id(rec.get("recommendation_id") or rec_uid)
    payload = dict(rec.get("payload") or {})
    payload["outcome"] = outcome
    props = {
        "recommendation_id": rec.get("recommendation_id") or rec_uid,
        "idempotency_key": rec.get("idempotency_key"),
        "source_kind": rec.get("source_kind") or "report",
        "report_type": rec.get("report_type"),
        "as_of": rec.get("as_of"),
        "action": rec.get("action") or "watch",
        "ticker": rec.get("ticker"),
        "instrument": rec.get("instrument") or rec.get("ticker") or "portfolio",
        "decision_state": rec.get("decision_state") or "generated",
        "status": rec.get("status"),
        "approval_id": str(rec.get("approval_id")) if rec.get("approval_id") is not None else None,
        "approval_required": bool(rec.get("approval_required")),
        "approval_status": rec.get("approval_status"),
        "outcome_status": status,
        "supersedes_recommendation_id": rec.get("supersedes_recommendation_id"),
        "account_id": rec.get("account_id"),
        "portfolio_id": rec.get("portfolio_id"),
        "policy_id": rec.get("policy_id"),
        "policy_gate_result_id": rec.get("policy_gate_result_id"),
        "policy_gate_decision": rec.get("policy_gate_decision") or rec.get("policy_gate_status"),
        "policy_gate_review_required": bool(rec.get("policy_gate_review_required")),
        "confidence": _as_float(rec.get("confidence"), 0.0),
        "horizon": rec.get("horizon"),
        "rationale_summary": str(rec.get("rationale") or rec.get("rationale_summary") or "")[:500] or None,
        "rationale_hash": _stable_hash(str(rec.get("rationale") or "")) if rec.get("rationale") else None,
        "source_quality": rec.get("source_quality"),
        "decision_quality": rec.get("decision_quality"),
        "decision_quality_gate": rec.get("decision_quality_gate"),
        "payload": payload,
        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
    }
    now = datetime.now(UTC).isoformat()
    objects.write_object(
        "Recommendation",
        rec_uid,
        props,
        now,
        actor=actor,
        provenance=f"pv:recommendation_outcome:{rec_uid}",
    )
    return _write_decision_outcome(
        objects=objects,
        actor=actor,
        outcome_key=_outcome_key_for_recommendation(rec),
        source_kind="recommendation",
        parent_uid=rec_uid,
        relation_type="recommendation_has_decision_outcome",
        rec_or_coa=rec,
        outcome_status=status,
        outcome_payload=outcome,
        decision_quality_snapshot=rec.get("decision_quality")
        if isinstance(rec.get("decision_quality"), dict)
        else None,
    )


def record_course_of_action_outcome(
    coa: Mapping[str, Any],
    status: str,
    outcome: dict[str, Any],
    *,
    objects: OntologyObjectService | None = None,
    actor: Mapping[str, Any] | None = None,
) -> str:
    """Persist CourseOfAction retrospective outcome and linked DecisionOutcome."""
    objects = objects or OntologyObjectService()
    actor = actor or actor_to_dict(system_actor("decision_outcome_evaluator"))
    coa_uid = str(coa.get("object_uid") or coa.get("course_of_action_id") or "")
    if not coa_uid.startswith("course_of_action:"):
        coa_uid = course_of_action_id(coa.get("course_of_action_id") or coa_uid)
    payload = dict(coa.get("payload") or {})
    payload["outcome"] = outcome
    props = {
        "course_of_action_id": coa.get("course_of_action_id") or coa_uid,
        "idempotency_key": coa.get("idempotency_key"),
        "source_kind": coa.get("source_kind") or "workflow",
        "source_type": coa.get("source_type"),
        "source_id": coa.get("source_id"),
        "decision_type": coa.get("decision_type"),
        "action": coa.get("action") or "watch",
        "actionability": coa.get("actionability") or "actionable",
        "decision_state": coa.get("decision_state") or "generated",
        "status": coa.get("status"),
        "ticker": coa.get("ticker"),
        "instrument_id": coa.get("instrument_id"),
        "position_uid": coa.get("position_uid"),
        "approval_id": str(coa.get("approval_id")) if coa.get("approval_id") is not None else None,
        "approval_required": bool(coa.get("approval_required")),
        "approval_status": coa.get("approval_status"),
        "outcome_status": status,
        "action_run_id": coa.get("action_run_id"),
        "executed_action_id": coa.get("executed_action_id"),
        "confidence": coa.get("confidence"),
        "horizon": coa.get("horizon"),
        "rationale_summary": coa.get("rationale_summary"),
        "source_quality": coa.get("source_quality"),
        "decision_quality": coa.get("decision_quality"),
        "decision_quality_gate": coa.get("decision_quality_gate"),
        "payload": payload,
        "as_of": coa.get("as_of"),
        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
    }
    now = datetime.now(UTC).isoformat()
    objects.write_object(
        "CourseOfAction",
        coa_uid,
        props,
        now,
        actor=actor,
        provenance=f"pv:course_of_action_outcome:{coa_uid}",
    )
    return _write_decision_outcome(
        objects=objects,
        actor=actor,
        outcome_key=_outcome_key_for_course_of_action(coa),
        source_kind="course_of_action",
        parent_uid=coa_uid,
        relation_type="course_of_action_has_decision_outcome",
        rec_or_coa=coa,
        outcome_status=status,
        outcome_payload=outcome,
        decision_quality_snapshot=coa.get("decision_quality")
        if isinstance(coa.get("decision_quality"), dict)
        else None,
    )


def _evaluate_directional_record(
    record: Mapping[str, Any],
    *,
    today: date,
    record_kind: str,
    update_fn,
) -> str | None:
    as_of = _parse_date(record.get("as_of"))
    if as_of is None:
        update_fn(record, "unavailable", {"reason": "missing as_of date"})
        return "unavailable"
    if today < as_of + timedelta(days=_horizon_days(record.get("horizon"))):
        return None
    action = str(record.get("action") or "")
    if action == "do_nothing":
        update_fn(
            record,
            "evaluated",
            {
                "evaluation_authority": "ai_draft_user_final",
                "final_label_status": "draft",
                "process_label": "inconclusive",
                "timing_vs_expected_onset": _timing_label(
                    as_of,
                    today,
                    record.get("horizon"),
                    record.get("expected_onset_window"),
                ),
                "opportunity_cost": record.get("opportunity_cost_json", []),
                "draft_postmortem": (
                    f"No-action {record_kind} reached its review horizon. "
                    "User should confirm whether inaction preserved optionality or missed an actionable opportunity."
                ),
                "objective_score_available": False,
            },
        )
        return "evaluated"
    ticker = record.get("ticker")
    direction = _expected_direction(action)
    if not ticker or direction is None:
        update_fn(
            record,
            "unavailable",
            {
                "reason": "broad or non-directional decision; manual review required",
                "process_label": "inconclusive",
                "opportunity_cost": record.get("opportunity_cost_json", []),
            },
        )
        return "unavailable"
    try:
        close = _download_close_series(str(ticker), as_of, today)
        benchmark_close = _download_close_series("SPY", as_of, today)
        start = float(close.iloc[0])
        end = float(close.iloc[-1])
        forward_return = _series_return_pct(close)
        benchmark_return = _series_return_pct(benchmark_close)
        relative_return = forward_return - benchmark_return
        max_adverse, max_favorable = _excursions_pct(close, direction)
        directionally_right = forward_return > 0 if direction == "up" else forward_return < 0
        relative_right = relative_return > 0 if direction == "up" else relative_return < 0
        source_quality = str(record.get("source_quality") or "")
        confidence = _as_float(record.get("confidence"), 0.0)
        process_quality = "good" if source_quality in {"ok", "degraded"} and confidence >= 0.5 else "bad"
        outcome_quality = "good" if directionally_right and relative_right else "bad"
        thesis_context = _thesis_and_kill_context(str(ticker))
        update_fn(
            record,
            "evaluated",
            {
                "evaluation_authority": "ai_draft_user_final",
                "final_label_status": "draft",
                "start_price": start,
                "end_price": end,
                "forward_return_pct": round(forward_return, 2),
                "benchmark": "SPY",
                "benchmark_return_pct": round(benchmark_return, 2),
                "benchmark_relative_return_pct": round(relative_return, 2),
                "max_adverse_move_pct": round(max_adverse, 2),
                "max_favorable_move_pct": round(max_favorable, 2),
                "expected_direction": direction,
                "directionally_right": directionally_right,
                "relative_directionally_right": relative_right,
                "sizing_quality": {
                    "approval_status": record.get("approval_status"),
                    "label": "unverified_execution"
                    if record.get("approval_status") != "approved"
                    else "requires_trade_fill_review",
                },
                "timing_vs_expected_onset": _timing_label(
                    as_of,
                    today,
                    record.get("horizon"),
                    record.get("expected_onset_window"),
                ),
                "process_label": _process_label(process_quality, outcome_quality),
                **thesis_context,
                "draft_postmortem": (
                    "Objective price and process-attribution fields computed. "
                    "User should confirm execution, catalyst, and thesis labels."
                ),
            },
        )
        return "evaluated"
    except Exception as exc:
        update_fn(record, "unavailable", {"reason": str(exc)})
        return "unavailable"


def evaluate_due_decisions(limit: int = 50) -> dict[str, Any]:
    """Evaluate due recommendations and course-of-action decisions at review horizon."""
    today = datetime.now(UTC).date()
    reads = OntologyRuntimeReadService()
    objects = OntologyObjectService()
    actor = actor_to_dict(system_actor("decision_outcome_evaluator"))

    rec_checked = rec_updated = rec_unavailable = 0
    for rec in reads.recommendations(outcome_status="pending", limit=limit):
        rec_checked += 1
        result = _evaluate_directional_record(
            rec,
            today=today,
            record_kind="recommendation",
            update_fn=lambda r, s, o: record_recommendation_outcome(r, s, o, objects=objects, actor=actor),
        )
        if result == "evaluated":
            rec_updated += 1
        elif result == "unavailable":
            rec_unavailable += 1

    coa_checked = coa_updated = coa_unavailable = 0
    coa_rows = reads.list_objects(
        "CourseOfAction",
        filters={"outcome_status": "pending"},
        limit=limit,
    )
    for coa in coa_rows:
        coa_checked += 1
        result = _evaluate_directional_record(
            coa,
            today=today,
            record_kind="course of action",
            update_fn=lambda r, s, o: record_course_of_action_outcome(r, s, o, objects=objects, actor=actor),
        )
        if result == "evaluated":
            coa_updated += 1
        elif result == "unavailable":
            coa_unavailable += 1

    return {
        "recommendations": {
            "checked": rec_checked,
            "updated": rec_updated,
            "unavailable": rec_unavailable,
        },
        "course_of_actions": {
            "checked": coa_checked,
            "updated": coa_updated,
            "unavailable": coa_unavailable,
        },
        "checked": rec_checked + coa_checked,
        "updated": rec_updated + coa_updated,
        "unavailable": rec_unavailable + coa_unavailable,
    }


def finalize_decision_outcome(
    decision_outcome_uid: str,
    *,
    decision: str,
    note: str | None = None,
    corrected_postmortem: str | None = None,
    lessons_learned: str | None = None,
    actor_id: str | None = None,
    objects: OntologyObjectService | None = None,
) -> dict[str, Any]:
    """Finalize a draft decision outcome after human review."""
    objects = objects or OntologyObjectService()
    reads = OntologyRuntimeReadService()
    uid = decision_outcome_uid
    if not uid.startswith("decision_outcome:"):
        uid = decision_outcome_id(uid)
    row = reads.get(uid)
    if not row:
        raise ValueError(f"DecisionOutcome not found: {decision_outcome_uid}")

    decision_norm = str(decision or "").strip().lower()
    if decision_norm not in {"confirm", "correct", "reject"}:
        raise ValueError("decision must be confirm, correct, or reject")
    if decision_norm in {"correct", "reject"} and not str(note or "").strip():
        raise ValueError("note is required for correct and reject decisions")
    if decision_norm == "correct" and not str(corrected_postmortem or "").strip():
        raise ValueError("corrected_postmortem is required for correct decisions")

    final_status = {"confirm": "confirmed", "correct": "corrected", "reject": "rejected"}[decision_norm]
    draft = str(row.get("draft_postmortem") or "")
    final_postmortem = (
        corrected_postmortem if decision_norm == "correct" else (draft if decision_norm == "confirm" else note)
    )

    now = datetime.now(UTC).isoformat()
    actor = actor_to_dict(system_actor(actor_id or "decision_outcome_reviewer"))
    props = dict(row)
    props.update(
        {
            "final_label_status": final_status,
            "final_postmortem": final_postmortem,
            "lessons_learned": lessons_learned or note,
            "finalized_by": actor_id or actor.get("actor_id"),
            "finalized_at": now,
        }
    )
    updated = objects.write_object(
        "DecisionOutcome",
        uid,
        props,
        now,
        actor=actor,
        provenance=f"pv:decision_outcome_finalize:{uid}",
    )

    source_kind = str(row.get("source_kind") or "")
    parent_id = row.get("recommendation_id") if source_kind == "recommendation" else row.get("course_of_action_id")
    if parent_id:
        parent_uid = recommendation_id(parent_id) if source_kind == "recommendation" else course_of_action_id(parent_id)
        parent = reads.get(parent_uid)
        if parent:
            payload = dict(parent.get("payload") or {})
            outcome = dict(payload.get("outcome") or {})
            outcome.update(
                {
                    "final_label_status": final_status,
                    "final_postmortem": final_postmortem,
                    "lessons_learned": lessons_learned or note,
                    "finalized_by": props.get("finalized_by"),
                    "finalized_at": now,
                }
            )
            payload["outcome"] = outcome
            parent_props = dict(parent)
            parent_props["payload"] = payload
            parent_type = "Recommendation" if source_kind == "recommendation" else "CourseOfAction"
            objects.write_object(
                parent_type,
                parent_uid,
                parent_props,
                now,
                actor=actor,
                provenance=f"pv:decision_outcome_finalize_parent:{parent_uid}",
            )

    return updated
