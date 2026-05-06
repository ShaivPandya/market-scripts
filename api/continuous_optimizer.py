"""Continuous optimization mission loop.

The optimizer is intentionally analysis-only: it normalizes course-of-action
evidence, diffs it against prior durable state, and stages review work through
the existing action registry when a material decision state changes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ontology.object_service import OntologyObjectService
from ontology.policy import system_actor
from ontology.runtime_read_service import OntologyRuntimeReadService


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _bucket(value: Any, edges: list[float]) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if numeric < edges[0]:
        return "low"
    if numeric < edges[1]:
        return "medium"
    if numeric < edges[2]:
        return "high"
    return "very_high"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric:
        return default
    return numeric


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _object_id(row: dict[str, Any]) -> str:
    return str(row.get("object_uid") or row.get("id") or row.get("run_id") or "")


def _business_key(prefix: str, *parts: Any) -> str:
    clean = [str(part or "").strip() for part in parts if str(part or "").strip()]
    return f"{prefix}:{':'.join(clean)}" if clean else f"{prefix}:{uuid4().hex}"


def _write_runtime_object(object_type: str, business_key: str, properties: dict[str, Any]) -> dict[str, Any]:
    props = dict(properties)
    props.setdefault("updated_at", _now_iso())
    row = OntologyObjectService().write_object(
        object_type,
        business_key,
        props,
        props.get("created_at") or props.get("started_at") or props.get("as_of") or props["updated_at"],
        actor=system_actor("continuous_optimizer"),
        provenance=f"pv:continuous_optimizer:{object_type}:{business_key}:{_stable_hash(props)}",
    )
    payload = dict(row.get("properties") or props)
    object_uid = str(row.get("object_uid") or business_key)
    payload["id"] = object_uid
    payload["object_uid"] = object_uid
    return payload


def _action_is_hold(action: str) -> bool:
    return action in {"Hold Long", "Hold Short", "Watch"}


def _severity_for_state(state: dict[str, Any]) -> str:
    action = str(state.get("action") or "")
    gate_status = str(state.get("gate_status") or "")
    priority = _as_float(state.get("priority_score"))
    confidence = _as_float(state.get("confidence"))
    risk_level = str((state.get("risk") or {}).get("risk_level") or "").lower()

    if risk_level in {"critical", "very_high"} or (gate_status == "review" and priority >= 2.5):
        return "urgent"
    if action in {"Exit Review", "Squeeze Review"}:
        return "urgent" if confidence >= 0.65 else "high"
    if action in {"Trim Long", "Press Short", "Cover Short"}:
        return "high" if confidence >= 0.55 or priority >= 1.5 else "normal"
    if action in {"Increase Long", "Research Long", "Research Short"}:
        return "normal" if confidence >= 0.45 else "low"
    if gate_status == "review":
        return "normal"
    return "low"


def _material_fingerprint(state: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    confidence_edges = list(thresholds.get("confidence_bucket_edges") or [0.35, 0.65, 0.8])
    priority_edges = list(thresholds.get("priority_bucket_edges") or [0.75, 1.5, 2.5])
    risk = _as_dict(state.get("risk"))
    return {
        "action": state.get("action"),
        "conviction_band": state.get("conviction_band"),
        "gate_status": state.get("gate_status"),
        "priority_bucket": _bucket(state.get("priority_score"), priority_edges),
        "confidence_bucket": _bucket(state.get("confidence"), confidence_edges),
        "risk_level": risk.get("risk_level"),
        "risk_gate": risk.get("risk_gate"),
    }


def _alert_type(previous: dict[str, Any] | None, current: dict[str, Any]) -> str:
    if previous is None:
        return "new_action_state"
    prev = _as_dict(previous.get("evidence"))
    cur = _as_dict(current.get("evidence"))
    prev_material = _as_dict(prev.get("material_state"))
    cur_material = _as_dict(cur.get("material_state"))
    if prev_material.get("risk_level") != cur_material.get("risk_level"):
        return "risk_gate_changed"
    if prev_material.get("gate_status") != cur_material.get("gate_status"):
        return "gate_changed"
    if prev_material.get("action") != cur_material.get("action"):
        return "action_changed"
    return "material_state_changed"


def _change_summary(previous: dict[str, Any] | None, current: dict[str, Any]) -> str:
    ticker = str(current.get("ticker") or "").upper()
    if previous is None:
        return f"{ticker}: new optimizer state is {current.get('action')} ({current.get('conviction_band')})."
    prev_evidence = _as_dict(previous.get("evidence"))
    prev_material = _as_dict(prev_evidence.get("material_state"))
    cur_evidence = _as_dict(current.get("evidence"))
    cur_material = _as_dict(cur_evidence.get("material_state"))
    changes = []
    for label, key in (
        ("action", "action"),
        ("band", "conviction_band"),
        ("gate", "gate_status"),
        ("priority", "priority_bucket"),
        ("confidence", "confidence_bucket"),
        ("risk", "risk_level"),
    ):
        before = prev_material.get(key)
        after = cur_material.get(key)
        if before != after:
            changes.append(f"{label} {before or 'n/a'} -> {after or 'n/a'}")
    suffix = "; ".join(changes) if changes else "material evidence changed"
    return f"{ticker}: {suffix}."


def _safe_source(name: str, fn) -> tuple[Any, dict[str, Any]]:
    started = datetime.now(UTC).isoformat()
    try:
        value = fn()
        return value, {"status": "ok", "checked_at": started}
    except Exception as exc:  # noqa: BLE001 - degraded command-center sources should not abort the run.
        return None, {"status": "degraded", "checked_at": started, "error": str(exc) or exc.__class__.__name__}


def _collect_context(tickers: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_freshness: dict[str, Any] = {}
    context: dict[str, Any] = {}
    reads = OntologyRuntimeReadService()

    portfolio_risk, source_freshness["portfolio_risk"] = _safe_source(
        "portfolio_risk",
        lambda: __import__("api.position_risk", fromlist=["get_latest_portfolio_risk"]).get_latest_portfolio_risk(),
    )
    context["portfolio_risk"] = portfolio_risk

    position_risk: dict[str, Any] = {}

    def _position_risks() -> dict[str, Any]:
        from api.position_risk import get_latest_position_risk

        return {ticker: get_latest_position_risk(ticker) for ticker in tickers}

    position_risk, source_freshness["position_risk"] = _safe_source("position_risk", _position_risks)
    context["position_risk"] = position_risk or {}

    latest_reports, source_freshness["reports"] = _safe_source("reports", lambda: reads.report_runs(limit=5))
    context["recent_report_runs"] = latest_reports or []

    recommendations, source_freshness["recommendations"] = _safe_source(
        "recommendations", lambda: reads.recommendations(limit=10)
    )
    context["recent_recommendations"] = recommendations or []

    triggers, source_freshness["watch_triggers"] = _safe_source(
        "watch_triggers", lambda: reads.watch_triggers(status="active")
    )
    context["active_watch_triggers"] = triggers or []

    workflows, source_freshness["workflow_runs"] = _safe_source("workflow_runs", lambda: reads.workflow_runs(limit=5))
    context["recent_workflow_runs"] = workflows or []

    def _thesis_pressure() -> list[dict[str, Any]]:
        latest = {str(row.get("ticker") or "").upper(): row for row in reads.latest_evaluations()}
        pressure = []
        for meta in reads.theses(limit=1000):
            ticker = str(meta.get("ticker") or "").upper()
            evaluation = latest.get(ticker)
            if not evaluation:
                continue
            action = str(evaluation.get("action") or "").lower()
            if action not in {"", "hold"} or evaluation.get("risk_flag"):
                pressure.append(
                    {
                        "ticker": ticker,
                        "action": action,
                        "risk_flag": evaluation.get("risk_flag"),
                        "confidence": evaluation.get("confidence"),
                        "evaluated_at": evaluation.get("evaluated_at"),
                    }
                )
        return pressure

    thesis, source_freshness["thesis_pressure"] = _safe_source("thesis_pressure", _thesis_pressure)
    context["thesis_pressure"] = thesis or []

    def _regime() -> Any:
        from api.signal_snapshot import get_signal_aggregator_snapshot_or_module_response

        return get_signal_aggregator_snapshot_or_module_response(lookback_weeks=156, include_raw_modules=False)

    regime, source_freshness["macro_signal_regime"] = _safe_source("macro_signal_regime", _regime)
    context["macro_signal_regime"] = regime

    return context, source_freshness


def _normalize_state(
    action: dict[str, Any], *, mission: dict[str, Any], run_id: str, context: dict[str, Any]
) -> dict[str, Any]:
    ticker = str(action.get("ticker") or "").upper()
    position_risk = (context.get("position_risk") or {}).get(ticker) if ticker else None
    risk_payload = {}
    if isinstance(position_risk, dict):
        risk_payload = {
            "risk_level": position_risk.get("risk_level"),
            "risk_score": position_risk.get("risk_score") or position_risk.get("score"),
            "risk_gate": position_risk.get("gate_status") or position_risk.get("risk_level"),
            "as_of": position_risk.get("as_of") or position_risk.get("computed_at"),
            "snapshot_id": position_risk.get("result_id") or position_risk.get("id"),
        }

    base_state = {
        "run_id": run_id,
        "mission_id": _object_id(mission) or str(mission.get("mission_id") or mission.get("id") or ""),
        "ticker": ticker,
        "asset": action.get("asset"),
        "direction": action.get("direction"),
        "action": action.get("action") or "Review",
        "conviction_band": action.get("conviction_band") or "none",
        "priority_score": _as_float(action.get("priority_score")),
        "scenario_score": _as_float(action.get("scenario_score")),
        "score_delta": _as_float(action.get("score_delta")),
        "confidence": _as_float(action.get("confidence")),
        "gate_status": action.get("gate_status") or "review",
        "risk": risk_payload,
    }
    severity = _severity_for_state(base_state)
    base_state["severity"] = severity
    material_state = _material_fingerprint(base_state, mission.get("thresholds") or {})
    evidence = {
        "deterministic_rationale": action.get("deterministic_rationale"),
        "gate_reasons": action.get("gate_reasons") or [],
        "warnings": action.get("warnings") or [],
        "data_coverage": action.get("data_coverage") or {},
        "factor_breakdown": action.get("factor_breakdown") or [],
        "sizing_implication": action.get("sizing_implication") or {},
        "scenario_score": base_state["scenario_score"],
        "score_delta": base_state["score_delta"],
        "baseline_score": action.get("baseline_score"),
        "material_state": material_state,
        "risk": risk_payload,
    }
    base_state["evidence"] = evidence
    base_state["source_links"] = {
        "portfolio_analyzer": {"kind": "course_of_action", "ticker": ticker},
        "position_risk": risk_payload.get("snapshot_id"),
    }
    base_state["state_hash"] = _stable_hash(material_state)
    return base_state


def _is_material_change(previous: dict[str, Any] | None, current: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    suppress_holds = bool(thresholds.get("suppress_low_severity_holds", True))
    if previous is None:
        return current.get("severity") != "low" and not (
            suppress_holds and _action_is_hold(str(current.get("action") or ""))
        )
    if previous.get("state_hash") == current.get("state_hash"):
        return False
    if current.get("severity") == "low" and suppress_holds and _action_is_hold(str(current.get("action") or "")):
        return False
    return True


def _stage_action_item(alert: dict[str, Any], snapshot: dict[str, Any]) -> str | None:
    from ontology.command_service import OntologyCommandContext, OntologyCommandService

    ticker = str(snapshot.get("ticker") or "").upper() or None
    action = str(snapshot.get("action") or "Review")
    severity = str(alert.get("severity") or "normal")
    urgency = "urgent" if severity == "urgent" else "high" if severity == "high" else "normal"
    action_type = "resize"
    if action in {"Research Long", "Research Short", "Review", "Squeeze Review", "Exit Review"}:
        action_type = "research" if action.startswith("Research") else "review"
    if action in {"Trim Long", "Cover Short", "Exit Review"}:
        action_type = "exit" if action == "Exit Review" else "resize"
    evidence = _as_dict(snapshot.get("evidence"))
    rationale = str(evidence.get("deterministic_rationale") or alert.get("change_summary") or "").strip()
    description = f"Continuous optimizer: {alert['change_summary']}"
    if rationale:
        description = f"{description}\n\nEvidence: {rationale}"
    approval = OntologyCommandService().propose_action(
        "create_action_item",
        {
            "ticker": ticker,
            "action_type": action_type,
            "description": description,
            "urgency": urgency,
        },
        OntologyCommandContext(
            actor=system_actor("continuous_optimizer"),
            source_type="workflow",
            source_id=str(alert.get("run_id")),
        ),
        reason=f"Review continuous optimizer alert {alert['id']}",
    )
    return str(approval["id"]) if approval and approval.get("id") is not None else None


def _get_mission(mission_id: Any) -> dict[str, Any] | None:
    reads = OntologyRuntimeReadService()
    if mission_id:
        key = str(mission_id)
        for candidate in (key, f"optimizationmission:{key}", f"optimization_mission:{key}"):
            mission = reads.get(candidate)
            if mission:
                return mission
        matches = reads.list_objects("OptimizationMission", filters={"mission_id": key}, limit=1)
        if matches:
            return matches[0]
        return None
    active = reads.list_objects("OptimizationMission", filters={"status": "active"}, limit=1)
    if active:
        return active[0]
    missions = reads.list_objects("OptimizationMission", limit=1)
    return missions[0] if missions else None


def _create_run(mission: dict[str, Any], input_hash: str) -> dict[str, Any]:
    run_uid = _business_key("optimization_run", uuid4().hex)
    now = _now_iso()
    mission_uid = _object_id(mission)
    return _write_runtime_object(
        "OptimizationRun",
        run_uid,
        {
            "id": run_uid,
            "run_id": run_uid,
            "mission_id": mission_uid,
            "mission_name": mission.get("name"),
            "status": "running",
            "started_at": now,
            "input_hash": input_hash,
        },
    )


def _previous_successful_run(mission_id: str, current_run_id: str) -> dict[str, Any] | None:
    runs = OntologyRuntimeReadService().list_objects("OptimizationRun", filters={"mission_id": mission_id}, limit=100)
    candidates = [
        run
        for run in runs
        if str(run.get("run_id") or run.get("id")) != current_run_id
        and str(run.get("status") or "") in {"succeeded", "completed"}
    ]
    candidates.sort(key=lambda row: str(row.get("completed_at") or row.get("started_at") or ""), reverse=True)
    return candidates[0] if candidates else None


def _snapshots_for_run(run_id: str) -> list[dict[str, Any]]:
    return OntologyRuntimeReadService().list_objects(
        "OptimizationActionSnapshot",
        filters={"run_id": run_id},
        limit=1000,
    )


def run_continuous_optimizer(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    mission = _get_mission(payload.get("mission_id"))
    if not mission:
        raise ValueError(f"Unknown optimization mission: {payload.get('mission_id')}")
    if mission.get("status") != "active" and not payload.get("force"):
        return {
            "status": "skipped",
            "reason": "mission_not_active",
            "mission": mission,
            "run": None,
            "alerts_created": 0,
        }

    scenario = _as_dict(mission.get("scenario"))
    mission_uid = _object_id(mission)
    input_payload = {
        "mission_id": mission_uid,
        "mission_name": mission.get("name"),
        "scenario": scenario,
        "source": payload.get("source") or "manual",
    }
    input_hash = _stable_hash(input_payload)
    run = _create_run(mission, input_hash=input_hash)
    run_id = str(run["run_id"])

    source_freshness: dict[str, Any] = {}
    try:
        from portfolio.portfolio_optimizer.portfolio_analyzer import get_data as get_analyzer_data

        analyzer_payload = get_analyzer_data(scenario=scenario)
        if analyzer_payload.get("error"):
            raise RuntimeError(str(analyzer_payload["error"]))
        course = analyzer_payload.get("course_of_action") if isinstance(analyzer_payload, dict) else None
        if not isinstance(course, dict) or not isinstance(course.get("action_queue"), list):
            raise RuntimeError("Portfolio analyzer did not return course_of_action.action_queue.")
        source_freshness["portfolio_analyzer"] = {
            "status": "ok",
            "as_of": (course.get("summary") or {}).get("as_of"),
            "checked_at": datetime.now(UTC).isoformat(),
        }
        actions = [row for row in course.get("action_queue") or [] if isinstance(row, dict)]
        tickers = [str(row.get("ticker") or "").upper() for row in actions if row.get("ticker")]
        context, secondary_freshness = _collect_context(tickers)
        source_freshness.update(secondary_freshness)
        degraded_sources = [
            name
            for name, freshness in source_freshness.items()
            if isinstance(freshness, dict) and freshness.get("status") not in {"ok", "fresh"}
        ]
        staging_allowed = not degraded_sources

        previous_run = _previous_successful_run(mission_uid, run_id)
        previous_by_ticker = {
            str(snapshot.get("ticker") or "").upper(): snapshot
            for snapshot in (_snapshots_for_run(str(previous_run["run_id"])) if previous_run else [])
        }

        current_snapshots = []
        alerts = []
        for action in actions:
            state = _normalize_state(action, mission=mission, run_id=run_id, context=context)
            snapshot_uid = _business_key("optimization_action_snapshot", run_id, state.get("ticker") or uuid4().hex)
            snapshot = _write_runtime_object(
                "OptimizationActionSnapshot",
                snapshot_uid,
                {
                    **state,
                    "id": snapshot_uid,
                    "snapshot_id": snapshot_uid,
                    "mission_id": mission_uid,
                    "created_at": _now_iso(),
                },
            )
            current_snapshots.append(snapshot)
            previous = previous_by_ticker.get(str(snapshot.get("ticker") or "").upper())
            if previous_run is None:
                continue
            if not _is_material_change(previous, snapshot, mission.get("thresholds") or {}):
                continue
            alert_uid = _business_key(
                "optimization_alert",
                run_id,
                snapshot.get("ticker") or uuid4().hex,
                _alert_type(previous, snapshot),
            )
            alert = _write_runtime_object(
                "OptimizationAlert",
                alert_uid,
                {
                    "id": alert_uid,
                    "alert_id": alert_uid,
                    "status": "open",
                    "created_at": _now_iso(),
                    "mission_id": mission_uid,
                    "run_id": run_id,
                    "ticker": snapshot.get("ticker"),
                    "alert_type": _alert_type(previous, snapshot),
                    "severity": snapshot.get("severity") or "normal",
                    "previous_snapshot_id": previous.get("id") if previous else None,
                    "current_snapshot_id": snapshot.get("id"),
                    "change_summary": _change_summary(previous, snapshot),
                    "evidence": {
                        "previous": previous.get("evidence") if previous else None,
                        "current": snapshot.get("evidence"),
                        "source_freshness": source_freshness,
                        "staging_blocked": (
                            {"reason": "degraded_sources", "sources": degraded_sources} if not staging_allowed else None
                        ),
                    },
                },
            )
            if staging_allowed and alert.get("severity") != "low":
                try:
                    approval_id = _stage_action_item(alert, snapshot)
                    if approval_id is not None:
                        alert = _write_runtime_object(
                            "OptimizationAlert",
                            str(alert["id"]),
                            {
                                **alert,
                                "approval_id": approval_id,
                                "action_item_approval_id": approval_id,
                            },
                        )
                except Exception as exc:  # noqa: BLE001 - alert remains open even if staging fails.
                    alert.setdefault("evidence", {})["staging_error"] = str(exc) or exc.__class__.__name__
            alerts.append(alert)

        summary = {
            "mission": mission.get("name"),
            "as_of": datetime.now(UTC).isoformat(),
            "source_quality": "degraded" if degraded_sources else "ok",
            "degraded_sources": degraded_sources,
            "action_count": len(current_snapshots),
            "alerts_created": len(alerts),
            "open_alerts": len([alert for alert in alerts if alert.get("status") == "open"]),
            "staged_approvals": len([alert for alert in alerts if alert.get("action_item_approval_id")]),
        }
        output_hash = _stable_hash({"snapshots": [s.get("state_hash") for s in current_snapshots], "alerts": alerts})
        completed = _write_runtime_object(
            "OptimizationRun",
            run_id,
            {
                **run,
                "status": "succeeded",
                "completed_at": _now_iso(),
                "summary": summary,
                "source_freshness": source_freshness,
                "input_hash": input_hash,
                "output_hash": output_hash,
            },
        )
        return {
            "status": "completed",
            "mission": mission,
            "run": completed,
            "summary": summary,
            "snapshots_created": len(current_snapshots),
            "alerts_created": len(alerts),
            "alerts": alerts,
        }
    except Exception as exc:
        summary = {
            "mission": mission.get("name"),
            "as_of": datetime.now(UTC).isoformat(),
            "source_quality": "failed",
            "alerts_created": 0,
        }
        _write_runtime_object(
            "OptimizationRun",
            run_id,
            {
                **run,
                "status": "failed",
                "completed_at": _now_iso(),
                "error": str(exc) or exc.__class__.__name__,
                "summary": summary,
                "source_freshness": source_freshness,
            },
        )
        raise
