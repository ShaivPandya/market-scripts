"""Safe-mode runner for builder-defined monitors and missions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from ontology.command_service import OntologyCommandContext, OntologyCommandService
from ontology.object_service import OntologyObjectService
from ontology.policy import system_actor
from ontology.runtime_read_service import OntologyRuntimeReadService


def _stable_hash(value: Any, length: int = 16) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _definition_id(definition: dict[str, Any], key: str) -> str:
    return str(definition.get("object_uid") or definition.get(key) or definition.get("id") or "").strip()


def _definition_ticker(definition: dict[str, Any]) -> str | None:
    scope = _as_dict(definition.get("scope"))
    value = definition.get("ticker") or scope.get("ticker")
    text = str(value or "").strip().upper()
    return text or None


def _source_requirement_review(definition: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    requirements = _as_list(definition.get("source_requirements"))
    if not requirements:
        return {"status": "ok", "blockers": [], "warnings": []}
    result_sources = {
        str(source).strip().lower() for source in _as_list(result.get("source_ids")) if str(source).strip()
    }
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        source_name = str(
            requirement.get("source_name") or requirement.get("id") or requirement.get("type") or ""
        ).strip()
        required = bool(requirement.get("required", True))
        if not source_name:
            continue
        matched = not result_sources or source_name.lower() in result_sources
        row = {"source_name": source_name, "required": required, "status": "ok" if matched else "missing"}
        if not matched and required:
            blockers.append(row)
        elif not matched:
            warnings.append(row)
    return {
        "status": "blocked" if blockers else "warning" if warnings else "ok",
        "blockers": blockers,
        "warnings": warnings,
    }


def evaluate_monitor_definition(definition: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one monitor definition without writing state."""

    from api.watch_trigger_monitor import evaluate_trigger

    machine_definition = _as_dict(definition.get("definition"))
    trigger_type = str(machine_definition.get("type") or definition.get("trigger_type") or "custom")
    if trigger_type and "type" not in machine_definition:
        machine_definition = {"type": trigger_type, **machine_definition}
    trigger = {
        "object_uid": _definition_id(definition, "monitor_id"),
        "condition": definition.get("condition") or definition.get("name"),
        "trigger_type": trigger_type,
        "ticker": _definition_ticker(definition),
        "definition": machine_definition,
    }
    result = evaluate_trigger(trigger)
    source_review = _source_requirement_review(definition, result)
    if source_review["status"] == "blocked":
        result = {
            **result,
            "fired": True,
            "hit_type": "source_blocked",
            "severity": definition.get("severity") or "medium",
            "evidence": "Required source configuration is missing or stale.",
            "source_requirement_review": source_review,
        }
    else:
        result["source_requirement_review"] = source_review
    return result


def _hit_payload_for_monitor(definition: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    definition_uid = _definition_id(definition, "monitor_id")
    hit_type = str(result.get("hit_type") or ("triggered" if result.get("fired") else "ok"))
    severity = str(result.get("severity") or definition.get("severity") or ("high" if result.get("fired") else "low"))
    evidence = str(result.get("evidence") or definition.get("condition") or definition.get("name") or "Monitor hit")
    fingerprint = _stable_hash(
        {
            "definition": definition_uid,
            "definition_hash": definition.get("definition_hash"),
            "result": result,
        }
    )
    return {
        "ticker": _definition_ticker(definition) or "UNKNOWN",
        "entity_type": "monitor_definition",
        "entity_id": definition_uid,
        "entity_label": definition.get("name"),
        "hit_type": hit_type,
        "severity": severity if severity in {"low", "medium", "high"} else "medium",
        "status": "open",
        "confidence": result.get("confidence") if isinstance(result.get("confidence"), (int, float)) else 0.75,
        "evidence": evidence,
        "source_ids": _as_list(result.get("source_ids")),
        "result": result,
        "fingerprint": f"{definition_uid}:{fingerprint}",
        "detected_at": result.get("as_of") or _now(),
    }


def _create_monitor_hit(payload: dict[str, Any], *, source_id: str) -> dict[str, Any]:
    service = OntologyCommandService()
    context = OntologyCommandContext(
        actor=system_actor("monitor_mission_runner"),
        source_type="workflow",
        source_id=source_id,
        request_mode="self_apply",
    )
    approval = service.propose_action("create_monitor_hit", payload, context, reason="Record builder monitor hit")
    return service.resolve_approval(str(approval["id"]), "approved", "Recorded by safe-mode monitor runner.", context)


def _stage_review_action(hit_payload: dict[str, Any], *, source_id: str) -> dict[str, Any]:
    service = OntologyCommandService()
    context = OntologyCommandContext(
        actor=system_actor("monitor_mission_runner"),
        source_type="workflow",
        source_id=source_id,
        request_mode="proposal",
    )
    payload = {
        "description": f"Review monitor hit: {hit_payload.get('entity_label') or hit_payload.get('entity_id')}",
        "action_type": "review",
        "ticker": hit_payload.get("ticker") if hit_payload.get("ticker") != "UNKNOWN" else None,
        "urgency": "high" if hit_payload.get("severity") == "high" else "normal",
        "alert_context": {
            "change_summary": hit_payload.get("entity_label") or hit_payload.get("entity_id"),
            "source": "monitor_hit",
            "ticker": hit_payload.get("ticker"),
        },
    }
    from decision_quality.proactive_alert_gate import apply_proactive_alert_gate

    payload, _gate_result = apply_proactive_alert_gate(
        "create_action_item",
        payload,
        source_type=context.source_type,
        alert_context=payload.get("alert_context"),
    )
    return service.propose_action(
        "create_action_item",
        payload,
        context,
        reason="Review builder monitor hit before any state change",
    )


def _write_workflow_artifact(run_id: str, key: str, value: Any) -> None:
    now = _now()
    artifact_id = f"{run_id}:{key}:{_stable_hash(value, 10)}"
    OntologyObjectService().write_object(
        "WorkflowArtifact",
        artifact_id,
        {
            "artifact_id": artifact_id,
            "workflow_run_id": run_id,
            "artifact_key": key,
            "artifact_value": value,
            "artifact_hash": _stable_hash(value),
            "state": "extracted",
            "metadata": {"runner": "monitor_mission_runner"},
            "ontology_run_id": "operational",
        },
        now,
        actor={"actor_type": "system", "actor_id": "monitor_mission_runner"},
        provenance=f"pv:monitor_mission_runner:artifact:{artifact_id}",
    )


def run_monitor_mission_runner(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run active builder definitions in safe mode."""

    from api.workflows import complete_workflow_run, create_workflow_run, fail_workflow_run

    payload = payload or {}
    reads = OntologyRuntimeReadService()
    monitor_filter = str(payload.get("monitor_id") or "").strip()
    mission_filter = str(payload.get("mission_id") or "").strip()
    monitors = reads.monitor_definitions(status="active", limit=500)
    missions = reads.mission_definitions(status="active", limit=500)
    if monitor_filter:
        monitors = [
            item
            for item in monitors
            if monitor_filter in {_definition_id(item, "monitor_id"), str(item.get("monitor_id") or "")}
        ]
    if mission_filter:
        missions = [
            item
            for item in missions
            if mission_filter in {_definition_id(item, "mission_id"), str(item.get("mission_id") or "")}
        ]

    run = create_workflow_run("monitor_mission_runner", None, actor=system_actor("monitor_mission_runner"))
    run_id = str(run["run_id"])
    checked = fired = hits = approvals = errors = 0
    artifacts: dict[str, Any] = {"monitors": [], "missions": []}
    try:
        for definition in monitors:
            checked += 1
            definition_uid = _definition_id(definition, "monitor_id")
            try:
                result = evaluate_monitor_definition(definition)
                artifacts["monitors"].append({"definition_id": definition_uid, "result": result})
                if result.get("fired"):
                    fired += 1
                    hit_payload = _hit_payload_for_monitor(definition, result)
                    source_id = f"{definition_uid}:{hit_payload['fingerprint']}"
                    hit = _create_monitor_hit(hit_payload, source_id=source_id)
                    hits += 1
                    review = _stage_review_action(hit_payload, source_id=source_id)
                    approvals += 1 if review.get("id") else 0
                    hit_payload["approval_id"] = hit.get("id")
            except Exception as exc:
                errors += 1
                artifacts["monitors"].append({"definition_id": definition_uid, "error": str(exc)})
        for mission in missions:
            checked += 1
            mission_uid = _definition_id(mission, "mission_id")
            result = {
                "fired": True,
                "hit_type": "needs_review",
                "severity": "medium",
                "evidence": f"Mission '{mission.get('name')}' completed in safe mode and requires human review.",
                "source_requirement_review": _source_requirement_review(mission, {}),
            }
            artifacts["missions"].append({"definition_id": mission_uid, "result": result})
            hit_payload = {
                "ticker": _definition_ticker(mission) or "UNKNOWN",
                "entity_type": "mission_definition",
                "entity_id": mission_uid,
                "entity_label": mission.get("name"),
                "hit_type": "needs_review",
                "severity": "medium",
                "status": "open",
                "confidence": 0.6,
                "evidence": result["evidence"],
                "source_ids": [],
                "result": result,
                "fingerprint": f"{mission_uid}:{_stable_hash({'definition_hash': mission.get('definition_hash'), 'run_id': run_id})}",
                "detected_at": _now(),
            }
            source_id = f"{mission_uid}:{hit_payload['fingerprint']}"
            _create_monitor_hit(hit_payload, source_id=source_id)
            hits += 1
            review = _stage_review_action(hit_payload, source_id=source_id)
            approvals += 1 if review.get("id") else 0
        _write_workflow_artifact(run_id, "monitor_mission_results", artifacts)
        complete_workflow_run(run_id, "Builder monitor/mission run completed in safe mode.", artifacts)
    except Exception as exc:
        fail_workflow_run(run_id, str(exc))
        raise
    return {
        "run_id": run_id,
        "checked": checked,
        "fired": fired,
        "hits": hits,
        "approvals": approvals,
        "errors": errors,
    }
