"""Decision-centered ontology writeback helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ontology.domain_write_service import OPERATIONAL_ONTOLOGY_RUN_ID
from ontology.object_service import OntologyObjectService
from ontology.schemas.identity import (
    account_id,
    asset_id,
    citation_id,
    course_of_action_comparison_id,
    course_of_action_id,
    evidence_id,
    object_version_ref_id,
    policy_gate_result_id,
    portfolio_id,
    portfolio_risk_snapshot_id,
    position_id,
    position_risk_snapshot_id,
    report_run_id,
    risk_metric_id,
    scenario_assumption_id,
    scenario_id,
    simulated_outcome_id,
    source_record_object_id,
    trade_proposal_id,
    workflow_artifact_id,
    workflow_run_id,
)
from ontology.schemas.identity import (
    action_run_id as action_run_uid,
)
from ontology.schemas.identity import (
    approval_id as approval_uid,
)
from ontology.schemas.identity import (
    executed_action_id as executed_action_uid,
)
from ontology.schemas.identity import (
    recommendation_id as recommendation_uid,
)

ACTIONABLE_RECOMMENDATION_ACTIONS = {
    "buy",
    "add",
    "short",
    "sell",
    "trim",
    "reduce",
    "exit",
    "hedge",
    "rebalance",
}


class DecisionOntologyWriteback:
    """Facade for governed decision artifact writes."""

    def __init__(self, object_service: OntologyObjectService | None = None):
        self.object_service = object_service or OntologyObjectService()

    def enabled(self) -> bool:
        return True

    def record_report_output(
        self,
        *,
        report_type: str,
        payload: Mapping[str, Any],
        report_run: Mapping[str, Any],
        persisted_recommendations: Sequence[Mapping[str, Any]] = (),
        actor: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | str | None = None,
    ) -> list[dict[str, Any]]:
        """Record a report run, report source record, and generated recommendations."""
        if not self.enabled():
            return []
        actor = actor or {"actor_type": "workflow", "actor_id": "report_sync"}
        report_id_value = str(report_run.get("report_id") or payload.get("report_id") or "")
        if not report_id_value:
            return []
        now = _now()
        valid_from = str(report_run.get("as_of") or payload.get("as_of") or report_run.get("synced_at") or now)
        provenance_id = _provenance_id(provenance, f"pv:report_sync:{report_id_value}")
        rows: list[dict[str, Any]] = []

        try:
            report_row = self.object_service.write_object(
                "ReportRun",
                report_id_value,
                {
                    "report_id": report_id_value,
                    "report_type": report_type,
                    "as_of": str(report_run.get("as_of") or payload.get("as_of") or valid_from),
                    "source": report_run.get("source") or "github_actions",
                    "source_run_id": report_run.get("source_run_id"),
                    "source_url": report_run.get("source_url"),
                    "status": report_run.get("status") or "completed",
                    "report_hash": report_run.get("report_hash"),
                    "input_hash": report_run.get("input_hash"),
                    "summary": _as_dict(report_run.get("summary_json") or report_run.get("summary")),
                    "artifact_paths": _as_dict(
                        report_run.get("artifact_paths_json") or report_run.get("artifact_paths")
                    ),
                    "issue_url": report_run.get("issue_url"),
                    "created_at": report_run.get("created_at"),
                    "updated_at": report_run.get("updated_at"),
                    "synced_at": report_run.get("synced_at") or now,
                    "error": report_run.get("error"),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                valid_from,
                actor=actor,
                provenance=provenance_id,
                input_hash=str(report_run.get("input_hash") or "") or None,
            )
            rows.append(report_row)

            source_record_id = f"report:{report_id_value}:payload"
            source_row = self.object_service.write_object(
                "SourceRecord",
                source_record_id,
                {
                    "source_record_id": source_record_id,
                    "vendor": str(report_run.get("source") or "github_actions"),
                    "source_name": f"{report_type}_report_sync",
                    "source_version": "1",
                    "dataset": "report_sync",
                    "record_kind": "report_payload",
                    "record_key_hash": _hash_text(report_id_value, length=32),
                    "payload_hash": _hash_value(payload),
                    "status": report_run.get("status") or "ok",
                    "quality": "ok" if not report_run.get("error") else "error",
                    "as_of": report_run.get("as_of") or payload.get("as_of"),
                    "load_time": report_run.get("synced_at") or now,
                    "artifact_uri": _artifact_uri(payload),
                    "provenance_event_id": provenance_id,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                valid_from,
                actor=actor,
                provenance=provenance_id,
                input_hash=str(report_run.get("input_hash") or "") or None,
            )
            rows.append(source_row)
            rows.extend(
                self._record_report_context_objects(
                    report_id_value=report_id_value,
                    payload=payload,
                    source_record_id=source_record_id,
                    actor=actor,
                    provenance_id=provenance_id,
                    valid_from=valid_from,
                )
            )

            for item in persisted_recommendations:
                record = item.get("record") if isinstance(item.get("record"), Mapping) else item
                if not isinstance(record, Mapping):
                    continue
                rows.extend(
                    self._record_recommendation_bundle(
                        report_id_value=report_id_value,
                        source_record_id=source_record_id,
                        record=record,
                        approval_id=_optional_int(item.get("approval_id")),
                        actor=actor,
                        provenance_id=provenance_id,
                        valid_from=str(record.get("as_of") or valid_from),
                    )
                )
        except Exception as exc:
            _handle_writeback_error("record_report_output", exc)
        return rows

    def record_workflow_artifact_proposal(
        self,
        *,
        run_id: str,
        artifact_key: str,
        artifact_index: int,
        artifact_value: Mapping[str, Any],
        approval_id: int | None,
        action_id: str | None = None,
        ticker: str | None = None,
        artifact_id: str | None = None,
        actor: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | str | None = None,
    ) -> list[dict[str, Any]]:
        """Persist a raw workflow artifact and link it to its approval proposal."""
        if not self.enabled():
            return []
        actor = actor or {"actor_type": "workflow", "actor_id": run_id}
        now = _now()
        artifact_id_value = artifact_id or f"{run_id}:{artifact_key}:{artifact_index}"
        provenance_id = _provenance_id(provenance, f"pv:workflow_artifact:{artifact_id_value}")
        rows: list[dict[str, Any]] = []
        try:
            workflow_artifact_row = self.object_service.write_object(
                "WorkflowArtifact",
                artifact_id_value,
                {
                    "artifact_id": artifact_id_value,
                    "workflow_run_id": run_id,
                    "artifact_key": artifact_key,
                    "artifact_index": artifact_index,
                    "artifact_value": dict(artifact_value),
                    "artifact_hash": _hash_value(artifact_value),
                    "state": "proposed" if approval_id is not None else "extracted",
                    "action_id": action_id,
                    "approval_id": approval_id,
                    "provenance_event_id": provenance_id,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                approval_id=approval_id,
                input_hash=_hash_value(artifact_value),
            )
            rows.append(workflow_artifact_row)
            rows.append(
                self.object_service.write_relation(
                    workflow_run_id(run_id),
                    workflow_artifact_id(artifact_id_value),
                    "workflow_run_produces_artifact",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "artifact_key": artifact_key},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
            if approval_id is not None:
                rows.append(
                    self.object_service.write_relation(
                        workflow_artifact_id(artifact_id_value),
                        approval_uid(approval_id),
                        "workflow_artifact_proposes_approval",
                        {
                            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                            "approval_id": str(approval_id),
                            "artifact_key": artifact_key,
                        },
                        now,
                        actor=actor,
                        provenance=provenance_id,
                        approval_id=approval_id,
                    )
                )
                rows.append(
                    self.object_service.write_relation(
                        approval_uid(approval_id),
                        workflow_artifact_id(artifact_id_value),
                        "approval_targets_workflow_artifact",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "target_object_type": "WorkflowArtifact"},
                        now,
                        actor=actor,
                        provenance=provenance_id,
                        approval_id=approval_id,
                    )
                )
        except Exception as exc:
            _handle_writeback_error("record_workflow_artifact_proposal", exc)
        return rows

    def apply_approved_decision(
        self,
        *,
        approval_id: int | None,
        action_run_id: int | None,
        action_id: str,
        output: Mapping[str, Any],
        mutated_versions: Sequence[Mapping[str, Any]],
        actor: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | str | None = None,
    ) -> list[dict[str, Any]]:
        """Record the immutable executed-action layer for an approved mutation."""
        if not self.enabled() or approval_id is None or action_run_id is None:
            return []
        actor = actor or {"actor_type": "approval_apply", "actor_id": None}
        now = _now()
        provenance_id = _provenance_id(provenance, f"pv:action_run:{action_run_id}")
        executed_id = f"{approval_id}:{action_run_id}:{action_id}"
        rows: list[dict[str, Any]] = []
        version_refs_raw = (_version_ref_payload(row) for row in mutated_versions)
        version_refs = [ref for ref in version_refs_raw if ref is not None]
        try:
            executed_row = self.object_service.write_object(
                "ExecutedAction",
                executed_id,
                {
                    "executed_action_id": executed_id,
                    "action_id": action_id,
                    "approval_id": approval_id,
                    "action_run_id": action_run_id,
                    "execution_mode": "approval_required",
                    "produced_object_versions": version_refs,
                    "mutated_object_versions": version_refs,
                    "applied_at": now,
                    "status": "applied",
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                action_run_id=action_run_id,
                approval_id=approval_id,
                input_hash=_hash_value({"action_id": action_id, "output": output}),
            )
            rows.append(executed_row)
            rows.append(
                self.object_service.write_relation(
                    action_run_uid(action_run_id),
                    executed_action_uid(executed_id),
                    "action_run_produces_executed_action",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    action_run_id=action_run_id,
                    approval_id=approval_id,
                )
            )
            rows.append(
                self.object_service.write_relation(
                    approval_uid(approval_id),
                    action_run_uid(action_run_id),
                    "approval_applies_action_run",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    action_run_id=action_run_id,
                    approval_id=approval_id,
                )
            )
            for ref in version_refs:
                ref_row = self.object_service.write_object(
                    "ObjectVersionRef",
                    ref["ref_id"],
                    {**ref, "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    action_run_id=action_run_id,
                    approval_id=approval_id,
                )
                rows.append(ref_row)
                rows.append(
                    self.object_service.write_relation(
                        executed_action_uid(executed_id),
                        object_version_ref_id(ref["ref_id"]),
                        "executed_action_mutates_object_version",
                        {
                            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                            "object_uid": ref["object_uid"],
                            "object_type": ref.get("object_type"),
                            "version_id": ref["version_id"],
                        },
                        now,
                        actor=actor,
                        provenance=provenance_id,
                        action_run_id=action_run_id,
                        approval_id=approval_id,
                    )
                )
        except Exception as exc:
            _handle_writeback_error("apply_approved_decision", exc)
        return rows

    def record_scenario_simulation(
        self,
        *,
        simulation: Mapping[str, Any],
        request_payload: Mapping[str, Any],
        actor: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        """Persist simulator output as ontology-backed COA artifacts."""
        if not self.enabled():
            return {"rows_written": 0, "artifact_ids": {}, "outcome_artifact_ids": {}}

        actor = actor or {"actor_type": "api", "actor_id": "scenario_simulator"}
        now = _now()
        simulation_id = str(simulation.get("simulation_id") or _hash_value(simulation))
        input_hash = str(simulation.get("input_hash") or _hash_value(request_payload))
        provenance_id = _provenance_id(provenance, f"pv:{simulation_id}")
        portfolio = _as_dict(simulation.get("portfolio"))
        request_portfolio = _as_dict(request_payload.get("portfolio"))
        request_position = _as_dict(request_payload.get("position"))
        portfolio_id_value = str(portfolio.get("portfolio_id") or request_portfolio.get("portfolio_id") or "").strip()
        account_id_value = str(portfolio.get("account_id") or request_portfolio.get("account_id") or "").strip()
        position_uid = str(request_position.get("position_uid") or request_position.get("object_uid") or "").strip()
        comparison_key = simulation_id
        comparison_uid = course_of_action_comparison_id(comparison_key)
        rows: list[dict[str, Any]] = []
        outcome_artifacts: dict[str, dict[str, Any]] = {}
        scenario_uids: dict[str, str] = {}
        scenario_link_keys: list[str] = []
        assumption_uids: list[str] = []
        policy_gate_uids: list[str] = []

        try:
            comparison_row = self.object_service.write_object(
                "CourseOfActionComparison",
                comparison_key,
                {
                    "comparison_id": comparison_key,
                    "objective": f"Compare scenario simulator candidate actions for {request_position.get('ticker') or 'position'}",
                    "scope_type": "position" if position_uid else "portfolio",
                    "scope_id": position_uid or portfolio_id_value or None,
                    "selected_course_of_action_id": None,
                    "decision_state": "generated",
                    "status": "open",
                    "ranking_summary": _as_dict(simulation.get("comparison")),
                    "selection_reason": None,
                    "as_of": simulation.get("generated_at") or now,
                    "created_at": now,
                    "updated_at": now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            rows.append(comparison_row)

            scenarios = _dicts(request_payload.get("scenarios"))
            for index, scenario in enumerate(scenarios):
                scenario_key = str(
                    scenario.get("scenario_id")
                    or scenario.get("id")
                    or f"{simulation_id}:scenario:{index + 1}:{scenario.get('name') or 'scenario'}"
                )
                scenario_uid = scenario_id(scenario_key)
                scenario_uids[scenario_key] = scenario_uid
                scenario_link_keys.append(scenario_key)
                scenario_uids[str(scenario.get("scenario_id") or scenario.get("id") or f"scenario:{index + 1}")] = (
                    scenario_uid
                )
                rows.append(
                    self.object_service.write_object(
                        "Scenario",
                        scenario_key,
                        {
                            "scenario_id": scenario_key,
                            "name": str(scenario.get("name") or scenario.get("label") or f"Scenario {index + 1}"),
                            "scenario_type": str(scenario.get("scenario_type") or scenario.get("type") or "stress"),
                            "scope_type": "position" if position_uid else "portfolio",
                            "scope_id": position_uid or portfolio_id_value or None,
                            "assumptions_hash": _hash_value(request_payload.get("assumptions") or scenario),
                            "result": _as_dict(scenario.get("result")),
                            "result_metrics": {
                                "price_move_pct": scenario.get("price_move_pct"),
                                "probability": scenario.get("probability"),
                                "stress_loss_pct": scenario.get("stress_loss_pct"),
                                "drawdown_pct": scenario.get("drawdown_pct"),
                                "daily_volatility_pct": scenario.get("daily_volatility_pct"),
                                "thesis_pressure": scenario.get("thesis_pressure"),
                            },
                            "loss_pct": _optional_float(scenario.get("stress_loss_pct")),
                            "generated_by_source": "api",
                            "generated_by_action": "scenario_simulator.evaluate",
                            "generated_by_run_id": simulation_id,
                            "as_of": simulation.get("generated_at") or now,
                            "status": "simulated",
                            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                        },
                        now,
                        actor=actor,
                        provenance=provenance_id,
                        input_hash=_hash_value(scenario),
                    )
                )

            assumptions = _dicts(request_payload.get("assumptions"))
            for index, assumption in enumerate(assumptions):
                assumption_key = str(
                    assumption.get("assumption_id")
                    or assumption.get("id")
                    or f"{simulation_id}:assumption:{index + 1}:{assumption.get('name') or 'assumption'}"
                )
                assumption_uid = scenario_assumption_id(assumption_key)
                assumption_uids.append(assumption_uid)
                scenario_ref = str(assumption.get("scenario_id") or "").strip()
                rows.append(
                    self.object_service.write_object(
                        "ScenarioAssumption",
                        assumption_key,
                        {
                            "assumption_id": assumption_key,
                            "scenario_id": scenario_ref or None,
                            "name": str(assumption.get("name") or f"Assumption {index + 1}"),
                            "value": assumption.get("value"),
                            "unit": assumption.get("unit"),
                            "direction": assumption.get("direction"),
                            "confidence": _optional_float(assumption.get("confidence")),
                            "source_record_ids": [
                                ref
                                for ref in _ids_from(
                                    assumption.get("source_record_refs") or assumption.get("source_refs"), "id"
                                )
                                if ref.startswith("source_record:")
                            ],
                            "as_of": assumption.get("as_of") or simulation.get("generated_at") or now,
                            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                        },
                        now,
                        actor=actor,
                        provenance=provenance_id,
                        input_hash=_hash_value(assumption),
                    )
                )
                linked_scenarios = [scenario_ref] if scenario_ref else list(scenario_link_keys)
                for linked in linked_scenarios:
                    target_scenario_uid = scenario_uids.get(linked) or scenario_id(linked)
                    rows.append(
                        self.object_service.write_relation(
                            target_scenario_uid,
                            assumption_uid,
                            "scenario_has_assumption",
                            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                            now,
                            actor=actor,
                            provenance=provenance_id,
                            input_hash=input_hash,
                        )
                    )

            for outcome in _dicts(simulation.get("outcomes")):
                candidate_id = str(outcome.get("candidate_id") or _hash_value(outcome))
                action = str(outcome.get("action") or "hold")
                course_key = f"{simulation_id}:{candidate_id}:{action}"
                course_uid = course_of_action_id(course_key)
                gate = _as_dict(outcome.get("policy_gate"))
                gate_uid = None
                if gate:
                    gate_key = f"{simulation_id}:{candidate_id}:policy_gate"
                    gate_uid = policy_gate_result_id(gate_key)
                    policy_gate_uids.append(gate_uid)
                    rows.append(
                        self.object_service.write_object(
                            "PolicyGateResult",
                            gate_key,
                            {
                                "gate_result_id": gate_key,
                                "decision": str(gate.get("decision") or "warn"),
                                "review_required": bool(gate.get("review_required")),
                                "approval_required": bool(gate.get("approval_required", True)),
                                "approval_mode": gate.get("approval_mode"),
                                "approval_requirements": _as_list(gate.get("approval_requirements")),
                                "rule_id": gate.get("rule_id"),
                                "reason": gate.get("reason"),
                                "remediation": gate.get("remediation"),
                                "matched_rules": _as_list(gate.get("matched_rules")),
                                "limit_overrides": _as_dict(gate.get("limit_overrides")),
                                "failure_reasons": _as_list(gate.get("failure_reasons")),
                                "warnings": _as_list(gate.get("warnings")),
                                "account_id": account_id_value or gate.get("account_id"),
                                "portfolio_id": portfolio_id_value or gate.get("portfolio_id"),
                                "policy_id": gate.get("policy_id"),
                                "policy_matrix_id": gate.get("policy_matrix_id"),
                                "evaluated_at": gate.get("evaluated_at") or now,
                                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                            },
                            now,
                            actor=actor,
                            provenance=provenance_id,
                            input_hash=_hash_value(gate),
                        )
                    )

                risk = _as_dict(outcome.get("risk"))
                uncertainty = _as_dict(outcome.get("uncertainty"))
                course_row = self.object_service.write_object(
                    "CourseOfAction",
                    course_key,
                    {
                        "course_of_action_id": course_key,
                        "idempotency_key": course_key,
                        "source_kind": "api",
                        "source_type": "api",
                        "source_id": "scenario_simulator.evaluate",
                        "decision_type": "scenario_simulation",
                        "action": action,
                        "actionability": _coa_actionability(action, gate, uncertainty),
                        "decision_state": "generated",
                        "status": "simulated",
                        "ticker": request_position.get("ticker"),
                        "instrument_id": request_position.get("instrument_id"),
                        "position_uid": position_uid or None,
                        "account_id": account_id_value or None,
                        "portfolio_id": portfolio_id_value or None,
                        "policy_gate_result_id": gate_uid,
                        "policy_gate_decision": gate.get("decision"),
                        "approval_required": bool(action != "hold"),
                        "approval_status": None,
                        "comparison_id": comparison_uid,
                        "confidence": _confidence_from_uncertainty(uncertainty),
                        "rationale_summary": outcome.get("rationale"),
                        "rationale_hash": _hash_text(str(outcome.get("rationale") or ""))
                        if outcome.get("rationale")
                        else None,
                        "source_quality": uncertainty.get("level"),
                        "sizing_summary": _as_dict(outcome.get("target_position")),
                        "effect_summary": _as_dict(outcome.get("exposure")),
                        "risk_summary": risk,
                        "policy_summary": gate,
                        "payload": _jsonable(dict(outcome)),
                        "as_of": simulation.get("generated_at") or now,
                        "created_at": now,
                        "updated_at": now,
                        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    },
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=_hash_value(outcome),
                )
                rows.append(course_row)
                rows.append(
                    self.object_service.write_relation(
                        comparison_uid,
                        course_uid,
                        "comparison_includes_course_of_action",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                        now,
                        actor=actor,
                        provenance=provenance_id,
                        input_hash=input_hash,
                    )
                )
                if portfolio_id_value:
                    rows.append(
                        self.object_service.write_relation(
                            course_uid,
                            portfolio_id(portfolio_id_value),
                            "course_of_action_targets_portfolio",
                            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                            now,
                            actor=actor,
                            provenance=provenance_id,
                            input_hash=input_hash,
                        )
                    )
                if position_uid:
                    rows.append(
                        self.object_service.write_relation(
                            course_uid,
                            position_uid if position_uid.startswith("position:") else position_id(position_uid),
                            "course_of_action_targets_position",
                            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                            now,
                            actor=actor,
                            provenance=provenance_id,
                            input_hash=input_hash,
                        )
                    )
                evidence_refs = _ids_from(_as_dict(outcome.get("provenance")).get("evidence_refs"), "id")
                for evidence_ref in evidence_refs:
                    rows.append(
                        self.object_service.write_relation(
                            course_uid,
                            evidence_ref if evidence_ref.startswith("evidence:") else evidence_id(evidence_ref),
                            "course_of_action_supported_by_evidence",
                            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                            now,
                            actor=actor,
                            provenance=provenance_id,
                            input_hash=input_hash,
                        )
                    )

                simulated_outcome_ids: list[str] = []
                for index, scenario_outcome in enumerate(_dicts(outcome.get("scenario_outcomes"))):
                    scenario_ref = str(scenario_outcome.get("scenario_id") or f"scenario:{index + 1}")
                    scenario_uid = scenario_uids.get(scenario_ref) or scenario_id(scenario_ref)
                    outcome_key = f"{simulation_id}:{candidate_id}:{scenario_ref}"
                    simulated_uid = simulated_outcome_id(outcome_key)
                    simulated_outcome_ids.append(simulated_uid)
                    rows.append(
                        self.object_service.write_object(
                            "SimulatedOutcome",
                            outcome_key,
                            {
                                "outcome_id": outcome_key,
                                "course_of_action_id": course_uid,
                                "scenario_id": scenario_uid,
                                "assumptions_hash": _hash_value(request_payload.get("assumptions") or {}),
                                "result": _as_dict(scenario_outcome),
                                "result_metrics": {
                                    "risk": risk,
                                    "liquidity": _as_dict(outcome.get("liquidity")),
                                    "thesis_pressure": _as_dict(outcome.get("thesis_pressure")),
                                    "uncertainty": uncertainty,
                                },
                                "expected_return_pct": _ratio_from_pct(
                                    scenario_outcome.get("target_return_pct_of_book")
                                ),
                                "loss_pct": _ratio_from_pct(scenario_outcome.get("loss_pct_of_book")),
                                "probability": _optional_float(scenario_outcome.get("probability")),
                                "confidence": _confidence_from_uncertainty(uncertainty),
                                "generated_by_source": "api",
                                "generated_by_action": "scenario_simulator.evaluate",
                                "generated_by_run_id": simulation_id,
                                "as_of": simulation.get("generated_at") or now,
                                "status": "simulated",
                                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                            },
                            now,
                            actor=actor,
                            provenance=provenance_id,
                            input_hash=_hash_value(scenario_outcome),
                        )
                    )
                    rows.append(
                        self.object_service.write_relation(
                            course_uid,
                            scenario_uid,
                            "course_of_action_uses_scenario",
                            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                            now,
                            actor=actor,
                            provenance=provenance_id,
                            input_hash=input_hash,
                        )
                    )
                    rows.append(
                        self.object_service.write_relation(
                            course_uid,
                            simulated_uid,
                            "course_of_action_has_simulated_outcome",
                            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                            now,
                            actor=actor,
                            provenance=provenance_id,
                            input_hash=input_hash,
                        )
                    )
                    if gate_uid:
                        rows.append(
                            self.object_service.write_relation(
                                gate_uid,
                                scenario_uid,
                                "policy_gate_uses_scenario",
                                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                                now,
                                actor=actor,
                                provenance=provenance_id,
                                input_hash=input_hash,
                            )
                        )

                outcome_artifacts[candidate_id] = {
                    "course_of_action_id": course_uid,
                    "simulated_outcome_ids": simulated_outcome_ids,
                    **({"policy_gate_result_id": gate_uid} if gate_uid else {}),
                }
        except Exception as exc:
            _handle_writeback_error("record_scenario_simulation", exc)

        return {
            "rows_written": len(rows),
            "artifact_ids": {
                "comparison_id": comparison_uid,
                "scenario_ids": list(dict.fromkeys(scenario_uids.values())),
                "assumption_ids": assumption_uids,
                "policy_gate_result_ids": policy_gate_uids,
            },
            "outcome_artifact_ids": outcome_artifacts,
        }

    def _record_recommendation_bundle(
        self,
        *,
        report_id_value: str,
        source_record_id: str,
        record: Mapping[str, Any],
        approval_id: int | None,
        actor: Mapping[str, Any],
        provenance_id: str,
        valid_from: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        recommendation_key = _recommendation_key(record)
        recommendation_properties = _recommendation_properties(record, approval_id=approval_id)
        recommendation_row = self.object_service.write_object(
            "Recommendation",
            recommendation_key,
            recommendation_properties,
            valid_from,
            actor=actor,
            provenance=provenance_id,
            approval_id=approval_id,
            input_hash=str(record.get("input_hash") or "") or None,
        )
        rows.append(recommendation_row)
        recommendation_object_uid = recommendation_uid(
            record.get("id")
            or record.get("idempotency_key")
            or f"{record.get('report_type')}:{record.get('as_of')}:{record.get('action')}:{record.get('ticker')}"
        )
        rows.append(
            self.object_service.write_relation(
                report_run_id(report_id_value),
                recommendation_object_uid,
                "report_run_produces_recommendation",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                valid_from,
                actor=actor,
                provenance=provenance_id,
                approval_id=approval_id,
            )
        )
        rows.append(
            self.object_service.write_relation(
                recommendation_object_uid,
                source_record_object_id(source_record_id),
                "recommendation_supported_by_source_record",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                valid_from,
                actor=actor,
                provenance=provenance_id,
                approval_id=approval_id,
            )
        )
        rows.extend(
            self._record_recommendation_evidence(
                recommendation_object_uid=recommendation_object_uid,
                recommendation_key=recommendation_key,
                record=record,
                actor=actor,
                provenance_id=provenance_id,
                valid_from=valid_from,
                approval_id=approval_id,
            )
        )
        rows.extend(
            self._record_recommendation_risk_snapshots(
                recommendation_object_uid=recommendation_object_uid,
                recommendation_key=recommendation_key,
                record=record,
                actor=actor,
                provenance_id=provenance_id,
                valid_from=valid_from,
                approval_id=approval_id,
            )
        )
        if record.get("account_id"):
            rows.append(
                self.object_service.write_relation(
                    recommendation_object_uid,
                    account_id(record["account_id"]),
                    "recommendation_targets_account",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
        if record.get("portfolio_id"):
            rows.append(
                self.object_service.write_relation(
                    recommendation_object_uid,
                    portfolio_id(record["portfolio_id"]),
                    "recommendation_targets_portfolio",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
        for metric_key in _ids_from(record.get("risk_metric_ids") or record.get("risk_metrics"), "metric_id"):
            rows.append(
                self.object_service.write_relation(
                    recommendation_object_uid,
                    risk_metric_id(metric_key),
                    "recommendation_uses_risk_metric",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
        for scenario_key in _ids_from(record.get("scenario_ids") or record.get("scenarios"), "scenario_id"):
            rows.append(
                self.object_service.write_relation(
                    recommendation_object_uid,
                    scenario_id(scenario_key),
                    "recommendation_uses_scenario",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
        gate = record.get("policy_gate_result")
        gate_id = record.get("policy_gate_result_id")
        if isinstance(gate, Mapping) or gate_id:
            gate_key = str(gate_id or _hash_value(gate or record))
            if gate_key.startswith("policy_gate_result:"):
                gate_key = gate_key.split(":", 1)[1]
            gate_payload = gate if isinstance(gate, Mapping) else {}
            rows.append(
                self.object_service.write_object(
                    "PolicyGateResult",
                    policy_gate_result_id(gate_key),
                    {
                        "gate_result_id": gate_key,
                        "decision": str(gate_payload.get("decision") or record.get("policy_gate_decision") or "warn"),
                        "review_required": bool(
                            gate_payload.get("review_required") or record.get("policy_gate_review_required")
                        ),
                        "failure_reasons": _as_list(
                            gate_payload.get("failure_reasons") or record.get("policy_gate_failures")
                        ),
                        "warnings": _as_list(gate_payload.get("warnings") or record.get("policy_gate_warnings")),
                        "account_id": record.get("account_id"),
                        "portfolio_id": record.get("portfolio_id"),
                        "policy_id": record.get("policy_id"),
                        "evaluated_at": gate_payload.get("evaluated_at") or _now(),
                        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    },
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
            rows.append(
                self.object_service.write_relation(
                    policy_gate_result_id(gate_key),
                    recommendation_object_uid,
                    "policy_gate_evaluates_recommendation",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
            rows.append(
                self.object_service.write_relation(
                    recommendation_object_uid,
                    policy_gate_result_id(gate_key),
                    "recommendation_has_policy_gate_result",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
        if _is_actionable(record):
            proposal_key = str(record.get("trade_proposal_id") or record.get("idempotency_key") or recommendation_key)
            rows.append(
                self.object_service.write_object(
                    "TradeProposal",
                    proposal_key,
                    {
                        "proposal_id": proposal_key,
                        "recommendation_id": recommendation_key,
                        "account_id": record.get("account_id"),
                        "portfolio_id": record.get("portfolio_id"),
                        "action": str(record.get("action") or "review"),
                        "instrument": str(record.get("instrument") or record.get("ticker") or "portfolio"),
                        "proposed_change": _as_dict(record.get("trade_proposal"))
                        or {
                            "target_change": record.get("target_change"),
                            "horizon": record.get("horizon"),
                        },
                        "sizing_summary": _as_dict(record.get("sizing_summary")),
                        "risk_summary": _as_dict(record.get("risk_summary")),
                        "policy_gate_result_id": _optional_int(record.get("policy_gate_result_id")),
                        "approval_id": approval_id,
                        "decision_state": "pending_approval" if approval_id is not None else "staged",
                        "status": "pending_approval" if approval_id is not None else "staged",
                        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    },
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
            proposal_uid = trade_proposal_id(proposal_key)
            rows.append(
                self.object_service.write_relation(
                    recommendation_object_uid,
                    proposal_uid,
                    "recommendation_has_trade_proposal",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
            rows.append(
                self.object_service.write_relation(
                    proposal_uid,
                    recommendation_object_uid,
                    "trade_proposal_derives_from_recommendation",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                )
            )
            ticker = str(record.get("ticker") or "").strip().upper()
            if ticker:
                rows.append(
                    self.object_service.write_relation(
                        proposal_uid,
                        asset_id(ticker),
                        "trade_proposal_targets_asset",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                        valid_from,
                        actor=actor,
                        provenance=provenance_id,
                        approval_id=approval_id,
                    )
                )
            if approval_id is not None:
                rows.append(
                    self.object_service.write_relation(
                        proposal_uid,
                        approval_uid(approval_id),
                        "trade_proposal_requires_approval",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "approval_id": str(approval_id)},
                        valid_from,
                        actor=actor,
                        provenance=provenance_id,
                        approval_id=approval_id,
                    )
                )
                rows.append(
                    self.object_service.write_relation(
                        approval_uid(approval_id),
                        recommendation_object_uid,
                        "approval_targets_recommendation",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "target_object_type": "Recommendation"},
                        valid_from,
                        actor=actor,
                        provenance=provenance_id,
                        approval_id=approval_id,
                    )
                )
                rows.append(
                    self.object_service.write_relation(
                        approval_uid(approval_id),
                        proposal_uid,
                        "approval_targets_trade_proposal",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "target_object_type": "TradeProposal"},
                        valid_from,
                        actor=actor,
                        provenance=provenance_id,
                        approval_id=approval_id,
                    )
                )
        return rows

    def _record_recommendation_evidence(
        self,
        *,
        recommendation_object_uid: str,
        recommendation_key: str,
        record: Mapping[str, Any],
        actor: Mapping[str, Any],
        provenance_id: str,
        valid_from: str,
        approval_id: int | None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for relation_type, evidence_items in (
            ("recommendation_supported_by_evidence", _as_list(record.get("evidence"))),
            ("recommendation_contradicted_by_evidence", _as_list(record.get("disconfirming_evidence"))),
        ):
            for index, item in enumerate(evidence_items):
                if item is None or item == "":
                    continue
                evidence_payload = _as_dict(item) if isinstance(item, (Mapping, str)) else {}
                summary = (
                    evidence_payload.get("summary")
                    or evidence_payload.get("text")
                    or evidence_payload.get("evidence")
                    or evidence_payload.get("description")
                    or (item if isinstance(item, str) else None)
                )
                if not str(summary or "").strip():
                    continue
                role = "supporting" if relation_type == "recommendation_supported_by_evidence" else "disconfirming"
                evidence_key = str(
                    evidence_payload.get("evidence_id") or f"{recommendation_key}:{role}:{index}:{_hash_value(item)}"
                )
                evidence_row = self.object_service.write_object(
                    "Evidence",
                    evidence_key,
                    {
                        "evidence_id": evidence_key,
                        "evidence_type": str(evidence_payload.get("evidence_type") or role),
                        "title": evidence_payload.get("title") or evidence_payload.get("source"),
                        "summary": _truncate(summary, 2000),
                        "source_record_id": evidence_payload.get("source_record_id"),
                        "document_artifact_id": evidence_payload.get("document_artifact_id"),
                        "confidence": _optional_float(evidence_payload.get("confidence")),
                        "observed_at": evidence_payload.get("observed_at") or record.get("as_of"),
                        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    },
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                    input_hash=_hash_value(item),
                )
                rows.append(evidence_row)
                evidence_uid_value = evidence_id(evidence_key)
                rows.append(
                    self.object_service.write_relation(
                        recommendation_object_uid,
                        evidence_uid_value,
                        relation_type,
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "relation_role": role},
                        valid_from,
                        actor=actor,
                        provenance=provenance_id,
                        approval_id=approval_id,
                        input_hash=_hash_value(item),
                    )
                )
                citation_payload = _citation_payload(evidence_payload)
                if citation_payload:
                    citation_key = str(
                        citation_payload.get("citation_id")
                        or f"{recommendation_key}:{role}:{index}:citation:{_hash_value(citation_payload)}"
                    )
                    citation_row = self.object_service.write_object(
                        "Citation",
                        citation_key,
                        {
                            "citation_id": citation_key,
                            "source_record_id": citation_payload.get("source_record_id")
                            or evidence_payload.get("source_record_id"),
                            "document_artifact_id": citation_payload.get("document_artifact_id")
                            or evidence_payload.get("document_artifact_id"),
                            "title": citation_payload.get("title") or evidence_payload.get("title"),
                            "url": citation_payload.get("url"),
                            "source_path": citation_payload.get("source_path"),
                            "quote_hash": citation_payload.get("quote_hash")
                            or _hash_text(str(citation_payload.get("quote") or summary), length=32),
                            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                        },
                        valid_from,
                        actor=actor,
                        provenance=provenance_id,
                        approval_id=approval_id,
                        input_hash=_hash_value(citation_payload),
                    )
                    rows.append(citation_row)
                    citation_uid_value = citation_id(citation_key)
                    for citation_relation in ("evidence_has_citation", "evidence_cites_citation"):
                        rows.append(
                            self.object_service.write_relation(
                                evidence_uid_value,
                                citation_uid_value,
                                citation_relation,
                                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                                valid_from,
                                actor=actor,
                                provenance=provenance_id,
                                approval_id=approval_id,
                                input_hash=_hash_value(citation_payload),
                            )
                        )
        return rows

    def _record_recommendation_risk_snapshots(
        self,
        *,
        recommendation_object_uid: str,
        recommendation_key: str,
        record: Mapping[str, Any],
        actor: Mapping[str, Any],
        provenance_id: str,
        valid_from: str,
        approval_id: int | None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        position_snapshot_id = str(record.get("risk_snapshot_id") or "").strip()
        if position_snapshot_id:
            props = _position_risk_snapshot_props(record, snapshot_id=position_snapshot_id)
            rows.append(
                self.object_service.write_object(
                    "PositionRiskSnapshot",
                    position_snapshot_id,
                    props,
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                    input_hash=_hash_value(props),
                )
            )
            rows.append(
                self.object_service.write_relation(
                    recommendation_object_uid,
                    position_risk_snapshot_id(position_snapshot_id),
                    "recommendation_uses_position_risk_snapshot",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                    input_hash=recommendation_key,
                )
            )
        portfolio_snapshot_id = str(record.get("portfolio_risk_snapshot_id") or "").strip()
        if portfolio_snapshot_id:
            props = _portfolio_risk_snapshot_props(record, snapshot_id=portfolio_snapshot_id)
            rows.append(
                self.object_service.write_object(
                    "PortfolioRiskSnapshot",
                    portfolio_snapshot_id,
                    props,
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                    input_hash=_hash_value(props),
                )
            )
            rows.append(
                self.object_service.write_relation(
                    recommendation_object_uid,
                    portfolio_risk_snapshot_id(portfolio_snapshot_id),
                    "recommendation_uses_portfolio_risk_snapshot",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    approval_id=approval_id,
                    input_hash=recommendation_key,
                )
            )
        return rows

    def _record_report_context_objects(
        self,
        *,
        report_id_value: str,
        payload: Mapping[str, Any],
        source_record_id: str,
        actor: Mapping[str, Any],
        provenance_id: str,
        valid_from: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        summary = _as_dict(payload.get("summary"))
        for index, item in enumerate(_dicts(payload.get("risk_metrics") or summary.get("risk_metrics"))):
            metric_key = str(
                item.get("metric_id")
                or f"report:{report_id_value}:risk_metric:{index}:{item.get('metric') or item.get('name') or 'metric'}"
            )
            rows.append(
                self.object_service.write_object(
                    "RiskMetric",
                    metric_key,
                    {
                        "metric_id": metric_key,
                        "metric": str(item.get("metric") or item.get("name") or "metric"),
                        "scope_type": item.get("scope_type"),
                        "scope_id": item.get("scope_id"),
                        "value": item.get("value"),
                        "unit": item.get("unit"),
                        "method": item.get("method"),
                        "window": item.get("window"),
                        "confidence": _optional_float(item.get("confidence")),
                        "source_record_ids": [source_record_id],
                        "as_of": item.get("as_of") or payload.get("as_of"),
                        "source": item.get("source") or "report_sync",
                        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    },
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    source_record_id=source_record_id,
                    input_hash=_hash_value(item),
                )
            )
        for index, item in enumerate(_dicts(payload.get("scenarios") or summary.get("scenarios"))):
            scenario_key = str(
                item.get("scenario_id")
                or f"report:{report_id_value}:scenario:{index}:{item.get('scenario_type') or item.get('name') or 'scenario'}"
            )
            rows.append(
                self.object_service.write_object(
                    "Scenario",
                    scenario_key,
                    {
                        "scenario_id": scenario_key,
                        "name": str(item.get("name") or item.get("scenario_type") or "scenario"),
                        "scenario_type": str(item.get("scenario_type") or "stress"),
                        "scope_type": item.get("scope_type"),
                        "scope_id": item.get("scope_id"),
                        "assumptions_hash": item.get("assumptions_hash")
                        or _hash_value(item.get("assumptions") or item),
                        "result": _as_dict(item.get("result")),
                        "result_metrics": _as_dict(item.get("result_metrics") or item.get("metrics")),
                        "loss_pct": _optional_float(item.get("loss_pct") or item.get("stress_loss")),
                        "generated_by_source": item.get("generated_by_source") or "report_sync",
                        "generated_by_action": item.get("generated_by_action"),
                        "generated_by_run_id": item.get("generated_by_run_id"),
                        "as_of": item.get("as_of") or payload.get("as_of"),
                        "status": str(item.get("status") or "generated"),
                        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    },
                    valid_from,
                    actor=actor,
                    provenance=provenance_id,
                    source_record_id=source_record_id,
                    input_hash=_hash_value(item),
                )
            )
        return rows


def record_report_output(**kwargs: Any) -> list[dict[str, Any]]:
    return DecisionOntologyWriteback().record_report_output(**kwargs)


def record_workflow_artifact_proposal(**kwargs: Any) -> list[dict[str, Any]]:
    return DecisionOntologyWriteback().record_workflow_artifact_proposal(**kwargs)


def apply_approved_decision(**kwargs: Any) -> list[dict[str, Any]]:
    return DecisionOntologyWriteback().apply_approved_decision(**kwargs)


def record_scenario_simulation(**kwargs: Any) -> dict[str, Any]:
    return DecisionOntologyWriteback().record_scenario_simulation(**kwargs)


def _recommendation_key(record: Mapping[str, Any]) -> str:
    return str(
        record.get("id")
        or record.get("idempotency_key")
        or _hash_value(
            {
                "report_type": record.get("report_type"),
                "as_of": record.get("as_of"),
                "action": record.get("action"),
                "ticker": record.get("ticker"),
                "instrument": record.get("instrument"),
            }
        )
    )


def _recommendation_properties(record: Mapping[str, Any], *, approval_id: int | None) -> dict[str, Any]:
    action = str(record.get("action") or "watch")
    is_actionable = action in ACTIONABLE_RECOMMENDATION_ACTIONS
    status = str(record.get("status") or record.get("recommendation_status") or "open")
    decision_state = "proposed" if approval_id is not None else "generated"
    if status in {"blocked", "error", "closed"} or action in {"watch", "hold", "avoid", "do_nothing"}:
        decision_state = "closed" if status in {"blocked", "error", "closed"} else "generated"
    return {
        "recommendation_id": _recommendation_key(record),
        "idempotency_key": record.get("idempotency_key"),
        "source_kind": "report",
        "report_type": record.get("report_type"),
        "as_of": record.get("as_of"),
        "action": action,
        "ticker": record.get("ticker"),
        "instrument": record.get("instrument") or record.get("ticker") or "portfolio",
        "decision_state": decision_state,
        "status": status,
        "approval_id": approval_id or _optional_int(record.get("approval_id")),
        "approval_required": bool(record.get("approval_required") or is_actionable),
        "approval_status": record.get("approval_status"),
        "outcome_status": record.get("outcome_status") or "pending",
        "supersedes_recommendation_id": record.get("supersedes_recommendation_id"),
        "account_id": record.get("account_id"),
        "portfolio_id": record.get("portfolio_id"),
        "policy_id": record.get("policy_id"),
        "policy_gate_result_id": _optional_int(record.get("policy_gate_result_id")),
        "policy_gate_decision": record.get("policy_gate_decision") or record.get("policy_gate_status"),
        "policy_gate_review_required": bool(record.get("policy_gate_review_required")),
        "confidence": _optional_float(record.get("confidence")),
        "horizon": record.get("horizon"),
        "rationale_summary": _truncate(record.get("rationale"), 500),
        "rationale_hash": _hash_text(str(record.get("rationale") or "")) if record.get("rationale") else None,
        "source_quality": record.get("source_quality"),
        "decision_quality": _as_dict(record.get("decision_quality")) or None,
        "decision_quality_gate": _as_dict(record.get("decision_quality_gate")) or None,
        "payload": _jsonable(dict(record)),
        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
    }


def _version_ref_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    temporal = _temporal(row)
    object_uid = str(row.get("object_uid") or temporal.get("object_uid") or "")
    version_id = str(row.get("version_id") or temporal.get("version_id") or "")
    if not object_uid or not version_id:
        return None
    ref_id = f"{object_uid}:{version_id}"
    return {
        "ref_id": ref_id,
        "object_uid": object_uid,
        "object_type": row.get("object_type"),
        "version_id": version_id,
        "valid_from": temporal.get("valid_from") or row.get("valid_from"),
        "tx_from": temporal.get("tx_from") or row.get("tx_from"),
        "temporal_confidence": temporal.get("temporal_confidence") or row.get("temporal_confidence"),
        "source_record_id": row.get("source_record_id"),
    }


def _temporal(row: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = row.get("_meta")
    if isinstance(meta, Mapping):
        temporal = meta.get("temporal")
        if isinstance(temporal, Mapping):
            return temporal
    return {}


def _is_actionable(record: Mapping[str, Any]) -> bool:
    return str(record.get("action") or "").lower() in ACTIONABLE_RECOMMENDATION_ACTIONS


def _coa_actionability(action: str, gate: Mapping[str, Any], uncertainty: Mapping[str, Any]) -> str:
    if str(gate.get("decision") or "").lower() == "blocked":
        return "blocked_by_policy"
    if str(uncertainty.get("level") or "").lower() == "high":
        return "missing_inputs"
    if action == "hold":
        return "watch_only"
    return "actionable"


def _confidence_from_uncertainty(uncertainty: Mapping[str, Any]) -> float:
    return {"low": 0.85, "medium": 0.65, "high": 0.35}.get(str(uncertainty.get("level") or "").lower(), 0.5)


def _ratio_from_pct(value: Any) -> float | None:
    number = _optional_float(value)
    if number is None:
        return None
    return number / 100.0


def _artifact_uri(payload: Mapping[str, Any]) -> str | None:
    paths = payload.get("artifact_paths")
    if isinstance(paths, Mapping):
        for value in paths.values():
            if value:
                return str(value)
    return None


def _provenance_id(provenance: Mapping[str, Any] | str | None, default: str) -> str:
    if isinstance(provenance, str) and provenance:
        return provenance
    if isinstance(provenance, Mapping):
        for key in ("provenance_event_id", "event_id", "id"):
            value = provenance.get(key)
            if value:
                return str(value)
    return default


def _handle_writeback_error(surface: str, exc: Exception) -> None:
    raise exc


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_text(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _hash_value(value: Any, *, length: int = 16) -> str:
    raw = json.dumps(_jsonable(value), sort_keys=True, default=str, separators=(",", ":"))
    return _hash_text(raw, length=length)


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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return _as_dict(decoded)
    return {}


def _citation_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    citation = value.get("citation")
    if isinstance(citation, Mapping):
        return {str(key): item for key, item in citation.items()}
    if any(value.get(key) for key in ("url", "source_path", "document_artifact_id", "source_record_id")):
        return {
            "title": value.get("title") or value.get("source"),
            "url": value.get("url"),
            "source_path": value.get("source_path"),
            "document_artifact_id": value.get("document_artifact_id"),
            "source_record_id": value.get("source_record_id"),
            "quote": value.get("quote") or value.get("summary") or value.get("text"),
        }
    return {}


def _position_risk_snapshot_props(record: Mapping[str, Any], *, snapshot_id: str) -> dict[str, Any]:
    risk_status = _as_dict(record.get("risk_source_status"))
    bindings = _as_list(record.get("risk_bindings"))
    payload = {
        "risk_bindings": bindings,
        "risk_source_status": risk_status,
    }
    return {
        "snapshot_id": snapshot_id,
        "ticker": record.get("ticker"),
        "portfolio_risk_snapshot_id": record.get("portfolio_risk_snapshot_id"),
        "as_of": record.get("as_of"),
        "computed_at": risk_status.get("computed_at"),
        "risk_score": _optional_float(record.get("risk_score")),
        "risk_level": record.get("risk_level"),
        "confidence": _optional_float(record.get("risk_confidence")),
        "quality": record.get("risk_quality") or risk_status.get("quality"),
        "source_status": risk_status,
        "payload": payload,
        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
    }


def _portfolio_risk_snapshot_props(record: Mapping[str, Any], *, snapshot_id: str) -> dict[str, Any]:
    risk_status = _as_dict(record.get("risk_source_status"))
    bindings = _as_list(record.get("risk_bindings"))
    return {
        "snapshot_id": snapshot_id,
        "as_of": record.get("as_of"),
        "computed_at": risk_status.get("computed_at"),
        "average_risk_score": _optional_float(risk_status.get("average_risk_score")),
        "max_risk_score": _optional_float(risk_status.get("max_risk_score")),
        "confidence": _optional_float(record.get("risk_confidence") or risk_status.get("confidence")),
        "quality": record.get("risk_quality") or risk_status.get("quality"),
        "position_count": _optional_int(risk_status.get("position_count")),
        "position_snapshot_ids": [
            str(item.get("risk_snapshot_id"))
            for item in bindings
            if isinstance(item, Mapping) and item.get("risk_snapshot_id")
        ],
        "source_status": risk_status,
        "payload": {"risk_bindings": bindings, "risk_source_status": risk_status},
        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _dicts(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, Mapping)]


def _ids_from(value: Any, id_key: str) -> list[str]:
    ids: list[str] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            item_value = item.get(id_key) or item.get("id") or item.get("key")
        else:
            item_value = item
        if item_value is not None and str(item_value).strip():
            ids.append(str(item_value).strip())
    return ids


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truncate(value: Any, max_chars: int) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:max_chars]
