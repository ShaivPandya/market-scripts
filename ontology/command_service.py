"""Ontology-primary command layer for governed investing mutations."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from decision_quality.actions import ACTIONABLE_ACTIONS as DECISION_ACTIONABLE_ACTIONS
from decision_quality.actions import normalize_action
from decision_quality.gates import apply_decision_quality_gates
from decision_quality.models import DecisionQualityGate, DecisionQualityGateReason, parse_decision_quality
from ontology.approval_workflow import (
    approval_requirement_progress,
    normalize_approval_decisions,
    normalize_approval_requirements,
    select_approval_requirement,
)
from ontology.object_service import OntologyObjectService
from ontology.policy import DEFAULT_ONTOLOGY_POLICY as POLICY
from ontology.policy import Actor, NodeResource, PolicyDecision, actor_to_dict, admin_actor, require_allowed
from ontology.schemas.identity import (
    account_id,
    action_item_id,
    action_run_id,
    approval_id,
    audit_event_id,
    citation_id,
    company_financial_profile_id,
    course_of_action_id,
    course_of_action_rationale_id,
    document_artifact_id,
    equity_overview_id,
    evidence_id,
    executed_decision_record_id,
    extrinsic_sensitivity_id,
    industry_force_assessment_id,
    instrument_id,
    issuer_id,
    management_quality_accomplishment_id,
    management_quality_assessment_id,
    management_quality_scorecard_row_id,
    management_quality_setback_id,
    mission_definition_id,
    monitor_definition_id,
    monitor_hit_id,
    policy_gate_result_id,
    portfolio_id,
    portfolio_risk_snapshot_id,
    position_id,
    position_risk_snapshot_id,
    recommendation_id,
    supply_chain_relationship_id,
    supply_demand_outlook_id,
    thesis_document_id,
    thesis_id,
    thesis_section_id,
    trade_proposal_id,
    watch_trigger_id,
)
from ontology.schemas.objects import normalize_course_of_action_action

OPERATIONAL_ONTOLOGY_RUN_ID = "operational"
logger = logging.getLogger(__name__)
ACTIONABLE_ACTIONS = set(DECISION_ACTIONABLE_ACTIONS)
COURSE_OF_ACTION_CREATE_ACTION_IDS = {"create_course_of_action", "create_recommendation"}
FINANCIAL_ACTION_IDS = {"update_portfolio_positions", "update_hedge_positions", *COURSE_OF_ACTION_CREATE_ACTION_IDS}
RESEARCH_ACTION_IDS = {
    "change_thesis_status",
    "create_catalyst",
    "update_catalyst_status",
    "create_kill_condition",
    "update_kill_condition_status",
    "create_thesis_claim",
    "update_thesis_claim",
    "save_thesis_content",
    "save_overview_content",
    "save_management_quality_content",
    "save_evaluation",
    "create_research_note",
    "create_portfolio_news_digest",
    "delete_portfolio_news_digest",
    "create_analyst_feedback",
    "finalize_decision_outcome",
    "create_action_item",
    "complete_action_item",
    "dismiss_action_item",
    "create_watch_trigger",
    "fire_watch_trigger",
    "cancel_watch_trigger",
    "replace_watch_trigger",
    "update_watch_trigger_check",
    "update_watch_trigger_definition",
    "create_monitor_definition",
    "update_monitor_definition",
    "disable_monitor_definition",
    "create_mission_definition",
    "update_mission_definition",
    "disable_mission_definition",
    "create_monitor_hit",
    "update_monitor_hit_status",
}
WATCH_TRIGGER_TARGET_ACTION_IDS = {
    "fire_watch_trigger",
    "cancel_watch_trigger",
    "replace_watch_trigger",
    "update_watch_trigger_check",
    "update_watch_trigger_definition",
}


class OntologyCommandError(Exception):
    message: str

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class OntologyCommandNotFound(OntologyCommandError):
    resource: str
    identifier: str

    def __init__(self, resource: str, identifier: str):
        super().__init__(f"{resource} not found: {identifier}")
        self.resource = resource
        self.identifier = identifier


class OntologyCommandConflict(OntologyCommandError):
    pass


class OntologyCommandValidationError(OntologyCommandError):
    pass


@dataclass(frozen=True, slots=True)
class OntologyCommandContext:
    actor: Actor
    source_type: str
    source_id: str
    request_mode: str = "proposal"

    @property
    def actor_type(self) -> str:
        return self.actor.actor_type

    @property
    def actor_id(self) -> str:
        return self.actor.actor_id


class OntologyCommandService:
    """Write approvals, recommendations, research, and position state through ontology only."""

    def __init__(self, object_service: OntologyObjectService | None = None):
        self.objects = object_service or OntologyObjectService()

    def _require_policy_allowed(
        self,
        decision: PolicyDecision,
        *,
        actor: Actor,
        provenance_id: str,
        object_refs: list[dict[str, Any]],
    ) -> None:
        if decision.allowed:
            return
        self._write_audit(
            "policy.denied",
            "policy",
            "denied",
            actor=actor,
            provenance_id=provenance_id,
            object_refs=object_refs,
            after_summary=dict(decision.audit),
        )
        require_allowed(decision)

    def list_approvals(
        self,
        *,
        actor: Actor | None = None,
        status: str | None = "pending",
        ticker: str | None = None,
        application_status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        actor = actor or admin_actor(source="ontology_command")
        read_decision = POLICY.check_action(actor, "approval.read", _approval_policy_context())
        self._require_policy_allowed(
            read_decision,
            actor=actor,
            provenance_id=_provenance_id("policy", "approval.read", actor.actor_id),
            object_refs=[{"type": "Approval", "id": "*"}],
        )
        filters: dict[str, Any] = {}
        if status:
            filters["status"] = status
        if ticker:
            filters["ticker"] = str(ticker).strip().upper()
        if application_status:
            filters["application_status"] = application_status
        rows = self.objects.query_objects("Approval", filters=filters, limit=limit)
        approvals: list[dict[str, Any]] = []
        for row in rows:
            approval = _flatten_object(row)
            object_decision = POLICY.check_object(actor, _approval_node_resource(approval), action="read")
            if object_decision.allowed:
                approvals.append(approval)
                continue
            self._write_audit(
                "policy.denied",
                "policy",
                "denied",
                actor=actor,
                provenance_id=_provenance_id("policy", "approval.read", actor.actor_id, approval.get("id")),
                object_refs=[{"type": "Approval", "id": str(approval.get("id") or "")}],
                after_summary=dict(object_decision.audit),
            )
        return approvals

    def get_approval(self, approval_uid: str, *, actor: Actor | None = None) -> dict[str, Any]:
        actor = actor or admin_actor(source="ontology_command")
        read_decision = POLICY.check_action(actor, "approval.read", _approval_policy_context(approval_id=approval_uid))
        self._require_policy_allowed(
            read_decision,
            actor=actor,
            provenance_id=_provenance_id("policy", "approval.read", actor.actor_id, approval_uid),
            object_refs=[{"type": "Approval", "id": str(approval_uid)}],
        )
        row = self.objects.get_object(_normalize_approval_uid(approval_uid))
        if not row:
            raise OntologyCommandNotFound("Approval", approval_uid)
        approval = _flatten_object(row)
        object_decision = POLICY.check_object(actor, _approval_node_resource(approval), action="read")
        self._require_policy_allowed(
            object_decision,
            actor=actor,
            provenance_id=_provenance_id("policy", "approval.read", actor.actor_id, approval_uid),
            object_refs=[{"type": "Approval", "id": str(approval.get("id") or approval_uid)}],
        )
        return approval

    def propose_action(
        self,
        action_id: str,
        payload: Mapping[str, Any],
        context: OntologyCommandContext,
        *,
        reason: str | None = None,
        entity_id: str | None = None,
        supersedes_approval_id: str | None = None,
    ) -> dict[str, Any]:
        action_id = _non_blank(action_id, "action_id")
        payload_dict = dict(payload)
        _validate_governed_action(action_id, payload_dict)
        from ontology.runtime_read_service import runtime_object_service

        with runtime_object_service(self.objects):
            payload_dict = _approval_payload_for_action(action_id, payload_dict)
            _normalize_create_recommendation_payload(action_id, payload_dict)
            self._prepare_action_payload(action_id, payload_dict)
            base_state_hash = _base_state_hash(action_id, payload_dict)
        if _recommendation_should_persist_without_approval(action_id, payload_dict):
            return self._persist_recommendation_without_approval(
                payload_dict,
                context,
                reason=reason,
            )
        policy_gate_result = self._evaluate_policy_gate(action_id, payload_dict, context)
        if _recommendation_policy_gate_blocks(policy_gate_result):
            _downgrade_recommendation_for_policy_gate(payload_dict, policy_gate_result)
            return self._persist_recommendation_without_approval(
                payload_dict,
                context,
                reason=reason,
            )
        approval_requirements = normalize_approval_requirements((policy_gate_result or {}).get("approval_requirements"))
        is_financial = _is_financial_action_for_payload(action_id, payload_dict)
        now = _now()
        input_hash = _stable_hash({"action_id": action_id, "payload": payload_dict})
        approval_scope = _approval_scope_from_payload(payload_dict)
        policy_decision = POLICY.check_action(
            context.actor,
            "approval.create",
            _approval_policy_context(
                action_id=action_id,
                payload=payload_dict,
                command_context=context,
                risk_class="financial" if is_financial else "research",
                policy_gate_result=policy_gate_result,
                account_id=approval_scope["account_id"],
                portfolio_id=approval_scope["portfolio_id"],
            ),
        )
        self._require_policy_allowed(
            policy_decision,
            actor=context.actor,
            provenance_id=_provenance_id("policy", "approval.create", context.actor.actor_id, input_hash),
            object_refs=[{"type": "Approval", "id": action_id}],
        )
        entity_type = _entity_type_for_action(action_id)
        normalized_supersedes_id = _normalize_approval_uid(supersedes_approval_id) if supersedes_approval_id else None
        uid_hash = input_hash
        if normalized_supersedes_id:
            uid_hash = _stable_hash(
                {
                    "action_id": action_id,
                    "payload": payload_dict,
                    "supersedes_approval_id": normalized_supersedes_id,
                }
            )
        approval_uid = approval_id(f"{entity_type}:{uid_hash}")
        provenance_id = _provenance_id("approval", approval_uid, input_hash)
        ticker = _ticker_from_payload(payload_dict)
        target_uid, target_type = _target_for_action(action_id, payload_dict)

        props = {
            "entity_type": entity_type,
            "entity_id": entity_id or target_uid,
            "ticker": ticker,
            "account_id": approval_scope["account_id"],
            "portfolio_id": approval_scope["portfolio_id"],
            "data_sensitivity": "portfolio_private" if is_financial else "research_private",
            "target_object_uid": target_uid,
            "target_object_type": target_type,
            "action_id": action_id,
            "action_schema_name": action_id,
            "action_schema_version": 1,
            "action_input_hash": input_hash,
            "proposed_change": payload_dict,
            "reason": reason,
            "source_type": context.source_type,
            "source_id": context.source_id,
            "status": "pending",
            "resolution_state": "pending",
            "application_state": "pending",
            "application_status": "pending",
            "application_attempts": 0,
            "risk_class": "financial" if is_financial else "research",
            "approval_required": bool((policy_gate_result or {}).get("approval_required", True)),
            "approval_mode": (policy_gate_result or {}).get("approval_mode") or _approval_mode_from_context(context),
            "approval_requirements": approval_requirements,
            "approval_decisions": [],
            "approval_policy_rule_id": (policy_gate_result or {}).get("rule_id"),
            "approval_policy_reason": (policy_gate_result or {}).get("reason"),
            "approval_note_required": True,
            "policy_gate_result": policy_gate_result,
            "policy_gate_result_id": policy_gate_result.get("policy_gate_result_id") if policy_gate_result else None,
            "policy_gate_decision": policy_gate_result.get("decision") if policy_gate_result else None,
            "policy_decision_id": policy_decision.decision_id,
            "policy_matched_rule": policy_decision.matched_rule,
            "policy_explanation": policy_decision.explanation,
            "base_state_hash": base_state_hash,
            "requested_by_actor_id": context.actor.actor_id,
            "created_at": now,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        }
        if normalized_supersedes_id:
            props["supersedes_approval_id"] = normalized_supersedes_id

        row = self.objects.write_object(
            "Approval",
            approval_uid,
            props,
            now,
            actor=actor_to_dict(context.actor),
            provenance=provenance_id,
            input_hash=input_hash,
        )
        self._write_audit(
            "approval.created",
            "approval",
            "succeeded",
            actor=context.actor,
            provenance_id=provenance_id,
            object_refs=[{"type": "Approval", "id": approval_uid}],
            after_summary={
                "action_id": action_id,
                "target_object_uid": target_uid,
                "approval_policy_rule_id": props.get("approval_policy_rule_id"),
                "approval_policy_reason": props.get("approval_policy_reason"),
                "policy_decision_id": props.get("policy_decision_id"),
                "policy_matched_rule": props.get("policy_matched_rule"),
            },
        )
        if action_id in COURSE_OF_ACTION_CREATE_ACTION_IDS:
            record = _dict(payload_dict.get("record") or payload_dict)
            self._write_course_of_action_bundle(
                record,
                approval_object_id=approval_uid,
                action_run_id=None,
                actor=actor_to_dict(context.actor),
                provenance_id=provenance_id,
                input_hash=input_hash,
                now=now,
                decision_state="under_review",
                approval_status="pending",
            )
        if normalized_supersedes_id:
            self._write_audit(
                "approval.replacement.created",
                "approval",
                "succeeded",
                actor=context.actor,
                provenance_id=provenance_id,
                object_refs=[
                    {"type": "Approval", "id": approval_uid},
                    {"type": "Approval", "id": normalized_supersedes_id},
                ],
                after_summary={
                    "action_id": action_id,
                    "supersedes_approval_id": normalized_supersedes_id,
                    "approval_policy_rule_id": props.get("approval_policy_rule_id"),
                    "approval_policy_reason": props.get("approval_policy_reason"),
                    "policy_decision_id": props.get("policy_decision_id"),
                    "policy_matched_rule": props.get("policy_matched_rule"),
                },
            )
        approval = _flatten_object(row)
        _refresh_temporal_read_models_after_command()
        return approval

    def _prepare_action_payload(self, action_id: str, payload: dict[str, Any]) -> None:
        if action_id not in WATCH_TRIGGER_TARGET_ACTION_IDS:
            return
        trigger_id = payload.get("trigger_id") or payload.get("id")
        trigger = _watch_trigger_context(self.objects, trigger_id)
        if not trigger:
            raise OntologyCommandNotFound("WatchTrigger", str(trigger_id or ""))
        payload["trigger_id"] = _normalize_watch_trigger_uid(trigger_id)
        if action_id == "replace_watch_trigger":
            for source_field, payload_field in (
                ("condition", "old_condition"),
                ("trigger_type", "old_trigger_type"),
                ("ticker", "old_ticker"),
                ("expires_at", "old_expires_at"),
                ("definition", "old_definition"),
            ):
                value = trigger.get(source_field)
                if value not in (None, ""):
                    payload[payload_field] = value
            return
        _merge_watch_trigger_context(payload, trigger)

    def _evaluate_policy_gate(
        self,
        action_id: str,
        payload: dict[str, Any],
        context: OntologyCommandContext,
    ) -> dict[str, Any] | None:
        from portfolio.policy_gate import PolicyGateBlockedError, ensure_policy_gate_for_action, is_financial_action

        if not is_financial_action(action_id, payload):
            return None

        try:
            gated_payload, gate = ensure_policy_gate_for_action(
                action_id,
                payload,
                context={
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "request_mode": context.request_mode,
                    "actor_id": context.actor.actor_id,
                    "actor_type": context.actor.actor_type,
                    "actor_roles": list(context.actor.roles),
                },
                object_service=self.objects,
                raise_on_blocked=action_id not in COURSE_OF_ACTION_CREATE_ACTION_IDS,
            )
        except PolicyGateBlockedError as exc:
            raise OntologyCommandValidationError(str(exc)) from exc
        payload.clear()
        payload.update(gated_payload)
        return gate

    def _persist_recommendation_without_approval(
        self,
        payload: dict[str, Any],
        context: OntologyCommandContext,
        *,
        reason: str | None,
    ) -> dict[str, Any]:
        action_id = "create_recommendation"
        now = _now()
        input_hash = _stable_hash({"action_id": action_id, "payload": payload})
        run_key = f"{action_id}:{now}"
        run_uid = action_run_id(run_key)
        provenance_id = _provenance_id("recommendation.persisted", run_uid, input_hash)
        actor = actor_to_dict(context.actor)
        run_props = {
            "action_id": action_id,
            "action_schema_name": action_id,
            "action_schema_version": 1,
            "actor_type": context.actor.actor_type,
            "actor_id": context.actor.actor_id,
            "source_type": context.source_type,
            "source_id": context.source_id,
            "approval_id": None,
            "input_hash": input_hash,
            "started_at": now,
            "provenance_event_id": provenance_id,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        }
        self.objects.write_object(
            "ActionRun",
            run_key,
            {
                **run_props,
                "status": "running",
                "execution_state": "running",
                "completed_at": None,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        version_refs = self._write_action_targets(
            action_id,
            payload,
            context,
            provenance_id=provenance_id,
            input_hash=input_hash,
            approval_object_id=None,
            action_run_id=run_uid,
        )
        self.objects.write_object(
            "ActionRun",
            run_key,
            {
                **run_props,
                "status": "succeeded",
                "execution_state": "succeeded",
                "completed_at": _now(),
                "output_hash": _stable_hash(version_refs),
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        record = _dict(payload.get("record") or payload)
        rec_key = str(record.get("recommendation_id") or record.get("idempotency_key") or _stable_hash(record))
        rec_uid = recommendation_id(rec_key)
        self._write_audit(
            "recommendation.persisted",
            "recommendation",
            "succeeded",
            actor=context.actor,
            provenance_id=provenance_id,
            object_refs=[{"type": "Recommendation", "id": rec_uid}, {"type": "ActionRun", "id": run_uid}],
            after_summary={
                "action": record.get("action"),
                "recommendation_status": record.get("recommendation_status"),
                "approval_required": False,
            },
        )
        try:
            row = self.objects.get_object(rec_uid)
        except Exception:
            row = None
        out = _flatten_object(row) if row else {"id": rec_uid, "object_uid": rec_uid, "payload": record}
        out.update(
            {
                "entity_type": "recommendation",
                "action_id": action_id,
                "approval_required": False,
                "approval_id": None,
                "application_status": "applied",
                "action_run_id": run_uid,
                "proposed_change": payload,
                "reason": reason,
            }
        )
        _refresh_temporal_read_models_after_command()
        return out

    def resolve_approval(
        self,
        approval_uid: str,
        status: str,
        note: str | None,
        context: OntologyCommandContext,
        requirement_id: str | None = None,
    ) -> dict[str, Any]:
        approval = self.get_approval(approval_uid, actor=context.actor)
        current_status = str(approval.get("status") or "pending").lower()
        if current_status != "pending":
            raise OntologyCommandConflict(f"Approval {approval['id']} is already {current_status}")
        status = str(status or "").strip().lower()
        if status not in {"approved", "rejected"}:
            raise OntologyCommandValidationError("Approval status must be approved or rejected.")
        resolve_decision = POLICY.check_action(
            context.actor,
            "approval.resolve",
            _approval_policy_context(
                approval=approval,
                approval_id=approval_uid,
                command_context=context,
                status=status,
            ),
        )
        self._require_policy_allowed(
            resolve_decision,
            actor=context.actor,
            provenance_id=_provenance_id("policy", "approval.resolve", context.actor.actor_id, approval_uid, status),
            object_refs=[{"type": "Approval", "id": str(approval.get("id") or approval_uid)}],
        )
        if status == "approved" and not str(note or "").strip():
            raise OntologyCommandValidationError("Approval note is required.")

        requirements = normalize_approval_requirements(approval.get("approval_requirements"))
        decisions = normalize_approval_decisions(approval.get("approval_decisions"))
        progress = approval_requirement_progress(requirements, decisions)
        application_status = str(approval.get("application_status") or "pending").strip().lower()
        retrying_completed_approval = status == "approved" and progress["completed"] and application_status == "failed"
        selected_requirement: dict[str, Any] | None = None
        updated_decisions = decisions
        if not retrying_completed_approval:
            selected_requirement, denial = select_approval_requirement(
                requirements,
                decisions,
                actor_id=context.actor.actor_id,
                actor_roles=context.actor.roles,
                requested_by_actor_id=str(approval.get("requested_by_actor_id") or "") or None,
                requirement_id=str(requirement_id or "").strip() or None,
            )
            if selected_requirement is None:
                if status == "rejected" and progress["completed"] and requirements:
                    selected_requirement = requirements[0]
                else:
                    raise OntologyCommandValidationError(denial or "No remaining approval requirements.")
            updated_decisions = [
                *decisions,
                {
                    "requirement_id": str(selected_requirement.get("id") or ""),
                    "actor_id": context.actor.actor_id,
                    "actor_type": context.actor.actor_type,
                    "actor_roles": list(context.actor.roles),
                    "decision": status,
                    "note": note,
                    "decided_at": _now(),
                },
            ]
        if status == "approved":
            if str(approval.get("policy_gate_decision") or "").strip().lower() == "blocked":
                raise OntologyCommandValidationError("Blocked policy gate results cannot be approved.")
            from ontology.runtime_read_service import runtime_object_service

            with runtime_object_service(self.objects):
                _ensure_fresh_base_state(approval)

        input_hash = str(approval.get("action_input_hash") or _stable_hash(approval))
        now = _now()
        provenance_id = _provenance_id("approval_resolution", approval["id"], status, input_hash)
        attempts = _int(approval.get("application_attempts"))
        if status == "rejected":
            props = {
                **{k: v for k, v in approval.items() if not k.startswith("_")},
                "approval_requirements": requirements,
                "approval_decisions": updated_decisions,
                "status": "rejected",
                "resolution_state": "rejected",
                "application_state": "not_applicable",
                "application_status": "not_applicable",
                "application_completed_at": now,
                "application_error": None,
                "resolved_by_actor_id": context.actor.actor_id,
                "resolved_at": now,
                "resolved_note": note,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            }
            props.pop("id", None)
            props.pop("object_uid", None)
            row = self.objects.write_object(
                "Approval",
                approval["id"],
                props,
                now,
                actor=actor_to_dict(context.actor),
                provenance=provenance_id,
                input_hash=input_hash,
            )
            resolved = _flatten_object(row)
            self._write_audit(
                "approval.rejected",
                "approval",
                "succeeded",
                actor=context.actor,
                provenance_id=provenance_id,
                object_refs=[{"type": "Approval", "id": approval["id"]}],
                after_summary={
                    "requirement_id": selected_requirement.get("id") if selected_requirement else None,
                    "note": note,
                    "approval_policy_rule_id": approval.get("approval_policy_rule_id"),
                    "approval_policy_reason": approval.get("approval_policy_reason"),
                },
            )
            _refresh_temporal_read_models_after_command()
            return resolved

        updated_progress = approval_requirement_progress(requirements, updated_decisions)
        if not updated_progress["completed"]:
            props = {
                **{k: v for k, v in approval.items() if not k.startswith("_")},
                "approval_requirements": requirements,
                "approval_decisions": updated_decisions,
                "status": "pending",
                "resolution_state": "pending",
                "application_state": "pending",
                "application_status": "pending",
                "application_error": None,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            }
            props.pop("id", None)
            props.pop("object_uid", None)
            row = self.objects.write_object(
                "Approval",
                approval["id"],
                props,
                now,
                actor=actor_to_dict(context.actor),
                provenance=provenance_id,
                input_hash=input_hash,
            )
            resolved = _flatten_object(row)
            self._write_audit(
                "approval.decision.recorded",
                "approval",
                "succeeded",
                actor=context.actor,
                provenance_id=provenance_id,
                object_refs=[{"type": "Approval", "id": approval["id"]}],
                after_summary={
                    "decision": "approved",
                    "requirement_id": selected_requirement.get("id") if selected_requirement else None,
                    "remaining_count": updated_progress["remaining_count"],
                    "approval_policy_rule_id": approval.get("approval_policy_rule_id"),
                    "approval_policy_reason": approval.get("approval_policy_reason"),
                },
            )
            _refresh_temporal_read_models_after_command()
            return resolved

        applying_props = {
            **{k: v for k, v in approval.items() if not k.startswith("_")},
            "approval_requirements": requirements,
            "approval_decisions": updated_decisions,
            "status": "pending",
            "resolution_state": "pending",
            "application_state": "applying",
            "application_status": "applying",
            "application_attempts": attempts + 1,
            "application_started_at": now,
            "application_completed_at": None,
            "application_error": None,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        }
        applying_props.pop("id", None)
        applying_props.pop("object_uid", None)
        row = self.objects.write_object(
            "Approval",
            approval["id"],
            applying_props,
            now,
            actor=actor_to_dict(context.actor),
            provenance=provenance_id,
            input_hash=input_hash,
        )
        resolved = _flatten_object(row)
        if selected_requirement is not None:
            self._write_audit(
                "approval.decision.recorded",
                "approval",
                "succeeded",
                actor=context.actor,
                provenance_id=provenance_id,
                object_refs=[{"type": "Approval", "id": approval["id"]}],
                after_summary={
                    "decision": "approved",
                    "requirement_id": selected_requirement.get("id"),
                    "remaining_count": 0,
                    "approval_policy_rule_id": approval.get("approval_policy_rule_id"),
                    "approval_policy_reason": approval.get("approval_policy_reason"),
                },
            )
        try:
            applied = self._apply_approval(
                resolved,
                context,
                provenance_id=provenance_id,
                input_hash=input_hash,
                note=note,
            )
        except OntologyCommandError as exc:
            error = exc.message
        except Exception as exc:
            error = str(exc).strip() or exc.__class__.__name__
        else:
            _refresh_temporal_read_models_after_command()
            return applied
        failed_now = _now()
        failed_action_id = str(resolved.get("action_id") or "").strip()
        if failed_action_id:
            failed_started_at = resolved.get("application_started_at") or failed_now
            self.objects.write_object(
                "ActionRun",
                f"{failed_action_id}:{failed_started_at}",
                {
                    "action_id": failed_action_id,
                    "action_schema_name": failed_action_id,
                    "action_schema_version": int(resolved.get("action_schema_version") or 1),
                    "actor_type": context.actor.actor_type,
                    "actor_id": context.actor.actor_id,
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "approval_id": approval["id"],
                    "input_hash": input_hash,
                    "status": "failed",
                    "execution_state": "failed",
                    "error": error[:1000],
                    "started_at": failed_started_at,
                    "completed_at": failed_now,
                    "provenance_event_id": provenance_id,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                failed_now,
                actor=actor_to_dict(context.actor),
                provenance=provenance_id,
                input_hash=input_hash,
            )
        failed_props = {
            **{k: v for k, v in resolved.items() if not k.startswith("_")},
            "status": "pending",
            "resolution_state": "pending",
            "application_state": "failed",
            "application_status": "failed",
            "application_completed_at": failed_now,
            "application_error": error[:1000],
            "resolved_by_actor_id": None,
            "resolved_at": None,
            "resolved_note": None,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        }
        failed_props.pop("id", None)
        failed_props.pop("object_uid", None)
        self.objects.write_object(
            "Approval",
            approval["id"],
            failed_props,
            failed_now,
            actor=actor_to_dict(context.actor),
            provenance=provenance_id,
            input_hash=input_hash,
        )
        self._write_audit(
            "approval.apply.failed",
            "approval",
            "failed",
            actor=context.actor,
            provenance_id=provenance_id,
            object_refs=[{"type": "Approval", "id": approval["id"]}],
            after_summary={"application_status": "failed", "error": error[:1000]},
        )
        _refresh_temporal_read_models_after_command()
        raise OntologyCommandConflict(f"Approval {approval['id']} application failed: {error}") from None

    def _apply_approval(
        self,
        approval: dict[str, Any],
        context: OntologyCommandContext,
        *,
        provenance_id: str,
        input_hash: str,
        note: str | None,
    ) -> dict[str, Any]:
        action_id = _non_blank(approval.get("action_id"), "action_id")
        payload = _dict(approval.get("proposed_change"))
        now = _now()
        run_key = f"{action_id}:{now}"
        run_uid = action_run_id(run_key)
        run_props = {
            "action_id": action_id,
            "action_schema_name": action_id,
            "action_schema_version": 1,
            "actor_type": context.actor.actor_type,
            "actor_id": context.actor.actor_id,
            "source_type": context.source_type,
            "source_id": context.source_id,
            "approval_id": approval["id"],
            "input_hash": input_hash,
            "started_at": now,
            "provenance_event_id": provenance_id,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        }
        run = self.objects.write_object(
            "ActionRun",
            run_key,
            {
                **run_props,
                "status": "running",
                "execution_state": "running",
                "completed_at": None,
            },
            now,
            actor=actor_to_dict(context.actor),
            provenance=provenance_id,
            input_hash=input_hash,
        )
        version_refs = self._write_action_targets(
            action_id,
            payload,
            context,
            provenance_id=provenance_id,
            input_hash=input_hash,
            approval_object_id=approval["id"],
            action_run_id=run_uid,
        )
        run = self.objects.write_object(
            "ActionRun",
            run_key,
            {
                **run_props,
                "status": "succeeded",
                "execution_state": "succeeded",
                "completed_at": _now(),
            },
            now,
            actor=actor_to_dict(context.actor),
            provenance=provenance_id,
            input_hash=input_hash,
        )
        decision_key = f"{approval['id']}:{run_uid}"
        decision = self.objects.write_object(
            "ExecutedDecisionRecord",
            decision_key,
            {
                "decision_record_id": decision_key,
                "approval_id": approval["id"],
                "action_run_id": run_uid,
                "action_id": action_id,
                "target_object_uid": approval.get("target_object_uid"),
                "target_object_type": approval.get("target_object_type"),
                "applied_object_versions": version_refs,
                "applied_at": now,
                "status": "recorded",
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor_to_dict(context.actor),
            provenance=provenance_id,
            input_hash=input_hash,
        )
        self.objects.write_relation(
            _flatten_object(decision)["id"],
            approval["id"],
            "executed_decision_applies_approval",
            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
            now,
            actor=actor_to_dict(context.actor),
            provenance=provenance_id,
            input_hash=input_hash,
        )
        self.objects.write_relation(
            _flatten_object(decision)["id"],
            _flatten_object(run)["id"],
            "executed_decision_records_action_run",
            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
            now,
            actor=actor_to_dict(context.actor),
            provenance=provenance_id,
            input_hash=input_hash,
        )
        applied_props = {
            **{k: v for k, v in approval.items() if not k.startswith("_") and k != "id"},
            "status": "approved",
            "resolution_state": "approved",
            "application_state": "applied",
            "application_status": "applied",
            "application_completed_at": now,
            "application_error": None,
            "resolved_by_actor_id": context.actor.actor_id,
            "resolved_at": now,
            "resolved_note": note,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        }
        applied_props.pop("object_uid", None)
        applied = self.objects.write_object(
            "Approval",
            approval["id"],
            applied_props,
            now,
            actor=actor_to_dict(context.actor),
            provenance=provenance_id,
            input_hash=input_hash,
        )
        self._write_audit(
            "approval.applied",
            "approval",
            "succeeded",
            actor=context.actor,
            provenance_id=provenance_id,
            object_refs=[{"type": "Approval", "id": approval["id"]}, {"type": "ActionRun", "id": run_uid}],
            after_summary={
                "mutated_object_versions": version_refs,
                "approval_policy_rule_id": approval.get("approval_policy_rule_id"),
                "approval_policy_reason": approval.get("approval_policy_reason"),
            },
        )
        return _flatten_object(applied)

    def _expire_absent_replacement_objects(
        self,
        object_type: str,
        submitted_tickers: set[str],
        now: str,
    ) -> list[str]:
        expired_tickers: list[str] = []
        for existing in self.objects.query_objects(object_type, limit=1000):
            flat = _flatten_object(existing)
            ticker = str(flat.get("ticker") or "").strip().upper()
            object_uid = str(flat.get("id") or existing.get("object_uid") or "").strip()
            if not ticker or not object_uid or ticker in submitted_tickers:
                continue
            self._expire_current_object_and_relations(object_uid, now)
            expired_tickers.append(ticker)
        return sorted(set(expired_tickers))

    def _expire_current_object_and_relations(self, object_uid: str, now: str) -> None:
        self.objects.expire_object(object_uid, tx_to=now)
        for relation in self.objects.query_relations(source_object_uid=object_uid, limit=1000):
            relation_uid = str(relation.get("relation_uid") or "").strip()
            if relation_uid:
                self.objects.expire_relation(relation_uid, tx_to=now)
        for relation in self.objects.query_relations(target_object_uid=object_uid, limit=1000):
            relation_uid = str(relation.get("relation_uid") or "").strip()
            if relation_uid:
                self.objects.expire_relation(relation_uid, tx_to=now)

    def _resolve_removed_position_pressure_alerts(
        self,
        removed_tickers: list[str],
        *,
        now: str,
        actor: Mapping[str, Any],
        provenance_id: str,
        input_hash: str,
    ) -> list[dict[str, Any]]:
        if not removed_tickers:
            return []
        removed = {ticker.upper() for ticker in removed_tickers}
        refs: list[dict[str, Any]] = []
        for alert in self.objects.query_objects("OptimizationAlert", filters={"status": "open"}, limit=1000):
            flat = _flatten_object(alert)
            ticker = str(flat.get("ticker") or "").strip().upper()
            alert_type = str(flat.get("alert_type") or "").strip().lower()
            if ticker not in removed or alert_type != "thesis_pressure":
                continue
            props = dict(flat)
            business_key = str(props.get("alert_id") or props.get("id") or props.get("object_uid") or "").strip()
            if not business_key:
                continue
            props["status"] = "resolved"
            props["resolved_at"] = now
            props["resolved_reason"] = "position_removed"
            props["updated_at"] = now
            resolved = self.objects.write_object(
                "OptimizationAlert",
                business_key,
                props,
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(resolved))
        return refs

    def _create_portfolio_news_digest(
        self,
        payload: Mapping[str, Any],
        context: OntologyCommandContext,
        *,
        actor: Mapping[str, Any],
        provenance_id: str,
        input_hash: str,
        now: str,
        approval_object_id: str | None,
        action_run_id: str | None,
    ) -> list[dict[str, Any]]:
        from api.routers.portfolio_news import _index_digest_best_effort
        from ontology.action_registry import CreatePortfolioNewsDigestInput
        from ontology.domain_write_service import domain_write_scope
        from portfolio.news_digests import save_digest

        typed = CreatePortfolioNewsDigestInput(**dict(payload))
        with domain_write_scope(
            action_id="create_portfolio_news_digest",
            actor_type=context.actor.actor_type,
            approval_id=approval_object_id,
            action_run_id=action_run_id,
            source_type=context.source_type,
            source_id=context.source_id,
        ):
            detail = save_digest(typed.content, filename=typed.filename)

        digest_id = _non_blank(detail.get("id"), "digest_id")
        row = self.objects.write_object(
            "DocumentArtifact",
            _strip_uid_prefix(digest_id, "document_artifact"),
            {
                "document_type": "news_digest",
                "document_id": digest_id,
                "title": detail.get("title") or detail.get("filename") or digest_id,
                "content_hash": detail.get("content_hash") or hashlib.sha256(typed.content.encode("utf-8")).hexdigest(),
                "artifact_uri": f"news_digests/{digest_id}.md",
                "status": "active",
                "source_type": context.source_type,
                "source_id": context.source_id,
                "created_at": detail.get("uploaded_at") or now,
                "updated_at": detail.get("updated_at") or now,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        try:
            _index_digest_best_effort(detail)
        except Exception:
            logger.debug("Failed to index news digest %s", digest_id, exc_info=True)
        return [_version_ref_from_row(row)]

    def _delete_portfolio_news_digest(
        self,
        payload: Mapping[str, Any],
        context: OntologyCommandContext,
        *,
        actor: Mapping[str, Any],
        provenance_id: str,
        input_hash: str,
        now: str,
        approval_object_id: str | None,
        action_run_id: str | None,
    ) -> list[dict[str, Any]]:
        from api.routers.portfolio_news import _delete_digest_index_best_effort
        from ontology.action_registry import DeletePortfolioNewsDigestInput
        from ontology.domain_write_service import domain_write_scope
        from portfolio.news_digests import delete_digest

        typed = DeletePortfolioNewsDigestInput(**dict(payload))
        with domain_write_scope(
            action_id="delete_portfolio_news_digest",
            actor_type=context.actor.actor_type,
            approval_id=approval_object_id,
            action_run_id=action_run_id,
            source_type=context.source_type,
            source_id=context.source_id,
        ):
            deleted = delete_digest(typed.digest_id)
        if not deleted:
            raise OntologyCommandNotFound("News digest", typed.digest_id)

        row = self.objects.write_object(
            "DocumentArtifact",
            _strip_uid_prefix(typed.digest_id, "document_artifact"),
            {
                "document_type": "news_digest",
                "document_id": typed.digest_id,
                "title": typed.digest_id,
                "content_hash": None,
                "artifact_uri": f"news_digests/{typed.digest_id}.md",
                "status": "deleted",
                "source_type": context.source_type,
                "source_id": context.source_id,
                "created_at": now,
                "updated_at": now,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        try:
            _delete_digest_index_best_effort(typed.digest_id)
        except Exception:
            logger.debug("Failed to delete news digest index %s", typed.digest_id, exc_info=True)
        return [_version_ref_from_row(row)]

    def _write_action_targets(
        self,
        action_id: str,
        payload: Mapping[str, Any],
        context: OntologyCommandContext,
        *,
        provenance_id: str,
        input_hash: str,
        approval_object_id: str | None = None,
        action_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        now = _now()
        actor = actor_to_dict(context.actor)
        refs: list[dict[str, Any]] = []
        if action_id == "update_portfolio_positions":
            positions = _list(payload.get("positions"))
            submitted_tickers = {_non_blank(_dict(position).get("ticker"), "ticker").upper() for position in positions}
            removed_tickers = self._expire_absent_replacement_objects("Position", submitted_tickers, now)
            self._ensure_default_account_portfolio(context, provenance_id=provenance_id, input_hash=input_hash)
            for position in positions:
                row = dict(position)
                ticker = _non_blank(row.get("ticker"), "ticker").upper()
                instr_key = str(row.get("instrument_id") or ticker)
                instr_uid = instrument_id(instr_key)
                self.objects.write_object(
                    "Instrument",
                    instr_key,
                    {
                        "instrument_id": instr_key,
                        "ticker": ticker,
                        "asset_class": row.get("asset") or "security",
                        "instrument_type": row.get("instrument_type") or "security",
                        "price_symbol": row.get("price_symbol") or ticker,
                        "fx_base_currency": row.get("fx_base_currency"),
                        "fx_quote_currency": row.get("fx_quote_currency"),
                        "status": "active",
                        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    },
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
                row.setdefault("asset", "equity")
                row.setdefault("direction", "long")
                row["account_id"] = str(row.get("account_id") or "default")
                row["portfolio_id"] = str(row.get("portfolio_id") or "default")
                row["instrument_id"] = instr_uid
                row["ontology_run_id"] = OPERATIONAL_ONTOLOGY_RUN_ID
                pos = self.objects.write_object(
                    "Position",
                    ticker,
                    row,
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
                pos_uid = _flatten_object(pos)["id"]
                self.objects.write_relation(
                    portfolio_id(row["portfolio_id"]),
                    pos_uid,
                    "portfolio_holds_position",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
                self.objects.write_relation(
                    pos_uid,
                    instr_uid,
                    "position_references_instrument",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
                refs.append(_version_ref_from_row(pos))
            refs.extend(
                self._resolve_removed_position_pressure_alerts(
                    removed_tickers,
                    now=now,
                    actor=actor,
                    provenance_id=provenance_id,
                    input_hash=input_hash,
                )
            )
            return refs
        if action_id == "update_hedge_positions":
            positions = _list(payload.get("positions"))
            submitted_tickers = {_non_blank(_dict(position).get("ticker"), "ticker").upper() for position in positions}
            self._expire_absent_replacement_objects("HedgePosition", submitted_tickers, now)
            for position in positions:
                row = dict(position)
                ticker = _non_blank(row.get("ticker"), "ticker").upper()
                row.setdefault("asset", "equity")
                row.setdefault("direction", "short")
                row["ontology_run_id"] = OPERATIONAL_ONTOLOGY_RUN_ID
                hedge = self.objects.write_object(
                    "HedgePosition",
                    ticker,
                    row,
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
                refs.append(_version_ref_from_row(hedge))
            return refs
        if action_id == "change_thesis_status":
            ticker = _non_blank(payload.get("ticker"), "ticker").upper()
            thesis_uid = thesis_id(ticker)
            row = self.objects.write_object(
                "Thesis",
                thesis_uid,
                {
                    "ticker": ticker,
                    "status": str(payload.get("new_status") or payload.get("status") or "under_review"),
                    "created_at": now,
                    "updated_at": now,
                    "instrument_id": instrument_id(ticker),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id == "create_portfolio_news_digest":
            refs.extend(
                self._create_portfolio_news_digest(
                    payload,
                    context,
                    actor=actor,
                    provenance_id=provenance_id,
                    input_hash=input_hash,
                    now=now,
                    approval_object_id=approval_object_id,
                    action_run_id=action_run_id,
                )
            )
            return refs
        if action_id == "delete_portfolio_news_digest":
            refs.extend(
                self._delete_portfolio_news_digest(
                    payload,
                    context,
                    actor=actor,
                    provenance_id=provenance_id,
                    input_hash=input_hash,
                    now=now,
                    approval_object_id=approval_object_id,
                    action_run_id=action_run_id,
                )
            )
            return refs
        if action_id in {"create_catalyst", "update_catalyst_status"}:
            catalyst_key = _canonical_object_key(payload.get("catalyst_id"))
            existing = _object_context_by_uid(self.objects, catalyst_key)
            ticker = _optional_ticker(payload) or _optional_ticker(existing) or "UNKNOWN"
            description = _non_blank(
                payload.get("description")
                or existing.get("description")
                or payload.get("evidence")
                or "Catalyst update",
                "description",
            )
            row = self.objects.write_object(
                "Catalyst",
                catalyst_key or f"{ticker}:{description}",
                {
                    "ticker": ticker,
                    "name": str(payload.get("name") or description[:120]),
                    "description": description,
                    "source": context.source_type,
                    "category": payload.get("category")
                    if payload.get("category") is not None
                    else existing.get("category"),
                    "target_date": payload.get("target_date")
                    if payload.get("target_date") is not None
                    else existing.get("target_date"),
                    "status": payload.get("status") or existing.get("status") or "pending",
                    "evidence": payload.get("evidence")
                    if payload.get("evidence") is not None
                    else existing.get("evidence"),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id in {"create_kill_condition", "update_kill_condition_status"}:
            kill_condition_key = _canonical_object_key(payload.get("kill_condition_id"))
            existing = _object_context_by_uid(self.objects, kill_condition_key)
            ticker = _optional_ticker(payload) or _optional_ticker(existing) or "UNKNOWN"
            condition = _non_blank(
                payload.get("condition")
                or existing.get("condition")
                or payload.get("status")
                or "Kill condition update",
                "condition",
            )
            row = self.objects.write_object(
                "KillCondition",
                kill_condition_key or f"{ticker}:{condition}",
                {
                    "ticker": ticker,
                    "condition": condition,
                    "metric": payload.get("metric") if payload.get("metric") is not None else existing.get("metric"),
                    "threshold": payload.get("threshold")
                    if payload.get("threshold") is not None
                    else existing.get("threshold"),
                    "status": payload.get("status") or "active",
                    "triggered_at": now
                    if str(payload.get("status") or existing.get("status") or "").strip().lower() == "triggered"
                    else existing.get("triggered_at"),
                    "created_at": existing.get("created_at") or now,
                    "updated_at": now,
                    "created_by": context.source_type,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id == "create_monitor_hit":
            fingerprint = str(payload.get("fingerprint") or _stable_hash(payload)).strip()
            hit_uid = monitor_hit_id(fingerprint)
            now_detected = str(payload.get("detected_at") or now)
            row = self.objects.write_object(
                "MonitorHit",
                hit_uid,
                {
                    "hit_id": hit_uid,
                    "ticker": _optional_ticker(payload) or "UNKNOWN",
                    "entity_type": str(payload.get("entity_type") or "catalyst"),
                    "entity_id": str(payload.get("entity_id") or ""),
                    "entity_label": payload.get("entity_label"),
                    "hit_type": str(payload.get("hit_type") or "needs_review"),
                    "severity": payload.get("severity"),
                    "status": str(payload.get("status") or "open"),
                    "confidence": payload.get("confidence"),
                    "evidence": payload.get("evidence"),
                    "source_ids": _list(payload.get("source_ids")),
                    "result": _dict(payload.get("result")),
                    "detected_at": now_detected,
                    "approval_id": payload.get("approval_id"),
                    "action_item_id": payload.get("action_item_id"),
                    "fingerprint": fingerprint,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id == "update_monitor_hit_status":
            hit_uid = _normalize_monitor_hit_uid(payload.get("hit_id"))
            existing = _object_context_by_uid(self.objects, hit_uid)
            if not existing:
                raise OntologyCommandNotFound("MonitorHit", str(payload.get("hit_id")))
            row = self.objects.write_object(
                "MonitorHit",
                hit_uid,
                {
                    "hit_id": hit_uid,
                    "ticker": existing.get("ticker") or _optional_ticker(payload) or "UNKNOWN",
                    "entity_type": existing.get("entity_type") or str(payload.get("entity_type") or "catalyst"),
                    "entity_id": existing.get("entity_id") or str(payload.get("entity_id") or ""),
                    "entity_label": existing.get("entity_label"),
                    "hit_type": existing.get("hit_type") or str(payload.get("hit_type") or "needs_review"),
                    "severity": existing.get("severity"),
                    "status": str(payload.get("status") or existing.get("status") or "open"),
                    "confidence": existing.get("confidence"),
                    "evidence": existing.get("evidence"),
                    "source_ids": _list(existing.get("source_ids")),
                    "result": _dict(existing.get("result")),
                    "detected_at": existing.get("detected_at") or now,
                    "approval_id": existing.get("approval_id"),
                    "action_item_id": existing.get("action_item_id"),
                    "fingerprint": existing.get("fingerprint"),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id in {"create_thesis_claim", "update_thesis_claim"}:
            claim_object_key = _canonical_object_key(
                payload.get("claim_id") or payload.get("thesis_claim_id") or payload.get("id")
            )
            existing = _object_context_by_uid(self.objects, claim_object_key)
            ticker = _optional_ticker(payload) or _optional_ticker(existing) or "UNKNOWN"
            claim = _non_blank(
                payload.get("claim") or existing.get("claim") or payload.get("status") or "Thesis claim update", "claim"
            )
            row = self.objects.write_object(
                "ThesisClaim",
                claim_object_key or f"{ticker}:{claim}",
                {
                    "ticker": ticker,
                    "claim": claim,
                    "expected_evidence": payload.get("expected_evidence"),
                    "disconfirming_evidence": payload.get("disconfirming_evidence"),
                    "source_requirements": _list(payload.get("source_requirements")),
                    "cadence": payload.get("cadence"),
                    "confidence": payload.get("confidence"),
                    "status": payload.get("status") or "active",
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "created_at": now,
                    "updated_at": now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            claim_ref = _version_ref_from_row(row)
            refs.append(claim_ref)
            claim_uid = str(claim_ref.get("object_uid") or "")
            if claim_uid:
                from ontology.evidence_ledger import write_claim_evidence_graph

                write_claim_evidence_graph(
                    self.objects,
                    claim_uid=claim_uid,
                    claim_key=claim_object_key or f"{ticker}:{claim}",
                    expected_evidence=payload.get("expected_evidence"),
                    disconfirming_evidence=payload.get("disconfirming_evidence"),
                    valid_from=now,
                    actor=actor,
                    provenance_id=provenance_id,
                    approval_id=approval_object_id,
                    input_hash=input_hash,
                )
            return refs
        if action_id == "create_research_note":
            content = str(
                payload.get("content")
                or payload.get("note")
                or payload.get("body")
                or payload.get("text")
                or payload.get("summary")
                or ""
            )
            note_ticker = _optional_ticker(payload)
            title = str(payload.get("title") or payload.get("name") or payload.get("headline") or "").strip()
            if not title:
                title = f"Research note {note_ticker or _stable_hash(payload)[:12]}"
            raw_document_id = (
                payload.get("document_id")
                or payload.get("note_id")
                or payload.get("research_note_id")
                or payload.get("id")
                or f"{note_ticker or 'general'}:{title}:{_stable_hash(content or payload)}"
            )
            document_id = _strip_uid_prefix(raw_document_id, "document_artifact")
            row = self.objects.write_object(
                "DocumentArtifact",
                document_id,
                {
                    "document_type": "research_note",
                    "document_id": document_id,
                    "title": title,
                    "ticker": note_ticker,
                    "content_hash": _stable_hash(content or payload),
                    "artifact_uri": payload.get("artifact_uri") or payload.get("source_path"),
                    "status": str(payload.get("status") or "active"),
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "created_at": now,
                    "updated_at": now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id == "create_analyst_feedback":
            target_uid = _non_blank(payload.get("target_object_uid"), "target_object_uid")
            target_type = _non_blank(payload.get("target_object_type"), "target_object_type")
            decision = _non_blank(payload.get("decision"), "decision")
            feedback_key = payload.get("feedback_id") or f"{target_uid}:{decision}:{_stable_hash(payload)}"
            row = self.objects.write_object(
                "AnalystFeedback",
                str(feedback_key),
                {
                    "feedback_id": str(feedback_key),
                    "target_object_uid": target_uid,
                    "target_object_type": target_type,
                    "decision": decision,
                    "note": payload.get("note") or payload.get("reason"),
                    "correction": _dict(payload.get("correction")),
                    "confidence": payload.get("confidence"),
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "approval_id": approval_object_id,
                    "created_by": context.actor.actor_id,
                    "created_at": now,
                    "status": "submitted",
                    "metadata": {"action_run_id": action_run_id},
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            self.objects.write_relation(
                "analyst_feedback_targets_object",
                str(row.get("object_uid")),
                target_uid,
                {
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    "target_object_uid": target_uid,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id == "finalize_decision_outcome":
            from ontology.decision_outcome_service import finalize_decision_outcome as finalize_outcome

            outcome_uid = _non_blank(payload.get("decision_outcome_id"), "decision_outcome_id")
            row = finalize_outcome(
                outcome_uid,
                decision=_non_blank(payload.get("decision"), "decision"),
                note=payload.get("note"),
                corrected_postmortem=payload.get("corrected_postmortem"),
                lessons_learned=payload.get("lessons_learned"),
                actor_id=context.actor.actor_id,
                objects=self.objects,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id == "save_evaluation":
            ticker = _non_blank(payload.get("ticker"), "ticker").upper()
            evaluated_at = str(payload.get("evaluated_at") or now)
            row = self.objects.write_object(
                "Evaluation",
                f"{ticker}:{evaluated_at}",
                {
                    "ticker": ticker,
                    "evaluated_at": evaluated_at,
                    "thesis_status": str(payload.get("thesis_status") or "under_review"),
                    "technical_read": str(payload.get("technical_read") or "unknown"),
                    "fundamental_read": str(payload.get("fundamental_read") or "unknown"),
                    "action": str(payload.get("action") or "review"),
                    "confidence": str(payload.get("confidence") or "unknown"),
                    "earnings_note": payload.get("earnings_note"),
                    "risk_flag": payload.get("risk_flag"),
                    "key_developments": _list(payload.get("key_developments")),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id == "save_thesis_content":
            refs.extend(
                self._write_thesis_content(
                    payload,
                    context,
                    actor=actor,
                    provenance_id=provenance_id,
                    input_hash=input_hash,
                    now=now,
                )
            )
            return refs
        if action_id == "save_management_quality_content":
            refs.extend(
                self._write_management_quality_content(
                    payload,
                    context,
                    actor=actor,
                    provenance_id=provenance_id,
                    input_hash=input_hash,
                    now=now,
                )
            )
            return refs
        if action_id == "create_action_item":
            description = _non_blank(payload.get("description"), "description")
            row = self.objects.write_object(
                "ActionItem",
                description,
                {
                    "description": description,
                    "action_type": str(payload.get("action_type") or "review"),
                    "ticker": _optional_ticker(payload),
                    "urgency": str(payload.get("urgency") or "normal"),
                    "status": "open",
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "created_at": now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id in {"complete_action_item", "dismiss_action_item"}:
            item_id = payload.get("item_id") or payload.get("id")
            item_key = _canonical_object_key(item_id, prefix="action_item")
            description = str(payload.get("description") or f"Action item {item_key}").strip()
            row = self.objects.write_object(
                "ActionItem",
                item_key or description,
                {
                    "description": description,
                    "action_type": str(payload.get("action_type") or "review"),
                    "ticker": _optional_ticker(payload),
                    "urgency": str(payload.get("urgency") or "normal"),
                    "status": "completed" if action_id == "complete_action_item" else "dismissed",
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "completed_at": now,
                    "resolution_note": payload.get("resolution_note"),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id == "create_watch_trigger":
            condition = _non_blank(payload.get("condition"), "condition")
            row = self.objects.write_object(
                "WatchTrigger",
                condition,
                {
                    "condition": condition,
                    "trigger_type": str(payload.get("trigger_type") or "custom"),
                    "ticker": _optional_ticker(payload),
                    "status": "active",
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "expires_at": payload.get("expires_at"),
                    "definition": _dict(payload.get("definition")),
                    "created_at": now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id in {
            "cancel_watch_trigger",
            "fire_watch_trigger",
            "update_watch_trigger_check",
            "update_watch_trigger_definition",
        }:
            trigger_id = payload.get("trigger_id") or payload.get("id")
            trigger_uid = _normalize_watch_trigger_uid(trigger_id)
            trigger = _watch_trigger_context(self.objects, trigger_uid)
            if not trigger:
                raise OntologyCommandNotFound("WatchTrigger", str(trigger_id))
            status = "cancelled"
            if action_id == "fire_watch_trigger":
                status = "fired"
            elif action_id in {"update_watch_trigger_check", "update_watch_trigger_definition"}:
                status = str(payload.get("status") or trigger.get("status") or "active")
            condition = str(
                trigger.get("condition") or payload.get("condition") or f"Watch trigger {trigger_uid}"
            ).strip()
            definition = (
                _dict(payload.get("definition"))
                if action_id == "update_watch_trigger_definition"
                else _dict(trigger.get("definition"))
            )
            evidence = payload.get("evidence") if "evidence" in payload else trigger.get("last_evidence")
            row = self.objects.write_object(
                "WatchTrigger",
                trigger_uid,
                {
                    "condition": condition,
                    "trigger_type": str(trigger.get("trigger_type") or payload.get("trigger_type") or "custom"),
                    "ticker": trigger.get("ticker") or _optional_ticker(payload),
                    "status": status,
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "expires_at": trigger.get("expires_at"),
                    "created_at": trigger.get("created_at"),
                    "last_result": _dict(payload.get("result")) or _dict(trigger.get("last_result")),
                    "last_evidence": evidence,
                    "definition": definition,
                    "last_checked_at": now
                    if action_id == "update_watch_trigger_check"
                    else trigger.get("last_checked_at"),
                    "fired_at": now if action_id == "fire_watch_trigger" else trigger.get("fired_at"),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id == "replace_watch_trigger":
            trigger_id = payload.get("trigger_id") or payload.get("id")
            trigger_uid = _normalize_watch_trigger_uid(trigger_id)
            trigger = _watch_trigger_context(self.objects, trigger_uid)
            if not trigger:
                raise OntologyCommandNotFound("WatchTrigger", str(trigger_id))
            cancelled = self.objects.write_object(
                "WatchTrigger",
                trigger_uid,
                {
                    "condition": str(trigger.get("condition") or f"Watch trigger {trigger_uid}").strip(),
                    "trigger_type": str(trigger.get("trigger_type") or "custom"),
                    "ticker": trigger.get("ticker"),
                    "status": "cancelled",
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "expires_at": trigger.get("expires_at"),
                    "definition": _dict(trigger.get("definition")),
                    "created_at": trigger.get("created_at"),
                    "last_checked_at": trigger.get("last_checked_at"),
                    "last_result": _dict(trigger.get("last_result")),
                    "last_evidence": trigger.get("last_evidence"),
                    "fired_at": trigger.get("fired_at"),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(cancelled))
            condition = _non_blank(payload.get("condition"), "condition")
            replacement_seed = {
                "old_trigger_id": trigger_uid,
                "condition": condition,
                "trigger_type": str(payload.get("trigger_type") or "custom"),
                "ticker": _optional_ticker(payload),
                "expires_at": payload.get("expires_at"),
                "definition": _dict(payload.get("definition")),
            }
            replacement_key = f"replacement:{trigger_uid}:{_stable_hash(replacement_seed)[:12]}"
            replacement = self.objects.write_object(
                "WatchTrigger",
                replacement_key,
                {
                    "trigger_id": replacement_key,
                    "condition": condition,
                    "trigger_type": str(payload.get("trigger_type") or "custom"),
                    "ticker": _optional_ticker(payload),
                    "status": "active",
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "expires_at": payload.get("expires_at"),
                    "definition": _dict(payload.get("definition")),
                    "created_at": now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(replacement))
            return refs
        if action_id in {"create_monitor_definition", "update_monitor_definition", "disable_monitor_definition"}:
            existing = {}
            if action_id == "create_monitor_definition":
                monitor_uid = monitor_definition_id(
                    payload.get("monitor_id") or payload.get("name") or _stable_hash(payload)[:12]
                )
                status = str(payload.get("status") or "active")
                version = 1
            else:
                monitor_uid = _normalize_monitor_definition_uid(payload.get("monitor_id"))
                existing = _monitor_definition_context(self.objects, monitor_uid)
                if not existing:
                    raise OntologyCommandNotFound("MonitorDefinition", str(payload.get("monitor_id")))
                status = (
                    "disabled"
                    if action_id == "disable_monitor_definition"
                    else str(payload.get("status") or existing.get("status") or "active")
                )
                version = int(existing.get("definition_version") or 1) + (
                    1 if action_id == "update_monitor_definition" else 0
                )
            merged_definition = _merge_definition_payload(existing, payload, kind="monitor")
            definition_hash = _definition_hash(merged_definition)
            row = self.objects.write_object(
                "MonitorDefinition",
                monitor_uid,
                {
                    **merged_definition,
                    "monitor_id": monitor_uid,
                    "status": status,
                    "owner_actor_id": existing.get("owner_actor_id") or context.actor.actor_id,
                    "definition_version": version,
                    "definition_hash": definition_hash,
                    "created_at": existing.get("created_at") or now,
                    "updated_at": now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id in {"create_mission_definition", "update_mission_definition", "disable_mission_definition"}:
            existing = {}
            if action_id == "create_mission_definition":
                mission_uid = mission_definition_id(
                    payload.get("mission_id") or payload.get("name") or _stable_hash(payload)[:12]
                )
                status = str(payload.get("status") or "active")
                version = 1
            else:
                mission_uid = _normalize_mission_definition_uid(payload.get("mission_id"))
                existing = _mission_definition_context(self.objects, mission_uid)
                if not existing:
                    raise OntologyCommandNotFound("MissionDefinition", str(payload.get("mission_id")))
                status = (
                    "disabled"
                    if action_id == "disable_mission_definition"
                    else str(payload.get("status") or existing.get("status") or "active")
                )
                version = int(existing.get("definition_version") or 1) + (
                    1 if action_id == "update_mission_definition" else 0
                )
            merged_definition = _merge_definition_payload(existing, payload, kind="mission")
            definition_hash = _definition_hash(merged_definition)
            row = self.objects.write_object(
                "MissionDefinition",
                mission_uid,
                {
                    **merged_definition,
                    "mission_id": mission_uid,
                    "status": status,
                    "owner_actor_id": existing.get("owner_actor_id") or context.actor.actor_id,
                    "definition_version": version,
                    "definition_hash": definition_hash,
                    "created_at": existing.get("created_at") or now,
                    "updated_at": now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            return refs
        if action_id in COURSE_OF_ACTION_CREATE_ACTION_IDS:
            record = _dict(payload.get("record") or payload)
            rec_key = str(record.get("recommendation_id") or record.get("idempotency_key") or _stable_hash(record))
            rec_uid = recommendation_id(rec_key)
            course_refs = self._write_course_of_action_bundle(
                record,
                approval_object_id=approval_object_id,
                action_run_id=action_run_id,
                recommendation_uid_value=rec_uid,
                recommendation_key=rec_key,
                actor=actor,
                provenance_id=provenance_id,
                input_hash=input_hash,
                now=now,
                decision_state="applied" if approval_object_id and action_run_id else "generated",
                approval_status="approved" if approval_object_id else "none",
            )
            refs.extend(course_refs)
            row = self.objects.write_object(
                "Recommendation",
                rec_key,
                {
                    "recommendation_id": rec_key,
                    "idempotency_key": record.get("idempotency_key"),
                    "source_kind": str(record.get("source_kind") or record.get("report_type") or "agent"),
                    "report_type": record.get("report_type"),
                    "as_of": record.get("as_of") or now,
                    "action": _non_blank(record.get("action"), "action"),
                    "ticker": _optional_ticker(record),
                    "instrument": record.get("instrument") or record.get("ticker"),
                    "decision_state": "approved" if record.get("action") in ACTIONABLE_ACTIONS else "generated",
                    "status": record.get("status") or record.get("recommendation_status"),
                    "approval_id": approval_object_id,
                    "approval_required": bool(record.get("action") in ACTIONABLE_ACTIONS),
                    "approval_status": "approved" if approval_object_id else "none",
                    "outcome_status": record.get("outcome_status") or "pending",
                    "account_id": record.get("account_id"),
                    "portfolio_id": record.get("portfolio_id"),
                    "policy_id": record.get("policy_id"),
                    "policy_gate_result_id": record.get("policy_gate_result_id"),
                    "policy_gate_decision": record.get("policy_gate_decision"),
                    "policy_gate_review_required": bool(record.get("policy_gate_review_required")),
                    "confidence": record.get("confidence"),
                    "horizon": record.get("horizon"),
                    "rationale_summary": record.get("rationale") or record.get("rationale_summary"),
                    "source_quality": record.get("critical_data_quality") or record.get("source_quality"),
                    "decision_quality": record.get("decision_quality"),
                    "decision_quality_gate": record.get("decision_quality_gate"),
                    "payload": record,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            refs.extend(
                self._write_recommendation_lineage(
                    rec_uid,
                    rec_key,
                    record,
                    approval_object_id=approval_object_id,
                    course_of_action_uid_value=course_of_action_id(_course_of_action_key(record)),
                    course_of_action_key=_course_of_action_key(record),
                    actor=actor,
                    provenance_id=provenance_id,
                    input_hash=input_hash,
                    now=now,
                )
            )
            return refs
        if action_id == "save_overview_content":
            refs.extend(
                self._write_overview_content(
                    payload,
                    context,
                    actor=actor,
                    provenance_id=provenance_id,
                    input_hash=input_hash,
                    now=now,
                )
            )
            return refs
        raise OntologyCommandValidationError(f"Ontology-primary action is not implemented: {action_id}")

    def _write_course_of_action_bundle(
        self,
        record: Mapping[str, Any],
        *,
        approval_object_id: str | None,
        action_run_id: str | None,
        actor: Mapping[str, Any],
        provenance_id: str,
        input_hash: str,
        now: str,
        decision_state: str,
        approval_status: str,
        recommendation_uid_value: str | None = None,
        recommendation_key: str | None = None,
    ) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        course_key = _course_of_action_key(record)
        course_uid = course_of_action_id(course_key)
        action = _course_of_action_action(record.get("action"))
        actionability = _course_of_action_actionability(record)
        rationale = str(record.get("rationale") or record.get("rationale_summary") or "").strip()
        gate = _dict(record.get("policy_gate_result"))
        gate_id = str(record.get("policy_gate_result_id") or gate.get("policy_gate_result_id") or "").strip()
        if gate_id.startswith("policy_gate_result:"):
            gate_id = gate_id.split(":", 1)[1]
        row = self.objects.write_object(
            "CourseOfAction",
            course_key,
            {
                "course_of_action_id": course_key,
                "idempotency_key": record.get("idempotency_key") or course_key,
                "source_kind": str(record.get("source_kind") or record.get("report_type") or "agent"),
                "source_type": record.get("source_type") or record.get("report_type") or "agent",
                "source_id": record.get("source_id") or record.get("report_id") or record.get("idempotency_key"),
                "decision_type": record.get("decision_type") or "investment_decision",
                "action": action,
                "actionability": actionability,
                "decision_state": decision_state,
                "status": record.get("status") or record.get("recommendation_status"),
                "ticker": _optional_ticker(record),
                "instrument_id": instrument_id(record.get("instrument") or record.get("ticker") or course_key),
                "position_uid": record.get("position_uid"),
                "account_id": record.get("account_id"),
                "portfolio_id": record.get("portfolio_id"),
                "policy_id": record.get("policy_id"),
                "policy_gate_result_id": gate_id or None,
                "policy_gate_decision": gate.get("decision") or record.get("policy_gate_decision"),
                "approval_id": approval_object_id,
                "approval_required": approval_object_id is not None or bool(record.get("approval_required")),
                "approval_status": approval_status,
                "action_run_id": action_run_id,
                "confidence": record.get("confidence"),
                "horizon": record.get("horizon"),
                "rationale_summary": rationale or None,
                "rationale_hash": _stable_hash(rationale) if rationale else None,
                "source_quality": record.get("critical_data_quality") or record.get("source_quality"),
                "sizing_summary": _dict(record.get("sizing_summary") or record.get("sizing")),
                "effect_summary": _dict(record.get("effect_summary") or record.get("trade_proposal")),
                "risk_summary": _dict(record.get("risk_summary") or record.get("risk")),
                "policy_summary": _dict(record.get("policy_summary") or gate),
                "decision_quality": record.get("decision_quality"),
                "decision_quality_gate": record.get("decision_quality_gate"),
                "payload": dict(record),
                "as_of": record.get("as_of") or now,
                "created_at": record.get("created_at") or now,
                "updated_at": now,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        refs.append(_version_ref_from_row(row))
        if rationale:
            rationale_key = f"{course_key}:rationale:{_stable_hash(rationale)}"
            rationale_row = self.objects.write_object(
                "CourseOfActionRationale",
                rationale_key,
                {
                    "rationale_id": rationale_key,
                    "course_of_action_id": course_uid,
                    "summary": rationale,
                    "evidence_summary": record.get("evidence_summary"),
                    "rationale_hash": _stable_hash(rationale),
                    "created_at": now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(rationale_row))
            self.objects.write_relation(
                course_uid,
                course_of_action_rationale_id(rationale_key),
                "course_of_action_has_rationale",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
        relation_writes: list[tuple[str, str, str, dict[str, Any]]] = []
        if record.get("account_id"):
            relation_writes.append(
                (
                    course_uid,
                    account_id(record["account_id"]),
                    "course_of_action_targets_account",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                )
            )
        if record.get("portfolio_id"):
            relation_writes.append(
                (
                    course_uid,
                    portfolio_id(record["portfolio_id"]),
                    "course_of_action_targets_portfolio",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                )
            )
        ticker_or_instrument = record.get("instrument") or record.get("ticker")
        if ticker_or_instrument:
            relation_writes.append(
                (
                    course_uid,
                    instrument_id(ticker_or_instrument),
                    "course_of_action_targets_instrument",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                )
            )
        if recommendation_uid_value:
            relation_writes.append(
                (
                    course_uid,
                    recommendation_uid_value,
                    "course_of_action_links_recommendation",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                )
            )
        if approval_object_id:
            relation_writes.extend(
                [
                    (
                        course_uid,
                        approval_object_id,
                        "course_of_action_requires_approval",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "approval_id": approval_object_id},
                    ),
                    (
                        approval_object_id,
                        course_uid,
                        "approval_targets_course_of_action",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "target_object_type": "CourseOfAction"},
                    ),
                ]
            )
        if action_run_id:
            relation_writes.append(
                (
                    action_run_id,
                    course_uid,
                    "action_run_applies_course_of_action",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                )
            )
        for source, target, relation_type, properties in relation_writes:
            self.objects.write_relation(
                source,
                target,
                relation_type,
                properties,
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
        return refs

    def _write_recommendation_lineage(
        self,
        recommendation_uid_value: str,
        recommendation_key: str,
        record: Mapping[str, Any],
        *,
        approval_object_id: str | None,
        course_of_action_uid_value: str | None,
        course_of_action_key: str | None,
        actor: Mapping[str, Any],
        provenance_id: str,
        input_hash: str,
        now: str,
    ) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        gate = _dict(record.get("policy_gate_result"))
        gate_id = str(record.get("policy_gate_result_id") or gate.get("policy_gate_result_id") or "").strip()
        if gate or gate_id:
            gate_key = _strip_uid_prefix(gate_id or _stable_hash(gate or record), "policy_gate_result")
            gate_uid = policy_gate_result_id(gate_key)
            gate_row = self.objects.write_object(
                "PolicyGateResult",
                gate_uid,
                {
                    "gate_result_id": gate_key,
                    "decision": gate.get("decision") or record.get("policy_gate_decision") or "warn",
                    "review_required": bool(gate.get("review_required") or record.get("policy_gate_review_required")),
                    "failure_reasons": _list(gate.get("failure_reasons") or record.get("policy_gate_failures")),
                    "warnings": _list(gate.get("warnings") or record.get("policy_gate_warnings")),
                    "account_id": record.get("account_id"),
                    "portfolio_id": record.get("portfolio_id"),
                    "policy_id": record.get("policy_id"),
                    "evaluated_at": gate.get("evaluated_at") or now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(gate_row))
            for relation_type, source, target in (
                ("policy_gate_evaluates_recommendation", gate_uid, recommendation_uid_value),
                ("recommendation_has_policy_gate_result", recommendation_uid_value, gate_uid),
            ):
                self.objects.write_relation(
                    source,
                    target,
                    relation_type,
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )

        if record.get("risk_snapshot_id"):
            snapshot_id = str(record["risk_snapshot_id"])
            risk_row = self.objects.write_object(
                "PositionRiskSnapshot",
                snapshot_id,
                {
                    "snapshot_id": snapshot_id,
                    "ticker": _optional_ticker(record),
                    "portfolio_risk_snapshot_id": record.get("portfolio_risk_snapshot_id"),
                    "as_of": record.get("as_of"),
                    "risk_score": _optional_float(record.get("risk_score")),
                    "risk_level": record.get("risk_level"),
                    "confidence": _optional_float(record.get("risk_confidence")),
                    "quality": record.get("risk_quality"),
                    "source_status": _dict(record.get("risk_source_status")),
                    "payload": {"risk_bindings": _list(record.get("risk_bindings"))},
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(risk_row))
            self.objects.write_relation(
                recommendation_uid_value,
                position_risk_snapshot_id(snapshot_id),
                "recommendation_uses_position_risk_snapshot",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            if course_of_action_uid_value:
                self.objects.write_relation(
                    course_of_action_uid_value,
                    position_risk_snapshot_id(snapshot_id),
                    "course_of_action_uses_position_risk_snapshot",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
        if record.get("portfolio_risk_snapshot_id"):
            snapshot_id = str(record["portfolio_risk_snapshot_id"])
            risk_row = self.objects.write_object(
                "PortfolioRiskSnapshot",
                snapshot_id,
                {
                    "snapshot_id": snapshot_id,
                    "as_of": record.get("as_of"),
                    "confidence": _optional_float(record.get("risk_confidence")),
                    "quality": record.get("risk_quality"),
                    "source_status": _dict(record.get("risk_source_status")),
                    "payload": {"risk_bindings": _list(record.get("risk_bindings"))},
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(risk_row))
            self.objects.write_relation(
                recommendation_uid_value,
                portfolio_risk_snapshot_id(snapshot_id),
                "recommendation_uses_portfolio_risk_snapshot",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            if course_of_action_uid_value:
                self.objects.write_relation(
                    course_of_action_uid_value,
                    portfolio_risk_snapshot_id(snapshot_id),
                    "course_of_action_uses_portfolio_risk_snapshot",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )

        for relation_type, evidence_items in (
            ("recommendation_supported_by_evidence", _list(record.get("evidence"))),
            ("recommendation_contradicted_by_evidence", _list(record.get("disconfirming_evidence"))),
        ):
            role = "supporting" if relation_type == "recommendation_supported_by_evidence" else "disconfirming"
            for index, item in enumerate(evidence_items):
                evidence_text = _evidence_summary(item)
                if not evidence_text:
                    continue
                evidence_key = f"{recommendation_key}:{role}:{index}:{_stable_hash(item)}"
                evidence_row = self.objects.write_object(
                    "Evidence",
                    evidence_key,
                    {
                        "evidence_id": evidence_key,
                        "evidence_type": role,
                        "summary": evidence_text,
                        "observed_at": record.get("as_of") or now,
                        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    },
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
                refs.append(_version_ref_from_row(evidence_row))
                evidence_uid_value = evidence_id(evidence_key)
                self.objects.write_relation(
                    recommendation_uid_value,
                    evidence_uid_value,
                    relation_type,
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "relation_role": role},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
                if course_of_action_uid_value:
                    self.objects.write_relation(
                        course_of_action_uid_value,
                        evidence_uid_value,
                        "course_of_action_supported_by_evidence"
                        if relation_type == "recommendation_supported_by_evidence"
                        else "course_of_action_contradicted_by_evidence",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "relation_role": role},
                        now,
                        actor=actor,
                        provenance=provenance_id,
                        input_hash=input_hash,
                    )
                citation = _dict(item).get("citation") if isinstance(item, Mapping) else None
                citation_payload = _dict(citation)
                if citation_payload:
                    citation_key = f"{evidence_key}:citation:{_stable_hash(citation_payload)}"
                    citation_row = self.objects.write_object(
                        "Citation",
                        citation_key,
                        {
                            "citation_id": citation_key,
                            "title": citation_payload.get("title"),
                            "url": citation_payload.get("url"),
                            "source_path": citation_payload.get("source_path"),
                            "quote_hash": _stable_hash(citation_payload.get("quote") or evidence_text),
                            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                        },
                        now,
                        actor=actor,
                        provenance=provenance_id,
                        input_hash=input_hash,
                    )
                    refs.append(_version_ref_from_row(citation_row))
                    for citation_relation in ("evidence_has_citation", "evidence_cites_citation"):
                        self.objects.write_relation(
                            evidence_uid_value,
                            citation_id(citation_key),
                            citation_relation,
                            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                            now,
                            actor=actor,
                            provenance=provenance_id,
                            input_hash=input_hash,
                        )

        if str(record.get("action") or "").lower() in ACTIONABLE_ACTIONS:
            proposal_key = str(record.get("trade_proposal_id") or record.get("idempotency_key") or recommendation_key)
            proposal_uid = trade_proposal_id(proposal_key)
            proposal_row = self.objects.write_object(
                "TradeProposal",
                proposal_key,
                {
                    "proposal_id": proposal_key,
                    "recommendation_id": recommendation_key,
                    "course_of_action_id": course_of_action_key,
                    "account_id": record.get("account_id"),
                    "portfolio_id": record.get("portfolio_id"),
                    "action": str(record.get("action") or "review"),
                    "instrument": str(record.get("instrument") or record.get("ticker") or "portfolio"),
                    "proposed_change": _dict(record.get("trade_proposal"))
                    or {"target_change": record.get("target_change"), "horizon": record.get("horizon")},
                    "policy_gate_result_id": record.get("policy_gate_result_id"),
                    "approval_id": approval_object_id,
                    "decision_state": "pending_approval" if approval_object_id else "staged",
                    "status": "pending_approval" if approval_object_id else "staged",
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(proposal_row))
            for relation_type, source, target, properties in (
                (
                    "recommendation_has_trade_proposal",
                    recommendation_uid_value,
                    proposal_uid,
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                ),
                (
                    "trade_proposal_derives_from_recommendation",
                    proposal_uid,
                    recommendation_uid_value,
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                ),
            ):
                self.objects.write_relation(
                    source,
                    target,
                    relation_type,
                    properties,
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
            if approval_object_id:
                self.objects.write_relation(
                    approval_object_id,
                    recommendation_uid_value,
                    "approval_targets_recommendation",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "target_object_type": "Recommendation"},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
                self.objects.write_relation(
                    approval_object_id,
                    proposal_uid,
                    "approval_targets_trade_proposal",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "target_object_type": "TradeProposal"},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
        return refs

    def _write_overview_content(
        self,
        payload: Mapping[str, Any],
        context: OntologyCommandContext,
        *,
        actor: Mapping[str, Any],
        provenance_id: str,
        input_hash: str,
        now: str,
    ) -> list[dict[str, Any]]:
        ticker = _non_blank(payload.get("ticker"), "ticker").upper()
        content = _non_blank(payload.get("content"), "content")
        preserve_exact = bool(payload.get("preserve_exact_content"))
        try:
            from api.routers.overview import parse_overview_markdown
            from portfolio.overview_content import save_overview_content

            parsed = parse_overview_markdown(content) or {}
            saved = save_overview_content(ticker, content, preserve_exact_content=preserve_exact)
        except Exception as exc:
            raise OntologyCommandValidationError(str(exc) or exc.__class__.__name__) from exc

        issuer_uid = issuer_id(ticker)
        instr_uid = instrument_id(ticker)
        doc_uid = document_artifact_id("overview", ticker)
        overview_uid = equity_overview_id(issuer_uid)
        content_hash = _stable_hash(saved.index_content)
        refs: list[dict[str, Any]] = []

        issuer_row = self.objects.write_object(
            "Issuer",
            issuer_uid,
            {"issuer_id": ticker, "name": ticker, "ticker": ticker, "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        refs.append(_version_ref_from_row(issuer_row))
        instr_row = self.objects.write_object(
            "Instrument",
            ticker,
            {
                "instrument_id": ticker,
                "ticker": ticker,
                "asset_class": "security",
                "instrument_type": "security",
                "issuer_id": issuer_uid,
                "status": "active",
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        refs.append(_version_ref_from_row(instr_row))
        doc_row = self.objects.write_object(
            "DocumentArtifact",
            doc_uid,
            {
                "document_type": "overview",
                "document_id": ticker,
                "title": f"{ticker} overview",
                "ticker": ticker,
                "content_hash": content_hash,
                "artifact_uri": saved.source_path,
                "status": "active",
                "source_type": context.source_type,
                "source_id": context.source_id,
                "created_at": now,
                "updated_at": now,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        refs.append(_version_ref_from_row(doc_row))
        overview_row = self.objects.write_object(
            "EquityOverview",
            overview_uid,
            {
                "overview_id": overview_uid,
                "issuer_id": issuer_uid,
                "ticker": ticker,
                "document_id": doc_uid,
                "content_hash": content_hash,
                "status": "active",
                "created_at": now,
                "updated_at": now,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        refs.append(_version_ref_from_row(overview_row))
        for relation_type, source, target, properties in (
            ("document_artifact_materializes_research_object", doc_uid, overview_uid, {"document_role": "markdown"}),
            ("research_object_uses_document", overview_uid, doc_uid, {"document_role": "rendered_markdown"}),
            ("equity_overview_covers_issuer", overview_uid, issuer_uid, {}),
            ("equity_overview_covers_instrument", overview_uid, instr_uid, {}),
        ):
            self.objects.write_relation(
                source,
                target,
                relation_type,
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, **properties},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )

        refs.extend(
            self._write_overview_children(
                ticker=ticker,
                issuer_uid=issuer_uid,
                overview_uid=overview_uid,
                parsed=parsed,
                actor=actor,
                provenance_id=provenance_id,
                input_hash=input_hash,
                now=now,
            )
        )
        _best_effort_index_document(
            "overview",
            saved.index_content,
            ticker,
            saved.source_path,
            f"overview-{ticker}",
            overview_uid,
            _version_ref_from_row(overview_row).get("version_id"),
        )
        return refs

    def _write_overview_children(
        self,
        *,
        ticker: str,
        issuer_uid: str,
        overview_uid: str,
        parsed: Mapping[str, Any],
        actor: Mapping[str, Any],
        provenance_id: str,
        input_hash: str,
        now: str,
    ) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        financials = _dict(parsed.get("financials"))
        if financials:
            profile_uid = company_financial_profile_id(f"{overview_uid}:financial_profile")
            row = self.objects.write_object(
                "CompanyFinancialProfile",
                profile_uid,
                {
                    "profile_id": profile_uid,
                    "overview_id": overview_uid,
                    "issuer_id": issuer_uid,
                    "ticker": ticker,
                    "revenue_growth": _dict(financials.get("revenue_growth")) or None,
                    "eps_growth": _dict(financials.get("eps_growth")) or None,
                    "debt": _dict(financials.get("debt")) or None,
                    "reinvestment": financials.get("reinvestment"),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            self.objects.write_relation(
                overview_uid,
                profile_uid,
                "equity_overview_has_financial_profile",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
        for index, item in enumerate(_list(parsed.get("sensitivity")), start=1):
            if not isinstance(item, Mapping) or not item.get("factor"):
                continue
            child_uid = extrinsic_sensitivity_id(f"{overview_uid}:sensitivity:{index}:{item.get('factor')}")
            row = self.objects.write_object(
                "ExtrinsicSensitivity",
                child_uid,
                {
                    "sensitivity_id": child_uid,
                    "overview_id": overview_uid,
                    "issuer_id": issuer_uid,
                    "ticker": ticker,
                    "factor": item.get("factor"),
                    "sensitivity": item.get("sensitivity"),
                    "capacity": item.get("capacity"),
                    "rationale": item.get("rationale"),
                    "ordinal": index,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            self.objects.write_relation(
                overview_uid,
                child_uid,
                "equity_overview_has_extrinsic_sensitivity",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
        for index, item in enumerate(_list(parsed.get("porters_five_forces")), start=1):
            if not isinstance(item, Mapping) or not item.get("force"):
                continue
            child_uid = industry_force_assessment_id(f"{overview_uid}:force:{index}:{item.get('force')}")
            row = self.objects.write_object(
                "IndustryForceAssessment",
                child_uid,
                {
                    "force_id": child_uid,
                    "overview_id": overview_uid,
                    "issuer_id": issuer_uid,
                    "ticker": ticker,
                    "force": item.get("force"),
                    "rating": item.get("rating"),
                    "description": item.get("description"),
                    "ordinal": index,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            self.objects.write_relation(
                overview_uid,
                child_uid,
                "equity_overview_has_industry_force",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
        for outlook_type, item in (
            ("supply", _dict(parsed.get("supply_outlook"))),
            ("demand", _dict(parsed.get("demand_outlook"))),
        ):
            if not item:
                continue
            child_uid = supply_demand_outlook_id(f"{overview_uid}:{outlook_type}")
            row = self.objects.write_object(
                "SupplyDemandOutlook",
                child_uid,
                {
                    "outlook_id": child_uid,
                    "overview_id": overview_uid,
                    "issuer_id": issuer_uid,
                    "ticker": ticker,
                    "outlook_type": outlook_type,
                    "rating": item.get("rating"),
                    "points": _list(item.get("points")),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
            self.objects.write_relation(
                overview_uid,
                child_uid,
                "equity_overview_has_supply_demand_outlook",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
        supply_chain = _dict(parsed.get("supply_chain"))
        for counterparty_role, items in (
            ("supplier", _list(supply_chain.get("suppliers"))),
            ("customer", _list(supply_chain.get("customers"))),
        ):
            for index, item in enumerate(items, start=1):
                if not isinstance(item, Mapping) or not item.get("name"):
                    continue
                child_uid = supply_chain_relationship_id(
                    f"{overview_uid}:{counterparty_role}:{index}:{item.get('name')}"
                )
                row = self.objects.write_object(
                    "SupplyChainRelationship",
                    child_uid,
                    {
                        "relationship_id": child_uid,
                        "overview_id": overview_uid,
                        "issuer_id": issuer_uid,
                        "ticker": ticker,
                        "counterparty_role": counterparty_role,
                        "counterparty_name": item.get("name"),
                        "relationship": item.get("relationship"),
                        "exposure": item.get("exposure"),
                        "notes": item.get("notes"),
                        "ordinal": index,
                        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    },
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
                refs.append(_version_ref_from_row(row))
                self.objects.write_relation(
                    overview_uid,
                    child_uid,
                    "equity_overview_has_supply_chain_relationship",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )
        return refs

    def _write_thesis_content(
        self,
        payload: Mapping[str, Any],
        context: OntologyCommandContext,
        *,
        actor: Mapping[str, Any],
        provenance_id: str,
        input_hash: str,
        now: str,
    ) -> list[dict[str, Any]]:
        ticker = _non_blank(payload.get("ticker"), "ticker").upper()
        content = _non_blank(payload.get("content"), "content")
        preserve_exact = bool(payload.get("preserve_exact_content"))
        try:
            from portfolio.thesis_content import write_thesis

            index_content = content if preserve_exact else content.strip()
            source_path = write_thesis(ticker, content if preserve_exact else f"{index_content}\n")
        except Exception as exc:
            raise OntologyCommandValidationError(str(exc) or exc.__class__.__name__) from exc

        issuer_uid = issuer_id(ticker)
        instr_uid = instrument_id(ticker)
        thesis_doc_uid = thesis_document_id(ticker)
        doc_uid = document_artifact_id("thesis", ticker)
        content_hash = _stable_hash(index_content)
        refs: list[dict[str, Any]] = []
        for object_type, business_key, props in (
            ("Issuer", issuer_uid, {"issuer_id": ticker, "name": ticker, "ticker": ticker}),
            (
                "Instrument",
                ticker,
                {
                    "instrument_id": ticker,
                    "ticker": ticker,
                    "asset_class": "security",
                    "instrument_type": "security",
                    "issuer_id": issuer_uid,
                    "status": "active",
                },
            ),
            (
                "Thesis",
                ticker,
                {
                    "ticker": ticker,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                    "instrument_id": instr_uid,
                },
            ),
            (
                "DocumentArtifact",
                doc_uid,
                {
                    "document_type": "thesis",
                    "document_id": ticker,
                    "title": f"{ticker} thesis",
                    "ticker": ticker,
                    "content_hash": content_hash,
                    "artifact_uri": source_path,
                    "status": "active",
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                    "created_at": now,
                    "updated_at": now,
                },
            ),
        ):
            row = self.objects.write_object(
                object_type,
                business_key,
                {**props, "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(row))
        thesis_doc_row = self.objects.write_object(
            "ThesisDocument",
            thesis_doc_uid,
            {
                "thesis_document_id": thesis_doc_uid,
                "ticker": ticker,
                "issuer_id": issuer_uid,
                "instrument_id": instr_uid,
                "document_id": doc_uid,
                "content_hash": content_hash,
                "status": "active",
                "created_at": now,
                "updated_at": now,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        refs.append(_version_ref_from_row(thesis_doc_row))
        for relation_type, source, target, properties in (
            ("document_artifact_materializes_research_object", doc_uid, thesis_doc_uid, {"document_role": "markdown"}),
            ("research_object_uses_document", thesis_doc_uid, doc_uid, {"document_role": "rendered_markdown"}),
            ("thesis_document_covers_issuer", thesis_doc_uid, issuer_uid, {}),
            ("thesis_document_covers_instrument", thesis_doc_uid, instr_uid, {}),
        ):
            self.objects.write_relation(
                source,
                target,
                relation_type,
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, **properties},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
        for index, section in enumerate(_markdown_sections(index_content), start=1):
            section_uid = thesis_section_id(f"{thesis_doc_uid}:section:{index}:{section['heading']}")
            section_row = self.objects.write_object(
                "ThesisSection",
                section_uid,
                {
                    "section_id": section_uid,
                    "thesis_document_id": thesis_doc_uid,
                    "ticker": ticker,
                    "heading": section["heading"],
                    "level": section["level"],
                    "content": section["content"],
                    "content_hash": _stable_hash(section["content"]),
                    "ordinal": index,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(section_row))
            self.objects.write_relation(
                thesis_doc_uid,
                section_uid,
                "thesis_document_has_section",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
        refs.extend(
            self._write_thesis_markdown_entities(
                ticker=ticker,
                thesis_uid=thesis_id(ticker),
                content=index_content,
                actor=actor,
                provenance_id=provenance_id,
                input_hash=input_hash,
                now=now,
            )
        )
        _best_effort_index_document(
            "thesis",
            index_content,
            ticker,
            source_path,
            f"thesis-{ticker}",
            thesis_doc_uid,
            _version_ref_from_row(thesis_doc_row).get("version_id"),
        )
        return refs

    def _write_thesis_markdown_entities(
        self,
        *,
        ticker: str,
        thesis_uid: str,
        content: str,
        actor: Mapping[str, Any],
        provenance_id: str,
        input_hash: str,
        now: str,
    ) -> list[dict[str, Any]]:
        """Project thesis markdown sections into ontology-native process objects."""
        from portfolio.thesis_backfill import _categorize_catalyst, _extract_label_and_description, _parse_bullets
        from portfolio.thesis_sync import _normalize_match_text, _parse_structured_claims, _parse_text_claims

        refs: list[dict[str, Any]] = []
        catalyst_by_label: dict[str, str] = {}
        kill_condition_by_label: dict[str, str] = {}

        position_uid = position_id(ticker)
        try:
            position_row = self.objects.get_object(position_uid)
        except Exception:
            position_row = None
        if position_row:
            self.objects.write_relation(
                position_uid,
                thesis_uid,
                "has_thesis",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )

        for bullet in _parse_bullets(content, "Key Catalysts"):
            if _blank_markdown_item(bullet):
                continue
            label, desc = _extract_label_and_description(bullet)
            description = f"{label}: {desc}" if desc != label else label
            row = self.objects.write_object(
                "Catalyst",
                f"{ticker}:{description}",
                {
                    "ticker": ticker,
                    "name": label,
                    "description": description,
                    "source": "thesis_markdown",
                    "category": _categorize_catalyst(label, desc),
                    "status": "pending",
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            ref = _version_ref_from_row(row)
            refs.append(ref)
            catalyst_uid = str(ref.get("object_uid") or "")
            if catalyst_uid:
                for key in {label, description}:
                    normalized = _normalize_match_text(str(key))
                    if normalized:
                        catalyst_by_label[normalized] = catalyst_uid
                self.objects.write_relation(
                    thesis_uid,
                    catalyst_uid,
                    "has_catalyst",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )

        for bullet in _parse_bullets(content, "Risk Factors"):
            if _blank_markdown_item(bullet):
                continue
            label, desc = _extract_label_and_description(bullet)
            condition = f"{label}: {desc}" if desc != label else label
            row = self.objects.write_object(
                "KillCondition",
                f"{ticker}:{condition}",
                {
                    "ticker": ticker,
                    "condition": condition,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                    "created_by": "thesis_markdown",
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            ref = _version_ref_from_row(row)
            refs.append(ref)
            kill_condition_uid = str(ref.get("object_uid") or "")
            if kill_condition_uid:
                for key in {label, condition}:
                    normalized = _normalize_match_text(str(key))
                    if normalized:
                        kill_condition_by_label[normalized] = kill_condition_uid
                self.objects.write_relation(
                    thesis_uid,
                    kill_condition_uid,
                    "thesis_has_kill_condition",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    now,
                    actor=actor,
                    provenance=provenance_id,
                    input_hash=input_hash,
                )

        parsed_claims = _parse_structured_claims(content)
        claim_records = parsed_claims if parsed_claims is not None else _parse_text_claims(content)
        for raw_record in claim_records:
            record = dict(raw_record)
            if _blank_markdown_item(record.get("claim")):
                continue
            linked_catalyst_labels = list(record.pop("linked_catalyst_labels", []) or [])
            linked_kill_condition_labels = list(record.pop("linked_kill_condition_labels", []) or [])
            record.pop("parsed_from_text", None)
            record.pop("id", None)
            claim = _non_blank(record.get("claim"), "claim")
            row = self.objects.write_object(
                "ThesisClaim",
                f"{ticker}:{claim}",
                {
                    "ticker": ticker,
                    "claim": claim,
                    "expected_evidence": record.get("expected_evidence"),
                    "disconfirming_evidence": record.get("disconfirming_evidence"),
                    "source_requirements": _list(record.get("source_requirements")),
                    "cadence": record.get("cadence"),
                    "confidence": record.get("confidence"),
                    "status": record.get("status") or "active",
                    "source_type": "thesis_markdown",
                    "source_id": f"thesis_markdown:{ticker}",
                    "created_at": now,
                    "updated_at": now,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            ref = _version_ref_from_row(row)
            refs.append(ref)
            claim_uid = str(ref.get("object_uid") or "")
            if not claim_uid:
                continue
            self.objects.write_relation(
                thesis_uid,
                claim_uid,
                "thesis_has_claim",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            for label in linked_catalyst_labels:
                target_catalyst_uid = catalyst_by_label.get(_normalize_match_text(str(label)))
                if target_catalyst_uid:
                    self.objects.write_relation(
                        claim_uid,
                        target_catalyst_uid,
                        "claim_links_catalyst",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                        now,
                        actor=actor,
                        provenance=provenance_id,
                        input_hash=input_hash,
                    )
            for label in linked_kill_condition_labels:
                target_kill_condition_uid = kill_condition_by_label.get(_normalize_match_text(str(label)))
                if target_kill_condition_uid:
                    self.objects.write_relation(
                        claim_uid,
                        target_kill_condition_uid,
                        "claim_links_kill_condition",
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                        now,
                        actor=actor,
                        provenance=provenance_id,
                        input_hash=input_hash,
                    )
            from ontology.evidence_ledger import write_claim_evidence_graph

            write_claim_evidence_graph(
                self.objects,
                claim_uid=claim_uid,
                claim_key=f"{ticker}:{claim}",
                expected_evidence=record.get("expected_evidence"),
                disconfirming_evidence=record.get("disconfirming_evidence"),
                valid_from=now,
                actor=actor,
                provenance_id=provenance_id,
                input_hash=input_hash,
            )

        return refs

    def _write_management_quality_content(
        self,
        payload: Mapping[str, Any],
        context: OntologyCommandContext,
        *,
        actor: Mapping[str, Any],
        provenance_id: str,
        input_hash: str,
        now: str,
    ) -> list[dict[str, Any]]:
        ticker = _non_blank(payload.get("ticker"), "ticker").upper()
        content = _non_blank(payload.get("content"), "content")
        preserve_exact = bool(payload.get("preserve_exact_content"))
        try:
            from api.routers.management_quality import parse_management_quality_markdown
            from portfolio.management_quality_content import save_management_quality_content

            parsed = parse_management_quality_markdown(content) or {}
            saved = save_management_quality_content(ticker, content, preserve_exact_content=preserve_exact)
            try:
                from api.retrieval import index_document

                index_document(
                    doc_type="management_quality",
                    content=saved.index_content,
                    ticker=ticker,
                    source_path=saved.source_path,
                    doc_id=f"management_quality-{ticker}",
                )
            except Exception:
                # Retrieval indexing should not prevent the authoritative ontology write.
                pass
        except Exception as exc:
            raise OntologyCommandValidationError(str(exc) or exc.__class__.__name__) from exc

        issuer_uid = issuer_id(ticker)
        assessment_uid = management_quality_assessment_id(issuer_uid)
        document_uid = document_artifact_id("management_quality", ticker)
        summary = _dict(parsed.get("summary"))
        owner = _dict(summary.get("owner_mindset"))
        business = _dict(summary.get("business_value_understanding"))
        follow = _dict(summary.get("follow_through"))

        refs: list[dict[str, Any]] = []
        issuer_row = self.objects.write_object(
            "Issuer",
            issuer_uid,
            {
                "issuer_id": ticker,
                "name": ticker,
                "ticker": ticker,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        refs.append(_version_ref_from_row(issuer_row))
        doc_row = self.objects.write_object(
            "DocumentArtifact",
            document_uid,
            {
                "document_type": "management_quality",
                "document_id": ticker,
                "title": f"{ticker} management quality",
                "ticker": ticker,
                "content_hash": _stable_hash(content),
                "artifact_uri": saved.source_path,
                "status": "active",
                "source_type": context.source_type,
                "source_id": context.source_id,
                "created_at": now,
                "updated_at": now,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        refs.append(_version_ref_from_row(doc_row))
        assessment_row = self.objects.write_object(
            "ManagementQualityAssessment",
            assessment_uid,
            {
                "assessment_id": assessment_uid,
                "issuer_id": issuer_uid,
                "ticker": ticker,
                "status": "active",
                "overall_rating": summary.get("overall_rating"),
                "bottom_line": summary.get("bottom_line"),
                "owner_mindset_rating": owner.get("rating"),
                "owner_mindset_text": owner.get("text"),
                "business_value_understanding_rating": business.get("rating"),
                "business_value_understanding_text": business.get("text"),
                "follow_through_rating": follow.get("rating"),
                "follow_through_text": follow.get("text"),
                "content_hash": _stable_hash(content),
                "document_id": document_uid,
                "source_type": context.source_type,
                "source_id": context.source_id,
                "created_at": now,
                "updated_at": now,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        refs.append(_version_ref_from_row(assessment_row))
        self.objects.write_relation(
            assessment_uid,
            issuer_uid,
            "management_quality_assesses_issuer",
            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        self.objects.write_relation(
            assessment_uid,
            document_uid,
            "research_object_uses_document",
            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "document_role": "rendered_markdown"},
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        self.objects.write_relation(
            document_uid,
            assessment_uid,
            "document_artifact_materializes_research_object",
            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "document_role": "markdown"},
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )

        for index, row in enumerate(_list(parsed.get("scorecard")), start=1):
            if not isinstance(row, Mapping):
                continue
            question = str(row.get("question") or "").strip()
            rating = str(row.get("rating") or "").strip()
            if not question or not rating:
                continue
            row_uid = management_quality_scorecard_row_id(f"{assessment_uid}:scorecard:{index}:{question}")
            scorecard = self.objects.write_object(
                "ManagementQualityScorecardRow",
                row_uid,
                {
                    "row_id": row_uid,
                    "assessment_id": assessment_uid,
                    "issuer_id": issuer_uid,
                    "ticker": ticker,
                    "question": question,
                    "rating": rating,
                    "evidence": row.get("evidence"),
                    "ordinal": index,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(scorecard))
            self.objects.write_relation(
                assessment_uid,
                _flatten_object(scorecard)["id"],
                "management_quality_has_scorecard_row",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )

        for index, row in enumerate(_list(parsed.get("accomplishments")), start=1):
            if not isinstance(row, Mapping) or not str(row.get("text") or "").strip():
                continue
            acc_uid = management_quality_accomplishment_id(
                f"{assessment_uid}:accomplishment:{index}:{row.get('title') or row.get('text')}"
            )
            acc = self.objects.write_object(
                "ManagementQualityAccomplishment",
                acc_uid,
                {
                    "accomplishment_id": acc_uid,
                    "assessment_id": assessment_uid,
                    "issuer_id": issuer_uid,
                    "ticker": ticker,
                    "title": row.get("title"),
                    "text": row.get("text"),
                    "ordinal": index,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(acc))
            self.objects.write_relation(
                assessment_uid,
                _flatten_object(acc)["id"],
                "management_quality_has_accomplishment",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )

        for index, row in enumerate(_list(parsed.get("setbacks")), start=1):
            if not isinstance(row, Mapping) or not str(row.get("text") or "").strip():
                continue
            setback_uid = management_quality_setback_id(
                f"{assessment_uid}:setback:{index}:{row.get('title') or row.get('text')}"
            )
            setback = self.objects.write_object(
                "ManagementQualitySetback",
                setback_uid,
                {
                    "setback_id": setback_uid,
                    "assessment_id": assessment_uid,
                    "issuer_id": issuer_uid,
                    "ticker": ticker,
                    "title": row.get("title"),
                    "text": row.get("text"),
                    "response_rating": row.get("response_rating"),
                    "response_text": row.get("response_text"),
                    "ordinal": index,
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
            refs.append(_version_ref_from_row(setback))
            self.objects.write_relation(
                assessment_uid,
                _flatten_object(setback)["id"],
                "management_quality_has_setback",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                now,
                actor=actor,
                provenance=provenance_id,
                input_hash=input_hash,
            )
        return refs

    def _ensure_default_account_portfolio(
        self,
        context: OntologyCommandContext,
        *,
        provenance_id: str,
        input_hash: str,
    ) -> None:
        now = _now()
        actor = actor_to_dict(context.actor)
        self.objects.write_object(
            "Account",
            "default",
            {
                "account_id": "default",
                "investor_id": "owner",
                "account_type": "single_owner",
                "tax_status": "unknown",
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )
        self.objects.write_object(
            "Portfolio",
            "default",
            {
                "portfolio_id": "default",
                "account_id": "default",
                "base_currency": "USD",
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            now,
            actor=actor,
            provenance=provenance_id,
            input_hash=input_hash,
        )

    def _write_audit(
        self,
        action_name: str,
        category: str,
        status: str,
        *,
        actor: Actor,
        provenance_id: str,
        object_refs: list[dict[str, Any]],
        after_summary: dict[str, Any] | None = None,
    ) -> None:
        now = _now()
        event_uid = audit_event_id(f"{action_name}:{_stable_hash(object_refs)}:{now}")
        try:
            self.objects.write_object(
                "AuditEvent",
                event_uid,
                {
                    "event_id": event_uid,
                    "occurred_at": now,
                    "actor_type": actor.actor_type,
                    "actor_id": actor.actor_id,
                    "action_name": action_name,
                    "action_category": category,
                    "status": status,
                    "object_refs": object_refs,
                    "after_summary": after_summary,
                    "lineage_root_id": provenance_id,
                    "retention_class": "audit_7y",
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                now,
                actor=actor_to_dict(actor),
                provenance=provenance_id,
            )
        except Exception:
            logger.warning("Failed to write ontology command audit event %s", action_name, exc_info=True)


def _approval_node_resource(approval: Mapping[str, Any]) -> NodeResource:
    return NodeResource(
        id=str(approval.get("id") or approval.get("object_uid") or ""),
        type="Approval",
        properties={key: value for key, value in dict(approval).items() if not str(key).startswith("_")},
    )


def _approval_scope_from_payload(payload: Mapping[str, Any] | None) -> dict[str, str]:
    raw = dict(payload or {})
    candidates: list[Mapping[str, Any]] = [raw]
    record = raw.get("record")
    if isinstance(record, Mapping):
        candidates.append(record)
    proposed_change = raw.get("proposed_change")
    if isinstance(proposed_change, Mapping):
        candidates.append(proposed_change)
        nested_record = proposed_change.get("record")
        if isinstance(nested_record, Mapping):
            candidates.append(nested_record)

    account_id = ""
    portfolio_id = ""
    for candidate in candidates:
        if not account_id:
            account_id = str(candidate.get("account_id") or candidate.get("owner_account_id") or "").strip()
        if not portfolio_id:
            portfolio_id = str(candidate.get("portfolio_id") or "").strip()
    return {
        "account_id": account_id or "default",
        "portfolio_id": portfolio_id or "default",
    }


def _approval_policy_context(
    *,
    action_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
    approval: Mapping[str, Any] | None = None,
    approval_id: str | None = None,
    command_context: OntologyCommandContext | None = None,
    risk_class: str | None = None,
    policy_gate_result: Mapping[str, Any] | None = None,
    account_id: str | None = None,
    portfolio_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    source = dict(approval or payload or {})
    scope = _approval_scope_from_payload(source)
    has_scope_source = bool(source) or bool(account_id) or bool(portfolio_id)
    resolved_risk_class = str(risk_class or source.get("risk_class") or "").strip().lower()
    data_sensitivity = "portfolio_private" if resolved_risk_class == "financial" else "research_private"
    gate = dict(policy_gate_result or source.get("policy_gate_result") or {})
    return {
        "purpose": "approval",
        "resource_type": "Approval",
        "resource_id": approval_id or source.get("id") or source.get("object_uid"),
        "governed_action_id": action_id or source.get("action_id"),
        "request_mode": command_context.request_mode if command_context else None,
        "source_type": command_context.source_type if command_context else source.get("source_type"),
        "source_id": command_context.source_id if command_context else source.get("source_id"),
        "status": status or source.get("status"),
        "risk_class": resolved_risk_class or None,
        "policy_gate_decision": gate.get("decision") or source.get("policy_gate_decision"),
        "policy_gate_result_id": gate.get("policy_gate_result_id") or source.get("policy_gate_result_id"),
        "account_id": account_id or source.get("account_id") or (scope["account_id"] if has_scope_source else None),
        "portfolio_id": portfolio_id
        or source.get("portfolio_id")
        or (scope["portfolio_id"] if has_scope_source else None),
        "data_sensitivity": source.get("data_sensitivity") or data_sensitivity,
    }


def _flatten_object(row: Mapping[str, Any]) -> dict[str, Any]:
    props = dict(row.get("properties") or row.get("properties_json") or {})
    out = {**props}
    out["id"] = str(row.get("object_uid") or props.get("id") or "")
    out["object_uid"] = out["id"]
    out["_meta"] = dict(row.get("_meta") or {})
    return out


def _version_ref_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    flat = _flatten_object(row)
    temporal = _dict(flat.get("_meta")).get("temporal") if isinstance(flat.get("_meta"), dict) else {}
    temporal = _dict(temporal)
    return {
        "object_uid": flat["id"],
        "object_type": row.get("object_type"),
        "version_id": temporal.get("version_id"),
        "valid_from": temporal.get("valid_from"),
    }


def _normalize_approval_uid(value: str) -> str:
    text = _non_blank(value, "approval_id")
    return text if text.startswith("approval:") else approval_id(text)


def _normalize_action_item_uid(value: Any) -> str:
    text = _non_blank(value, "item_id")
    return text if text.startswith("action_item:") else action_item_id(text)


def _normalize_monitor_hit_uid(value: Any) -> str:
    text = _non_blank(value, "hit_id")
    return text if text.startswith("monitor_hit:") else monitor_hit_id(text)


def _normalize_watch_trigger_uid(value: Any) -> str:
    text = _non_blank(value, "trigger_id")
    return text if text.startswith("watch_trigger:") else watch_trigger_id(text)


def _normalize_monitor_definition_uid(value: Any) -> str:
    text = _non_blank(value, "monitor_id")
    return text if text.startswith("monitor_definition:") else monitor_definition_id(text)


def _normalize_mission_definition_uid(value: Any) -> str:
    text = _non_blank(value, "mission_id")
    return text if text.startswith("mission_definition:") else mission_definition_id(text)


def _entity_type_for_action(action_id: str) -> str:
    if action_id == "update_portfolio_positions":
        return "portfolio_positions"
    if action_id == "update_hedge_positions":
        return "hedge_positions"
    if action_id == "create_action_item":
        return "action_item"
    if action_id in {"complete_action_item", "dismiss_action_item"}:
        return "action_item_status"
    if action_id == "create_watch_trigger":
        return "watch_trigger"
    if action_id in {"cancel_watch_trigger", "fire_watch_trigger"}:
        return "watch_trigger_status"
    if action_id == "replace_watch_trigger":
        return "watch_trigger"
    if action_id == "update_watch_trigger_check":
        return "watch_trigger_check"
    if action_id == "update_watch_trigger_definition":
        return "watch_trigger_definition"
    if action_id == "create_monitor_definition":
        return "monitor_definition"
    if action_id in {"update_monitor_definition", "disable_monitor_definition"}:
        return "monitor_definition_status" if action_id == "disable_monitor_definition" else "monitor_definition"
    if action_id == "create_mission_definition":
        return "mission_definition"
    if action_id in {"update_mission_definition", "disable_mission_definition"}:
        return "mission_definition_status" if action_id == "disable_mission_definition" else "mission_definition"
    if action_id in COURSE_OF_ACTION_CREATE_ACTION_IDS:
        return "course_of_action"
    if action_id == "create_monitor_hit":
        return "monitor_hit"
    if action_id == "update_monitor_hit_status":
        return "monitor_hit_status"
    if action_id == "create_recommendation":
        return "recommendation"
    if action_id == "create_portfolio_news_digest":
        return "news_digest_create"
    if action_id == "delete_portfolio_news_digest":
        return "news_digest_delete"
    if action_id == "create_analyst_feedback":
        return "analyst_feedback"
    if action_id == "finalize_decision_outcome":
        return "decision_outcome"
    if action_id in RESEARCH_ACTION_IDS:
        return "research_object"
    return "ontology_action"


def _target_for_action(action_id: str, payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    ticker = _ticker_from_payload(payload)
    if action_id in {"complete_action_item", "dismiss_action_item"}:
        item_id = payload.get("item_id") or payload.get("id")
        if item_id:
            return _normalize_action_item_uid(item_id), "ActionItem"
    if action_id in WATCH_TRIGGER_TARGET_ACTION_IDS:
        trigger_id = payload.get("trigger_id") or payload.get("id")
        if trigger_id:
            return _normalize_watch_trigger_uid(trigger_id), "WatchTrigger"
    if action_id in {"update_monitor_definition", "disable_monitor_definition"}:
        monitor_id = payload.get("monitor_id") or payload.get("id")
        if monitor_id:
            return _normalize_monitor_definition_uid(monitor_id), "MonitorDefinition"
    if action_id in {"update_mission_definition", "disable_mission_definition"}:
        mission_id = payload.get("mission_id") or payload.get("id")
        if mission_id:
            return _normalize_mission_definition_uid(mission_id), "MissionDefinition"
    if action_id in COURSE_OF_ACTION_CREATE_ACTION_IDS:
        record = _dict(payload.get("record") or payload)
        ticker = _ticker_from_payload(record)
    if ticker and action_id == "save_management_quality_content":
        return management_quality_assessment_id(issuer_id(ticker)), "ManagementQualityAssessment"
    if ticker and action_id == "save_overview_content":
        return equity_overview_id(issuer_id(ticker)), "EquityOverview"
    if ticker and action_id in RESEARCH_ACTION_IDS:
        return thesis_id(ticker), "Thesis"
    if ticker and action_id in {"update_portfolio_positions", "update_hedge_positions"}:
        return portfolio_id(payload.get("portfolio_id") or "default"), "Portfolio"
    if action_id in COURSE_OF_ACTION_CREATE_ACTION_IDS:
        rec = _dict(payload.get("record") or payload)
        return course_of_action_id(_course_of_action_key(rec)), "CourseOfAction"
    if action_id == "delete_portfolio_news_digest":
        digest_id = str(payload.get("digest_id") or "").strip()
        if digest_id:
            return document_artifact_id("news_digest", digest_id), "DocumentArtifact"
    if action_id == "create_analyst_feedback":
        target_uid = str(payload.get("target_object_uid") or "").strip()
        target_type = str(payload.get("target_object_type") or "").strip()
        if target_uid and target_type:
            return target_uid, target_type
    if action_id == "finalize_decision_outcome":
        outcome_uid = str(payload.get("decision_outcome_id") or "").strip()
        if outcome_uid:
            from ontology.schemas.identity import decision_outcome_id as outcome_uid_for

            return outcome_uid_for(outcome_uid), "DecisionOutcome"
    if action_id in {"create_monitor_hit", "update_monitor_hit_status"}:
        entity_id = payload.get("entity_id") or payload.get("hit_id")
        entity_type = str(payload.get("entity_type") or "").strip().lower()
        if entity_id and entity_type == "catalyst":
            return _canonical_object_key(entity_id), "Catalyst"
        if entity_id and entity_type == "kill_condition":
            return _canonical_object_key(entity_id), "KillCondition"
        hit_id = payload.get("hit_id")
        if hit_id:
            return _normalize_monitor_hit_uid(hit_id), "MonitorHit"
    return None, None


def _validate_governed_action(action_id: str, payload: Mapping[str, Any]) -> None:
    if action_id not in FINANCIAL_ACTION_IDS | RESEARCH_ACTION_IDS:
        raise OntologyCommandValidationError(f"Unsupported ontology-primary action: {action_id}")
    if action_id == "update_portfolio_positions" and not _list(payload.get("positions")):
        raise OntologyCommandValidationError("At least one position is required.")
    if action_id in {"save_management_quality_content", "save_overview_content", "save_thesis_content"}:
        _non_blank(payload.get("ticker"), "ticker")
        _non_blank(payload.get("content"), "content")
    if action_id in COURSE_OF_ACTION_CREATE_ACTION_IDS:
        record = _dict(payload.get("record") or payload)
        _non_blank(record.get("action"), "action")
    if action_id == "create_analyst_feedback":
        _non_blank(payload.get("target_object_uid"), "target_object_uid")
        _non_blank(payload.get("target_object_type"), "target_object_type")
        _non_blank(payload.get("decision"), "decision")


def _normalize_create_recommendation_payload(action_id: str, payload: dict[str, Any]) -> None:
    if action_id not in COURSE_OF_ACTION_CREATE_ACTION_IDS:
        return
    record = _dict(payload.get("record") or payload)
    action = normalize_action(record.get("action"))
    record["action"] = action
    record["recommendation_status"] = _recommendation_status_value(record.get("recommendation_status"))
    if action in ACTIONABLE_ACTIONS:
        decision_quality, parse_errors = parse_decision_quality(record.get("decision_quality"))
        gate = apply_decision_quality_gates(
            decision_quality,
            current_action=action,
            recommendation_status=record["recommendation_status"],
            data_quality=_recommendation_data_quality(record),
            parse_errors=parse_errors,
        )
        gate = _add_required_recommendation_gate_reasons(gate, record, action)
        _apply_recommendation_gate(record, gate, decision_quality)
    else:
        record["approval_required"] = False
        if _recommendation_has_blocking_data_quality(record) and record["recommendation_status"] == "clear":
            record["recommendation_status"] = "review_required"
        if isinstance(record.get("decision_quality"), Mapping):
            decision_quality, parse_errors = parse_decision_quality(record.get("decision_quality"))
            gate = apply_decision_quality_gates(
                decision_quality,
                current_action=action,
                recommendation_status=record["recommendation_status"],
                data_quality=_recommendation_data_quality(record),
                parse_errors=parse_errors,
            )
            _apply_recommendation_gate(record, gate, decision_quality)
            record["approval_required"] = False
    payload["record"] = record


def _course_of_action_key(record: Mapping[str, Any]) -> str:
    return str(
        record.get("course_of_action_id")
        or record.get("recommendation_id")
        or record.get("idempotency_key")
        or _stable_hash(record)
    )


def _course_of_action_action(value: Any) -> str:
    normalized = normalize_course_of_action_action(normalize_action(value))
    if normalized == "reduce":
        return "trim"
    if normalized == "hedge":
        return "rebalance"
    if normalized in {"avoid", "do_nothing"}:
        return "watch" if normalized == "avoid" else "hold"
    if normalized not in {"hold", "watch", "buy", "add", "sell", "trim", "exit", "short", "cover", "rebalance"}:
        return "watch"
    return normalized


def _course_of_action_actionability(record: Mapping[str, Any]) -> str:
    action = normalize_action(record.get("action"))
    status = str(record.get("recommendation_status") or record.get("status") or "").strip().lower()
    if status == "blocked":
        return "blocked_by_policy"
    if action in ACTIONABLE_ACTIONS:
        return "actionable"
    if action in {"hold", "do_nothing"}:
        return "do_nothing"
    if action in {"watch", "avoid"}:
        return "watch_only"
    return "missing_inputs"


def _recommendation_status_value(value: Any) -> str:
    status = str(value or "clear").strip().lower() or "clear"
    return status if status in {"clear", "review_required", "blocked", "error"} else "clear"


def _recommendation_data_quality(record: Mapping[str, Any]) -> dict[str, Any]:
    summary = _dict(record.get("source_quality_summary") or record.get("data_quality"))
    critical = (
        record.get("critical_data_quality") or summary.get("critical_data_quality") or summary.get("overall_status")
    )
    source = record.get("source_quality") or summary.get("source_quality") or summary.get("quality")
    return {
        **summary,
        "critical_data_quality": critical,
        "overall_status": summary.get("overall_status") or critical,
        "source_quality": source,
        "quality": summary.get("quality") or source,
    }


def _recommendation_has_source_quality(record: Mapping[str, Any]) -> bool:
    quality = _recommendation_data_quality(record)
    return any(
        str(quality.get(field) or "").strip() for field in ("critical_data_quality", "source_quality", "quality")
    )


def _recommendation_has_blocking_data_quality(record: Mapping[str, Any]) -> bool:
    quality = _recommendation_data_quality(record)
    values = {str(value or "").strip().lower() for value in quality.values()}
    return bool(values & {"stale", "failed"})


def _add_required_recommendation_gate_reasons(
    gate: DecisionQualityGate,
    record: Mapping[str, Any],
    original_action: str,
) -> DecisionQualityGate:
    reasons = list(gate.reasons)
    if not str(record.get("rationale") or record.get("rationale_summary") or "").strip():
        reasons.append(
            DecisionQualityGateReason(
                code="MISSING_RATIONALE",
                severity="blocker",
                message="Actionable recommendations require a rationale before approval review.",
            )
        )
    if not _recommendation_has_source_quality(record):
        reasons.append(
            DecisionQualityGateReason(
                code="MISSING_SOURCE_QUALITY",
                severity="blocker",
                message="Actionable recommendations require explicit source quality before approval review.",
            )
        )
    if reasons == list(gate.reasons):
        return gate
    final_action = gate.final_action
    final_status = gate.final_recommendation_status
    if any(reason.severity == "blocker" for reason in reasons) and original_action in ACTIONABLE_ACTIONS:
        final_action = _recommendation_fallback_action(original_action)
        final_status = "review_required" if final_status == "clear" else final_status
    return gate.model_copy(
        update={
            "status": _recommendation_gate_status(reasons, original_action, final_action),
            "final_action": final_action,
            "final_recommendation_status": final_status,
            "reasons": reasons,
        }
    )


def _apply_recommendation_gate(
    record: dict[str, Any],
    gate: DecisionQualityGate,
    decision_quality: Any,
) -> None:
    record["action"] = gate.final_action
    record["recommendation_status"] = gate.final_recommendation_status
    record["decision_quality"] = decision_quality.model_dump(mode="json") if decision_quality else None
    record["decision_quality_gate"] = gate.model_dump(mode="json")
    record["approval_required"] = bool(gate.final_action in ACTIONABLE_ACTIONS)
    if gate.confidence_cap is not None:
        try:
            raw_confidence = record.get("confidence")
            record["confidence"] = (
                min(float(raw_confidence), gate.confidence_cap) if raw_confidence is not None else gate.confidence_cap
            )
        except (TypeError, ValueError):
            record["confidence"] = gate.confidence_cap


def _recommendation_gate_status(
    reasons: list[DecisionQualityGateReason],
    original_action: str,
    final_action: str,
) -> str:
    if any(reason.severity == "blocker" for reason in reasons):
        return "blocked" if final_action == original_action else "downgraded"
    if reasons:
        return "downgraded" if final_action != original_action else "pass"
    return "pass"


def _recommendation_fallback_action(original_action: str) -> str:
    if original_action in {"short", "sell"}:
        return "avoid"
    if original_action == "do_nothing":
        return "do_nothing"
    return "watch"


def _recommendation_should_persist_without_approval(action_id: str, payload: Mapping[str, Any]) -> bool:
    if action_id not in COURSE_OF_ACTION_CREATE_ACTION_IDS:
        return False
    record = _dict(payload.get("record") or payload)
    return normalize_action(record.get("action")) not in ACTIONABLE_ACTIONS


def _recommendation_policy_gate_blocks(policy_gate_result: Mapping[str, Any] | None) -> bool:
    decision = str((policy_gate_result or {}).get("decision") or "").strip().lower()
    return decision in {"blocked", "error"}


def _downgrade_recommendation_for_policy_gate(
    payload: dict[str, Any],
    policy_gate_result: Mapping[str, Any] | None,
) -> None:
    record = _dict(payload.get("record") or payload)
    action = normalize_action(record.get("action"))
    if action in ACTIONABLE_ACTIONS:
        record["action"] = _recommendation_fallback_action(action)
    record["recommendation_status"] = "blocked"
    record["approval_required"] = False
    blocked_reasons = [str(item) for item in _list(record.get("blocked_reasons")) if str(item).strip()]
    policy_reason = f"policy_gate:{str((policy_gate_result or {}).get('decision') or 'blocked')}"
    if policy_reason not in blocked_reasons:
        blocked_reasons.append(policy_reason)
    record["blocked_reasons"] = blocked_reasons
    payload["record"] = record


def _is_financial_action_for_payload(action_id: str, payload: Mapping[str, Any]) -> bool:
    try:
        from portfolio.policy_gate import is_financial_action

        return is_financial_action(action_id, payload)
    except Exception:
        return action_id in FINANCIAL_ACTION_IDS


def _approval_mode_from_context(context: OntologyCommandContext) -> str:
    request_mode = str(context.request_mode or "").strip().lower()
    if request_mode in {"self_apply", "break_glass"}:
        return request_mode
    return "approval_required"


def _approval_payload_for_action(action_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError as PydanticValidationError

    from ontology.action_registry import ActionValidationError, get_action

    try:
        action = get_action(action_id)
        typed = action.input_model.model_validate(dict(payload))
    except ActionValidationError as exc:
        raise OntologyCommandValidationError(exc.message) from exc
    except PydanticValidationError as exc:
        errors = exc.errors()
        first: Mapping[str, Any]
        if errors:
            first = errors[0]
        else:
            first = {}
        loc = ".".join(str(part) for part in first.get("loc", ()) if part != "__root__")
        msg = str(first.get("msg") or "Invalid action input")
        raise OntologyCommandValidationError(f"{loc}: {msg}" if loc else msg) from exc

    if action.approval_spec and action.approval_spec.payload_builder:
        return cast(dict[str, Any], action.approval_spec.payload_builder(typed))
    return cast(dict[str, Any], typed.model_dump())


def _base_state_hash(action_id: str, payload: Mapping[str, Any]) -> str | None:
    from ontology.action_registry import compute_action_base_state_hash

    try:
        return compute_action_base_state_hash(action_id, dict(payload))
    except Exception:
        return _stable_hash({"action_id": action_id, "payload": dict(payload)})


def _ensure_fresh_base_state(approval: Mapping[str, Any]) -> None:
    stored_hash = str(approval.get("base_state_hash") or "").strip()
    action_id = str(approval.get("action_id") or "").strip()
    proposed_change = _dict(approval.get("proposed_change"))
    if not stored_hash or not action_id or not proposed_change:
        return
    current_hash = _base_state_hash(action_id, proposed_change)
    if current_hash and current_hash != stored_hash:
        raise OntologyCommandConflict(
            "Approval base state changed before application; refresh and create a new proposal"
        )


def _provenance_id(*parts: Any) -> str:
    return "pv:ontology_command:" + _stable_hash(parts)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _refresh_temporal_read_models_after_command() -> None:
    from api.postgres import use_postgres_state
    from ontology.read_model import TemporalReadModelRepository

    fail_closed = use_postgres_state()
    try:
        TemporalReadModelRepository().refresh()
    except Exception:
        if not fail_closed:
            logger.debug("ontology read model refresh skipped outside Postgres state", exc_info=True)
            return
        logger.exception("ontology read model refresh failed after command write")
        raise


def _non_blank(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise OntologyCommandValidationError(f"{field} is required.")
    return text


def _blank_markdown_item(value: Any) -> bool:
    text = str(value or "").strip()
    return not text or text.upper() == "TBD"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _strip_uid_prefix(value: Any, prefix: str) -> str:
    text = str(value or "").strip()
    marker = f"{prefix}:"
    return text.removeprefix(marker).strip() if text.startswith(marker) else text


def _canonical_object_key(value: Any, *, prefix: str | None = None) -> str:
    text = str(value or "").strip()
    if prefix:
        return _strip_uid_prefix(text, prefix)
    return text


def _object_context_by_uid(objects: Any, object_uid: Any) -> dict[str, Any]:
    raw_id = str(object_uid or "").strip()
    if not raw_id:
        return {}
    try:
        row = objects.get_object(raw_id)
    except Exception:
        return {}
    return _flatten_object(row) if row else {}


def _watch_trigger_context(objects: Any, trigger_id: Any) -> dict[str, Any]:
    try:
        return _object_context_by_uid(objects, _normalize_watch_trigger_uid(trigger_id))
    except OntologyCommandValidationError:
        return {}


def _monitor_definition_context(objects: Any, monitor_id: Any) -> dict[str, Any]:
    try:
        return _object_context_by_uid(objects, _normalize_monitor_definition_uid(monitor_id))
    except OntologyCommandValidationError:
        return {}


def _mission_definition_context(objects: Any, mission_id: Any) -> dict[str, Any]:
    try:
        return _object_context_by_uid(objects, _normalize_mission_definition_uid(mission_id))
    except OntologyCommandValidationError:
        return {}


def _merge_definition_payload(existing: Mapping[str, Any], payload: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    monitor_fields = (
        "name",
        "description",
        "template_id",
        "scope",
        "trigger_type",
        "condition",
        "definition",
        "thresholds",
        "source_requirements",
        "cadence",
        "severity",
        "output_policy",
        "approval_behavior",
    )
    mission_fields = (
        "name",
        "description",
        "template_id",
        "scope",
        "mission_type",
        "workflow_name",
        "schedule",
        "source_requirements",
        "thresholds",
        "steps",
        "output_policy",
        "approval_behavior",
    )
    fields = mission_fields if kind == "mission" else monitor_fields
    merged: dict[str, Any] = {}
    for field in fields:
        if field in payload and payload.get(field) is not None:
            value = payload.get(field)
        else:
            value = existing.get(field)
        if value is not None:
            merged[field] = value
    merged.setdefault("template_id", "custom")
    merged.setdefault("scope", {})
    merged.setdefault("thresholds", {})
    merged.setdefault("source_requirements", [])
    merged.setdefault("output_policy", {})
    merged.setdefault("approval_behavior", "hit_only_then_human_review")
    if kind == "mission":
        merged.setdefault("mission_type", "monitor_review")
        merged.setdefault("schedule", {})
        merged.setdefault("steps", [])
    else:
        merged.setdefault("trigger_type", "custom")
        merged.setdefault("definition", {})
        merged.setdefault("cadence", {})
        merged.setdefault("severity", "medium")
    return merged


def _definition_hash(payload: Mapping[str, Any]) -> str:
    hashed = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "updated_at", "last_run_at", "definition_hash", "definition_version"}
    }
    return _stable_hash(hashed)[:24]


def _merge_watch_trigger_context(payload: dict[str, Any], trigger: Mapping[str, Any]) -> None:
    for field in ("condition", "trigger_type", "ticker", "expires_at", "created_at"):
        value = trigger.get(field)
        if value not in (None, ""):
            payload[field] = value
    if "definition" not in payload or payload.get("definition") is None:
        payload["definition"] = _dict(trigger.get("definition"))
    if "last_result" not in payload:
        payload["last_result"] = _dict(trigger.get("last_result"))
    if "last_evidence" not in payload and trigger.get("last_evidence") not in (None, ""):
        payload["last_evidence"] = trigger.get("last_evidence")
    if "last_checked_at" not in payload and trigger.get("last_checked_at") not in (None, ""):
        payload["last_checked_at"] = trigger.get("last_checked_at")
    if "fired_at" not in payload and trigger.get("fired_at") not in (None, ""):
        payload["fired_at"] = trigger.get("fired_at")
    if "status" not in payload and trigger.get("status") not in (None, ""):
        payload["status"] = trigger.get("status")


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _evidence_summary(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("summary", "text", "evidence", "description", "title"):
            if str(value.get(key) or "").strip():
                return str(value.get(key)).strip()
        return None
    text = str(value or "").strip()
    return text or None


def _markdown_sections(content: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_heading: str | None = None
    current_level = 2
    current_lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            marker = stripped.split(" ", 1)[0]
            if marker and set(marker) == {"#"} and len(marker) <= 6 and " " in stripped:
                if current_heading is not None:
                    sections.append(
                        {
                            "heading": current_heading,
                            "level": current_level,
                            "content": "\n".join(current_lines).strip(),
                        }
                    )
                current_level = len(marker)
                current_heading = stripped.split(" ", 1)[1].strip()
                current_lines = []
                continue
        if current_heading is not None:
            current_lines.append(line)
    if current_heading is not None:
        sections.append(
            {
                "heading": current_heading,
                "level": current_level,
                "content": "\n".join(current_lines).strip(),
            }
        )
    return [section for section in sections if section["heading"]]


def _best_effort_index_document(
    doc_type: str,
    content: str,
    ticker: str,
    source_path: str,
    doc_id: str,
    object_uid: str,
    object_version_id: str | None,
) -> None:
    try:
        from api.retrieval import index_document

        index_document(
            doc_type=doc_type,
            content=content,
            ticker=ticker,
            source_path=source_path,
            doc_id=doc_id,
            object_uid=object_uid,
            object_version_id=object_version_id,
        )
    except Exception:
        logger.warning("Failed to index %s document %s", doc_type, doc_id, exc_info=True)


def _ticker_from_payload(payload: Mapping[str, Any]) -> str | None:
    return _optional_ticker(payload)


def _optional_ticker(payload: Mapping[str, Any]) -> str | None:
    raw = payload.get("ticker") or payload.get("instrument")
    text = str(raw or "").strip().upper()
    return text or None
