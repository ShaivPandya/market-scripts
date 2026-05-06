"""Decision-centered ontology writeback helpers.

The functions here are intentionally safe to call from legacy report, workflow,
and approval paths. They no-op unless ontology shadow or primary writes are
enabled, and they raise only when primary writes are configured.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ontology.domain_write_service import (
    OPERATIONAL_ONTOLOGY_RUN_ID,
    ontology_primary_writes_enabled,
    ontology_shadow_writes_enabled,
)
from ontology.object_service import OntologyObjectService
from ontology.schemas.identity import (
    account_id,
    asset_id,
    object_version_ref_id,
    policy_gate_result_id,
    portfolio_id,
    report_run_id,
    risk_metric_id,
    scenario_id,
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

logger = logging.getLogger(__name__)

ACTIONABLE_RECOMMENDATION_ACTIONS = {"buy", "sell", "reduce", "exit", "rebalance", "hedge"}


class DecisionOntologyWriteback:
    """Facade for governed decision artifact writes."""

    def __init__(self, object_service: OntologyObjectService | None = None):
        self.object_service = object_service or OntologyObjectService()

    def enabled(self) -> bool:
        return ontology_shadow_writes_enabled()

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
            record.get("legacy_id")
            or record.get("id")
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


def _recommendation_key(record: Mapping[str, Any]) -> str:
    return str(
        record.get("id")
        or record.get("legacy_id")
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
        "legacy_id": _optional_int(record.get("id") or record.get("legacy_id")),
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
    if ontology_primary_writes_enabled():
        raise exc
    logger.warning("Decision ontology writeback failed in %s: %s", surface, exc, exc_info=True)


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
