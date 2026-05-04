"""Backfill legacy operational stores into temporal ontology tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from ontology.domain_write_service import OPERATIONAL_ONTOLOGY_RUN_ID, DomainOntologyWriteService, OntologyMutation
from ontology.object_service import object_uid_for
from ontology.schemas.identity import (
    action_run_id,
    approval_id,
    audit_event_id,
    executed_action_id,
    policy_gate_result_id,
    recommendation_id,
    report_run_id,
    thesis_id,
    trade_proposal_id,
    workflow_artifact_id,
    workflow_run_id,
)
from ontology.temporal_repository import SourceRecordWrite, TemporalOntologyRepository, payload_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill legacy operational DB rows into temporal ontology.")
    parser.add_argument(
        "--cutover-time",
        default=None,
        help="Transaction-time timestamp for backfilled rows. Defaults to current UTC time.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect row counts without writing temporal rows.")
    args = parser.parse_args()

    cutover_time = args.cutover_time or datetime.now(UTC).isoformat()
    summary = backfill_operational_legacy_state(cutover_time=cutover_time, dry_run=bool(args.dry_run))
    print(json.dumps(summary, sort_keys=True, default=str))


def backfill_operational_legacy_state(*, cutover_time: str, dry_run: bool = False) -> dict[str, Any]:
    repo = TemporalOntologyRepository()
    service = DomainOntologyWriteService()
    counts: dict[str, int] = {}

    def write(table: str, record_kind: str, record_key: str, object_type: str, business_key: str, row: dict) -> None:
        valid_from = _valid_from(row, cutover_time)
        counts[f"{table}:seen"] = counts.get(f"{table}:seen", 0) + 1
        if dry_run:
            return
        source = repo.write_source_record_version(
            SourceRecordWrite(
                vendor="legacy",
                source_name=table,
                source_version="1",
                dataset=table,
                record_kind=record_kind,
                record_key=record_key,
                payload_hash=payload_hash(row),
                payload=row,
                valid_from=valid_from,
                tx_from=cutover_time,
                status="ok",
                quality="reconstructed",
            )
        )
        result = service.write_object(
            OntologyMutation(object_type, business_key, row, valid_from),
            actor={"actor_type": "system", "actor_id": "ontology.backfill_operational"},
            temporal_confidence="backfilled",
            source_record_id=str(source["source_record_id"]),
        )
        counts[f"{object_type}:written"] = counts.get(f"{object_type}:written", 0) + 1
        _write_owner_relations(service, object_type, row, result, valid_from=valid_from)

    for row in _portfolio_rows():
        role = str(row.get("role") or "position")
        object_type = "HedgePosition" if role == "hedge" else "Position"
        write(
            "positions",
            role,
            f"legacy:positions:{row.get('ticker')}:{role}",
            object_type,
            str(row.get("ticker") or ""),
            _position_row(row, role=role),
        )

    for row in _thesis_meta_rows():
        write("thesis_meta", "thesis", f"legacy:thesis_meta:{row.get('ticker')}", "Thesis", str(row.get("ticker")), row)

    for row in _thesis_evaluation_rows():
        write(
            "thesis_evaluations",
            "evaluation",
            f"legacy:thesis_evaluations:{row.get('ticker')}:{row.get('evaluated_at')}",
            "Evaluation",
            f"{row.get('ticker')}-{row.get('evaluated_at')}",
            _evaluation_row(row),
        )

    core_mappings: tuple[tuple[str, str, str, str, Any], ...] = (
        ("catalysts", "catalyst", "Catalyst", "description", _catalyst_row),
        ("kill_conditions", "kill_condition", "KillCondition", "id", _legacy_id_row),
        ("thesis_claims", "thesis_claim", "ThesisClaim", "id", _thesis_claim_row),
        ("action_items", "action_item", "ActionItem", "id", _legacy_id_row),
        ("watch_triggers", "watch_trigger", "WatchTrigger", "id", _watch_trigger_row),
        ("research_notes", "research_note", "ResearchNote", "id", _legacy_id_row),
        ("pending_approvals", "approval", "Approval", "id", _pending_approval_row),
        ("action_runs", "action_run", "ActionRun", "id", _action_run_row),
        ("action_events", "action_event", "ActionEvent", "id", _action_event_row),
        ("workflow_runs", "workflow_run", "WorkflowRun", "run_id", _workflow_run_row),
        ("workflow_artifact_records", "workflow_artifact", "WorkflowArtifact", "artifact_id", _workflow_artifact_row),
        ("report_runs", "report_run", "ReportRun", "report_id", _report_run_row),
        ("recommendations", "recommendation", "Recommendation", "id", _recommendation_row),
        ("policy_gate_results", "policy_gate_result", "PolicyGateResult", "id", _policy_gate_result_row),
        ("audit_events", "audit_event", "AuditEvent", "event_id", _audit_event_row),
        ("source_record_refs", "source_record", "SourceRecord", "record_ref_id", _source_record_ref_row),
    )
    for table, record_kind, object_type, key_field, mapper in core_mappings:
        for row in _core_table_rows(table):
            mapped = mapper(row)
            record_key = f"legacy:{table}:{row.get(key_field)}"
            write(table, record_kind, record_key, object_type, str(row.get(key_field) or ""), mapped)

    for row in _core_table_rows("recommendations"):
        proposal = _json_value(row.get("trade_proposal_json"), default={})
        if not proposal:
            continue
        counts["trade_proposals:seen"] = counts.get("trade_proposals:seen", 0) + 1
        if dry_run:
            continue
        proposal_id = str(row.get("idempotency_key") or row.get("id") or _stable_hash(proposal))
        result = service.write_object(
            OntologyMutation(
                "TradeProposal",
                proposal_id,
                _trade_proposal_row(row, proposal_id=proposal_id, proposal=proposal),
                _valid_from(row, cutover_time),
            ),
            actor={"actor_type": "system", "actor_id": "ontology.backfill_operational"},
            provenance_event_id=f"pv:backfill:trade_proposal:{proposal_id}",
            temporal_confidence="backfilled",
        )
        counts["TradeProposal:written"] = counts.get("TradeProposal:written", 0) + 1
        if result:
            _write_trade_proposal_relations(service, row, proposal_id, valid_from=_valid_from(row, cutover_time))

    for row in _core_table_rows("action_runs"):
        if not row.get("approval_id") or str(row.get("status") or "") != "succeeded":
            continue
        counts["executed_actions:seen"] = counts.get("executed_actions:seen", 0) + 1
        if dry_run:
            continue
        executed_id = f"{row.get('approval_id')}:{row.get('id')}:{row.get('action_id')}"
        service.write_object(
            OntologyMutation(
                "ExecutedAction",
                executed_id,
                _executed_action_row(row, executed_id=executed_id),
                _valid_from(row, cutover_time),
            ),
            actor={"actor_type": "system", "actor_id": "ontology.backfill_operational"},
            provenance_event_id=f"pv:backfill:executed_action:{executed_id}",
            action_run_id_value=_optional_int(row.get("id")),
            approval_id=_optional_int(row.get("approval_id")),
            temporal_confidence="backfilled",
        )
        counts["ExecutedAction:written"] = counts.get("ExecutedAction:written", 0) + 1
        _write_executed_action_relations(service, row, executed_id, valid_from=_valid_from(row, cutover_time))

    if not dry_run:
        _write_decision_relations(service, valid_from=cutover_time, counts=counts)

    counts["cutover_time"] = cutover_time  # type: ignore[assignment]
    counts["dry_run"] = int(bool(dry_run))
    return counts


def _write_owner_relations(
    service: DomainOntologyWriteService,
    object_type: str,
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    valid_from: str,
) -> None:
    object_uid = _result_object_uid(result)
    ticker = str(row.get("ticker") or "").strip().upper()
    if not object_uid or not ticker:
        return
    relation_type: str | None = None
    if object_type == "Evaluation":
        relation_type = "evaluated_by"
    elif object_type == "Catalyst":
        relation_type = "has_catalyst"
    elif object_type == "KillCondition":
        relation_type = "thesis_has_kill_condition"
    elif object_type == "ThesisClaim":
        relation_type = "thesis_has_claim"
    if relation_type is None:
        return
    service.object_service.write_relation(
        thesis_id(ticker),
        object_uid,
        relation_type,
        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
        valid_from,
        actor={"actor_type": "system", "actor_id": "ontology.backfill_operational"},
    )


def _portfolio_rows() -> Iterable[dict[str, Any]]:
    from portfolio.portfolio_db import get_positions

    return [dict(row) for row in get_positions(include_hedges=True)]


def _thesis_meta_rows() -> Iterable[dict[str, Any]]:
    from portfolio.thesis_db import get_all_thesis_meta

    return [dict(row) for row in get_all_thesis_meta()]


def _thesis_evaluation_rows() -> Iterable[dict[str, Any]]:
    from portfolio import thesis_db

    conn = thesis_db._get_conn()  # type: ignore[attr-defined]
    with thesis_db._lock:  # type: ignore[attr-defined]
        rows = conn.execute("SELECT * FROM thesis_evaluations ORDER BY ticker, evaluated_at").fetchall()
    return [_row(row) for row in rows]


def _core_table_rows(table: str) -> Iterable[dict[str, Any]]:
    from portfolio import core_db

    conn = core_db._get_conn()  # type: ignore[attr-defined]
    with core_db._lock:  # type: ignore[attr-defined]
        try:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        except Exception:
            return []
    return [_row(row) for row in rows]


def _position_row(row: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    out = _identity_row(row)
    out["role"] = role
    out.setdefault("timeframe", "operational")
    out.setdefault("risk_score", 0.0)
    out.setdefault("risk_level", "low")
    out.setdefault("volatility_cluster", 0.0)
    out.setdefault("breadth_stress", 0.0)
    out.setdefault("sector_stress", 0.0)
    out.setdefault("macro_regime", 0.0)
    return out


def _evaluation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    out["key_developments"] = _json_value(out.pop("key_developments", None), default=[])
    return out


def _catalyst_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    out["legacy_id"] = row.get("id")
    out["name"] = str(out.get("name") or out.get("description") or "Catalyst")[:120]
    out["source"] = str(out.get("created_by") or "legacy")
    return out


def _thesis_claim_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    out["legacy_id"] = row.get("id")
    out["source_requirements"] = _json_value(out.pop("source_requirements_json", None), default=[])
    out["linked_catalyst_ids"] = _json_value(out.pop("linked_catalyst_ids_json", None), default=[])
    out["linked_kill_condition_ids"] = _json_value(out.pop("linked_kill_condition_ids_json", None), default=[])
    return out


def _watch_trigger_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    out["legacy_id"] = row.get("id")
    out["definition"] = _json_value(out.pop("definition_json", None), default=None)
    out["last_result"] = _json_value(out.pop("last_result_json", None), default=None)
    return out


def _pending_approval_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    return {
        "legacy_id": row.get("id"),
        "entity_type": out.get("entity_type") or "approval",
        "entity_id": out.get("entity_id"),
        "ticker": out.get("ticker"),
        "action_id": out.get("action_id"),
        "action_schema_name": out.get("action_schema_name"),
        "action_schema_version": out.get("action_schema_version"),
        "action_input_hash": out.get("action_input_hash"),
        "proposed_change": _json_value(out.get("proposed_change"), default={}),
        "reason": out.get("reason"),
        "source_type": out.get("source_type"),
        "source_id": out.get("source_id"),
        "status": out.get("status") or "pending",
        "application_status": out.get("application_status"),
        "created_at": out.get("created_at"),
        "resolved_at": out.get("resolved_at"),
        "resolved_note": out.get("resolved_note"),
    }


def _action_run_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    return {
        "legacy_id": row.get("id"),
        "action_id": out.get("action_id") or "unknown",
        "action_schema_name": out.get("action_schema_name"),
        "action_schema_version": out.get("action_schema_version") or 1,
        "actor_type": out.get("actor_type") or "unknown",
        "actor_id": out.get("actor_id"),
        "source_type": out.get("source_type"),
        "source_id": out.get("source_id"),
        "approval_id": out.get("approval_id"),
        "parent_action_run_id": out.get("parent_action_run_id"),
        "input_hash": out.get("input_hash"),
        "status": out.get("status") or "running",
        "error": out.get("error"),
        "started_at": out.get("started_at"),
        "completed_at": out.get("completed_at"),
        "provenance_event_id": out.get("provenance_event_id"),
    }


def _action_event_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    out["legacy_id"] = row.get("id")
    out["payload"] = _json_value(out.pop("payload_json", None), default=None)
    return out


def _workflow_run_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    out["artifacts"] = _json_value(out.get("artifacts"), default=None)
    out["tool_sections"] = _json_value(out.get("tool_sections"), default=None)
    return out


def _workflow_artifact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    return {
        "artifact_id": out.get("artifact_id") or "unknown",
        "workflow_run_id": out.get("workflow_run_id"),
        "artifact_key": out.get("artifact_key") or "artifact",
        "artifact_index": out.get("artifact_index"),
        "artifact_value": _json_value(out.get("summary_json"), default=None),
        "approval_id": out.get("approval_id"),
        "provenance_event_id": out.get("provenance_event_id"),
    }


def _report_run_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    out["summary"] = _json_value(out.pop("summary_json", None), default=None)
    out["artifact_paths"] = _json_value(out.pop("artifact_paths_json", None), default=None)
    return out


def _recommendation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    action = out.get("action") or "do_nothing"
    return {
        "recommendation_id": row.get("id") or out.get("idempotency_key"),
        "legacy_id": row.get("id"),
        "idempotency_key": out.get("idempotency_key"),
        "source_kind": "report",
        "report_type": out.get("report_type"),
        "as_of": out.get("as_of"),
        "action": action,
        "ticker": out.get("ticker"),
        "instrument": out.get("instrument"),
        "decision_state": "proposed" if out.get("approval_id") else "generated",
        "status": out.get("status"),
        "approval_id": out.get("approval_id"),
        "approval_required": action in {"buy", "sell", "reduce", "exit", "rebalance", "hedge"},
        "approval_status": out.get("approval_status"),
        "outcome_status": out.get("outcome_status"),
        "payload": {key: value for key, value in out.items() if key not in {"schema_version"}},
    }


def _policy_gate_result_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_value(row.get("result_json"), default={})
    return {
        "gate_result_id": str(row.get("id") or result.get("policy_gate_result_id") or _stable_hash(result)),
        "decision": result.get("decision") or row.get("decision") or "error",
        "review_required": bool(row.get("review_required") or result.get("review_required")),
        "failure_reasons": _json_value(result.get("failure_reasons"), default=[]),
        "warnings": _json_value(result.get("warnings"), default=[]),
        "account_id": row.get("account_id") or result.get("account_id"),
        "portfolio_id": row.get("portfolio_id") or result.get("portfolio_id"),
        "policy_id": row.get("policy_id") or result.get("policy_id"),
        "evaluated_at": row.get("created_at") or result.get("evaluated_at"),
    }


def _audit_event_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    return {
        "event_id": row.get("event_id") or row.get("id") or _stable_hash(row),
        "occurred_at": out.get("occurred_at"),
        "actor_type": out.get("actor_type") or "system",
        "actor_id": out.get("actor_id"),
        "action_name": out.get("action_name") or "unknown",
        "action_category": out.get("action_category") or "unknown",
        "status": out.get("status") or "unknown",
        "object_refs": _json_value(out.get("object_refs_json"), default=[]),
        "before_summary": _json_value(out.get("before_summary_json"), default=None),
        "after_summary": _json_value(out.get("after_summary_json"), default=None),
        "source_lineage": _json_value(out.get("source_lineage_json"), default=None),
        "metadata": _json_value(out.get("metadata_json"), default=None),
        "lineage_root_id": out.get("lineage_root_id"),
        "retention_class": out.get("retention_class") or "audit_365d",
    }


def _source_record_ref_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_record_id": row.get("record_ref_id") or _stable_hash(row),
        "vendor": "legacy",
        "source_name": row.get("source_name") or "legacy",
        "source_version": "1",
        "dataset": row.get("source_name") or "legacy",
        "record_kind": row.get("record_kind") or "record",
        "record_key_hash": row.get("record_key_hash") or _stable_hash(row.get("record_ref_id")),
        "payload_hash": row.get("record_hash") or _stable_hash(row),
        "status": "ok",
        "quality": "reconstructed",
        "as_of": row.get("as_of"),
        "load_time": row.get("created_at"),
        "provenance_event_id": row.get("adapter_run_event_id"),
    }


def _trade_proposal_row(
    row: Mapping[str, Any],
    *,
    proposal_id: str,
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "recommendation_id": row.get("id") or row.get("idempotency_key"),
        "account_id": row.get("account_id"),
        "portfolio_id": row.get("portfolio_id"),
        "action": row.get("action") or proposal.get("action") or "review",
        "instrument": row.get("instrument") or row.get("ticker") or "portfolio",
        "proposed_change": dict(proposal),
        "sizing_summary": _json_value(proposal.get("sizing_summary"), default={})
        if isinstance(proposal, Mapping)
        else {},
        "risk_summary": _json_value(proposal.get("risk_summary"), default={}) if isinstance(proposal, Mapping) else {},
        "policy_gate_result_id": _optional_int(row.get("policy_gate_result_id")),
        "approval_id": _optional_int(row.get("approval_id")),
        "decision_state": "pending_approval" if row.get("approval_id") else "staged",
        "status": "pending_approval" if row.get("approval_id") else "staged",
    }


def _executed_action_row(row: Mapping[str, Any], *, executed_id: str) -> dict[str, Any]:
    return {
        "executed_action_id": executed_id,
        "action_id": row.get("action_id") or "unknown",
        "approval_id": _optional_int(row.get("approval_id")),
        "action_run_id": _optional_int(row.get("id")),
        "execution_mode": "approval_required",
        "produced_object_versions": [],
        "mutated_object_versions": [],
        "applied_at": row.get("completed_at") or row.get("started_at"),
        "status": "applied" if row.get("status") == "succeeded" else str(row.get("status") or "unknown"),
    }


def _write_decision_relations(
    service: DomainOntologyWriteService,
    *,
    valid_from: str,
    counts: dict[str, int],
) -> None:
    for row in _core_table_rows("recommendations"):
        recommendation_uid = recommendation_id(row.get("id") or row.get("idempotency_key"))
        report_id = row.get("report_id")
        if report_id:
            _write_relation(
                service,
                report_run_id(report_id),
                recommendation_uid,
                "report_run_produces_recommendation",
                valid_from,
                counts,
            )
        gate_id = row.get("policy_gate_result_id")
        if gate_id:
            _write_relation(
                service,
                policy_gate_result_id(gate_id),
                recommendation_uid,
                "policy_gate_evaluates_recommendation",
                valid_from,
                counts,
            )
        approval = row.get("approval_id")
        if approval:
            _write_relation(
                service,
                approval_id(approval),
                recommendation_uid,
                "approval_targets_recommendation",
                valid_from,
                counts,
                approval_id_value=_optional_int(approval),
            )

    for row in _core_table_rows("workflow_artifact_records"):
        artifact = row.get("artifact_id")
        run_id = row.get("workflow_run_id")
        approval = row.get("approval_id")
        if artifact and run_id:
            _write_relation(
                service,
                workflow_run_id(run_id),
                workflow_artifact_id(artifact),
                "workflow_run_produces_artifact",
                valid_from,
                counts,
                properties={"artifact_key": row.get("artifact_key")},
            )
        if artifact and approval:
            _write_relation(
                service,
                workflow_artifact_id(artifact),
                approval_id(approval),
                "workflow_artifact_proposes_approval",
                valid_from,
                counts,
                approval_id_value=_optional_int(approval),
                properties={"approval_id": str(approval), "artifact_key": row.get("artifact_key")},
            )
            _write_relation(
                service,
                approval_id(approval),
                workflow_artifact_id(artifact),
                "approval_targets_workflow_artifact",
                valid_from,
                counts,
                approval_id_value=_optional_int(approval),
            )

    for row in _core_table_rows("action_runs"):
        approval = row.get("approval_id")
        run_id = row.get("id")
        if approval and run_id:
            _write_relation(
                service,
                approval_id(approval),
                action_run_id(run_id),
                "approval_applies_action_run",
                valid_from,
                counts,
                action_run_id_value=_optional_int(run_id),
                approval_id_value=_optional_int(approval),
            )

    for row in _core_table_rows("audit_events"):
        event = row.get("event_id") or row.get("id")
        refs = _json_value(row.get("object_refs_json"), default=[])
        for ref in refs if isinstance(refs, list) else []:
            if isinstance(ref, Mapping) and ref.get("type") == "action_run" and ref.get("id"):
                _write_relation(
                    service,
                    audit_event_id(event),
                    action_run_id(ref["id"]),
                    "audit_event_observes_action_run",
                    valid_from,
                    counts,
                )


def _write_trade_proposal_relations(
    service: DomainOntologyWriteService,
    row: Mapping[str, Any],
    proposal_id: str,
    *,
    valid_from: str,
) -> None:
    counts: dict[str, int] = {}
    proposal_uid = trade_proposal_id(proposal_id)
    recommendation = row.get("id") or row.get("idempotency_key")
    if recommendation:
        _write_relation(
            service,
            proposal_uid,
            recommendation_id(recommendation),
            "trade_proposal_derives_from_recommendation",
            valid_from,
            counts,
        )
    if row.get("approval_id"):
        _write_relation(
            service,
            proposal_uid,
            approval_id(row["approval_id"]),
            "trade_proposal_requires_approval",
            valid_from,
            counts,
            approval_id_value=_optional_int(row.get("approval_id")),
            properties={"approval_id": str(row.get("approval_id"))},
        )
    if row.get("policy_gate_result_id"):
        _write_relation(
            service,
            policy_gate_result_id(row["policy_gate_result_id"]),
            proposal_uid,
            "policy_gate_evaluates_trade_proposal",
            valid_from,
            counts,
        )


def _write_executed_action_relations(
    service: DomainOntologyWriteService,
    row: Mapping[str, Any],
    executed_id: str,
    *,
    valid_from: str,
) -> None:
    counts: dict[str, int] = {}
    if row.get("id"):
        _write_relation(
            service,
            action_run_id(row["id"]),
            executed_action_id(executed_id),
            "action_run_produces_executed_action",
            valid_from,
            counts,
            action_run_id_value=_optional_int(row.get("id")),
            approval_id_value=_optional_int(row.get("approval_id")),
        )
    if row.get("approval_id") and row.get("id"):
        _write_relation(
            service,
            approval_id(row["approval_id"]),
            action_run_id(row["id"]),
            "approval_applies_action_run",
            valid_from,
            counts,
            action_run_id_value=_optional_int(row.get("id")),
            approval_id_value=_optional_int(row.get("approval_id")),
        )


def _write_relation(
    service: DomainOntologyWriteService,
    source_uid: str,
    target_uid: str,
    relation_type: str,
    valid_from: str,
    counts: dict[str, int],
    *,
    properties: Mapping[str, Any] | None = None,
    action_run_id_value: int | None = None,
    approval_id_value: int | None = None,
) -> None:
    payload = {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, **dict(properties or {})}
    try:
        service.object_service.write_relation(
            source_uid,
            target_uid,
            relation_type,
            payload,
            valid_from,
            actor={"actor_type": "system", "actor_id": "ontology.backfill_operational"},
            provenance=f"pv:backfill:relation:{relation_type}:{_stable_hash(source_uid + target_uid)}",
            action_run_id=action_run_id_value,
            approval_id=approval_id_value,
            temporal_confidence="backfilled",
        )
        counts[f"{relation_type}:written"] = counts.get(f"{relation_type}:written", 0) + 1
    except Exception:
        counts[f"{relation_type}:failed"] = counts.get(f"{relation_type}:failed", 0) + 1


def _identity_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in dict(row).items() if value is not None and str(key) != "id"}


def _legacy_id_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _identity_row(row)
    out["legacy_id"] = row.get("id")
    return out


def _valid_from(row: Mapping[str, Any], fallback: str) -> str:
    for key in ("valid_from", "updated_at", "created_at", "started_at", "evaluated_at", "as_of", "synced_at"):
        value = row.get(key)
        if value:
            return str(value)
    return fallback


def _json_value(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _row(row: Any) -> dict[str, Any]:
    return dict(row)


def _result_object_uid(row: Mapping[str, Any]) -> str | None:
    if row.get("object_uid"):
        return str(row["object_uid"])
    meta = row.get("_meta")
    if isinstance(meta, Mapping):
        temporal = meta.get("temporal")
        if isinstance(temporal, Mapping) and temporal.get("object_uid"):
            return str(temporal["object_uid"])
    object_type = str(row.get("object_type") or "")
    properties = row.get("properties")
    if object_type and isinstance(properties, Mapping):
        return object_uid_for(object_type, str(row.get("business_key") or ""), properties)
    return None


if __name__ == "__main__":
    main()
