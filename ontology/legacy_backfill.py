"""Migration-only reader for legacy SQLite exports.

This module is intentionally not a runtime compatibility layer. It exists so a
maintenance-window cutover can read old SQLite exports, write audited ontology
objects, and then deploy the Postgres-only runtime.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ontology.command_service import OntologyCommandContext, OntologyCommandService
from ontology.object_service import OntologyObjectService
from ontology.policy import system_actor
from ontology.schemas.identity import provenance_event_id, provenance_link_id
from ontology.temporal_repository import SnapshotVersionWrite, TemporalOntologyRepository, payload_hash


class LegacyBackfillDisabled(RuntimeError):
    pass


def _enabled() -> bool:
    return (os.getenv("TALISMAN_ENABLE_LEGACY_BACKFILL") or "").strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    if not _enabled():
        raise LegacyBackfillDisabled(
            "Legacy SQLite reads are allowed only for maintenance-window backfill. "
            "Set TALISMAN_ENABLE_LEGACY_BACKFILL=true in the migration job."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def backfill_audit_minimum(
    *,
    portfolio_db_path: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill current portfolio positions through ontology command service.

    Additional audit-minimum domains should be fed through the same command
    boundary, not by importing legacy runtime modules.
    """

    positions = _read_positions(Path(portfolio_db_path))
    if dry_run:
        return {"dry_run": True, "positions": len(positions)}
    service = OntologyCommandService()
    context = OntologyCommandContext(
        actor=system_actor("legacy_backfill"),
        source_type="migration",
        source_id="legacy_backfill.audit_minimum",
    )
    approval = service.propose_action(
        "update_portfolio_positions",
        {"positions": positions},
        context,
        reason="Audit-minimum maintenance-window backfill from legacy portfolio export.",
    )
    applied = service.resolve_approval(approval["id"], "approved", "Approved migration backfill.", context)
    return {"dry_run": False, "positions": len(positions), "approval_id": applied["id"]}


def backfill_runtime_objects(
    *,
    core_db_path: str | Path,
    portfolio_db_path: str | Path | None = None,
    thesis_db_path: str | Path | None = None,
    snapshot_db_path: str | Path | None = None,
    dry_run: bool = False,
    provenance_event_id_value: str = "pv:legacy_backfill:runtime_objects",
) -> dict[str, Any]:
    """Backfill current legacy runtime objects into temporal ontology versions.

    This is cutover-only scaffolding. It reads a legacy SQLite export under the
    explicit `TALISMAN_ENABLE_LEGACY_BACKFILL` gate and writes first-class
    ontology objects with `temporal_confidence='backfilled'`.
    """

    path = Path(core_db_path)
    mutations: list[tuple[str, str, dict[str, Any], str]] = []
    link_relations: list[tuple[str, str, dict[str, Any], str]] = []
    snapshot_versions: list[SnapshotVersionWrite] = []
    cutover_time = _now()
    with _connect(path) as conn:
        mutations.extend(_runtime_object_rows(conn, cutover_time=cutover_time))
        link_relations.extend(_provenance_link_relation_rows(conn, cutover_time=cutover_time))
    if portfolio_db_path is not None:
        with _connect(Path(portfolio_db_path)) as conn:
            mutations.extend(_portfolio_rows(conn, cutover_time=cutover_time))
    if thesis_db_path is not None:
        with _connect(Path(thesis_db_path)) as conn:
            mutations.extend(_thesis_rows(conn, cutover_time=cutover_time))
    if snapshot_db_path is not None:
        with _connect(Path(snapshot_db_path)) as conn:
            snapshot_versions.extend(
                _snapshot_versions(conn, cutover_time=cutover_time, provenance_event_id_value=provenance_event_id_value)
            )
    counts: dict[str, int] = {}
    for object_type, *_ in mutations:
        counts[object_type] = counts.get(object_type, 0) + 1
    if dry_run:
        return {
            "dry_run": True,
            "objects": counts,
            "relations": len(link_relations),
            "computed_snapshots": len(snapshot_versions),
        }

    service = OntologyObjectService()
    actor = {"actor_type": "system", "actor_id": "legacy_backfill"}
    service.write_object(
        "ProvenanceEvent",
        provenance_event_id_value,
        {
            "event_id": provenance_event_id_value,
            "event_type": "legacy_backfill",
            "event_name": "runtime_objects",
            "status": "succeeded",
            "started_at": cutover_time,
            "completed_at": cutover_time,
            "actor_type": "system",
            "actor_id": "legacy_backfill",
            "criticality": "operational",
            "retention_class": "financial_lineage_7y",
        },
        cutover_time,
        actor=actor,
        provenance=provenance_event_id_value,
        temporal_confidence="backfilled",
    )
    written = 0
    for object_type, business_key, properties, valid_from in mutations:
        service.write_object(
            object_type,
            business_key,
            properties,
            valid_from,
            actor=actor,
            provenance=provenance_event_id_value,
            temporal_confidence="backfilled",
        )
        written += 1
    for source_uid, target_uid, properties, valid_from in link_relations:
        service.write_relation(
            source_uid,
            target_uid,
            "provenance_event_records_link",
            properties,
            valid_from,
            actor=actor,
            provenance=provenance_event_id_value,
            temporal_confidence="backfilled",
        )
    repository = TemporalOntologyRepository()
    for snapshot_version in snapshot_versions:
        repository.write_computed_snapshot_version(snapshot_version)
    return {
        "dry_run": False,
        "objects": counts,
        "object_versions": written,
        "relations": len(link_relations),
        "computed_snapshots": len(snapshot_versions),
    }


def _read_positions(path: Path) -> list[dict[str, Any]]:
    with _connect(path) as conn:
        rows = conn.execute("SELECT * FROM positions ORDER BY ticker").fetchall()
    return [_jsonable(dict(row)) for row in rows]


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    out[key] = json.loads(stripped)
                    continue
                except json.JSONDecodeError:
                    pass
        out[key] = value
    return out


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


_JSON_FIELD_DEFAULTS: dict[str, Any] = {
    "artifact_paths": {},
    "artifacts": {},
    "before_summary": None,
    "after_summary": None,
    "blocked_reasons": [],
    "data_quality": {},
    "definition": {},
    "disclosures": [],
    "disconfirming_evidence": [],
    "evidence": {},
    "factor_scores": {},
    "failures": [],
    "last_result": {},
    "linked_catalyst_ids": [],
    "linked_kill_condition_ids": [],
    "metadata": {},
    "missing_information": [],
    "object_refs": [],
    "payload": {},
    "policy_gate_result": {},
    "portfolio_fit": {},
    "proposed_change": {},
    "rankings": [],
    "raw_result": {},
    "recommendation_record": {},
    "result": {},
    "scope_statuses": [],
    "source_config": {},
    "source_freshness": {},
    "source_lineage": None,
    "source_links": {},
    "source_requirements": [],
    "summary": {},
    "thresholds": {},
    "tool_sections": [],
    "warnings": [],
}

_LEGACY_TABLES = {
    "action_events",
    "action_items",
    "action_runs",
    "audit_events",
    "catalysts",
    "computed_snapshots",
    "idea_comparison_rankings",
    "idea_comparison_runs",
    "idea_evaluations",
    "investment_ideas",
    "kill_conditions",
    "optimization_action_snapshots",
    "optimization_alerts",
    "optimization_missions",
    "optimization_runs",
    "pending_approvals",
    "policy_gate_results",
    "positions",
    "provenance_events",
    "provenance_links",
    "recommendations",
    "report_runs",
    "research_notes",
    "source_record_refs",
    "thesis_claims",
    "thesis_evaluations",
    "thesis_meta",
    "thesis_status_history",
    "watch_triggers",
    "workflow_artifact_records",
    "workflow_runs",
}


def _runtime_object_rows(conn: sqlite3.Connection, *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    rows.extend(_optimization_rows(conn, cutover_time=cutover_time))
    rows.extend(_idea_rows(conn, cutover_time=cutover_time))
    rows.extend(_provenance_rows(conn, cutover_time=cutover_time))
    rows.extend(_core_operational_rows(conn, cutover_time=cutover_time))
    return rows


def _optimization_rows(conn: sqlite3.Connection, *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    for row in _select_all(conn, "optimization_missions"):
        props = _rename_json_fields(
            row, {"scenario_json": "scenario", "source_config_json": "source_config", "thresholds_json": "thresholds"}
        )
        rows.append(
            ("OptimizationMission", f"optimization_mission:{props.get('id')}", props, _valid_from(props, cutover_time))
        )
    for row in _select_all(conn, "optimization_runs"):
        props = _rename_json_fields(row, {"summary_json": "summary", "source_freshness_json": "source_freshness"})
        rows.append(("OptimizationRun", str(props.get("run_id")), props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "optimization_action_snapshots"):
        props = _rename_json_fields(row, {"evidence_json": "evidence", "source_links_json": "source_links"})
        rows.append(
            (
                "OptimizationActionSnapshot",
                f"optimization_action_snapshot:{props.get('id')}",
                props,
                _valid_from(props, cutover_time),
            )
        )
    for row in _select_all(conn, "optimization_alerts"):
        props = _rename_json_fields(row, {"evidence_json": "evidence"})
        if "dismissed_note" in props:
            props["dismissal_note"] = props.pop("dismissed_note")
        rows.append(
            ("OptimizationAlert", f"optimization_alert:{props.get('id')}", props, _valid_from(props, cutover_time))
        )
    return rows


def _idea_rows(conn: sqlite3.Connection, *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    for row in _select_all(conn, "investment_ideas"):
        props = _rename_json_fields(row, {"tags_json": "tags", "metadata_json": "metadata"})
        rows.append(("InvestmentIdea", f"investment_idea:{props.get('id')}", props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "idea_evaluations"):
        props = _rename_json_fields(
            row,
            {
                "factor_scores_json": "factor_scores",
                "missing_information_json": "missing_information",
                "data_quality_json": "data_quality",
                "evidence_json": "evidence",
                "disconfirming_evidence_json": "disconfirming_evidence",
                "portfolio_fit_json": "portfolio_fit",
                "recommendation_record_json": "recommendation_record",
            },
        )
        props.pop("raw_result_json", None)
        rows.append(("IdeaEvaluation", f"idea_evaluation:{props.get('id')}", props, _valid_from(props, cutover_time)))
    rankings_by_run: dict[str, list[dict[str, Any]]] = {}
    for ranking in _select_all(conn, "idea_comparison_rankings"):
        rankings_by_run.setdefault(str(ranking.get("run_id") or ""), []).append(ranking)
    for row in _select_all(conn, "idea_comparison_runs"):
        props = _rename_json_fields(row, {"scope_statuses_json": "scope_statuses", "raw_result_json": "raw_result"})
        props["rankings"] = rankings_by_run.get(str(props.get("run_id") or ""), [])
        rows.append(("IdeaComparisonRun", str(props.get("run_id")), props, _valid_from(props, cutover_time)))
    return rows


def _provenance_rows(conn: sqlite3.Connection, *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    for row in _select_all(conn, "provenance_events"):
        props = _rename_json_fields(row, {"summary_json": "summary", "metadata_json": "metadata"})
        rows.append(("ProvenanceEvent", str(props.get("id")), props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "provenance_links"):
        props = _rename_json_fields(row, {"metadata_json": "metadata"})
        rows.append(("ProvenanceLink", str(props.get("id")), props, _valid_from(props, cutover_time)))
    return rows


def _core_operational_rows(
    conn: sqlite3.Connection, *, cutover_time: str
) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    for row in _select_all(conn, "catalysts"):
        props = {
            "ticker": row.get("ticker"),
            "name": row.get("description") or f"legacy catalyst {row.get('id')}",
            "description": row.get("description"),
            "source": row.get("created_by") or "legacy",
            "category": row.get("category"),
            "target_date": row.get("target_date"),
            "status": row.get("status"),
            "evidence": row.get("evidence"),
            "ontology_run_id": "operational",
        }
        rows.append(("Catalyst", f"catalyst:{row.get('id')}", props, _valid_from(row, cutover_time)))
    for row in _select_all(conn, "kill_conditions"):
        props = _with_legacy_id(row, "id")
        props["ontology_run_id"] = "operational"
        rows.append(
            ("KillCondition", f"kill_condition:{props.get('legacy_id')}", props, _valid_from(props, cutover_time))
        )
    for row in _select_all(conn, "thesis_claims"):
        props = _with_legacy_id(
            _rename_json_fields(
                row,
                {
                    "source_requirements_json": "source_requirements",
                    "linked_catalyst_ids_json": "linked_catalyst_ids",
                    "linked_kill_condition_ids_json": "linked_kill_condition_ids",
                },
            ),
            "id",
        )
        props["ontology_run_id"] = "operational"
        rows.append(("ThesisClaim", f"thesis_claim:{props.get('legacy_id')}", props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "research_notes"):
        props = _with_legacy_id(row, "id")
        props["ontology_run_id"] = "operational"
        rows.append(
            ("ResearchNote", f"research_note:{props.get('legacy_id')}", props, _valid_from(props, cutover_time))
        )
    for row in _select_all(conn, "action_items"):
        props = _with_legacy_id(row, "id")
        props["ontology_run_id"] = "operational"
        rows.append(("ActionItem", f"action_item:{props.get('legacy_id')}", props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "watch_triggers"):
        props = _with_legacy_id(
            _rename_json_fields(row, {"definition_json": "definition", "last_result_json": "last_result"}), "id"
        )
        props["ontology_run_id"] = "operational"
        rows.append(
            ("WatchTrigger", f"watch_trigger:{props.get('legacy_id')}", props, _valid_from(props, cutover_time))
        )
    for row in _select_all(conn, "workflow_runs"):
        props = _without(row, {"lineage_completeness"})
        props["artifacts"] = props.get("artifacts") if isinstance(props.get("artifacts"), dict) else {}
        props["tool_sections"] = props.get("tool_sections") if isinstance(props.get("tool_sections"), list) else []
        props["ontology_run_id"] = "operational"
        rows.append(("WorkflowRun", str(props.get("run_id")), props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "workflow_artifact_records"):
        summary = _jsonable_value(row.get("summary_json"))
        props = {
            "artifact_id": row.get("artifact_id"),
            "workflow_run_id": row.get("workflow_run_id"),
            "artifact_key": row.get("artifact_key"),
            "artifact_index": row.get("artifact_index"),
            "artifact_value": summary if isinstance(summary, (dict, list, str)) else None,
            "artifact_hash": row.get("artifact_hash"),
            "approval_id": _optional_text(row.get("approval_id")),
            "provenance_event_id": row.get("provenance_event_id"),
            "metadata": {
                "redaction_policy": row.get("redaction_policy"),
                "retention_class": row.get("retention_class"),
                "created_at": row.get("created_at"),
            },
            "ontology_run_id": "operational",
        }
        rows.append(("WorkflowArtifact", str(props.get("artifact_id")), props, _valid_from(row, cutover_time)))
    for row in _select_all(conn, "report_runs"):
        props = _rename_json_fields(row, {"summary_json": "summary", "artifact_paths_json": "artifact_paths"})
        props["ontology_run_id"] = "operational"
        rows.append(("ReportRun", str(props.get("report_id")), props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "recommendations"):
        rows.extend(_recommendation_rows(row, cutover_time=cutover_time))
    for row in _select_all(conn, "pending_approvals"):
        props = _approval_props(row)
        rows.append(("Approval", f"approval:{props.get('legacy_id')}", props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "action_runs"):
        props = _action_run_props(row)
        rows.append(("ActionRun", f"action_run:{props.get('legacy_id')}", props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "action_events"):
        props = _with_legacy_id(_rename_json_fields(row, {"payload_json": "payload"}), "id")
        props["action_run_id"] = _optional_text(props.get("action_run_id")) or ""
        props["ontology_run_id"] = "operational"
        rows.append(("ActionEvent", f"action_event:{props.get('legacy_id')}", props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "policy_gate_results"):
        props = _policy_gate_props(row)
        rows.append(("PolicyGateResult", str(props.get("gate_result_id")), props, _valid_from(row, cutover_time)))
    for row in _select_all(conn, "audit_events"):
        props = _audit_event_props(row)
        rows.append(("AuditEvent", str(props.get("event_id")), props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "source_record_refs"):
        props = _source_record_props(row)
        rows.append(("SourceRecord", str(props.get("source_record_id")), props, _valid_from(row, cutover_time)))
    return rows


def _portfolio_rows(conn: sqlite3.Connection, *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    allowed = {
        "ticker",
        "asset",
        "direction",
        "contrarian",
        "conviction",
        "cost_basis",
        "shares",
        "quantity",
        "instrument_type",
        "price_symbol",
        "contract_multiplier",
        "role",
    }
    for row in _select_all(conn, "positions"):
        props = {key: row.get(key) for key in allowed if key in row}
        props["ontology_run_id"] = "operational"
        object_type = "HedgePosition" if str(row.get("role") or "").lower() == "hedge" else "Position"
        rows.append((object_type, str(row.get("ticker")), props, _valid_from(row, cutover_time)))
    return rows


def _thesis_rows(conn: sqlite3.Connection, *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    for row in _select_all(conn, "thesis_meta"):
        props = {**row, "ontology_run_id": "operational"}
        rows.append(("Thesis", str(row.get("ticker")), props, _valid_from(props, cutover_time)))
    for row in _select_all(conn, "thesis_evaluations"):
        props = _without(row, {"id"})
        props["ontology_run_id"] = "operational"
        rows.append(
            (
                "Evaluation",
                f"{props.get('ticker')}:{props.get('evaluated_at')}",
                props,
                _valid_from(props, cutover_time),
            )
        )
    for row in _select_all(conn, "thesis_status_history"):
        event_id = f"legacy_thesis_status:{row.get('id')}"
        props = {
            "event_id": event_id,
            "occurred_at": row.get("changed_at"),
            "actor_type": "system",
            "action_name": "thesis.status_changed",
            "action_category": "state_transition",
            "status": "succeeded",
            "object_refs": [{"object_type": "Thesis", "object_id": f"thesis:{row.get('ticker')}"}],
            "before_summary": {"status": row.get("old_status")},
            "after_summary": {"status": row.get("new_status")},
            "metadata": {"legacy_id": row.get("id"), "reason": row.get("reason")},
            "retention_class": "audit_365d",
            "ontology_run_id": "operational",
        }
        rows.append(("AuditEvent", event_id, props, _valid_from(row, cutover_time)))
    return rows


def _snapshot_versions(
    conn: sqlite3.Connection,
    *,
    cutover_time: str,
    provenance_event_id_value: str,
) -> list[SnapshotVersionWrite]:
    versions: list[SnapshotVersionWrite] = []
    for row in _select_all(conn, "computed_snapshots"):
        payload = row.get("payload_json") if isinstance(row.get("payload_json"), dict) else None
        status = "ok" if str(row.get("status") or "").lower() == "ok" else "error"
        error = _optional_text(row.get("error"))
        valid_from = str(row.get("as_of_date") or row.get("fetched_at") or cutover_time)
        versions.append(
            SnapshotVersionWrite(
                snapshot_key=str(row.get("snapshot_key")),
                payload_hash=payload_hash(payload if payload is not None else {"error": error}),
                payload=payload,
                artifact_uri=_optional_text(row.get("artifact_uri")),
                as_of=_optional_text(row.get("as_of_date")),
                load_time=_optional_text(row.get("fetched_at")) or cutover_time,
                valid_from=valid_from,
                status=status,
                quality="backfilled" if status == "ok" else "degraded",
                error=error,
                provenance_event_id=provenance_event_id_value,
            )
        )
    return versions


def _provenance_link_relation_rows(
    conn: sqlite3.Connection, *, cutover_time: str
) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    for row in _select_all(conn, "provenance_links"):
        valid_from = str(row.get("created_at") or cutover_time)
        rows.append(
            (
                provenance_event_id(row.get("event_id")),
                provenance_link_id(row.get("id")),
                {
                    "ontology_run_id": "operational",
                    "link_type": row.get("link_type"),
                    "source_ref_type": row.get("source_ref_type"),
                    "source_ref_id": row.get("source_ref_id"),
                    "target_ref_type": row.get("target_ref_type"),
                    "target_ref_id": row.get("target_ref_id"),
                },
                valid_from,
            )
        )
    return rows


def _recommendation_rows(row: dict[str, Any], *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    raw = _rename_json_fields(
        row,
        {
            "blocked_reasons_json": "blocked_reasons",
            "what_changed_json": "what_changed",
            "evidence_json": "evidence",
            "disconfirming_evidence_json": "disconfirming_evidence",
            "alternatives_json": "alternatives",
            "opportunity_cost_json": "opportunity_cost",
            "outcome_json": "outcome",
            "source_quality_summary_json": "source_quality_summary",
            "policy_gate_failures_json": "policy_gate_failures",
            "policy_gate_warnings_json": "policy_gate_warnings",
            "policy_gate_disclosures_json": "policy_gate_disclosures",
            "trade_proposal_json": "trade_proposal",
            "risk_source_status_json": "risk_source_status",
            "risk_bindings_json": "risk_bindings",
        },
    )
    recommendation_key = str(raw.get("idempotency_key") or raw.get("id"))
    props = {
        "recommendation_id": recommendation_key,
        "legacy_id": raw.get("id"),
        "idempotency_key": raw.get("idempotency_key"),
        "source_kind": "report",
        "report_type": raw.get("report_type"),
        "as_of": raw.get("as_of"),
        "action": raw.get("action"),
        "ticker": raw.get("ticker"),
        "instrument": raw.get("instrument"),
        "decision_state": _recommendation_decision_state(raw),
        "status": raw.get("status"),
        "approval_id": _optional_text(raw.get("approval_id")),
        "approval_required": bool(raw.get("approval_id")) or str(raw.get("recommendation_status")) == "review_required",
        "approval_status": raw.get("approval_status"),
        "outcome_status": raw.get("outcome_status"),
        "account_id": raw.get("account_id"),
        "portfolio_id": raw.get("portfolio_id"),
        "policy_id": raw.get("policy_id"),
        "policy_gate_result_id": _optional_text(raw.get("policy_gate_result_id")),
        "policy_gate_decision": raw.get("policy_gate_decision"),
        "policy_gate_review_required": bool(raw.get("policy_gate_review_required")),
        "confidence": raw.get("confidence"),
        "horizon": raw.get("horizon"),
        "rationale_summary": raw.get("rationale"),
        "rationale_hash": raw.get("prompt_hash") or raw.get("input_hash"),
        "source_quality": raw.get("source_quality"),
        "payload": raw,
        "ontology_run_id": "operational",
    }
    rows = [("Recommendation", f"recommendation:{recommendation_key}", props, _valid_from(raw, cutover_time))]
    trade_proposal = raw.get("trade_proposal")
    if isinstance(trade_proposal, dict) and trade_proposal:
        proposal_id = str(trade_proposal.get("proposal_id") or f"recommendation:{recommendation_key}")
        proposal_props = {
            "proposal_id": proposal_id,
            "recommendation_id": recommendation_key,
            "account_id": raw.get("account_id"),
            "portfolio_id": raw.get("portfolio_id"),
            "action": trade_proposal.get("action") or raw.get("action"),
            "instrument": trade_proposal.get("instrument") or raw.get("instrument") or raw.get("ticker"),
            "proposed_change": trade_proposal,
            "sizing_summary": trade_proposal.get("sizing_summary")
            if isinstance(trade_proposal.get("sizing_summary"), dict)
            else {},
            "risk_summary": trade_proposal.get("risk_summary")
            if isinstance(trade_proposal.get("risk_summary"), dict)
            else {},
            "policy_gate_result_id": _optional_text(raw.get("policy_gate_result_id")),
            "approval_id": _optional_text(raw.get("approval_id")),
            "decision_state": "staged",
            "status": "staged",
            "ontology_run_id": "operational",
        }
        rows.append(("TradeProposal", f"trade_proposal:{proposal_id}", proposal_props, _valid_from(raw, cutover_time)))
    return rows


def _approval_props(row: dict[str, Any]) -> dict[str, Any]:
    raw = _rename_json_fields(row, {"proposed_change": "proposed_change"})
    status = str(raw.get("status") or "pending")
    application_status = str(raw.get("application_status") or "pending")
    return {
        "legacy_id": raw.get("id"),
        "entity_type": raw.get("entity_type"),
        "entity_id": _optional_text(raw.get("entity_id")),
        "ticker": raw.get("ticker"),
        "action_id": raw.get("action_id"),
        "action_schema_name": raw.get("action_schema_name"),
        "action_schema_version": raw.get("action_schema_version"),
        "action_input_hash": raw.get("action_input_hash"),
        "proposed_change": raw.get("proposed_change") if isinstance(raw.get("proposed_change"), dict) else {},
        "reason": raw.get("reason"),
        "source_type": raw.get("source_type"),
        "source_id": raw.get("source_id"),
        "status": status,
        "resolution_state": status if status in {"pending", "approved", "rejected", "expired"} else "pending",
        "application_state": application_status
        if application_status in {"pending", "applying", "applied", "failed", "not_applicable"}
        else "pending",
        "application_status": application_status,
        "risk_class": raw.get("risk_class"),
        "base_state_hash": raw.get("base_state_hash"),
        "supersedes_approval_id": _optional_text(raw.get("supersedes_approval_id")),
        "requested_by_actor_id": raw.get("requested_by_actor_id"),
        "resolved_by_actor_id": raw.get("resolved_by_actor_id"),
        "created_at": raw.get("created_at"),
        "resolved_at": raw.get("resolved_at"),
        "resolved_note": raw.get("resolved_note"),
        "ontology_run_id": "operational",
    }


def _action_run_props(row: dict[str, Any]) -> dict[str, Any]:
    raw = dict(row)
    status = str(raw.get("status") or "running")
    return {
        "legacy_id": raw.get("id"),
        "action_id": raw.get("action_id"),
        "action_schema_name": raw.get("action_schema_name"),
        "action_schema_version": raw.get("action_schema_version") or 1,
        "actor_type": raw.get("actor_type"),
        "actor_id": raw.get("actor_id"),
        "source_type": raw.get("source_type"),
        "source_id": raw.get("source_id"),
        "approval_id": _optional_text(raw.get("approval_id")),
        "parent_action_run_id": _optional_text(raw.get("parent_action_run_id")),
        "input_hash": raw.get("input_hash"),
        "output_hash": raw.get("output_hash"),
        "status": status,
        "execution_state": status if status in {"running", "succeeded", "failed", "rolled_back", "denied"} else None,
        "error": raw.get("error"),
        "started_at": raw.get("started_at"),
        "completed_at": raw.get("completed_at"),
        "provenance_event_id": raw.get("provenance_event_id"),
        "ontology_run_id": "operational",
    }


def _policy_gate_props(row: dict[str, Any]) -> dict[str, Any]:
    result = _jsonable_value(row.get("result_json"))
    result_dict = result if isinstance(result, dict) else {}
    failures = result_dict.get("failure_reasons") or result_dict.get("failures") or []
    warnings = result_dict.get("warnings") or []
    return {
        "gate_result_id": row.get("id"),
        "decision": row.get("decision"),
        "review_required": bool(row.get("review_required")),
        "failure_reasons": failures if isinstance(failures, list) else [],
        "warnings": warnings if isinstance(warnings, list) else [],
        "account_id": row.get("account_id"),
        "portfolio_id": row.get("portfolio_id"),
        "policy_id": row.get("policy_id"),
        "evaluated_at": row.get("created_at"),
        "ontology_run_id": "operational",
    }


def _audit_event_props(row: dict[str, Any]) -> dict[str, Any]:
    props = _rename_json_fields(
        row,
        {
            "object_refs_json": "object_refs",
            "before_summary_json": "before_summary",
            "after_summary_json": "after_summary",
            "source_lineage_json": "source_lineage",
            "metadata_json": "metadata",
        },
    )
    object_refs = props.get("object_refs") if isinstance(props.get("object_refs"), list) else []
    if not object_refs and props.get("object_type") and props.get("object_id"):
        object_refs = [{"object_type": props.get("object_type"), "object_id": props.get("object_id")}]
    metadata = props.get("metadata") if isinstance(props.get("metadata"), dict) else {}
    for key in ("id", "request_id", "criticality", "idempotency_key", "producer_name", "producer_version", "error"):
        if props.get(key) is not None:
            metadata[key] = props.get(key)
    return {
        "event_id": props.get("event_id"),
        "occurred_at": props.get("occurred_at"),
        "actor_type": props.get("actor_type") or "system",
        "actor_id": props.get("actor_id"),
        "action_name": props.get("action_name"),
        "action_category": props.get("action_category"),
        "status": props.get("status"),
        "object_refs": object_refs,
        "before_summary": props.get("before_summary") if isinstance(props.get("before_summary"), dict) else None,
        "after_summary": props.get("after_summary") if isinstance(props.get("after_summary"), dict) else None,
        "source_lineage": props.get("source_lineage") if isinstance(props.get("source_lineage"), dict) else None,
        "metadata": metadata,
        "lineage_root_id": props.get("lineage_root_id"),
        "retention_class": props.get("retention_class") or "audit_365d",
        "ontology_run_id": "operational",
    }


def _source_record_props(row: dict[str, Any]) -> dict[str, Any]:
    summary = _jsonable_value(row.get("summary_json"))
    return {
        "source_record_id": row.get("record_ref_id"),
        "vendor": row.get("source_name"),
        "source_name": row.get("source_name"),
        "source_version": "unknown",
        "dataset": row.get("source_name"),
        "record_kind": row.get("record_kind"),
        "record_key_hash": row.get("record_key_hash"),
        "payload_hash": row.get("record_hash"),
        "status": "ok",
        "quality": "ok",
        "as_of": row.get("as_of"),
        "load_time": row.get("created_at"),
        "provenance_event_id": row.get("adapter_run_event_id"),
        "metadata": {
            "adapter_run_event_id": row.get("adapter_run_event_id"),
            "summary": summary,
            "redaction_policy": row.get("redaction_policy"),
            "retention_class": row.get("retention_class"),
        },
        "ontology_run_id": "operational",
    }


def _recommendation_decision_state(row: dict[str, Any]) -> str:
    if row.get("approval_status") == "approved":
        return "approved"
    if row.get("approval_status") == "rejected":
        return "rejected"
    if row.get("approval_id") or row.get("approval_status") == "pending":
        return "under_review"
    return "generated"


def _with_legacy_id(row: dict[str, Any], key: str) -> dict[str, Any]:
    props = dict(row)
    props["legacy_id"] = props.pop(key, None)
    return props


def _without(row: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in keys}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _jsonable_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _select_all(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    if table_name not in _LEGACY_TABLES:
        raise ValueError(f"Unsupported legacy table: {table_name}")
    if not _table_exists(conn, table_name):
        return []
    quoted = '"' + table_name.replace('"', '""') + '"'
    rows = conn.execute(f"SELECT * FROM {quoted}").fetchall()
    return [_jsonable(dict(row)) for row in rows]


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)).fetchone()
    return row is not None


def _rename_json_fields(row: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    props = dict(row)
    for source, target in mapping.items():
        value = props.pop(source, None)
        props[target] = value if value is not None else _json_default(target)
    return props


def _json_default(target: str) -> Any:
    value = _JSON_FIELD_DEFAULTS.get(target)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _valid_from(props: dict[str, Any], fallback: str) -> str:
    for key in ("created_at", "started_at", "evaluated_at", "as_of", "updated_at"):
        value = props.get(key)
        if value:
            return str(value)
    return fallback
