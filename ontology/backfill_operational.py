"""Backfill legacy operational stores into temporal ontology tables."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from ontology.domain_write_service import OPERATIONAL_ONTOLOGY_RUN_ID, DomainOntologyWriteService, OntologyMutation
from ontology.object_service import object_uid_for
from ontology.schemas.identity import thesis_id
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
    )
    for table, record_kind, object_type, key_field, mapper in core_mappings:
        for row in _core_table_rows(table):
            mapped = mapper(row)
            record_key = f"legacy:{table}:{row.get(key_field)}"
            write(table, record_kind, record_key, object_type, str(row.get(key_field) or ""), mapped)

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
    return {
        "legacy_id": row.get("id"),
        "report_type": out.get("report_type"),
        "as_of": out.get("as_of"),
        "action": out.get("action") or "do_nothing",
        "ticker": out.get("ticker"),
        "instrument": out.get("instrument"),
        "status": out.get("status"),
        "approval_id": out.get("approval_id"),
        "approval_status": out.get("approval_status"),
        "outcome_status": out.get("outcome_status"),
        "payload": {key: value for key, value in out.items() if key not in {"schema_version"}},
    }


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
