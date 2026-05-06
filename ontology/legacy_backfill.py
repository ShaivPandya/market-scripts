"""Migration-only reader for legacy SQLite exports.

This module is intentionally not a runtime compatibility layer. It exists so a
maintenance-window cutover can read old SQLite exports, write audited ontology
objects, and then deploy the Postgres-only runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from api.provenance import (
    DEFAULT_REDACTION_POLICY,
    FINANCIAL_RETENTION_CLASS,
    LINK_RELATION_TYPES,
    REF_AGENT_SESSION,
    REF_COMPUTED_SNAPSHOT_VERSION,
    REF_MODEL_CALL,
    REF_ONTOLOGY_OBJECT_VERSION,
    REF_ONTOLOGY_RUN,
    REF_RELATION_VERSION,
    REF_SCHEMA_DEFINITION,
    REF_TOOL_CALL,
    ref_object_uid_for,
)
from ontology.command_service import OntologyCommandContext, OntologyCommandService
from ontology.object_service import OntologyObjectService
from ontology.policy import system_actor
from ontology.schemas.identity import (
    citation_id,
    document_artifact_id,
    evidence_id,
    factor_score_id,
    idea_comparison_ranking_id,
    idea_comparison_run_id,
    idea_evaluation_id,
    investment_idea_id,
    issuer_id,
    management_quality_accomplishment_id,
    management_quality_assessment_id,
    management_quality_scorecard_row_id,
    management_quality_setback_id,
    missing_information_requirement_id,
    optimization_action_snapshot_id,
    optimization_alert_id,
    optimization_mission_id,
    optimization_run_id,
    source_freshness_id,
)
from ontology.temporal_repository import SnapshotVersionWrite, TemporalOntologyRepository, payload_hash


class LegacyBackfillDisabled(RuntimeError):
    pass


class LegacyBackfillUnmappedRefs(RuntimeError):
    def __init__(self, unmapped_refs: list[dict[str, Any]]):
        super().__init__(f"Legacy provenance backfill has unmapped refs: {unmapped_refs}")
        self.unmapped_refs = unmapped_refs


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
    management_quality_dir: str | Path | None = None,
    dry_run: bool = False,
    provenance_event_id_value: str = "pv:legacy_backfill:runtime_objects",
) -> dict[str, Any]:
    """Backfill current legacy runtime objects into temporal ontology versions.

    This is cutover-only scaffolding. It reads a legacy SQLite export under the
    explicit `TALISMAN_ENABLE_LEGACY_BACKFILL` gate and writes
    ontology objects with `temporal_confidence='backfilled'`.
    """

    path = Path(core_db_path)
    mutations: list[tuple[str, str, dict[str, Any], str]] = []
    link_relations: list[tuple[str, str, str, dict[str, Any], str]] = []
    snapshot_versions: list[SnapshotVersionWrite] = []
    cutover_time = _now()
    with _connect(path) as conn:
        mutations.extend(_runtime_object_rows(conn, cutover_time=cutover_time))
        link_relations.extend(_domain_relation_rows(conn, cutover_time=cutover_time))
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
    if management_quality_dir is not None:
        mq_mutations, mq_relations = _management_quality_rows(Path(management_quality_dir), cutover_time=cutover_time)
        mutations.extend(mq_mutations)
        link_relations.extend(mq_relations)
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
            "redaction_policy": DEFAULT_REDACTION_POLICY,
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
    for relation_type, source_uid, target_uid, properties, valid_from in link_relations:
        service.write_relation(
            source_uid,
            target_uid,
            relation_type,
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


def _legacy_default_mission_key(value: Any, name: Any = None) -> str:
    text = str(value or "").strip()
    normalized_name = str(name or "").strip().lower()
    if text == "1" or normalized_name in {"default", "daily command center"}:
        return "default"
    return text or "default"


def _idea_object_uid(value: Any) -> str:
    return investment_idea_id(value)


def _idea_evaluation_uid(value: Any) -> str:
    return idea_evaluation_id(value)


def _comparison_run_uid(value: Any) -> str:
    return idea_comparison_run_id(value)


def _comparison_ranking_uid(value: Any) -> str:
    return idea_comparison_ranking_id(value)


def _optimization_mission_uid(value: Any, name: Any = None) -> str:
    return optimization_mission_id(_legacy_default_mission_key(value, name))


def _optimization_run_uid(value: Any) -> str:
    return optimization_run_id(value)


def _optimization_snapshot_uid(row: dict[str, Any]) -> str:
    run_uid = _optimization_run_uid(row.get("run_id"))
    ticker = str(row.get("ticker") or "").strip().upper()
    return optimization_action_snapshot_id(f"{run_uid}:{ticker or row.get('id')}")


def _optimization_alert_uid(row: dict[str, Any], current_snapshot_uid: str | None = None) -> str:
    run_uid = _optimization_run_uid(row.get("run_id"))
    key = f"{run_uid}:{current_snapshot_uid or row.get('current_snapshot_id') or row.get('id')}:{row.get('alert_type')}"
    return optimization_alert_id(key)


def _source_freshness_category(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"ok", "fresh"}:
        return "fresh"
    if normalized in {"stale", "degraded", "failed", "error"}:
        return normalized
    return "unknown"


def _legacy_snapshot_uid(conn: sqlite3.Connection, legacy_id: Any) -> str | None:
    if legacy_id in (None, ""):
        return None
    try:
        raw = conn.execute("SELECT * FROM optimization_action_snapshots WHERE id = ?", (legacy_id,)).fetchone()
    except sqlite3.Error:
        raw = None
    if raw is None:
        return optimization_action_snapshot_id(legacy_id)
    return _optimization_snapshot_uid(_jsonable(dict(raw)))


def _stable_child_key(*parts: Any) -> str:
    raw = json.dumps([str(part) for part in parts], sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


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
    rows.extend(_provenance_ref_rows(conn, cutover_time=cutover_time))
    rows.extend(_core_operational_rows(conn, cutover_time=cutover_time))
    return rows


def _domain_relation_rows(
    conn: sqlite3.Connection, *, cutover_time: str
) -> list[tuple[str, str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, str, dict[str, Any], str]] = []

    for row in _select_all(conn, "idea_evaluations"):
        evaluation_uid = _idea_evaluation_uid(row.get("id"))
        idea_uid = _idea_object_uid(row.get("idea_id"))
        valid_from = _valid_from(row, cutover_time)
        rows.append(_domain_relation("idea_has_evaluation", idea_uid, evaluation_uid, valid_from))

        factor_scores = _jsonable_value(row.get("factor_scores_json"))
        if isinstance(factor_scores, dict):
            for factor_name in factor_scores:
                rows.append(
                    _domain_relation(
                        "research_object_has_factor_score",
                        evaluation_uid,
                        factor_score_id(f"{evaluation_uid}:{factor_name}"),
                        valid_from,
                    )
                )
        missing_information = _jsonable_value(row.get("missing_information_json"))
        if isinstance(missing_information, list):
            for index, raw_missing in enumerate(missing_information):
                if isinstance(raw_missing, dict):
                    field = str(raw_missing.get("field") or raw_missing.get("name") or "unspecified")
                elif isinstance(raw_missing, str):
                    field = raw_missing
                else:
                    continue
                rows.append(
                    _domain_relation(
                        "research_object_has_missing_information",
                        evaluation_uid,
                        missing_information_requirement_id(f"{evaluation_uid}:missing:{index}:{field}"),
                        valid_from,
                    )
                )
        for relation_type, relation_role, evidence_rows in (
            ("research_object_supported_by_evidence", "supporting", _jsonable_value(row.get("evidence_json"))),
            (
                "research_object_disconfirmed_by_evidence",
                "disconfirming",
                _jsonable_value(row.get("disconfirming_evidence_json")),
            ),
        ):
            if not isinstance(evidence_rows, list):
                continue
            for index, raw_evidence in enumerate(evidence_rows):
                if not isinstance(raw_evidence, dict):
                    continue
                evidence_uid = evidence_id(
                    f"{evaluation_uid}:{relation_role}:{index}:{_stable_child_key(raw_evidence)}"
                )
                rows.append(_domain_relation(relation_type, evaluation_uid, evidence_uid, valid_from))
                citation_value = raw_evidence.get("url") or raw_evidence.get("source_path")
                if citation_value:
                    rows.append(
                        _domain_relation("evidence_has_citation", evidence_uid, citation_id(citation_value), valid_from)
                    )
        if row.get("recommendation_id") not in (None, ""):
            rows.append(
                _domain_relation(
                    "research_object_links_recommendation",
                    evaluation_uid,
                    f"recommendation:{row.get('recommendation_id')}",
                    valid_from,
                )
            )
        for key in ("approval_id", "action_approval_id"):
            if row.get(key) not in (None, ""):
                rows.append(
                    _domain_relation(
                        "research_object_links_approval", evaluation_uid, f"approval:{row.get(key)}", valid_from
                    )
                )

    for ranking in _select_all(conn, "idea_comparison_rankings"):
        ranking_uid = _comparison_ranking_uid(ranking.get("id"))
        comparison_uid = _comparison_run_uid(ranking.get("run_id"))
        valid_from = _valid_from(ranking, cutover_time)
        rows.append(_domain_relation("comparison_run_has_ranking", comparison_uid, ranking_uid, valid_from))
        rows.append(
            _domain_relation("ranking_targets_idea", ranking_uid, _idea_object_uid(ranking.get("idea_id")), valid_from)
        )
        rows.append(
            _domain_relation(
                "ranking_uses_evaluation", ranking_uid, _idea_evaluation_uid(ranking.get("evaluation_id")), valid_from
            )
        )

    for run in _select_all(conn, "optimization_runs"):
        run_uid = _optimization_run_uid(run.get("run_id"))
        mission_uid = _optimization_mission_uid(run.get("mission_id"), run.get("mission_name"))
        valid_from = _valid_from(run, cutover_time)
        rows.append(_domain_relation("optimization_mission_has_run", mission_uid, run_uid, valid_from))
        source_freshness = _jsonable_value(run.get("source_freshness_json"))
        if isinstance(source_freshness, dict):
            for source_name in source_freshness:
                rows.append(
                    _domain_relation(
                        "optimization_object_has_source_freshness",
                        run_uid,
                        source_freshness_id(f"{run_uid}:source:{source_name}"),
                        valid_from,
                    )
                )

    for snapshot in _select_all(conn, "optimization_action_snapshots"):
        snapshot_uid = _optimization_snapshot_uid(snapshot)
        valid_from = _valid_from(snapshot, cutover_time)
        rows.append(
            _domain_relation(
                "optimization_run_has_snapshot", _optimization_run_uid(snapshot.get("run_id")), snapshot_uid, valid_from
            )
        )
        ticker = str(snapshot.get("ticker") or "").strip().upper()
        if ticker:
            rows.append(
                _domain_relation(
                    "optimization_snapshot_targets_position", snapshot_uid, f"position:{ticker}", valid_from
                )
            )
            rows.append(
                _domain_relation(
                    "optimization_snapshot_targets_instrument", snapshot_uid, f"instrument:{ticker.lower()}", valid_from
                )
            )

    for alert in _select_all(conn, "optimization_alerts"):
        current_snapshot_uid = _legacy_snapshot_uid(conn, alert.get("current_snapshot_id"))
        previous_snapshot_uid = _legacy_snapshot_uid(conn, alert.get("previous_snapshot_id"))
        alert_uid = _optimization_alert_uid(alert, current_snapshot_uid=current_snapshot_uid)
        valid_from = _valid_from(alert, cutover_time)
        if current_snapshot_uid:
            rows.append(
                _domain_relation("optimization_alert_current_snapshot", alert_uid, current_snapshot_uid, valid_from)
            )
        if previous_snapshot_uid:
            rows.append(
                _domain_relation("optimization_alert_previous_snapshot", alert_uid, previous_snapshot_uid, valid_from)
            )
        for approval_key in ("approval_id", "action_item_approval_id"):
            if alert.get(approval_key) not in (None, ""):
                rows.append(
                    _domain_relation(
                        "optimization_alert_links_approval",
                        alert_uid,
                        f"approval:{alert.get(approval_key)}",
                        valid_from,
                    )
                )
        if alert.get("action_item_id") not in (None, ""):
            rows.append(
                _domain_relation(
                    "optimization_alert_links_action_item",
                    alert_uid,
                    f"action_item:{alert.get('action_item_id')}",
                    valid_from,
                )
            )

    return rows


def _domain_relation(
    relation_type: str,
    source_uid: str,
    target_uid: str,
    valid_from: str,
    properties: dict[str, Any] | None = None,
) -> tuple[str, str, str, dict[str, Any], str]:
    return (
        relation_type,
        source_uid,
        target_uid,
        {"ontology_run_id": "operational", **(properties or {})},
        valid_from,
    )


def _optimization_rows(conn: sqlite3.Connection, *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    for row in _select_all(conn, "optimization_missions"):
        props = _rename_json_fields(
            row, {"scenario_json": "scenario", "source_config_json": "source_config", "thresholds_json": "thresholds"}
        )
        legacy_id = props.pop("id", None)
        mission_key = _legacy_default_mission_key(legacy_id, props.get("name"))
        props.update({"legacy_id": legacy_id, "mission_id": mission_key})
        rows.append(
            ("OptimizationMission", _optimization_mission_uid(mission_key), props, _valid_from(props, cutover_time))
        )
    for row in _select_all(conn, "optimization_runs"):
        props = _rename_json_fields(row, {"summary_json": "summary", "source_freshness_json": "source_freshness"})
        props["mission_id"] = _optimization_mission_uid(props.get("mission_id"), props.get("mission_name"))
        if str(props.get("status") or "").lower() == "succeeded":
            props["status"] = "completed"
        run_uid = _optimization_run_uid(props.get("run_id"))
        rows.append(("OptimizationRun", run_uid, props, _valid_from(props, cutover_time)))
        source_freshness = props.get("source_freshness")
        if isinstance(source_freshness, dict):
            for source_name, raw in source_freshness.items():
                payload = raw if isinstance(raw, dict) else {"status": str(raw)}
                key = f"{run_uid}:source:{source_name}"
                rows.append(
                    (
                        "SourceFreshness",
                        source_freshness_id(key),
                        {
                            "freshness_id": key,
                            "parent_uid": run_uid,
                            "parent_type": "OptimizationRun",
                            "source_name": str(source_name),
                            "status": str(payload.get("status") or "unknown"),
                            "checked_at": payload.get("checked_at"),
                            "as_of": payload.get("as_of"),
                            "freshness_category": _source_freshness_category(payload.get("status")),
                            "error": payload.get("error"),
                            "metadata": payload,
                        },
                        _valid_from(props, cutover_time),
                    )
                )
    for row in _select_all(conn, "optimization_action_snapshots"):
        props = _rename_json_fields(row, {"evidence_json": "evidence", "source_links_json": "source_links"})
        legacy_id = props.pop("id", None)
        props["legacy_id"] = legacy_id
        props["mission_id"] = _optimization_mission_uid(props.get("mission_id"))
        props["run_id"] = _optimization_run_uid(props.get("run_id"))
        snapshot_uid = _optimization_snapshot_uid({"id": legacy_id, **props})
        props["snapshot_id"] = snapshot_uid
        rows.append(
            (
                "OptimizationActionSnapshot",
                snapshot_uid,
                props,
                _valid_from(props, cutover_time),
            )
        )
    for row in _select_all(conn, "optimization_alerts"):
        props = _rename_json_fields(row, {"evidence_json": "evidence"})
        legacy_id = props.pop("id", None)
        props["legacy_id"] = legacy_id
        props["mission_id"] = _optimization_mission_uid(props.get("mission_id"))
        props["run_id"] = _optimization_run_uid(props.get("run_id"))
        current_snapshot_uid = _legacy_snapshot_uid(conn, props.get("current_snapshot_id"))
        previous_snapshot_uid = _legacy_snapshot_uid(conn, props.get("previous_snapshot_id"))
        props["current_snapshot_id"] = current_snapshot_uid
        props["previous_snapshot_id"] = previous_snapshot_uid
        alert_uid = _optimization_alert_uid({"id": legacy_id, **props}, current_snapshot_uid=current_snapshot_uid)
        props["alert_id"] = alert_uid
        if "dismissed_note" in props:
            props["dismissal_note"] = props.pop("dismissed_note")
        rows.append(("OptimizationAlert", alert_uid, props, _valid_from(props, cutover_time)))
    return rows


def _idea_rows(conn: sqlite3.Connection, *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    for row in _select_all(conn, "investment_ideas"):
        props = _rename_json_fields(row, {"tags_json": "tags", "metadata_json": "metadata"})
        legacy_id = props.pop("id", None)
        props["legacy_id"] = legacy_id
        props["idea_id"] = _idea_object_uid(legacy_id)
        if props.get("latest_evaluation_id") not in (None, ""):
            props["latest_evaluation_id"] = _idea_evaluation_uid(props.get("latest_evaluation_id"))
        if props.get("accepted_recommendation_id") not in (None, ""):
            props["accepted_recommendation_id"] = f"recommendation:{props.get('accepted_recommendation_id')}"
        rows.append(("InvestmentIdea", _idea_object_uid(legacy_id), props, _valid_from(props, cutover_time)))
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
        legacy_id = props.pop("id", None)
        evaluation_uid = _idea_evaluation_uid(legacy_id)
        props["legacy_id"] = legacy_id
        props["evaluation_id"] = evaluation_uid
        props["idea_id"] = _idea_object_uid(props.get("idea_id"))
        if props.get("recommendation_id") not in (None, ""):
            props["recommendation_id"] = f"recommendation:{props.get('recommendation_id')}"
        if props.get("approval_id") not in (None, ""):
            props["approval_id"] = f"approval:{props.get('approval_id')}"
        if props.get("action_approval_id") not in (None, ""):
            props["action_approval_id"] = f"approval:{props.get('action_approval_id')}"
        rows.append(("IdeaEvaluation", evaluation_uid, props, _valid_from(props, cutover_time)))
        factor_scores = props.get("factor_scores")
        if isinstance(factor_scores, dict):
            for factor_name, raw_factor in factor_scores.items():
                factor = raw_factor if isinstance(raw_factor, dict) else {"score": raw_factor}
                key = f"{evaluation_uid}:{factor_name}"
                rows.append(
                    (
                        "FactorScore",
                        factor_score_id(key),
                        {
                            "factor_score_id": key,
                            "parent_uid": evaluation_uid,
                            "parent_type": "IdeaEvaluation",
                            "factor_name": str(factor_name),
                            "score": factor.get("score"),
                            "status": factor.get("status"),
                            "rationale": factor.get("rationale"),
                            "missing": factor.get("missing") if isinstance(factor.get("missing"), list) else [],
                            "created_at": props.get("created_at") or props.get("evaluated_at"),
                        },
                        _valid_from(props, cutover_time),
                    )
                )
        missing_information = props.get("missing_information")
        if isinstance(missing_information, list):
            for index, raw_missing in enumerate(missing_information):
                if isinstance(raw_missing, str):
                    missing = {"field": raw_missing, "severity": "medium", "reason": raw_missing}
                elif isinstance(raw_missing, dict):
                    missing = raw_missing
                else:
                    continue
                field = str(missing.get("field") or missing.get("name") or "unspecified")
                key = f"{evaluation_uid}:missing:{index}:{field}"
                rows.append(
                    (
                        "MissingInformationRequirement",
                        missing_information_requirement_id(key),
                        {
                            "requirement_id": key,
                            "parent_uid": evaluation_uid,
                            "parent_type": "IdeaEvaluation",
                            "field": field,
                            "severity": str(missing.get("severity") or "medium"),
                            "reason": missing.get("reason") or missing.get("message"),
                            "status": str(missing.get("status") or "open"),
                            "created_at": props.get("created_at") or props.get("evaluated_at"),
                        },
                        _valid_from(props, cutover_time),
                    )
                )
        for relation_role, evidence_rows in (
            ("supporting", props.get("evidence")),
            ("disconfirming", props.get("disconfirming_evidence")),
        ):
            if not isinstance(evidence_rows, list):
                continue
            for index, raw_evidence in enumerate(evidence_rows):
                if not isinstance(raw_evidence, dict):
                    continue
                key = f"{evaluation_uid}:{relation_role}:{index}:{_stable_child_key(raw_evidence)}"
                rows.append(
                    (
                        "Evidence",
                        evidence_id(key),
                        {
                            "evidence_id": key,
                            "evidence_type": str(raw_evidence.get("evidence_type") or relation_role),
                            "title": raw_evidence.get("title") or raw_evidence.get("source"),
                            "summary": raw_evidence.get("summary") or raw_evidence.get("text"),
                            "source_record_id": raw_evidence.get("source_record_id"),
                            "document_artifact_id": raw_evidence.get("document_artifact_id"),
                            "confidence": raw_evidence.get("confidence"),
                            "observed_at": raw_evidence.get("observed_at") or props.get("evaluated_at"),
                        },
                        _valid_from(props, cutover_time),
                    )
                )
                citation_value = raw_evidence.get("url") or raw_evidence.get("source_path")
                if citation_value:
                    rows.append(
                        (
                            "Citation",
                            citation_id(citation_value),
                            {
                                "citation_id": str(citation_value),
                                "title": raw_evidence.get("title") or raw_evidence.get("source"),
                                "url": raw_evidence.get("url"),
                                "source_path": raw_evidence.get("source_path"),
                                "document_artifact_id": raw_evidence.get("document_artifact_id"),
                            },
                            _valid_from(props, cutover_time),
                        )
                    )
    rankings_by_run: dict[str, list[dict[str, Any]]] = {}
    for ranking in _select_all(conn, "idea_comparison_rankings"):
        rankings_by_run.setdefault(str(ranking.get("run_id") or ""), []).append(ranking)
        ranking_uid = _comparison_ranking_uid(ranking.get("id"))
        comparison_uid = _comparison_run_uid(ranking.get("run_id"))
        rows.append(
            (
                "IdeaComparisonRanking",
                ranking_uid,
                {
                    **_without(ranking, {"id"}),
                    "legacy_id": ranking.get("id"),
                    "ranking_id": ranking_uid,
                    "comparison_run_id": comparison_uid,
                    "run_id": comparison_uid,
                    "idea_id": _idea_object_uid(ranking.get("idea_id")),
                    "evaluation_id": _idea_evaluation_uid(ranking.get("evaluation_id")),
                },
                _valid_from(ranking, cutover_time),
            )
        )
    for row in _select_all(conn, "idea_comparison_runs"):
        props = _rename_json_fields(row, {"scope_statuses_json": "scope_statuses", "raw_result_json": "raw_result"})
        legacy_id = props.pop("id", None)
        original_run_id = str(props.get("run_id") or legacy_id or "")
        comparison_uid = _comparison_run_uid(original_run_id)
        embedded_rankings = []
        for ranking in rankings_by_run.get(original_run_id, []):
            embedded_rankings.append(
                {
                    **_without(ranking, {"id"}),
                    "id": _comparison_ranking_uid(ranking.get("id")),
                    "idea_id": _idea_object_uid(ranking.get("idea_id")),
                    "evaluation_id": _idea_evaluation_uid(ranking.get("evaluation_id")),
                    "run_id": comparison_uid,
                }
            )
        props.update(
            {
                "legacy_id": legacy_id,
                "comparison_run_id": comparison_uid,
                "run_id": comparison_uid,
                "rankings": embedded_rankings,
            }
        )
        rows.append(("IdeaComparisonRun", comparison_uid, props, _valid_from(props, cutover_time)))
    return rows


def _provenance_rows(conn: sqlite3.Connection, *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    for row in _select_all(conn, "provenance_events"):
        props = _rename_json_fields(row, {"summary_json": "summary", "metadata_json": "metadata"})
        props.setdefault("redaction_policy", DEFAULT_REDACTION_POLICY)
        props.setdefault("retention_class", FINANCIAL_RETENTION_CLASS)
        props.setdefault("lineage_root_id", props.get("id"))
        rows.append(("ProvenanceEvent", str(props.get("id")), props, _valid_from(props, cutover_time)))
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


def _management_quality_rows(
    directory: Path, *, cutover_time: str
) -> tuple[list[tuple[str, str, dict[str, Any], str]], list[tuple[str, str, str, dict[str, Any], str]]]:
    mutations: list[tuple[str, str, dict[str, Any], str]] = []
    relations: list[tuple[str, str, str, dict[str, Any], str]] = []
    if not directory.exists():
        return mutations, relations
    try:
        from api.routers.management_quality import parse_management_quality_markdown
    except Exception:
        parse_management_quality_markdown = None

    for path in sorted(directory.glob("*.md")):
        ticker = path.stem.strip().upper()
        if not ticker:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")
        parsed = None
        if parse_management_quality_markdown is not None:
            try:
                parsed = parse_management_quality_markdown(content)
            except Exception:
                parsed = None
        parsed_dict = parsed if isinstance(parsed, dict) else {}
        summary = parsed_dict.get("summary") if isinstance(parsed_dict.get("summary"), dict) else {}
        issuer_uid = issuer_id(ticker)
        assessment_uid = management_quality_assessment_id(issuer_uid)
        doc_uid = document_artifact_id("management_quality", ticker)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        mutations.extend(
            [
                (
                    "Issuer",
                    issuer_uid,
                    {"issuer_id": issuer_uid, "name": ticker, "ticker": ticker, "ontology_run_id": "operational"},
                    cutover_time,
                ),
                (
                    "DocumentArtifact",
                    doc_uid,
                    {
                        "document_type": "management_quality",
                        "document_id": f"management_quality:{ticker}",
                        "title": f"{ticker} management quality",
                        "ticker": ticker,
                        "content_hash": content_hash,
                        "artifact_uri": str(path),
                        "status": "active",
                        "source_type": "legacy_markdown",
                        "source_id": str(path),
                        "created_at": cutover_time,
                        "updated_at": cutover_time,
                    },
                    cutover_time,
                ),
                (
                    "ManagementQualityAssessment",
                    assessment_uid,
                    {
                        "assessment_id": assessment_uid,
                        "issuer_id": issuer_uid,
                        "ticker": ticker,
                        "status": "active",
                        "overall_rating": summary.get("overall_rating"),
                        "bottom_line": summary.get("bottom_line"),
                        "owner_mindset_rating": _summary_rating(summary, "owner_mindset"),
                        "owner_mindset_text": _summary_text(summary, "owner_mindset"),
                        "business_value_understanding_rating": _summary_rating(summary, "business_value_understanding"),
                        "business_value_understanding_text": _summary_text(summary, "business_value_understanding"),
                        "follow_through_rating": _summary_rating(summary, "follow_through"),
                        "follow_through_text": _summary_text(summary, "follow_through"),
                        "content_hash": content_hash,
                        "document_id": doc_uid,
                        "source_type": "legacy_markdown",
                        "source_id": str(path),
                        "created_at": cutover_time,
                        "updated_at": cutover_time,
                    },
                    cutover_time,
                ),
            ]
        )
        relations.append(
            _domain_relation("management_quality_assesses_issuer", assessment_uid, issuer_uid, cutover_time)
        )
        relations.append(
            _domain_relation(
                "research_object_uses_document",
                assessment_uid,
                doc_uid,
                cutover_time,
                {"document_role": "rendered_markdown"},
            )
        )

        scorecard = parsed_dict.get("scorecard") if isinstance(parsed_dict.get("scorecard"), list) else []
        for index, row in enumerate(scorecard):
            if not isinstance(row, dict) or not row.get("question") or not row.get("rating"):
                continue
            row_key = f"{assessment_uid}:scorecard:{index}:{_stable_child_key(row)}"
            row_uid = management_quality_scorecard_row_id(row_key)
            mutations.append(
                (
                    "ManagementQualityScorecardRow",
                    row_uid,
                    {
                        "row_id": row_key,
                        "assessment_id": assessment_uid,
                        "issuer_id": issuer_uid,
                        "ticker": ticker,
                        "question": row.get("question"),
                        "rating": row.get("rating"),
                        "evidence": row.get("evidence"),
                        "ordinal": index,
                    },
                    cutover_time,
                )
            )
            relations.append(
                _domain_relation("management_quality_has_scorecard_row", assessment_uid, row_uid, cutover_time)
            )

        accomplishments = (
            parsed_dict.get("accomplishments") if isinstance(parsed_dict.get("accomplishments"), list) else []
        )
        for index, row in enumerate(accomplishments):
            if not isinstance(row, dict) or not row.get("text"):
                continue
            accomplishment_key = f"{assessment_uid}:accomplishment:{index}:{_stable_child_key(row)}"
            accomplishment_uid = management_quality_accomplishment_id(accomplishment_key)
            mutations.append(
                (
                    "ManagementQualityAccomplishment",
                    accomplishment_uid,
                    {
                        "accomplishment_id": accomplishment_key,
                        "assessment_id": assessment_uid,
                        "issuer_id": issuer_uid,
                        "ticker": ticker,
                        "title": row.get("title"),
                        "text": row.get("text"),
                        "period": row.get("period"),
                        "ordinal": index,
                    },
                    cutover_time,
                )
            )
            relations.append(
                _domain_relation(
                    "management_quality_has_accomplishment", assessment_uid, accomplishment_uid, cutover_time
                )
            )

        setbacks = parsed_dict.get("setbacks") if isinstance(parsed_dict.get("setbacks"), list) else []
        for index, row in enumerate(setbacks):
            if not isinstance(row, dict) or not row.get("text"):
                continue
            setback_key = f"{assessment_uid}:setback:{index}:{_stable_child_key(row)}"
            setback_uid = management_quality_setback_id(setback_key)
            mutations.append(
                (
                    "ManagementQualitySetback",
                    setback_uid,
                    {
                        "setback_id": setback_key,
                        "assessment_id": assessment_uid,
                        "issuer_id": issuer_uid,
                        "ticker": ticker,
                        "title": row.get("title"),
                        "text": row.get("text"),
                        "response_rating": row.get("response_rating"),
                        "response_text": row.get("response_text"),
                        "ordinal": index,
                    },
                    cutover_time,
                )
            )
            relations.append(
                _domain_relation("management_quality_has_setback", assessment_uid, setback_uid, cutover_time)
            )
    return mutations, relations


def _summary_rating(summary: dict[str, Any], key: str) -> str | None:
    value = summary.get(key)
    return str(value.get("rating")) if isinstance(value, dict) and value.get("rating") is not None else None


def _summary_text(summary: dict[str, Any], key: str) -> str | None:
    value = summary.get(key)
    return str(value.get("text")) if isinstance(value, dict) and value.get("text") is not None else None


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


def _provenance_ref_rows(conn: sqlite3.Connection, *, cutover_time: str) -> list[tuple[str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    seen: set[tuple[str, str]] = set()
    unmapped: list[dict[str, Any]] = []
    for row in _select_all(conn, "provenance_links"):
        for ref_type_key, ref_id_key, ref_version_key in (
            ("source_ref_type", "source_ref_id", "source_ref_version"),
            ("target_ref_type", "target_ref_id", "target_ref_version"),
        ):
            ref_type = str(row.get(ref_type_key) or "")
            ref_id = str(row.get(ref_id_key) or "")
            try:
                mutation = _ref_object_mutation(
                    ref_type,
                    ref_id,
                    _optional_text(row.get(ref_version_key)),
                    cutover_time,
                )
            except Exception as exc:  # noqa: BLE001 - backfill reports all unmapped legacy refs together.
                unmapped.append(
                    {
                        "id": row.get("id"),
                        "reason": str(exc),
                        "ref_role": ref_type_key.removesuffix("_ref_type"),
                        "ref_type": ref_type,
                        "ref_id": ref_id,
                    }
                )
                continue
            if mutation is None:
                continue
            object_type, business_key, *_ = mutation
            identity = (object_type, business_key)
            if identity not in seen:
                rows.append(mutation)
                seen.add(identity)
    if unmapped:
        raise LegacyBackfillUnmappedRefs(unmapped)
    return rows


def _provenance_link_relation_rows(
    conn: sqlite3.Connection, *, cutover_time: str
) -> list[tuple[str, str, str, dict[str, Any], str]]:
    rows: list[tuple[str, str, str, dict[str, Any], str]] = []
    unmapped: list[dict[str, Any]] = []
    for row in _select_all(conn, "provenance_links"):
        valid_from = str(row.get("created_at") or cutover_time)
        link_type = str(row.get("link_type") or "")
        relation_type = LINK_RELATION_TYPES.get(link_type)
        if relation_type is None:
            unmapped.append({"id": row.get("id"), "reason": "unsupported_link_type", "link_type": link_type})
            continue
        try:
            source_uid = ref_object_uid_for(str(row.get("source_ref_type") or ""), row.get("source_ref_id"))
            target_uid = ref_object_uid_for(str(row.get("target_ref_type") or ""), row.get("target_ref_id"))
        except Exception as exc:  # noqa: BLE001 - backfill reports all unmapped legacy refs together.
            unmapped.append({"id": row.get("id"), "reason": str(exc), "row": row})
            continue
        rows.append(
            (
                relation_type,
                source_uid,
                target_uid,
                {
                    "event_id": row.get("event_id"),
                    "ontology_run_id": "operational",
                    "source_ref_type": row.get("source_ref_type"),
                    "source_ref_id": row.get("source_ref_id"),
                    "source_ref_version": row.get("source_ref_version"),
                    "target_ref_type": row.get("target_ref_type"),
                    "target_ref_id": row.get("target_ref_id"),
                    "target_ref_version": row.get("target_ref_version"),
                    "redaction_policy": DEFAULT_REDACTION_POLICY,
                    "retention_class": FINANCIAL_RETENTION_CLASS,
                    "lineage_root_id": row.get("lineage_root_id") or row.get("event_id"),
                    "metadata": row.get("metadata") or row.get("metadata_json"),
                },
                valid_from,
            )
        )
    if unmapped:
        raise LegacyBackfillUnmappedRefs(unmapped)
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


def _ref_object_mutation(
    ref_type: str,
    ref_id: str,
    ref_version: str | None,
    cutover_time: str,
) -> tuple[str, str, dict[str, Any], str] | None:
    if not ref_id:
        raise LegacyBackfillUnmappedRefs([{"ref_type": ref_type, "ref_id": ref_id, "reason": "missing_ref_id"}])
    if ref_type == REF_ONTOLOGY_OBJECT_VERSION:
        return (
            "ObjectVersionRef",
            ref_object_uid_for(ref_type, ref_id),
            {
                "ref_id": ref_id,
                "object_uid": ref_id,
                "version_id": ref_version or ref_id,
                "temporal_confidence": "backfilled",
                "ontology_run_id": "operational",
            },
            cutover_time,
        )
    if ref_type == REF_RELATION_VERSION:
        return (
            "RelationVersionRef",
            ref_object_uid_for(ref_type, ref_id),
            {
                "ref_id": ref_id,
                "relation_uid": ref_id,
                "version_id": ref_version or ref_id,
                "ontology_run_id": "operational",
            },
            cutover_time,
        )
    if ref_type == REF_SCHEMA_DEFINITION:
        schema_version_value = 1
        if ref_version:
            try:
                schema_version_value = int(ref_version)
            except (TypeError, ValueError):
                schema_version_value = 1
        return (
            "SchemaDefinitionRef",
            ref_object_uid_for(ref_type, ref_id),
            {
                "ref_id": ref_id,
                "schema_kind": "unknown",
                "schema_name": ref_id,
                "schema_version_value": schema_version_value,
                "ontology_run_id": "operational",
            },
            cutover_time,
        )
    if ref_type == REF_ONTOLOGY_RUN:
        return ("OntologyRunRef", ref_object_uid_for(ref_type, ref_id), {"run_id": ref_id}, cutover_time)
    if ref_type == REF_AGENT_SESSION:
        return ("AgentSessionRef", ref_object_uid_for(ref_type, ref_id), {"session_id": ref_id}, cutover_time)
    if ref_type == REF_MODEL_CALL:
        return ("ModelCallRef", ref_object_uid_for(ref_type, ref_id), {"call_id": ref_id}, cutover_time)
    if ref_type == REF_TOOL_CALL:
        return ("ToolCallRef", ref_object_uid_for(ref_type, ref_id), {"call_id": ref_id}, cutover_time)
    if ref_type == REF_COMPUTED_SNAPSHOT_VERSION:
        return (
            "ComputedSnapshotRef",
            ref_object_uid_for(ref_type, ref_id),
            {"snapshot_key": ref_id, "snapshot_id": ref_version},
            cutover_time,
        )
    ref_object_uid_for(ref_type, ref_id)
    return None


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
