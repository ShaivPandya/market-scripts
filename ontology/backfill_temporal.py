"""Backfill legacy ontology snapshots into temporal ontology tables."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ontology.repository import OntologyRepository
from ontology.temporal_repository import ObjectVersionWrite, RelationVersionWrite, TemporalOntologyRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill legacy ontology snapshot runs into temporal Postgres tables."
    )
    parser.add_argument("--db-path", type=Path, default=None, help="Legacy SQLite ontology DB path.")
    parser.add_argument("--run-id", default=None, help="Specific ontology run_id to backfill.")
    parser.add_argument("--all-runs", action="store_true", help="Backfill all known runs instead of the latest run.")
    parser.add_argument(
        "--cutover-time",
        default=None,
        help="Transaction-time timestamp to assign to backfilled rows. Defaults to current UTC time.",
    )
    args = parser.parse_args()

    legacy_repo = OntologyRepository(db_path=args.db_path)
    temporal_repo = TemporalOntologyRepository()
    cutover_time = args.cutover_time or datetime.now(UTC).isoformat()
    runs = _select_runs(legacy_repo, run_id=args.run_id, all_runs=bool(args.all_runs))
    node_count = 0
    edge_count = 0

    for run in runs:
        run_id = str(run["run_id"])
        as_of = str(run.get("as_of") or cutover_time)
        graph = legacy_repo.fetch_snapshot_graph(run_id, schema_mode="upgraded")
        for node in graph["nodes"]:
            temporal_repo.write_object_version(
                ObjectVersionWrite(
                    object_uid=str(node["id"]),
                    object_type=str(node["type"]),
                    business_key=str(node["id"]),
                    schema_name=str(node.get("schema_name") or node["type"]),
                    schema_version=int(node.get("schema_version") or 1),
                    properties=dict(node.get("properties") or {}),
                    valid_from=as_of,
                    tx_from=cutover_time,
                    provenance_event_id=run.get("provenance_event_id"),
                    temporal_confidence="backfilled",
                )
            )
            node_count += 1
        for edge in graph["edges"]:
            temporal_repo.write_relation_version(
                RelationVersionWrite(
                    relation_uid=_relation_uid(edge),
                    source_object_uid=str(edge["source_id"]),
                    target_object_uid=str(edge["target_id"]),
                    relation_type=str(edge["relation_type"]),
                    relation_schema_name=str(edge.get("relation_schema_name") or edge["relation_type"]),
                    relation_schema_version=int(edge.get("relation_schema_version") or 1),
                    properties=dict(edge.get("properties") or {}),
                    valid_from=as_of,
                    tx_from=cutover_time,
                    provenance_event_id=run.get("provenance_event_id"),
                    temporal_confidence="backfilled",
                )
            )
            edge_count += 1

    print(
        {
            "runs": len(runs),
            "object_versions": node_count,
            "relation_versions": edge_count,
            "cutover_time": cutover_time,
        }
    )


def _select_runs(repo: OntologyRepository, *, run_id: str | None, all_runs: bool) -> list[dict[str, Any]]:
    if run_id:
        run = repo.get_run(run_id)
        if run is None:
            raise SystemExit(f"Ontology run not found: {run_id}")
        return [run]
    if all_runs:
        return [repo.get_run(str(row["run_id"])) or row for row in repo.list_runs(limit=500)]
    latest = repo.get_latest_run()
    if latest is None:
        raise SystemExit("No ontology runs found to backfill.")
    return [latest]


def _relation_uid(edge: dict[str, Any]) -> str:
    return f"{edge['relation_type']}:{edge['source_id']}->{edge['target_id']}"


if __name__ == "__main__":
    main()
