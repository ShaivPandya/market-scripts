from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from api.audit import emit_audit_event
from api.postgres import use_postgres_state
from api.postgres_compat import PostgresCompatConnection
from ontology.models import OntologyEdge, OntologyNode
from ontology.schema_definitions import (
    SCHEMA_KIND_ONTOLOGY_EDGE_PROPERTIES,
    SCHEMA_KIND_ONTOLOGY_OBJECT,
    SCHEMA_KIND_ONTOLOGY_RELATION,
    create_ontology_binding_tables,
    create_schema_registry_tables,
    current_definition_hash,
    ontology_schema_definitions,
    seed_schema_definitions,
)
from ontology.schemas.identity import canonical_ticker, position_id, sector_id
from ontology.schemas.registry import (
    OntologySchemaValidationError,
    normalize_edge,
    normalize_graph,
    normalize_node,
    validate_edge_relation,
    validate_graph_relations,
)
from ontology.schemas.relations import RELATION_TYPE_SQL_VALUES, RelationCardinality, get_relation_definition

logger = logging.getLogger("uvicorn.error")
SchemaMode = Literal["stored", "upgraded"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = _REPO_ROOT / "data_cache" / "ontology" / "ontology.sqlite3"


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _source_status_counts(source_status: dict[str, Any]) -> dict[str, int]:
    counts = {"ok": 0, "partial": 0, "error": 0, "other": 0}
    for state in source_status.values():
        status = str(state.get("status") if isinstance(state, dict) else "error")
        if status in counts:
            counts[status] += 1
        else:
            counts["other"] += 1
    return counts


def _emit_ontology_audit(
    action_name: str,
    *,
    status: str,
    object_refs: list[dict[str, Any]] | None = None,
    after_summary: dict[str, Any] | None = None,
    source_lineage: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    emit_audit_event(
        action_name,
        "ontology",
        status,
        object_refs=object_refs,
        after_summary=after_summary,
        source_lineage=source_lineage,
        error=error,
    )


def _candidate_db_paths() -> list[Path]:
    env_path = (os.getenv("ONTOLOGY_DB_PATH") or "").strip()
    paths: list[Path] = []
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.append(DEFAULT_DB_PATH)
    paths.append(
        Path(os.getenv("TMPDIR") or "/tmp") / "market-scripts" / "data_cache" / "ontology" / "ontology.sqlite3"
    )
    deduped: list[Path] = []
    for path in paths:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _resolve_default_db_path() -> Path:
    last_error: Exception | None = None
    for candidate in _candidate_db_paths():
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception as exc:
            last_error = exc
            logger.warning(
                "ontology repository init: failed to create db parent at %s; trying next fallback",
                str(candidate.parent),
                exc_info=True,
            )
    raise PermissionError("Unable to initialize ontology DB path") from last_error


class OntologyRepository:
    """Persist ontology graph rows and materialized snapshot runs.

    The repository stores queryable graph snapshots and legacy live graph
    tables. Canonical portfolio, thesis, and process state remains in the
    backing stores that ingestion reads from.
    """

    def __init__(self, db_path: Path | None = None):
        self._use_postgres = db_path is None and use_postgres_state()
        self.db_path = (
            None if self._use_postgres else Path(db_path) if db_path is not None else _resolve_default_db_path()
        )
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection | PostgresCompatConnection:
        if self._use_postgres:
            return PostgresCompatConnection(
                table_map={
                    "nodes": "ontology_nodes",
                    "edges": "ontology_edges",
                    "snapshot_nodes": "ontology_snapshot_nodes",
                    "snapshot_edges": "ontology_snapshot_edges",
                }
            )
        assert self.db_path is not None
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        if self._use_postgres:
            return
        with self._connect() as conn:
            # Legacy tables are intentionally preserved for additive migration.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    schema_name TEXT NOT NULL DEFAULT 'legacy',
                    schema_version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL CHECK (relation_type IN ({RELATION_TYPE_SQL_VALUES})),
                    properties_json TEXT NOT NULL,
                    schema_name TEXT NOT NULL DEFAULT 'legacy',
                    schema_version INTEGER NOT NULL DEFAULT 0,
                    relation_schema_name TEXT NOT NULL DEFAULT 'legacy',
                    relation_schema_version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (source_id, target_id, relation_type),
                    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ontology_runs (
                    run_id TEXT PRIMARY KEY,
                    as_of TEXT NOT NULL,
                    source_status_json TEXT NOT NULL,
                    required_modules_json TEXT NOT NULL,
                    optional_modules_json TEXT NOT NULL,
                    component_scores_json TEXT NOT NULL,
                    provenance_event_id TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshot_nodes (
                    run_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    schema_name TEXT NOT NULL DEFAULT 'legacy',
                    schema_version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (run_id, id),
                    FOREIGN KEY (run_id) REFERENCES ontology_runs(run_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS snapshot_edges (
                    run_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL CHECK (relation_type IN ({RELATION_TYPE_SQL_VALUES})),
                    properties_json TEXT NOT NULL,
                    schema_name TEXT NOT NULL DEFAULT 'legacy',
                    schema_version INTEGER NOT NULL DEFAULT 0,
                    relation_schema_name TEXT NOT NULL DEFAULT 'legacy',
                    relation_schema_version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (run_id, source_id, target_id, relation_type),
                    FOREIGN KEY (run_id) REFERENCES ontology_runs(run_id) ON DELETE CASCADE,
                    FOREIGN KEY (run_id, source_id) REFERENCES snapshot_nodes(run_id, id) ON DELETE CASCADE,
                    FOREIGN KEY (run_id, target_id) REFERENCES snapshot_nodes(run_id, id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON ontology_runs(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_nodes_run_type ON snapshot_nodes(run_id, type)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshot_edges_run_source ON snapshot_edges(run_id, source_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshot_edges_run_target ON snapshot_edges(run_id, target_id)"
            )
            for table in ("nodes", "edges", "snapshot_nodes", "snapshot_edges"):
                _ensure_schema_columns(conn, table)
            for table in ("edges", "snapshot_edges"):
                _ensure_relation_schema_columns(conn, table)
            _ensure_ontology_run_provenance_column(conn)
            create_schema_registry_tables(conn)
            create_ontology_binding_tables(conn)
            seed_schema_definitions(conn, ontology_schema_definitions())
            _ensure_relation_indexes(conn)
            _ensure_snapshot_query_indexes(conn)

    def upsert_nodes(self, nodes: list[OntologyNode]) -> None:
        if not nodes:
            return
        normalized_nodes = [normalize_node(n, allow_legacy=_allow_legacy_schemas()) for n in nodes]
        rows = [
            (
                n.id,
                n.type,
                n.label,
                json.dumps(n.properties, default=str),
                n.schema_name,
                n.schema_version,
            )
            for n in normalized_nodes
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO nodes(id, type, label, properties_json, schema_name, schema_version, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                  type=excluded.type,
                  label=excluded.label,
                  properties_json=excluded.properties_json,
                  schema_name=excluded.schema_name,
                  schema_version=excluded.schema_version,
                  updated_at=datetime('now')
                """,
                rows,
            )
        _emit_ontology_audit(
            "ontology.nodes.upserted",
            status="succeeded",
            object_refs=[{"type": "ontology_node", "id": n.id} for n in normalized_nodes[:5]],
            after_summary={
                "node_count": len(normalized_nodes),
                "node_ids": [n.id for n in normalized_nodes[:10]],
                "node_types": sorted({n.type for n in normalized_nodes}),
            },
        )

    def upsert_edges(self, edges: list[OntologyEdge]) -> None:
        if not edges:
            return
        normalized_edges: list[OntologyEdge] = []
        with self._connect() as conn:
            normalized_edges = _normalize_live_edges_for_storage(conn, edges)
            rows = [
                (
                    e.source_id,
                    e.target_id,
                    e.relation_type,
                    json.dumps(e.properties, default=str),
                    e.schema_name,
                    e.schema_version,
                    e.relation_schema_name,
                    e.relation_schema_version,
                )
                for e in normalized_edges
            ]
            try:
                conn.executemany(
                    """
                    INSERT INTO edges(
                        source_id,
                        target_id,
                        relation_type,
                        properties_json,
                        schema_name,
                        schema_version,
                        relation_schema_name,
                        relation_schema_version,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                      properties_json=excluded.properties_json,
                      schema_name=excluded.schema_name,
                      schema_version=excluded.schema_version,
                      relation_schema_name=excluded.relation_schema_name,
                      relation_schema_version=excluded.relation_schema_version,
                      updated_at=datetime('now')
                    """,
                    rows,
                )
            except Exception as exc:
                _emit_ontology_audit(
                    "ontology.edges.upserted",
                    status="failed",
                    after_summary={"edge_count": len(edges)},
                    error=str(exc),
                )
                _raise_edge_integrity_error(exc)
        _emit_ontology_audit(
            "ontology.edges.upserted",
            status="succeeded",
            object_refs=[
                {
                    "type": "ontology_edge",
                    "id": f"{e.source_id}:{e.relation_type}:{e.target_id}",
                }
                for e in normalized_edges[:5]
            ],
            after_summary={
                "edge_count": len(normalized_edges),
                "relation_types": sorted({e.relation_type for e in normalized_edges}),
            },
        )

    def upsert_graph(self, nodes: list[OntologyNode], edges: list[OntologyEdge]) -> None:
        normalized = normalize_graph(nodes, edges, allow_legacy=_allow_legacy_schemas())
        _emit_ontology_audit(
            "ontology.graph.upsert.started",
            status="started",
            after_summary={"node_count": len(normalized.nodes), "edge_count": len(normalized.edges)},
        )
        self.upsert_nodes(normalized.nodes)
        self.upsert_edges(normalized.edges)
        _emit_ontology_audit(
            "ontology.graph.upserted",
            status="succeeded",
            after_summary={"node_count": len(normalized.nodes), "edge_count": len(normalized.edges)},
        )

    def save_snapshot(
        self,
        *,
        run_id: str,
        as_of: str,
        source_status: dict[str, Any],
        required_modules: Sequence[str],
        optional_modules: Sequence[str],
        component_scores: dict[str, float],
        nodes: list[OntologyNode],
        edges: list[OntologyEdge],
    ) -> None:
        """Save one materialized semantic/risk graph run for read/query paths."""
        normalized = normalize_graph(nodes, edges, run_id=run_id, allow_legacy=_allow_legacy_schemas())
        nodes = normalized.nodes
        edges = normalized.edges
        node_rows = [
            (
                run_id,
                n.id,
                n.type,
                n.label,
                json.dumps(n.properties, default=str),
                n.schema_name,
                n.schema_version,
            )
            for n in nodes
        ]
        edge_rows = [
            (
                run_id,
                e.source_id,
                e.target_id,
                e.relation_type,
                json.dumps(e.properties, default=str),
                e.schema_name,
                e.schema_version,
                e.relation_schema_name,
                e.relation_schema_version,
            )
            for e in edges
        ]
        binding_rows = _schema_binding_rows(run_id, nodes, edges)

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO ontology_runs(
                        run_id,
                        as_of,
                        source_status_json,
                        required_modules_json,
                        optional_modules_json,
                        component_scores_json,
                        provenance_event_id,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(run_id) DO UPDATE SET
                        as_of=excluded.as_of,
                        source_status_json=excluded.source_status_json,
                        required_modules_json=excluded.required_modules_json,
                        optional_modules_json=excluded.optional_modules_json,
                        component_scores_json=excluded.component_scores_json,
                        provenance_event_id=excluded.provenance_event_id
                    """,
                    (
                        run_id,
                        as_of,
                        json.dumps(source_status, default=str),
                        json.dumps(list(required_modules), default=str),
                        json.dumps(list(optional_modules), default=str),
                        json.dumps(component_scores, default=str),
                        _ontology_run_provenance_id(run_id),
                    ),
                )
                conn.execute("DELETE FROM snapshot_nodes WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM snapshot_edges WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM ontology_run_schema_bindings WHERE run_id = ?", (run_id,))
                if node_rows:
                    conn.executemany(
                        """
                        INSERT INTO snapshot_nodes(
                            run_id, id, type, label, properties_json, schema_name, schema_version, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                        """,
                        node_rows,
                    )
                if edge_rows:
                    conn.executemany(
                        """
                        INSERT INTO snapshot_edges(
                            run_id,
                            source_id,
                            target_id,
                            relation_type,
                            properties_json,
                            schema_name,
                            schema_version,
                            relation_schema_name,
                            relation_schema_version,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                        """,
                        edge_rows,
                    )
                if binding_rows:
                    conn.executemany(
                        """
                        INSERT INTO ontology_run_schema_bindings(
                            run_id,
                            schema_kind,
                            schema_name,
                            schema_version,
                            definition_hash
                        )
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(run_id, schema_kind, schema_name, schema_version) DO UPDATE SET
                            definition_hash=excluded.definition_hash
                        """,
                        binding_rows,
                    )
            except Exception as exc:
                _emit_ontology_audit(
                    "ontology.snapshot.saved",
                    status="failed",
                    object_refs=[{"type": "ontology_run", "id": run_id}],
                    after_summary={"run_id": run_id, "node_count": len(nodes), "edge_count": len(edges)},
                    source_lineage={
                        "run_id": run_id,
                        "as_of": as_of,
                        "required_modules": list(required_modules),
                        "optional_modules": list(optional_modules),
                        "source_status_counts": _source_status_counts(source_status),
                        "component_scores_hash": _stable_hash(component_scores),
                    },
                    error=str(exc),
                )
                _raise_edge_integrity_error(exc)
        _emit_ontology_audit(
            "ontology.snapshot.saved",
            status="succeeded",
            object_refs=[{"type": "ontology_run", "id": run_id}],
            after_summary={
                "run_id": run_id,
                "as_of": as_of,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "required_module_count": len(required_modules),
                "optional_module_count": len(optional_modules),
            },
            source_lineage={
                "run_id": run_id,
                "as_of": as_of,
                "required_modules": list(required_modules),
                "optional_modules": list(optional_modules),
                "source_status_counts": _source_status_counts(source_status),
                "component_scores_hash": _stable_hash(component_scores),
                "source_status": source_status,
            },
        )
        _record_snapshot_provenance(run_id, source_status, nodes, edges, binding_rows)

    def prune_runs_older_than(self, *, days: int) -> int:
        if days <= 0:
            return 0
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM ontology_runs WHERE created_at < datetime('now', ?)",
                (f"-{days} days",),
            ).fetchone()
            to_delete = int(row["cnt"]) if row else 0
            conn.execute(
                "DELETE FROM ontology_runs WHERE created_at < datetime('now', ?)",
                (f"-{days} days",),
            )
        _emit_ontology_audit(
            "ontology.runs.pruned",
            status="succeeded",
            after_summary={"retention_days": days, "deleted_count": to_delete},
        )
        return to_delete

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  run_id,
                  as_of,
                  source_status_json,
                  required_modules_json,
                  optional_modules_json,
                  component_scores_json,
                  provenance_event_id,
                  created_at
                FROM ontology_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "as_of": row["as_of"],
            "source_status": _load_json(row["source_status_json"]),
            "required_modules": _load_json_list(row["required_modules_json"]),
            "optional_modules": _load_json_list(row["optional_modules_json"]),
            "component_scores": _load_json(row["component_scores_json"]),
            "provenance_event_id": row["provenance_event_id"],
            "created_at": row["created_at"],
            "schema_bindings": self._get_schema_bindings(run_id),
        }

    def get_latest_run(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  run_id,
                  as_of,
                  source_status_json,
                  required_modules_json,
                  optional_modules_json,
                  component_scores_json,
                  provenance_event_id,
                  created_at
                FROM ontology_runs
                ORDER BY created_at DESC, run_id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "as_of": row["as_of"],
            "source_status": _load_json(row["source_status_json"]),
            "required_modules": _load_json_list(row["required_modules_json"]),
            "optional_modules": _load_json_list(row["optional_modules_json"]),
            "component_scores": _load_json(row["component_scores_json"]),
            "provenance_event_id": row["provenance_event_id"],
            "created_at": row["created_at"],
            "schema_bindings": self._get_schema_bindings(str(row["run_id"])),
        }

    def _get_schema_bindings(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT schema_kind, schema_name, schema_version, definition_hash
                    FROM ontology_run_schema_bindings
                    WHERE run_id = ?
                    ORDER BY schema_kind, schema_name, schema_version
                    """,
                    (run_id,),
                ).fetchall()
            except Exception:
                return []
        return [
            {
                "schema_kind": str(row["schema_kind"]),
                "schema_name": str(row["schema_name"]),
                "schema_version": int(row["schema_version"] or 0),
                "definition_hash": str(row["definition_hash"]),
            }
            for row in rows
        ]

    def list_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  run_id,
                  as_of,
                  source_status_json,
                  required_modules_json,
                  provenance_event_id,
                  created_at
                FROM ontology_runs
                ORDER BY created_at DESC, run_id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            source_status = _load_json(row["source_status_json"])
            required_modules = _load_json_list(row["required_modules_json"])
            required_ok = True
            for module in required_modules:
                state = source_status.get(module) if isinstance(source_status, dict) else {}
                status = state.get("status") if isinstance(state, dict) else "error"
                if str(status or "error") != "ok":
                    required_ok = False
                    break
            out.append(
                {
                    "run_id": row["run_id"],
                    "as_of": row["as_of"],
                    "created_at": row["created_at"],
                    "provenance_event_id": row["provenance_event_id"],
                    "required_modules_ok": required_ok,
                }
            )
        return out

    def snapshot_has_positions(self, run_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 AS found
                FROM snapshot_nodes
                WHERE run_id = ?
                  AND type = 'Position'
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return row is not None

    def query_snapshot_positions_page(
        self,
        run_id: str,
        *,
        filters: dict[str, Any] | None,
        page: int,
        page_size: int,
        schema_mode: SchemaMode,
    ) -> dict[str, Any]:
        _validate_schema_mode(schema_mode)
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 100))
        offset = (safe_page - 1) * safe_page_size
        parts = _build_snapshot_position_query_parts(run_id, filters, use_postgres=self._use_postgres)

        with self._connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS total_results {parts['from_sql']} WHERE {parts['where_sql']}",
                tuple(parts["params"]),
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT
                  p.id AS position_id,
                  p.label AS position_label,
                  p.properties_json AS position_props,
                  p.schema_name AS position_schema_name,
                  p.schema_version AS position_schema_version,
                  p.updated_at AS position_updated_at,
                  a.id AS asset_id,
                  a.label AS asset_label,
                  a.properties_json AS asset_props,
                  a.schema_name AS asset_schema_name,
                  a.schema_version AS asset_schema_version,
                  a.updated_at AS asset_updated_at,
                  s.id AS sector_id,
                  s.label AS sector_label,
                  s.properties_json AS sector_props,
                  s.schema_name AS sector_schema_name,
                  s.schema_version AS sector_schema_version,
                  s.updated_at AS sector_updated_at,
                  pa.properties_json AS position_asset_edge_props,
                  pa.schema_name AS position_asset_edge_schema_name,
                  pa.schema_version AS position_asset_edge_schema_version,
                  pa.relation_schema_name AS position_asset_edge_relation_schema_name,
                  pa.relation_schema_version AS position_asset_edge_relation_schema_version,
                  pa.updated_at AS position_asset_edge_updated_at,
                  ase.properties_json AS asset_sector_edge_props,
                  ase.schema_name AS asset_sector_edge_schema_name,
                  ase.schema_version AS asset_sector_edge_schema_version,
                  ase.relation_schema_name AS asset_sector_edge_relation_schema_name,
                  ase.relation_schema_version AS asset_sector_edge_relation_schema_version,
                  ase.updated_at AS asset_sector_edge_updated_at,
                  {parts["risk_score_sort_expr"]} AS risk_score_value
                {parts["from_sql"]}
                WHERE {parts["where_sql"]}
                ORDER BY risk_score_value DESC, p.id ASC
                LIMIT ? OFFSET ?
                """,
                tuple([*parts["params"], safe_page_size, offset]),
            ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "position_id": row["position_id"],
                    "position_label": row["position_label"],
                    "position_props": _node_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        id_key="position_id",
                        type_value="Position",
                        label_key="position_label",
                        properties_key="position_props",
                        schema_name_key="position_schema_name",
                        schema_version_key="position_schema_version",
                    ),
                    "position_schema_name": row["position_schema_name"],
                    "position_schema_version": int(row["position_schema_version"] or 0),
                    "position_updated_at": row["position_updated_at"],
                    "asset_id": row["asset_id"],
                    "asset_label": row["asset_label"],
                    "asset_props": _node_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        id_key="asset_id",
                        type_value="Asset",
                        label_key="asset_label",
                        properties_key="asset_props",
                        schema_name_key="asset_schema_name",
                        schema_version_key="asset_schema_version",
                    ),
                    "asset_schema_name": row["asset_schema_name"],
                    "asset_schema_version": int(row["asset_schema_version"] or 0)
                    if row["asset_id"] is not None
                    else None,
                    "asset_updated_at": row["asset_updated_at"],
                    "sector_id": row["sector_id"],
                    "sector_label": row["sector_label"],
                    "sector_props": _node_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        id_key="sector_id",
                        type_value="Sector",
                        label_key="sector_label",
                        properties_key="sector_props",
                        schema_name_key="sector_schema_name",
                        schema_version_key="sector_schema_version",
                    ),
                    "sector_schema_name": row["sector_schema_name"],
                    "sector_schema_version": (
                        int(row["sector_schema_version"] or 0) if row["sector_id"] is not None else None
                    ),
                    "sector_updated_at": row["sector_updated_at"],
                    "position_asset_edge_props": _edge_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        source_id_key="position_id",
                        target_id_key="asset_id",
                        relation_type_value="references_asset",
                        properties_key="position_asset_edge_props",
                        schema_name_key="position_asset_edge_schema_name",
                        schema_version_key="position_asset_edge_schema_version",
                    ),
                    "position_asset_edge_schema_name": row["position_asset_edge_schema_name"],
                    "position_asset_edge_schema_version": (
                        int(row["position_asset_edge_schema_version"] or 0)
                        if row["position_asset_edge_updated_at"] is not None
                        else None
                    ),
                    "position_asset_edge_relation_schema_name": _row_value(
                        row, "position_asset_edge_relation_schema_name", "legacy"
                    ),
                    "position_asset_edge_relation_schema_version": int(
                        _row_value(row, "position_asset_edge_relation_schema_version", 0) or 0
                    ),
                    "position_asset_edge_updated_at": row["position_asset_edge_updated_at"],
                    "asset_sector_edge_props": _edge_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        source_id_key="asset_id",
                        target_id_key="sector_id",
                        relation_type_value="belongs_to_sector",
                        properties_key="asset_sector_edge_props",
                        schema_name_key="asset_sector_edge_schema_name",
                        schema_version_key="asset_sector_edge_schema_version",
                    ),
                    "asset_sector_edge_schema_name": row["asset_sector_edge_schema_name"],
                    "asset_sector_edge_schema_version": (
                        int(row["asset_sector_edge_schema_version"] or 0)
                        if row["asset_sector_edge_updated_at"] is not None
                        else None
                    ),
                    "asset_sector_edge_relation_schema_name": _row_value(
                        row, "asset_sector_edge_relation_schema_name", "legacy"
                    ),
                    "asset_sector_edge_relation_schema_version": int(
                        _row_value(row, "asset_sector_edge_relation_schema_version", 0) or 0
                    ),
                    "asset_sector_edge_updated_at": row["asset_sector_edge_updated_at"],
                }
            )

        return {
            "rows": out,
            "total_results": int(total_row["total_results"] or 0) if total_row is not None else 0,
            "page": safe_page,
            "page_size": safe_page_size,
        }

    def aggregate_snapshot_positions(
        self,
        run_id: str,
        *,
        filters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        parts = _build_snapshot_position_query_parts(run_id, filters, use_postgres=self._use_postgres)
        risk_expr = parts["risk_score_expr"]
        asset_expr = parts["asset_bucket_expr"]

        with self._connect() as conn:
            counts = conn.execute(
                f"""
                SELECT
                  COUNT(*) AS position_count,
                  SUM(CASE WHEN {risk_expr} >= 0.75 THEN 1 ELSE 0 END) AS high_count,
                  SUM(CASE WHEN {risk_expr} >= 0.5 AND {risk_expr} < 0.75 THEN 1 ELSE 0 END) AS medium_count,
                  SUM(CASE WHEN {risk_expr} < 0.5 OR {risk_expr} IS NULL THEN 1 ELSE 0 END) AS low_count,
                  AVG({risk_expr}) AS average_risk_score
                {parts["from_sql"]}
                WHERE {parts["where_sql"]}
                """,
                tuple(parts["params"]),
            ).fetchone()
            asset_rows = conn.execute(
                f"""
                SELECT
                  {asset_expr} AS asset_name,
                  COUNT(*) AS asset_count
                {parts["from_sql"]}
                WHERE {parts["where_sql"]}
                GROUP BY {asset_expr}
                ORDER BY {asset_expr}
                """,
                tuple(parts["params"]),
            ).fetchall()

        return {
            "position_count": int(_row_value(counts, "position_count", 0) or 0),
            "risk_buckets": {
                "high": int(_row_value(counts, "high_count", 0) or 0),
                "medium": int(_row_value(counts, "medium_count", 0) or 0),
                "low": int(_row_value(counts, "low_count", 0) or 0),
            },
            "asset_exposure_counts": {
                str(row["asset_name"] or "unknown"): int(row["asset_count"] or 0) for row in asset_rows
            },
            "average_risk_score": round(float(_row_value(counts, "average_risk_score", 0.0) or 0.0), 4),
        }

    def fetch_snapshot_position_signal_evidence_batch(
        self,
        run_id: str,
        position_ids: Sequence[str],
        *,
        schema_mode: SchemaMode,
    ) -> dict[str, list[dict[str, Any]]]:
        _validate_schema_mode(schema_mode)
        normalized_ids = _normalized_ids(position_ids)
        if not normalized_ids:
            return {}
        placeholders = ", ".join("?" for _ in normalized_ids)
        sql = f"""
        SELECT
          ps.source_id AS position_id,
          s.id AS signal_id,
          s.label AS signal_label,
          s.schema_name AS signal_schema_name,
          s.schema_version AS signal_schema_version,
          s.updated_at AS signal_updated_at,
          s.properties_json AS signal_props,
          ps.schema_name AS edge_schema_name,
          ps.schema_version AS edge_schema_version,
          ps.relation_schema_name AS relation_schema_name,
          ps.relation_schema_version AS relation_schema_version,
          ps.updated_at AS edge_updated_at,
          ps.properties_json AS edge_props
        FROM snapshot_edges ps
        JOIN snapshot_nodes s
          ON s.run_id = ps.run_id
         AND s.id = ps.target_id
         AND s.type = 'Signal'
        WHERE ps.run_id = ?
          AND ps.relation_type = 'exposed_to_signal'
          AND ps.source_id IN ({placeholders})
        ORDER BY ps.source_id, s.id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, tuple([run_id, *normalized_ids])).fetchall()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["position_id"]), []).append(
                {
                    "position_id": row["position_id"],
                    "signal_id": row["signal_id"],
                    "signal_label": row["signal_label"],
                    "signal_props": _node_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        id_key="signal_id",
                        type_value="Signal",
                        label_key="signal_label",
                        properties_key="signal_props",
                        schema_name_key="signal_schema_name",
                        schema_version_key="signal_schema_version",
                    ),
                    "signal_schema_name": row["signal_schema_name"],
                    "signal_schema_version": int(row["signal_schema_version"] or 0),
                    "signal_updated_at": row["signal_updated_at"],
                    "edge_props": _edge_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        source_id_key="position_id",
                        target_id_key="signal_id",
                        relation_type_value="exposed_to_signal",
                        properties_key="edge_props",
                        schema_name_key="edge_schema_name",
                        schema_version_key="edge_schema_version",
                    ),
                    "edge_schema_name": row["edge_schema_name"],
                    "edge_schema_version": int(row["edge_schema_version"] or 0),
                    "edge_relation_schema_name": _row_value(row, "relation_schema_name", "legacy"),
                    "edge_relation_schema_version": int(_row_value(row, "relation_schema_version", 0) or 0),
                    "edge_updated_at": row["edge_updated_at"],
                }
            )
        return grouped

    def fetch_snapshot_position_thesis_context_batch(
        self,
        run_id: str,
        position_ids: Sequence[str],
        *,
        schema_mode: SchemaMode,
    ) -> dict[str, dict[str, Any]]:
        _validate_schema_mode(schema_mode)
        normalized_ids = _normalized_ids(position_ids)
        if not normalized_ids:
            return {}

        placeholders = ", ".join("?" for _ in normalized_ids)
        thesis_sql = f"""
        SELECT
          ht.source_id AS position_id,
          ht.target_id AS thesis_id,
          ht.properties_json AS has_thesis_edge_props,
          ht.schema_name AS has_thesis_edge_schema_name,
          ht.schema_version AS has_thesis_edge_schema_version,
          ht.relation_schema_name AS has_thesis_edge_relation_schema_name,
          ht.relation_schema_version AS has_thesis_edge_relation_schema_version,
          ht.updated_at AS has_thesis_edge_updated_at,
          t.label AS thesis_label,
          t.properties_json AS thesis_props,
          t.schema_name AS thesis_schema_name,
          t.schema_version AS thesis_schema_version,
          t.updated_at AS thesis_updated_at
        FROM snapshot_edges ht
        JOIN snapshot_nodes t
          ON t.run_id = ht.run_id
         AND t.id = ht.target_id
         AND t.type = 'Thesis'
        WHERE ht.run_id = ?
          AND ht.relation_type = 'has_thesis'
          AND ht.source_id IN ({placeholders})
        ORDER BY ht.source_id, ht.target_id
        """
        with self._connect() as conn:
            thesis_rows = conn.execute(thesis_sql, tuple([run_id, *normalized_ids])).fetchall()

            grouped: dict[str, dict[str, Any]] = {position_id: {} for position_id in normalized_ids}
            thesis_ids: list[str] = []
            for row in thesis_rows:
                position_key = str(row["position_id"])
                thesis_node = _node_payload_from_row(
                    row,
                    prefix="thesis",
                    node_type="Thesis",
                    run_id=run_id,
                    schema_mode=schema_mode,
                )
                if thesis_node is None:
                    continue
                thesis_edge = _edge_payload_from_row(
                    row,
                    prefix="has_thesis_edge",
                    source_id_key="position_id",
                    target_id_key="thesis_id",
                    relation_type="has_thesis",
                    run_id=run_id,
                    schema_mode=schema_mode,
                )
                grouped[position_key] = {
                    "thesis": {"node": thesis_node, "edge": thesis_edge},
                    "evaluations": [],
                    "catalysts": [],
                }
                thesis_ids.append(thesis_node["id"])

            normalized_thesis_ids = _normalized_ids(thesis_ids)
            if not normalized_thesis_ids:
                return grouped

            thesis_placeholders = ", ".join("?" for _ in normalized_thesis_ids)
            evaluation_sql = f"""
            SELECT
              eb.source_id AS thesis_id,
              eb.target_id AS evaluation_id,
              eb.properties_json AS evaluated_by_edge_props,
              eb.schema_name AS evaluated_by_edge_schema_name,
              eb.schema_version AS evaluated_by_edge_schema_version,
              eb.relation_schema_name AS evaluated_by_edge_relation_schema_name,
              eb.relation_schema_version AS evaluated_by_edge_relation_schema_version,
              eb.updated_at AS evaluated_by_edge_updated_at,
              e.label AS evaluation_label,
              e.properties_json AS evaluation_props,
              e.schema_name AS evaluation_schema_name,
              e.schema_version AS evaluation_schema_version,
              e.updated_at AS evaluation_updated_at
            FROM snapshot_edges eb
            JOIN snapshot_nodes e
              ON e.run_id = eb.run_id
             AND e.id = eb.target_id
             AND e.type = 'Evaluation'
            WHERE eb.run_id = ?
              AND eb.relation_type = 'evaluated_by'
              AND eb.source_id IN ({thesis_placeholders})
            ORDER BY eb.source_id, eb.target_id
            """
            catalyst_sql = f"""
            SELECT
              hc.source_id AS thesis_id,
              hc.target_id AS catalyst_id,
              hc.properties_json AS has_catalyst_edge_props,
              hc.schema_name AS has_catalyst_edge_schema_name,
              hc.schema_version AS has_catalyst_edge_schema_version,
              hc.relation_schema_name AS has_catalyst_edge_relation_schema_name,
              hc.relation_schema_version AS has_catalyst_edge_relation_schema_version,
              hc.updated_at AS has_catalyst_edge_updated_at,
              c.label AS catalyst_label,
              c.properties_json AS catalyst_props,
              c.schema_name AS catalyst_schema_name,
              c.schema_version AS catalyst_schema_version,
              c.updated_at AS catalyst_updated_at
            FROM snapshot_edges hc
            JOIN snapshot_nodes c
              ON c.run_id = hc.run_id
             AND c.id = hc.target_id
             AND c.type = 'Catalyst'
            WHERE hc.run_id = ?
              AND hc.relation_type = 'has_catalyst'
              AND hc.source_id IN ({thesis_placeholders})
            ORDER BY hc.source_id, hc.target_id
            """
            evaluation_rows = conn.execute(evaluation_sql, tuple([run_id, *normalized_thesis_ids])).fetchall()
            catalyst_rows = conn.execute(catalyst_sql, tuple([run_id, *normalized_thesis_ids])).fetchall()

        positions_by_thesis = {
            ctx["thesis"]["node"]["id"]: position_key
            for position_key, ctx in grouped.items()
            if isinstance(ctx.get("thesis"), dict) and isinstance(ctx["thesis"].get("node"), dict)
        }
        for row in evaluation_rows:
            position_key = positions_by_thesis.get(str(row["thesis_id"]))
            if position_key is None:
                continue
            node = _node_payload_from_row(
                row,
                prefix="evaluation",
                node_type="Evaluation",
                run_id=run_id,
                schema_mode=schema_mode,
            )
            if node is None:
                continue
            edge = _edge_payload_from_row(
                row,
                prefix="evaluated_by_edge",
                source_id_key="thesis_id",
                target_id_key="evaluation_id",
                relation_type="evaluated_by",
                run_id=run_id,
                schema_mode=schema_mode,
            )
            grouped[position_key]["evaluations"].append({"node": node, "edge": edge})

        for row in catalyst_rows:
            position_key = positions_by_thesis.get(str(row["thesis_id"]))
            if position_key is None:
                continue
            node = _node_payload_from_row(
                row,
                prefix="catalyst",
                node_type="Catalyst",
                run_id=run_id,
                schema_mode=schema_mode,
            )
            if node is None:
                continue
            edge = _edge_payload_from_row(
                row,
                prefix="has_catalyst_edge",
                source_id_key="thesis_id",
                target_id_key="catalyst_id",
                relation_type="has_catalyst",
                run_id=run_id,
                schema_mode=schema_mode,
            )
            grouped[position_key]["catalysts"].append({"node": node, "edge": edge})

        return grouped

    def fetch_snapshot_graph(self, run_id: str, *, schema_mode: SchemaMode) -> dict[str, list[dict[str, Any]]]:
        _validate_schema_mode(schema_mode)
        with self._connect() as conn:
            node_rows = conn.execute(
                """
                SELECT id, type, label, properties_json, schema_name, schema_version, updated_at
                FROM snapshot_nodes
                WHERE run_id = ?
                ORDER BY type, id
                """,
                (run_id,),
            ).fetchall()
            edge_rows = conn.execute(
                """
                SELECT
                  source_id,
                  target_id,
                  relation_type,
                  properties_json,
                  schema_name,
                  schema_version,
                  relation_schema_name,
                  relation_schema_version,
                  updated_at
                FROM snapshot_edges
                WHERE run_id = ?
                ORDER BY relation_type, source_id
                """,
                (run_id,),
            ).fetchall()

        nodes = [
            {
                "id": r["id"],
                "type": r["type"],
                "label": r["label"],
                "properties": _node_properties_for_mode(r, schema_mode=schema_mode, run_id=run_id),
                "schema_name": r["schema_name"],
                "schema_version": int(r["schema_version"] or 0),
                "updated_at": r["updated_at"],
            }
            for r in node_rows
        ]
        edges = [
            {
                "source_id": r["source_id"],
                "target_id": r["target_id"],
                "relation_type": r["relation_type"],
                "properties": _edge_properties_for_mode(r, schema_mode=schema_mode, run_id=run_id),
                "schema_name": r["schema_name"],
                "schema_version": int(r["schema_version"] or 0),
                "relation_schema_name": _row_value(r, "relation_schema_name", "legacy"),
                "relation_schema_version": int(_row_value(r, "relation_schema_version", 0) or 0),
                "updated_at": r["updated_at"],
            }
            for r in edge_rows
        ]
        return {"nodes": nodes, "edges": edges}

    def fetch_snapshot_position_asset_sector_rows(
        self, run_id: str, *, schema_mode: SchemaMode
    ) -> list[dict[str, Any]]:
        _validate_schema_mode(schema_mode)
        sql = """
        SELECT
          p.id AS position_id,
          p.label AS position_label,
          p.schema_name AS position_schema_name,
          p.schema_version AS position_schema_version,
          p.properties_json AS position_props,
          a.id AS asset_id,
          a.label AS asset_label,
          a.schema_name AS asset_schema_name,
          a.schema_version AS asset_schema_version,
          a.properties_json AS asset_props,
          s.id AS sector_id,
          s.label AS sector_label,
          s.schema_name AS sector_schema_name,
          s.schema_version AS sector_schema_version,
          s.properties_json AS sector_props
        FROM snapshot_nodes p
        LEFT JOIN snapshot_edges pa
          ON pa.run_id = p.run_id
         AND pa.source_id = p.id
         AND pa.relation_type = 'references_asset'
        LEFT JOIN snapshot_nodes a
          ON a.run_id = p.run_id
         AND a.id = pa.target_id
        LEFT JOIN snapshot_edges ase
          ON ase.run_id = p.run_id
         AND ase.source_id = a.id
         AND ase.relation_type = 'belongs_to_sector'
        LEFT JOIN snapshot_nodes s
          ON s.run_id = p.run_id
         AND s.id = ase.target_id
        WHERE p.run_id = ?
          AND p.type = 'Position'
        ORDER BY p.id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (run_id,)).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "position_id": row["position_id"],
                    "position_props": _node_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        id_key="position_id",
                        type_value="Position",
                        label_key="position_label",
                        properties_key="position_props",
                        schema_name_key="position_schema_name",
                        schema_version_key="position_schema_version",
                    ),
                    "asset_id": row["asset_id"],
                    "asset_props": _node_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        id_key="asset_id",
                        type_value="Asset",
                        label_key="asset_label",
                        properties_key="asset_props",
                        schema_name_key="asset_schema_name",
                        schema_version_key="asset_schema_version",
                    ),
                    "sector_id": row["sector_id"],
                    "sector_props": _node_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        id_key="sector_id",
                        type_value="Sector",
                        label_key="sector_label",
                        properties_key="sector_props",
                        schema_name_key="sector_schema_name",
                        schema_version_key="sector_schema_version",
                    ),
                }
            )
        return out

    def fetch_snapshot_position_signal_evidence(
        self,
        run_id: str,
        position_id: str,
        *,
        schema_mode: SchemaMode,
    ) -> list[dict[str, Any]]:
        _validate_schema_mode(schema_mode)
        sql = """
        SELECT
          s.id AS signal_id,
          s.label AS signal_label,
          s.schema_name AS signal_schema_name,
          s.schema_version AS signal_schema_version,
          s.properties_json AS signal_props,
          ps.schema_name AS edge_schema_name,
          ps.schema_version AS edge_schema_version,
          ps.relation_schema_name AS relation_schema_name,
          ps.relation_schema_version AS relation_schema_version,
          ps.properties_json AS edge_props
        FROM snapshot_edges ps
        JOIN snapshot_nodes s
          ON s.run_id = ps.run_id
         AND s.id = ps.target_id
         AND s.type = 'Signal'
        WHERE ps.run_id = ?
          AND ps.source_id = ?
          AND ps.relation_type = 'exposed_to_signal'
        ORDER BY s.id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (run_id, position_id)).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "signal_id": row["signal_id"],
                    "signal_props": _node_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        id_key="signal_id",
                        type_value="Signal",
                        label_key="signal_label",
                        properties_key="signal_props",
                        schema_name_key="signal_schema_name",
                        schema_version_key="signal_schema_version",
                    ),
                    "edge_props": _edge_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        source_id_key=None,
                        target_id_key="signal_id",
                        relation_type_value="exposed_to_signal",
                        properties_key="edge_props",
                        schema_name_key="edge_schema_name",
                        schema_version_key="edge_schema_version",
                    ),
                    "edge_relation_schema_name": _row_value(row, "relation_schema_name", "legacy"),
                    "edge_relation_schema_version": int(_row_value(row, "relation_schema_version", 0) or 0),
                }
            )
        return out

    def fetch_snapshot_all_position_signal_evidence(
        self,
        run_id: str,
        *,
        schema_mode: SchemaMode,
    ) -> dict[str, list[dict[str, Any]]]:
        _validate_schema_mode(schema_mode)
        sql = """
        SELECT
          ps.source_id AS position_id,
          s.id AS signal_id,
          s.label AS signal_label,
          s.schema_name AS signal_schema_name,
          s.schema_version AS signal_schema_version,
          s.properties_json AS signal_props,
          ps.schema_name AS edge_schema_name,
          ps.schema_version AS edge_schema_version,
          ps.relation_schema_name AS relation_schema_name,
          ps.relation_schema_version AS relation_schema_version,
          ps.properties_json AS edge_props
        FROM snapshot_edges ps
        JOIN snapshot_nodes s
          ON s.run_id = ps.run_id
         AND s.id = ps.target_id
         AND s.type = 'Signal'
        WHERE ps.run_id = ?
          AND ps.relation_type = 'exposed_to_signal'
        ORDER BY ps.source_id, s.id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (run_id,)).fetchall()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["position_id"], []).append(
                {
                    "signal_id": row["signal_id"],
                    "signal_props": _node_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        id_key="signal_id",
                        type_value="Signal",
                        label_key="signal_label",
                        properties_key="signal_props",
                        schema_name_key="signal_schema_name",
                        schema_version_key="signal_schema_version",
                    ),
                    "edge_props": _edge_properties_for_mode(
                        row,
                        schema_mode=schema_mode,
                        run_id=run_id,
                        source_id_key="position_id",
                        target_id_key="signal_id",
                        relation_type_value="exposed_to_signal",
                        properties_key="edge_props",
                        schema_name_key="edge_schema_name",
                        schema_version_key="edge_schema_version",
                    ),
                    "edge_relation_schema_name": _row_value(row, "relation_schema_name", "legacy"),
                    "edge_relation_schema_version": int(_row_value(row, "relation_schema_version", 0) or 0),
                }
            )
        return grouped

    def fetch_graph(self) -> dict[str, list[dict[str, Any]]]:
        with self._connect() as conn:
            node_rows = conn.execute(
                """
                SELECT id, type, label, properties_json, schema_name, schema_version, updated_at
                FROM nodes
                ORDER BY type, id
                """
            ).fetchall()
            edge_rows = conn.execute(
                """
                SELECT
                  source_id,
                  target_id,
                  relation_type,
                  properties_json,
                  schema_name,
                  schema_version,
                  relation_schema_name,
                  relation_schema_version,
                  updated_at
                FROM edges
                ORDER BY relation_type, source_id
                """
            ).fetchall()

        nodes = [
            {
                "id": r["id"],
                "type": r["type"],
                "label": r["label"],
                "properties": _load_node_properties(r),
                "schema_name": r["schema_name"],
                "schema_version": int(r["schema_version"] or 0),
                "updated_at": r["updated_at"],
            }
            for r in node_rows
        ]
        edges = [
            {
                "source_id": r["source_id"],
                "target_id": r["target_id"],
                "relation_type": r["relation_type"],
                "properties": _load_edge_properties(r),
                "schema_name": r["schema_name"],
                "schema_version": int(r["schema_version"] or 0),
                "relation_schema_name": _row_value(r, "relation_schema_name", "legacy"),
                "relation_schema_version": int(_row_value(r, "relation_schema_version", 0) or 0),
                "updated_at": r["updated_at"],
            }
            for r in edge_rows
        ]
        return {"nodes": nodes, "edges": edges}

    def fetch_position_asset_sector_rows(self) -> list[dict[str, Any]]:
        sql = """
        SELECT
          p.id AS position_id,
          p.label AS position_label,
          p.schema_name AS position_schema_name,
          p.schema_version AS position_schema_version,
          p.properties_json AS position_props,
          a.id AS asset_id,
          a.label AS asset_label,
          a.schema_name AS asset_schema_name,
          a.schema_version AS asset_schema_version,
          a.properties_json AS asset_props,
          s.id AS sector_id,
          s.label AS sector_label,
          s.schema_name AS sector_schema_name,
          s.schema_version AS sector_schema_version,
          s.properties_json AS sector_props
        FROM nodes p
        LEFT JOIN edges pa
          ON pa.source_id = p.id
         AND pa.relation_type = 'references_asset'
        LEFT JOIN nodes a
          ON a.id = pa.target_id
        LEFT JOIN edges ase
          ON ase.source_id = a.id
         AND ase.relation_type = 'belongs_to_sector'
        LEFT JOIN nodes s
          ON s.id = ase.target_id
        WHERE p.type = 'Position'
        ORDER BY p.id
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "position_id": row["position_id"],
                    "position_props": _load_node_properties(
                        row,
                        id_key="position_id",
                        type_value="Position",
                        label_key="position_label",
                        properties_key="position_props",
                        schema_name_key="position_schema_name",
                        schema_version_key="position_schema_version",
                    ),
                    "asset_id": row["asset_id"],
                    "asset_props": _load_node_properties(
                        row,
                        id_key="asset_id",
                        type_value="Asset",
                        label_key="asset_label",
                        properties_key="asset_props",
                        schema_name_key="asset_schema_name",
                        schema_version_key="asset_schema_version",
                    ),
                    "sector_id": row["sector_id"],
                    "sector_props": _load_node_properties(
                        row,
                        id_key="sector_id",
                        type_value="Sector",
                        label_key="sector_label",
                        properties_key="sector_props",
                        schema_name_key="sector_schema_name",
                        schema_version_key="sector_schema_version",
                    ),
                }
            )
        return out

    def fetch_position_signal_evidence(self, position_id: str) -> list[dict[str, Any]]:
        sql = """
        SELECT
          s.id AS signal_id,
          s.label AS signal_label,
          s.schema_name AS signal_schema_name,
          s.schema_version AS signal_schema_version,
          s.properties_json AS signal_props,
          ps.schema_name AS edge_schema_name,
          ps.schema_version AS edge_schema_version,
          ps.relation_schema_name AS relation_schema_name,
          ps.relation_schema_version AS relation_schema_version,
          ps.properties_json AS edge_props
        FROM edges ps
        JOIN nodes s
          ON s.id = ps.target_id
         AND s.type = 'Signal'
        WHERE ps.source_id = ?
          AND ps.relation_type = 'exposed_to_signal'
        ORDER BY s.id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (position_id,)).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "signal_id": row["signal_id"],
                    "signal_props": _load_node_properties(
                        row,
                        id_key="signal_id",
                        type_value="Signal",
                        label_key="signal_label",
                        properties_key="signal_props",
                        schema_name_key="signal_schema_name",
                        schema_version_key="signal_schema_version",
                    ),
                    "edge_props": _load_edge_properties(
                        row,
                        source_id_key=None,
                        target_id_key="signal_id",
                        relation_type_value="exposed_to_signal",
                        properties_key="edge_props",
                        schema_name_key="edge_schema_name",
                        schema_version_key="edge_schema_version",
                    ),
                    "edge_relation_schema_name": _row_value(row, "relation_schema_name", "legacy"),
                    "edge_relation_schema_version": int(_row_value(row, "relation_schema_version", 0) or 0),
                }
            )
        return out

    def backfill_schema_versions(self, *, dry_run: bool = True) -> dict[str, Any]:
        """Convert legacy ontology property bags to typed v1 payloads.

        The write mode rewrites each graph scope atomically: current graph rows
        first, then each snapshot run independently. Evaluation and catalyst IDs
        may be canonicalized, so dependent edges are rewritten with the node ID
        map returned by the schema registry.
        """
        report: dict[str, Any] = {"dry_run": dry_run, "scopes": [], "warnings": [], "errors": []}
        with self._connect() as conn:
            live_nodes = _fetch_node_envelopes(conn, table="nodes")
            live_edges = _fetch_edge_envelopes(conn, table="edges")
            live_graph = _normalize_backfill_scope(
                report,
                scope={"scope": "live"},
                nodes=live_nodes,
                edges=live_edges,
                dry_run=dry_run,
            )
            if live_graph is not None and not dry_run:
                conn.execute("DELETE FROM edges")
                conn.execute("DELETE FROM nodes")
                _insert_live_nodes(conn, live_graph.nodes)
                _insert_live_edges(conn, live_graph.edges)

            run_rows = conn.execute("SELECT run_id FROM ontology_runs ORDER BY run_id").fetchall()
            for run_row in run_rows:
                run_id = str(run_row["run_id"])
                nodes = _fetch_node_envelopes(conn, table="snapshot_nodes", run_id=run_id)
                edges = _fetch_edge_envelopes(conn, table="snapshot_edges", run_id=run_id)
                graph = _normalize_backfill_scope(
                    report,
                    scope={"scope": "snapshot", "run_id": run_id},
                    nodes=nodes,
                    edges=edges,
                    dry_run=dry_run,
                    run_id=run_id,
                )
                if graph is not None and not dry_run:
                    conn.execute("DELETE FROM snapshot_edges WHERE run_id = ?", (run_id,))
                    conn.execute("DELETE FROM snapshot_nodes WHERE run_id = ?", (run_id,))
                    _insert_snapshot_nodes(conn, run_id, graph.nodes)
                    _insert_snapshot_edges(conn, run_id, graph.edges)

        return report


def _normalize_backfill_scope(
    report: dict[str, Any],
    *,
    scope: dict[str, Any],
    nodes: list[OntologyNode],
    edges: list[OntologyEdge],
    dry_run: bool,
    run_id: str | None = None,
) -> Any:
    try:
        graph = normalize_graph(
            nodes,
            edges,
            run_id=run_id,
            allow_legacy=True,
            skip_optional_invalid=True,
            require_core_edges=True,
        )
    except OntologySchemaValidationError as exc:
        error = {**scope, "error": str(exc)}
        report["errors"].append(error)
        report["scopes"].append({**scope, "nodes": 0, "edges": 0, "rewritten_ids": 0, "error": str(exc)})
        if not dry_run:
            raise OntologySchemaValidationError(f"Cannot backfill ontology {scope}: {exc}") from exc
        return None

    report["scopes"].append(
        {
            **scope,
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "rewritten_ids": sum(1 for old, new in graph.node_id_map.items() if old != new),
        }
    )
    report["warnings"].extend(graph.warnings)
    return graph


def _load_json(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _load_json_list(raw: Any) -> list[str]:
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        return [str(v) for v in parsed if isinstance(v, (str, int, float))]
    except Exception:
        return []


def _allow_legacy_schemas() -> bool:
    value = (os.getenv("ONTOLOGY_STRICT_SCHEMAS") or "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def _ontology_run_provenance_id(run_id: str) -> str:
    try:
        from api import provenance

        return provenance.deterministic_id("pv:ontology_run", run_id)
    except Exception:
        return f"pv:ontology_run:{str(run_id).replace('+', '_')}"


def _ensure_schema_columns(conn: sqlite3.Connection, table: str) -> None:
    existing = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if "schema_name" not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN schema_name TEXT NOT NULL DEFAULT 'legacy'")
    if "schema_version" not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 0")


def _ensure_relation_schema_columns(conn: sqlite3.Connection, table: str) -> None:
    existing = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if "relation_schema_name" not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN relation_schema_name TEXT NOT NULL DEFAULT 'legacy'")
    if "relation_schema_version" not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN relation_schema_version INTEGER NOT NULL DEFAULT 0")


def _ensure_ontology_run_provenance_column(conn: sqlite3.Connection) -> None:
    existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(ontology_runs)").fetchall()}
    if "provenance_event_id" not in existing:
        conn.execute("ALTER TABLE ontology_runs ADD COLUMN provenance_event_id TEXT")


def _ensure_relation_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique_source_relation
        ON edges(source_id, relation_type)
        WHERE relation_type IN ('references_asset', 'belongs_to_sector')
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique_target_relation
        ON edges(target_id, relation_type)
        WHERE relation_type IN ('emits_signal', 'evaluated_by', 'has_catalyst')
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique_has_thesis_source
        ON edges(source_id, relation_type)
        WHERE relation_type = 'has_thesis'
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique_has_thesis_target
        ON edges(target_id, relation_type)
        WHERE relation_type = 'has_thesis'
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_edges_unique_run_source_relation
        ON snapshot_edges(run_id, source_id, relation_type)
        WHERE relation_type IN ('references_asset', 'belongs_to_sector')
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_edges_unique_run_target_relation
        ON snapshot_edges(run_id, target_id, relation_type)
        WHERE relation_type IN ('emits_signal', 'evaluated_by', 'has_catalyst')
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_edges_unique_has_thesis_run_source
        ON snapshot_edges(run_id, source_id, relation_type)
        WHERE relation_type = 'has_thesis'
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_edges_unique_has_thesis_run_target
        ON snapshot_edges(run_id, target_id, relation_type)
        WHERE relation_type = 'has_thesis'
        """
    )


def _ensure_snapshot_query_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_snapshot_nodes_run_type_id
        ON snapshot_nodes(run_id, type, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_snapshot_edges_run_relation_source_target
        ON snapshot_edges(run_id, relation_type, source_id, target_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_snapshot_nodes_position_asset_lookup
        ON snapshot_nodes(
            run_id,
            lower(COALESCE(json_extract(properties_json, '$.asset'), '')),
            id
        )
        WHERE type = 'Position'
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_snapshot_nodes_position_risk_sort
        ON snapshot_nodes(
            run_id,
            CAST(COALESCE(json_extract(properties_json, '$.risk_score'), 0) AS REAL) DESC,
            id
        )
        WHERE type = 'Position'
        """
    )


def _row_value(row: Any, key: str | None, default: Any = None) -> Any:
    if key is None:
        return default
    try:
        if hasattr(row, "keys") and key not in row.keys():
            return default
        return row[key]
    except Exception:
        return default


def _validate_schema_mode(schema_mode: SchemaMode) -> None:
    if schema_mode not in {"stored", "upgraded"}:
        raise ValueError("schema_mode must be 'stored' or 'upgraded'")


def _node_properties_for_mode(row: Any, *, schema_mode: SchemaMode, **kwargs: Any) -> dict[str, Any]:
    if schema_mode == "stored":
        return _load_json(_row_value(row, kwargs.get("properties_key", "properties_json")))
    return _load_node_properties(row, **kwargs)


def _edge_properties_for_mode(row: Any, *, schema_mode: SchemaMode, **kwargs: Any) -> dict[str, Any]:
    if schema_mode == "stored":
        return _load_json(_row_value(row, kwargs.get("properties_key", "properties_json")))
    return _load_edge_properties(row, **kwargs)


def _load_node_properties(
    row: Any,
    *,
    run_id: str | None = None,
    id_key: str = "id",
    type_key: str = "type",
    type_value: str | None = None,
    label_key: str = "label",
    properties_key: str = "properties_json",
    schema_name_key: str = "schema_name",
    schema_version_key: str = "schema_version",
) -> dict[str, Any]:
    node_id = _row_value(row, id_key)
    if node_id is None:
        return {}
    props = _load_json(_row_value(row, properties_key))
    node_type = type_value or _row_value(row, type_key)
    label = _row_value(row, label_key, str(node_id))
    if not node_type:
        return props
    allow_legacy = _allow_legacy_schemas()
    try:
        normalized = normalize_node(
            OntologyNode(
                id=str(node_id),
                type=node_type,
                label=str(label or node_id),
                properties=props,
                schema_name=str(_row_value(row, schema_name_key, props.get("schema_name") or "legacy")),
                schema_version=int(_row_value(row, schema_version_key, props.get("schema_version") or 0) or 0),
            ),
            run_id=run_id,
            allow_legacy=allow_legacy,
        )
        return normalized.properties
    except Exception:
        if not allow_legacy:
            raise
        return props


def _load_edge_properties(
    row: Any,
    *,
    run_id: str | None = None,
    source_id_key: str | None = "source_id",
    target_id_key: str | None = "target_id",
    relation_type_key: str = "relation_type",
    relation_type_value: str | None = None,
    properties_key: str = "properties_json",
    schema_name_key: str = "schema_name",
    schema_version_key: str = "schema_version",
) -> dict[str, Any]:
    props = _load_json(_row_value(row, properties_key))
    relation_type = relation_type_value or _row_value(row, relation_type_key)
    if not relation_type:
        return props
    allow_legacy = _allow_legacy_schemas()
    try:
        normalized = normalize_edge(
            OntologyEdge(
                source_id=str(_row_value(row, source_id_key, "unknown")),
                target_id=str(_row_value(row, target_id_key, "unknown")),
                relation_type=str(relation_type),
                properties=props,
                schema_name=str(_row_value(row, schema_name_key, props.get("schema_name") or "legacy")),
                schema_version=int(_row_value(row, schema_version_key, props.get("schema_version") or 0) or 0),
            ),
            run_id=run_id,
            allow_legacy=allow_legacy,
        )
        return normalized.properties
    except Exception:
        if not allow_legacy:
            raise
        return props


def _fetch_node_envelopes(
    conn: sqlite3.Connection | PostgresCompatConnection,
    *,
    table: str,
    run_id: str | None = None,
) -> list[OntologyNode]:
    where = " WHERE run_id = ?" if run_id is not None else ""
    rows = conn.execute(
        f"SELECT id, type, label, properties_json, schema_name, schema_version FROM {table}{where} ORDER BY id",
        (run_id,) if run_id is not None else (),
    ).fetchall()
    return [
        OntologyNode(
            id=str(row["id"]),
            type=row["type"],
            label=str(row["label"]),
            properties=_load_json(row["properties_json"]),
            schema_name=str(row["schema_name"] or "legacy"),
            schema_version=int(row["schema_version"] or 0),
        )
        for row in rows
    ]


def _fetch_edge_envelopes(
    conn: sqlite3.Connection | PostgresCompatConnection,
    *,
    table: str,
    run_id: str | None = None,
) -> list[OntologyEdge]:
    where = " WHERE run_id = ?" if run_id is not None else ""
    rows = conn.execute(
        f"""
        SELECT
            source_id,
            target_id,
            relation_type,
            properties_json,
            schema_name,
            schema_version,
            relation_schema_name,
            relation_schema_version
        FROM {table}{where}
        ORDER BY source_id, target_id, relation_type
        """,
        (run_id,) if run_id is not None else (),
    ).fetchall()
    return [
        OntologyEdge(
            source_id=str(row["source_id"]),
            target_id=str(row["target_id"]),
            relation_type=str(row["relation_type"]),
            properties=_load_json(row["properties_json"]),
            schema_name=str(row["schema_name"] or "legacy"),
            schema_version=int(row["schema_version"] or 0),
            relation_schema_name=str(_row_value(row, "relation_schema_name", "legacy") or "legacy"),
            relation_schema_version=int(_row_value(row, "relation_schema_version", 0) or 0),
        )
        for row in rows
    ]


def _insert_live_nodes(conn: sqlite3.Connection | PostgresCompatConnection, nodes: list[OntologyNode]) -> None:
    if not nodes:
        return
    conn.executemany(
        """
        INSERT INTO nodes(id, type, label, properties_json, schema_name, schema_version, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        [
            (n.id, n.type, n.label, json.dumps(n.properties, default=str), n.schema_name, n.schema_version)
            for n in nodes
        ],
    )


def _insert_live_edges(conn: sqlite3.Connection | PostgresCompatConnection, edges: list[OntologyEdge]) -> None:
    if not edges:
        return
    conn.executemany(
        """
        INSERT INTO edges(
            source_id,
            target_id,
            relation_type,
            properties_json,
            schema_name,
            schema_version,
            relation_schema_name,
            relation_schema_version,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        [
            (
                e.source_id,
                e.target_id,
                e.relation_type,
                json.dumps(e.properties, default=str),
                e.schema_name,
                e.schema_version,
                e.relation_schema_name,
                e.relation_schema_version,
            )
            for e in edges
        ],
    )


def _insert_snapshot_nodes(
    conn: sqlite3.Connection | PostgresCompatConnection,
    run_id: str,
    nodes: list[OntologyNode],
) -> None:
    if not nodes:
        return
    conn.executemany(
        """
        INSERT INTO snapshot_nodes(
            run_id, id, type, label, properties_json, schema_name, schema_version, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        [
            (run_id, n.id, n.type, n.label, json.dumps(n.properties, default=str), n.schema_name, n.schema_version)
            for n in nodes
        ],
    )


def _insert_snapshot_edges(
    conn: sqlite3.Connection | PostgresCompatConnection,
    run_id: str,
    edges: list[OntologyEdge],
) -> None:
    if not edges:
        return
    conn.executemany(
        """
        INSERT INTO snapshot_edges(
            run_id,
            source_id,
            target_id,
            relation_type,
            properties_json,
            schema_name,
            schema_version,
            relation_schema_name,
            relation_schema_version,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        [
            (
                run_id,
                e.source_id,
                e.target_id,
                e.relation_type,
                json.dumps(e.properties, default=str),
                e.schema_name,
                e.schema_version,
                e.relation_schema_name,
                e.relation_schema_version,
            )
            for e in edges
        ],
    )


def _schema_binding_rows(
    run_id: str, nodes: list[OntologyNode], edges: list[OntologyEdge]
) -> list[tuple[str, str, str, int, str]]:
    keys: set[tuple[str, str, int]] = set()
    for node in nodes:
        keys.add((SCHEMA_KIND_ONTOLOGY_OBJECT, node.schema_name, int(node.schema_version)))
    for edge in edges:
        keys.add((SCHEMA_KIND_ONTOLOGY_EDGE_PROPERTIES, edge.schema_name, int(edge.schema_version)))
        keys.add((SCHEMA_KIND_ONTOLOGY_RELATION, edge.relation_schema_name, int(edge.relation_schema_version)))
    return [
        (
            run_id,
            schema_kind,
            schema_name,
            schema_version,
            current_definition_hash(schema_kind, schema_name, schema_version),
        )
        for schema_kind, schema_name, schema_version in sorted(keys)
    ]


def _record_snapshot_provenance(
    run_id: str,
    source_status: dict[str, Any],
    nodes: list[OntologyNode],
    edges: list[OntologyEdge],
    binding_rows: list[tuple[str, str, str, int, str]],
) -> None:
    try:
        from api import provenance
    except Exception:
        return

    event_id = _ontology_run_provenance_id(run_id)
    for source_name, state in source_status.items():
        lineage = state.get("lineage") if isinstance(state, dict) else None
        adapter_event_id = lineage.get("provenance_event_id") if isinstance(lineage, dict) else None
        if not adapter_event_id:
            continue
        provenance.link_refs(
            event_id=event_id,
            source_ref_type="ontology_run",
            source_ref_id=run_id,
            target_ref_type="source_adapter_run",
            target_ref_id=str(adapter_event_id),
            link_type="used",
            metadata={
                "source_name": source_name,
                "status": state.get("status") if isinstance(state, dict) else None,
                "quality": state.get("quality") if isinstance(state, dict) else None,
            },
        )

    for _run_id, schema_kind, schema_name, schema_version, definition_hash in binding_rows:
        provenance.link_refs(
            event_id=event_id,
            source_ref_type="ontology_run",
            source_ref_id=run_id,
            target_ref_type="schema_definition",
            target_ref_id=f"{schema_kind}:{schema_name}",
            target_ref_version=str(schema_version),
            link_type="schema_bound",
            metadata={"definition_hash": definition_hash},
        )

    for node in nodes:
        provenance.link_refs(
            event_id=event_id,
            source_ref_type="ontology_run",
            source_ref_id=run_id,
            target_ref_type="ontology_object_version",
            target_ref_id=f"{run_id}:{node.id}",
            target_ref_version=f"{node.schema_name}:{node.schema_version}",
            link_type="produced",
            metadata={"node_id": node.id, "node_type": node.type, "schema_name": node.schema_name},
        )

    for edge in edges:
        relation_ref = f"{run_id}:{edge.source_id}:{edge.relation_type}:{edge.target_id}"
        provenance.link_refs(
            event_id=event_id,
            source_ref_type="ontology_run",
            source_ref_id=run_id,
            target_ref_type="relation_version",
            target_ref_id=relation_ref,
            target_ref_version=f"{edge.relation_schema_name}:{edge.relation_schema_version}",
            link_type="produced",
            metadata={
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation_type": edge.relation_type,
                "schema_name": edge.schema_name,
                "schema_version": edge.schema_version,
            },
        )


def _normalize_live_edges_for_storage(
    conn: sqlite3.Connection | PostgresCompatConnection,
    edges: list[OntologyEdge],
) -> list[OntologyEdge]:
    endpoint_ids = {edge.source_id for edge in edges} | {edge.target_id for edge in edges}
    node_types = _fetch_node_type_map(conn, endpoint_ids)
    normalized_edges = [
        validate_edge_relation(edge, node_types, allow_legacy=_allow_legacy_schemas()) for edge in edges
    ]

    cardinality_relations = {
        edge.relation_type
        for edge in normalized_edges
        if get_relation_definition(edge.relation_type).cardinality != RelationCardinality.MANY_TO_MANY
    }
    if not cardinality_relations:
        return normalized_edges

    existing_edges = [
        edge for edge in _fetch_edge_envelopes(conn, table="edges") if edge.relation_type in cardinality_relations
    ]
    combined = {(edge.source_id, edge.target_id, edge.relation_type): edge for edge in existing_edges}
    combined.update({(edge.source_id, edge.target_id, edge.relation_type): edge for edge in normalized_edges})
    combined_endpoint_ids = {edge.source_id for edge in combined.values()} | {
        edge.target_id for edge in combined.values()
    }
    combined_node_types = _fetch_node_type_map(conn, combined_endpoint_ids)
    relation_nodes = [
        OntologyNode(id=node_id, type=node_type, label=node_id, properties={})
        for node_id, node_type in combined_node_types.items()
    ]
    report = validate_graph_relations(
        relation_nodes,
        list(combined.values()),
        require_core_edges=False,
        skip_optional_invalid=False,
    )
    report.raise_for_errors()
    return normalized_edges


def _fetch_node_type_map(
    conn: sqlite3.Connection | PostgresCompatConnection,
    node_ids: set[str],
) -> dict[str, str]:
    if not node_ids:
        return {}
    placeholders = ", ".join("?" for _ in node_ids)
    rows = conn.execute(
        f"SELECT id, type FROM nodes WHERE id IN ({placeholders})",
        tuple(sorted(node_ids)),
    ).fetchall()
    return {str(row["id"]): str(row["type"]) for row in rows}


def _raise_edge_integrity_error(exc: Exception) -> None:
    module = exc.__class__.__module__
    if isinstance(exc, sqlite3.IntegrityError) or module.startswith("psycopg"):
        raise OntologySchemaValidationError(f"Ontology edge integrity violation: {exc}") from exc
    raise exc


def _build_snapshot_position_query_parts(
    run_id: str,
    filters: dict[str, Any] | None,
    *,
    use_postgres: bool,
) -> dict[str, Any]:
    normalized = _normalize_snapshot_position_filters(filters)
    risk_score_expr = _snapshot_json_float_expr("p.properties_json", "risk_score", use_postgres=use_postgres)
    risk_score_sort_expr = f"COALESCE({risk_score_expr}, 0.0)"
    asset_expr = _snapshot_json_text_expr("p.properties_json", "asset", use_postgres=use_postgres)
    asset_bucket_expr = f"COALESCE(NULLIF(lower({asset_expr}), ''), 'unknown')"
    from_sql = """
        FROM snapshot_nodes p
        LEFT JOIN snapshot_edges pa
          ON pa.run_id = p.run_id
         AND pa.source_id = p.id
         AND pa.relation_type = 'references_asset'
        LEFT JOIN snapshot_nodes a
          ON a.run_id = p.run_id
         AND a.id = pa.target_id
        LEFT JOIN snapshot_edges ase
          ON ase.run_id = p.run_id
         AND ase.source_id = a.id
         AND ase.relation_type = 'belongs_to_sector'
        LEFT JOIN snapshot_nodes s
          ON s.run_id = p.run_id
         AND s.id = ase.target_id
    """
    where_clauses = ["p.run_id = ?", "p.type = 'Position'"]
    params: list[Any] = [run_id]

    if normalized["position_ids"]:
        placeholders = ", ".join("?" for _ in normalized["position_ids"])
        where_clauses.append(f"p.id IN ({placeholders})")
        params.extend(normalized["position_ids"])
    if normalized["sector_ids"]:
        placeholders = ", ".join("?" for _ in normalized["sector_ids"])
        where_clauses.append(f"s.id IN ({placeholders})")
        params.extend(normalized["sector_ids"])
    if normalized["assets"]:
        placeholders = ", ".join("?" for _ in normalized["assets"])
        where_clauses.append(f"lower({asset_expr}) IN ({placeholders})")
        params.extend(normalized["assets"])
    if normalized["min_risk_score"] is not None:
        where_clauses.append(f"{risk_score_expr} >= ?")
        params.append(float(normalized["min_risk_score"]))

    return {
        "from_sql": from_sql,
        "where_sql": " AND ".join(where_clauses),
        "params": params,
        "risk_score_expr": risk_score_expr,
        "risk_score_sort_expr": risk_score_sort_expr,
        "asset_bucket_expr": asset_bucket_expr,
    }


def _normalize_snapshot_position_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    raw = filters if isinstance(filters, dict) else {}
    position_ids: list[str] = []
    for ticker in raw.get("tickers", []) if isinstance(raw.get("tickers"), list) else []:
        try:
            position_ids.append(position_id(canonical_ticker(ticker)))
        except Exception:
            continue
    sector_ids: list[str] = []
    for sector_name in raw.get("sectors", []) if isinstance(raw.get("sectors"), list) else []:
        text = str(sector_name or "").strip()
        if text:
            sector_ids.append(sector_id(text))
    assets = [
        str(asset).strip().lower()
        for asset in (raw.get("assets", []) if isinstance(raw.get("assets"), list) else [])
        if str(asset or "").strip()
    ]
    return {
        "position_ids": _normalized_ids(position_ids),
        "sector_ids": _normalized_ids(sector_ids),
        "assets": _normalized_ids(assets),
        "min_risk_score": _to_float(raw.get("min_risk_score")),
    }


def _normalized_ids(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _snapshot_json_text_expr(column: str, field: str, *, use_postgres: bool) -> str:
    if use_postgres:
        return f"({column}::jsonb ->> '{field}')"
    return f"json_extract({column}, '$.{field}')"


def _snapshot_json_float_expr(column: str, field: str, *, use_postgres: bool) -> str:
    text_expr = _snapshot_json_text_expr(column, field, use_postgres=use_postgres)
    if use_postgres:
        return f"NULLIF({text_expr}, '')::double precision"
    return f"CAST({text_expr} AS REAL)"


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _node_payload_from_row(
    row: Any,
    *,
    prefix: str,
    node_type: str,
    run_id: str,
    schema_mode: SchemaMode,
) -> dict[str, Any] | None:
    node_id = _row_value(row, f"{prefix}_id")
    if node_id is None:
        return None
    return {
        "id": str(node_id),
        "type": node_type,
        "label": str(_row_value(row, f"{prefix}_label", node_id) or node_id),
        "properties": _node_properties_for_mode(
            row,
            schema_mode=schema_mode,
            run_id=run_id,
            id_key=f"{prefix}_id",
            type_value=node_type,
            label_key=f"{prefix}_label",
            properties_key=f"{prefix}_props",
            schema_name_key=f"{prefix}_schema_name",
            schema_version_key=f"{prefix}_schema_version",
        ),
        "schema_name": _row_value(row, f"{prefix}_schema_name"),
        "schema_version": int(_row_value(row, f"{prefix}_schema_version", 0) or 0),
        "updated_at": _row_value(row, f"{prefix}_updated_at"),
    }


def _edge_payload_from_row(
    row: Any,
    *,
    prefix: str,
    source_id_key: str,
    target_id_key: str,
    relation_type: str,
    run_id: str,
    schema_mode: SchemaMode,
) -> dict[str, Any] | None:
    source_id = _row_value(row, source_id_key)
    target_id = _row_value(row, target_id_key)
    updated_at = _row_value(row, f"{prefix}_updated_at")
    if source_id is None or target_id is None or updated_at is None:
        return None
    return {
        "source_id": str(source_id),
        "target_id": str(target_id),
        "relation_type": relation_type,
        "properties": _edge_properties_for_mode(
            row,
            schema_mode=schema_mode,
            run_id=run_id,
            source_id_key=source_id_key,
            target_id_key=target_id_key,
            relation_type_value=relation_type,
            properties_key=f"{prefix}_props",
            schema_name_key=f"{prefix}_schema_name",
            schema_version_key=f"{prefix}_schema_version",
        ),
        "schema_name": _row_value(row, f"{prefix}_schema_name"),
        "schema_version": int(_row_value(row, f"{prefix}_schema_version", 0) or 0),
        "relation_schema_name": _row_value(row, f"{prefix}_relation_schema_name", "legacy"),
        "relation_schema_version": int(_row_value(row, f"{prefix}_relation_schema_version", 0) or 0),
        "updated_at": updated_at,
    }
