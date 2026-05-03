from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from api.postgres import use_postgres_state
from api.postgres_compat import PostgresCompatConnection
from ontology.models import OntologyEdge, OntologyNode
from ontology.schemas.registry import normalize_edge, normalize_graph, normalize_node

logger = logging.getLogger("uvicorn.error")

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = _REPO_ROOT / "data_cache" / "ontology" / "ontology.sqlite3"


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
                """
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    schema_name TEXT NOT NULL DEFAULT 'legacy',
                    schema_version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (source_id, target_id, relation_type)
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
                """
                CREATE TABLE IF NOT EXISTS snapshot_edges (
                    run_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    schema_name TEXT NOT NULL DEFAULT 'legacy',
                    schema_version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (run_id, source_id, target_id, relation_type),
                    FOREIGN KEY (run_id) REFERENCES ontology_runs(run_id) ON DELETE CASCADE
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

    def upsert_edges(self, edges: list[OntologyEdge]) -> None:
        if not edges:
            return
        normalized_edges = [normalize_edge(e, allow_legacy=_allow_legacy_schemas()) for e in edges]
        rows = [
            (
                e.source_id,
                e.target_id,
                e.relation_type,
                json.dumps(e.properties, default=str),
                e.schema_name,
                e.schema_version,
            )
            for e in normalized_edges
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO edges(
                    source_id, target_id, relation_type, properties_json, schema_name, schema_version, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                  properties_json=excluded.properties_json,
                  schema_name=excluded.schema_name,
                  schema_version=excluded.schema_version,
                  updated_at=datetime('now')
                """,
                rows,
            )

    def upsert_graph(self, nodes: list[OntologyNode], edges: list[OntologyEdge]) -> None:
        normalized = normalize_graph(nodes, edges, allow_legacy=_allow_legacy_schemas())
        self.upsert_nodes(normalized.nodes)
        self.upsert_edges(normalized.edges)

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
            )
            for e in edges
        ]

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ontology_runs(
                    run_id,
                    as_of,
                    source_status_json,
                    required_modules_json,
                    optional_modules_json,
                    component_scores_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(run_id) DO UPDATE SET
                    as_of=excluded.as_of,
                    source_status_json=excluded.source_status_json,
                    required_modules_json=excluded.required_modules_json,
                    optional_modules_json=excluded.optional_modules_json,
                    component_scores_json=excluded.component_scores_json
                """,
                (
                    run_id,
                    as_of,
                    json.dumps(source_status, default=str),
                    json.dumps(list(required_modules), default=str),
                    json.dumps(list(optional_modules), default=str),
                    json.dumps(component_scores, default=str),
                ),
            )
            conn.execute("DELETE FROM snapshot_nodes WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM snapshot_edges WHERE run_id = ?", (run_id,))
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
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    edge_rows,
                )

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
            "created_at": row["created_at"],
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
            "created_at": row["created_at"],
        }

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
                    "required_modules_ok": required_ok,
                }
            )
        return out

    def fetch_snapshot_graph(self, run_id: str) -> dict[str, list[dict[str, Any]]]:
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
                SELECT source_id, target_id, relation_type, properties_json, schema_name, schema_version, updated_at
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
                "properties": _load_node_properties(r, run_id=run_id),
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
                "properties": _load_edge_properties(r, run_id=run_id),
                "schema_name": r["schema_name"],
                "schema_version": int(r["schema_version"] or 0),
                "updated_at": r["updated_at"],
            }
            for r in edge_rows
        ]
        return {"nodes": nodes, "edges": edges}

    def fetch_snapshot_position_asset_sector_rows(self, run_id: str) -> list[dict[str, Any]]:
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
                    "position_props": _load_node_properties(
                        row,
                        run_id=run_id,
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
                        run_id=run_id,
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

    def fetch_snapshot_position_signal_evidence(self, run_id: str, position_id: str) -> list[dict[str, Any]]:
        sql = """
        SELECT
          s.id AS signal_id,
          s.label AS signal_label,
          s.schema_name AS signal_schema_name,
          s.schema_version AS signal_schema_version,
          s.properties_json AS signal_props,
          ps.schema_name AS edge_schema_name,
          ps.schema_version AS edge_schema_version,
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
                    "signal_props": _load_node_properties(
                        row,
                        run_id=run_id,
                        id_key="signal_id",
                        type_value="Signal",
                        label_key="signal_label",
                        properties_key="signal_props",
                        schema_name_key="signal_schema_name",
                        schema_version_key="signal_schema_version",
                    ),
                    "edge_props": _load_edge_properties(
                        row,
                        run_id=run_id,
                        source_id_key=None,
                        target_id_key="signal_id",
                        relation_type_value="exposed_to_signal",
                        properties_key="edge_props",
                        schema_name_key="edge_schema_name",
                        schema_version_key="edge_schema_version",
                    ),
                }
            )
        return out

    def fetch_snapshot_all_position_signal_evidence(self, run_id: str) -> dict[str, list[dict[str, Any]]]:
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
                    "signal_props": _load_node_properties(
                        row,
                        run_id=run_id,
                        id_key="signal_id",
                        type_value="Signal",
                        label_key="signal_label",
                        properties_key="signal_props",
                        schema_name_key="signal_schema_name",
                        schema_version_key="signal_schema_version",
                    ),
                    "edge_props": _load_edge_properties(
                        row,
                        run_id=run_id,
                        source_id_key="position_id",
                        target_id_key="signal_id",
                        relation_type_value="exposed_to_signal",
                        properties_key="edge_props",
                        schema_name_key="edge_schema_name",
                        schema_version_key="edge_schema_version",
                    ),
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
                SELECT source_id, target_id, relation_type, properties_json, schema_name, schema_version, updated_at
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
        report: dict[str, Any] = {"dry_run": dry_run, "scopes": [], "warnings": []}
        with self._connect() as conn:
            live_nodes = _fetch_node_envelopes(conn, table="nodes")
            live_edges = _fetch_edge_envelopes(conn, table="edges")
            live_graph = normalize_graph(live_nodes, live_edges, allow_legacy=True, skip_optional_invalid=True)
            report["scopes"].append(
                {
                    "scope": "live",
                    "nodes": len(live_graph.nodes),
                    "edges": len(live_graph.edges),
                    "rewritten_ids": sum(1 for old, new in live_graph.node_id_map.items() if old != new),
                }
            )
            report["warnings"].extend(live_graph.warnings)
            if not dry_run:
                conn.execute("DELETE FROM edges")
                conn.execute("DELETE FROM nodes")
                _insert_live_nodes(conn, live_graph.nodes)
                _insert_live_edges(conn, live_graph.edges)

            run_rows = conn.execute("SELECT run_id FROM ontology_runs ORDER BY run_id").fetchall()
            for run_row in run_rows:
                run_id = str(run_row["run_id"])
                nodes = _fetch_node_envelopes(conn, table="snapshot_nodes", run_id=run_id)
                edges = _fetch_edge_envelopes(conn, table="snapshot_edges", run_id=run_id)
                graph = normalize_graph(nodes, edges, run_id=run_id, allow_legacy=True, skip_optional_invalid=True)
                report["scopes"].append(
                    {
                        "scope": "snapshot",
                        "run_id": run_id,
                        "nodes": len(graph.nodes),
                        "edges": len(graph.edges),
                        "rewritten_ids": sum(1 for old, new in graph.node_id_map.items() if old != new),
                    }
                )
                report["warnings"].extend(graph.warnings)
                if not dry_run:
                    conn.execute("DELETE FROM snapshot_edges WHERE run_id = ?", (run_id,))
                    conn.execute("DELETE FROM snapshot_nodes WHERE run_id = ?", (run_id,))
                    _insert_snapshot_nodes(conn, run_id, graph.nodes)
                    _insert_snapshot_edges(conn, run_id, graph.edges)

        return report


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


def _ensure_schema_columns(conn: sqlite3.Connection, table: str) -> None:
    existing = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if "schema_name" not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN schema_name TEXT NOT NULL DEFAULT 'legacy'")
    if "schema_version" not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 0")


def _row_value(row: Any, key: str | None, default: Any = None) -> Any:
    if key is None:
        return default
    try:
        if hasattr(row, "keys") and key not in row.keys():
            return default
        return row[key]
    except Exception:
        return default


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
        SELECT source_id, target_id, relation_type, properties_json, schema_name, schema_version
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
        INSERT INTO edges(source_id, target_id, relation_type, properties_json, schema_name, schema_version, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        [
            (
                e.source_id,
                e.target_id,
                e.relation_type,
                json.dumps(e.properties, default=str),
                e.schema_name,
                e.schema_version,
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
            run_id, source_id, target_id, relation_type, properties_json, schema_name, schema_version, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
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
            )
            for e in edges
        ],
    )
