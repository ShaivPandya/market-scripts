from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ontology.models import OntologyEdge, OntologyNode

DEFAULT_DB_PATH = Path("data_cache") / "ontology" / "ontology.sqlite3"


class OntologyRepository:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            # Legacy tables are intentionally preserved for additive migration.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
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

    def upsert_nodes(self, nodes: list[OntologyNode]) -> None:
        if not nodes:
            return
        rows = [
            (
                n.id,
                n.type,
                n.label,
                json.dumps(n.properties, default=str),
            )
            for n in nodes
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO nodes(id, type, label, properties_json, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                  type=excluded.type,
                  label=excluded.label,
                  properties_json=excluded.properties_json,
                  updated_at=datetime('now')
                """,
                rows,
            )

    def upsert_edges(self, edges: list[OntologyEdge]) -> None:
        if not edges:
            return
        rows = [
            (
                e.source_id,
                e.target_id,
                e.relation_type,
                json.dumps(e.properties, default=str),
            )
            for e in edges
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO edges(source_id, target_id, relation_type, properties_json, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                  properties_json=excluded.properties_json,
                  updated_at=datetime('now')
                """,
                rows,
            )

    def upsert_graph(self, nodes: list[OntologyNode], edges: list[OntologyEdge]) -> None:
        self.upsert_nodes(nodes)
        self.upsert_edges(edges)

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
        node_rows = [
            (
                run_id,
                n.id,
                n.type,
                n.label,
                json.dumps(n.properties, default=str),
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
                    INSERT INTO snapshot_nodes(run_id, id, type, label, properties_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    """,
                    node_rows,
                )
            if edge_rows:
                conn.executemany(
                    """
                    INSERT INTO snapshot_edges(
                        run_id, source_id, target_id, relation_type, properties_json, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
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

    def fetch_snapshot_graph(self, run_id: str) -> dict[str, list[dict[str, Any]]]:
        with self._connect() as conn:
            node_rows = conn.execute(
                """
                SELECT id, type, label, properties_json, updated_at
                FROM snapshot_nodes
                WHERE run_id = ?
                ORDER BY type, id
                """,
                (run_id,),
            ).fetchall()
            edge_rows = conn.execute(
                """
                SELECT source_id, target_id, relation_type, properties_json, updated_at
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
                "properties": _load_json(r["properties_json"]),
                "updated_at": r["updated_at"],
            }
            for r in node_rows
        ]
        edges = [
            {
                "source_id": r["source_id"],
                "target_id": r["target_id"],
                "relation_type": r["relation_type"],
                "properties": _load_json(r["properties_json"]),
                "updated_at": r["updated_at"],
            }
            for r in edge_rows
        ]
        return {"nodes": nodes, "edges": edges}

    def fetch_snapshot_position_asset_sector_rows(self, run_id: str) -> list[dict[str, Any]]:
        sql = """
        SELECT
          p.id AS position_id,
          p.properties_json AS position_props,
          a.id AS asset_id,
          a.properties_json AS asset_props,
          s.id AS sector_id,
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
                    "position_props": _load_json(row["position_props"]),
                    "asset_id": row["asset_id"],
                    "asset_props": _load_json(row["asset_props"]),
                    "sector_id": row["sector_id"],
                    "sector_props": _load_json(row["sector_props"]),
                }
            )
        return out

    def fetch_snapshot_position_signal_evidence(self, run_id: str, position_id: str) -> list[dict[str, Any]]:
        sql = """
        SELECT
          s.id AS signal_id,
          s.properties_json AS signal_props,
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
                    "signal_props": _load_json(row["signal_props"]),
                    "edge_props": _load_json(row["edge_props"]),
                }
            )
        return out

    def fetch_graph(self) -> dict[str, list[dict[str, Any]]]:
        with self._connect() as conn:
            node_rows = conn.execute(
                "SELECT id, type, label, properties_json, updated_at FROM nodes ORDER BY type, id"
            ).fetchall()
            edge_rows = conn.execute(
                "SELECT source_id, target_id, relation_type, properties_json, updated_at FROM edges ORDER BY relation_type, source_id"
            ).fetchall()

        nodes = [
            {
                "id": r["id"],
                "type": r["type"],
                "label": r["label"],
                "properties": _load_json(r["properties_json"]),
                "updated_at": r["updated_at"],
            }
            for r in node_rows
        ]
        edges = [
            {
                "source_id": r["source_id"],
                "target_id": r["target_id"],
                "relation_type": r["relation_type"],
                "properties": _load_json(r["properties_json"]),
                "updated_at": r["updated_at"],
            }
            for r in edge_rows
        ]
        return {"nodes": nodes, "edges": edges}

    def fetch_position_asset_sector_rows(self) -> list[dict[str, Any]]:
        sql = """
        SELECT
          p.id AS position_id,
          p.properties_json AS position_props,
          a.id AS asset_id,
          a.properties_json AS asset_props,
          s.id AS sector_id,
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
                    "position_props": _load_json(row["position_props"]),
                    "asset_id": row["asset_id"],
                    "asset_props": _load_json(row["asset_props"]),
                    "sector_id": row["sector_id"],
                    "sector_props": _load_json(row["sector_props"]),
                }
            )
        return out

    def fetch_position_signal_evidence(self, position_id: str) -> list[dict[str, Any]]:
        sql = """
        SELECT
          s.id AS signal_id,
          s.properties_json AS signal_props,
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
                    "signal_props": _load_json(row["signal_props"]),
                    "edge_props": _load_json(row["edge_props"]),
                }
            )
        return out


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
