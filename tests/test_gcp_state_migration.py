from __future__ import annotations

import sqlite3
from pathlib import Path

from api.gcp_state_migration import SOURCE_DBS, StateMigrator, _normalize_pending_approval_rows, snapshot_sqlite_db


def test_snapshot_sqlite_db_includes_uncheckpointed_wal_rows(tmp_path):
    source = tmp_path / "live.sqlite3"
    conn = sqlite3.connect(source)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    conn.commit()
    conn.execute("INSERT INTO items (name) VALUES ('wal-row')")
    conn.commit()

    assert source.with_name(source.name + "-wal").exists()

    snapshot = snapshot_sqlite_db(source, tmp_path / "snapshot.sqlite3")

    with sqlite3.connect(snapshot) as snap_conn:
        rows = snap_conn.execute("SELECT name FROM items").fetchall()
    conn.close()
    assert rows == [("wal-row",)]


def test_retrieval_migration_contract_is_pinned():
    migration = Path("migrations/versions/20260429_0001_gcp_state_schema.py").read_text(encoding="utf-8")

    assert SOURCE_DBS["retrieval"] == "data_cache/retrieval/embeddings.db"
    assert "embedding vector(384) NOT NULL" in migration
    assert "USING hnsw (embedding vector_cosine_ops)" in migration


def test_pending_approval_migration_backfills_application_state():
    rows = _normalize_pending_approval_rows(
        [
            {"id": 1, "status": "approved", "resolved_at": "2026-05-03T01:00:00+00:00", "created_at": "old"},
            {"id": 2, "status": "rejected", "resolved_at": None, "created_at": "created"},
            {"id": 3, "status": "pending", "created_at": "created"},
        ]
    )

    assert rows[0]["application_status"] == "applied"
    assert rows[0]["application_completed_at"] == "2026-05-03T01:00:00+00:00"
    assert rows[1]["application_status"] == "not_applicable"
    assert rows[1]["application_completed_at"] == "created"
    assert rows[2]["application_status"] == "pending"
    assert rows[2]["application_attempts"] == 0


def test_ontology_migration_seeds_schema_definitions_for_pre_registry_sqlite(tmp_path):
    db = tmp_path / "source" / SOURCE_DBS["ontology"]
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ontology_runs (
                run_id TEXT PRIMARY KEY,
                as_of TEXT NOT NULL,
                source_status_json TEXT NOT NULL,
                required_modules_json TEXT NOT NULL,
                optional_modules_json TEXT NOT NULL,
                component_scores_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE snapshot_nodes (
                run_id TEXT NOT NULL,
                id TEXT NOT NULL,
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE snapshot_edges (
                run_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    class _FakeMigrator(StateMigrator):
        def __init__(self):
            super().__init__(snapshot_root=tmp_path / "source", run_id="run")
            self.upserts = []

        def _source_completed(self, source_name, source_sha256):  # noqa: ANN001, ANN201
            return False

        def _upsert_rows(self, table, columns, conflict, rows):  # noqa: ANN001, ANN201
            self.upserts.append((table, columns, list(rows)))

        def _record_source(self, result, status="completed"):  # noqa: ANN001, ANN201
            self.result = result

    migrator = _FakeMigrator()
    migrator.migrate_ontology()

    schema_upserts = [item for item in migrator.upserts if item[0] == "schema_definitions"]
    assert schema_upserts
    rows = schema_upserts[0][2]
    assert any(row["schema_kind"] == "ontology_object" and row["schema_name"] == "Position" for row in rows)
    edge_upsert = [item for item in migrator.upserts if item[0] == "ontology_edges"][0]
    assert "relation_schema_name" in edge_upsert[1]
