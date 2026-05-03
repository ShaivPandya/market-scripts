from __future__ import annotations

import sqlite3
from pathlib import Path

from api.gcp_state_migration import SOURCE_DBS, _normalize_pending_approval_rows, snapshot_sqlite_db


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
