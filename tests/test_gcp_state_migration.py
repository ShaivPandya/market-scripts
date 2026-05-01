from __future__ import annotations

import sqlite3
from pathlib import Path

from api.gcp_state_migration import SOURCE_DBS, snapshot_sqlite_db


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
