from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.description = None
        self.rowcount = 0
        self._rows: list[dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()):
        self.conn.queries.append((sql, params))
        if "retrieval_chunks" in sql and "JOIN retrieval_documents" in sql:
            self.description = [
                SimpleNamespace(name=name)
                for name in [
                    "chunk_id",
                    "doc_id",
                    "chunk_index",
                    "content",
                    "heading",
                    "doc_type",
                    "ticker",
                    "source_path",
                    "created_at",
                    "score",
                ]
            ]
            self._rows = [
                {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "chunk_index": 0,
                    "content": "hello world",
                    "heading": "Heading",
                    "doc_type": "thesis",
                    "ticker": "MU",
                    "source_path": "gs://bucket/live/theses/MU.md",
                    "created_at": "2026-04-29T00:00:00+00:00",
                    "score": 0.91,
                }
            ]
        else:
            self.description = []
            self._rows = []
        return self

    def executemany(self, sql: str, params_seq):
        self.conn.queries.append((sql, tuple(params_seq)))
        self.rowcount = len(params_seq)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self):
        self.queries: list[tuple[str, tuple[Any, ...]]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def cursor(self):
        return _FakeCursor(self)

    def execute(self, sql: str, params: tuple[Any, ...] = ()):
        cur = _FakeCursor(self)
        return cur.execute(sql, params)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class _FakePool:
    def __init__(self, conn):
        self.conn = conn
        self.timeouts: list[float] = []
        self.returned: list[_FakeConn] = []
        self.closed = False

    def getconn(self, *, timeout):
        self.timeouts.append(timeout)
        return self.conn

    def putconn(self, conn):
        self.returned.append(conn)

    def close(self):
        self.closed = True


def test_portfolio_legacy_writes_are_blocked_in_production(monkeypatch):
    from portfolio import portfolio_db

    fake = _FakeConn()

    @contextmanager
    def fake_connect():
        yield fake

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(portfolio_db, "connect", fake_connect)

    with pytest.raises(RuntimeError, match="Legacy domain write blocked"):
        portfolio_db.save_positions([{"ticker": "MU", "asset": "equity", "direction": "long"}])

    assert fake.queries == []


def test_postgres_connect_uses_pool_when_enabled(monkeypatch):
    from api import postgres

    fake_conn = _FakeConn()
    fake_pool = _FakePool(fake_conn)

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("POSTGRES_POOL_ENABLED", "true")
    monkeypatch.setenv("POSTGRES_POOL_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setattr(postgres, "_POOLS", {})
    monkeypatch.setattr(postgres, "_new_pool", lambda *args, **kwargs: fake_pool)

    with postgres.connect() as conn:
        assert conn is fake_conn

    assert fake_pool.timeouts == [2.5]
    assert fake_pool.returned == [fake_conn]
    assert fake_conn.rollbacks == 1
    assert fake_conn.closed is False


def test_postgres_connect_can_bypass_pool(monkeypatch):
    from api import postgres

    fake_conn = _FakeConn()

    monkeypatch.setenv("POSTGRES_POOL_ENABLED", "false")
    monkeypatch.setattr(postgres, "open_connection", lambda *args, **kwargs: fake_conn)

    with postgres.connect() as conn:
        assert conn is fake_conn

    assert fake_conn.closed is True


def test_postgres_open_connection_remains_direct_when_pool_enabled(monkeypatch):
    from api import postgres

    fake_conn = _FakeConn()

    monkeypatch.setenv("POSTGRES_POOL_ENABLED", "true")
    monkeypatch.setattr(postgres, "_new_direct_connection", lambda *args, **kwargs: fake_conn)

    assert postgres.open_connection() is fake_conn


def test_retrieval_search_uses_pgvector_tables(monkeypatch):
    from api import retrieval

    fake = _FakeConn()
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(retrieval, "_embed_single", lambda _query: [0.0] * 384)
    monkeypatch.setattr(retrieval, "open_connection", lambda register_pgvector=False: fake)

    results = retrieval.search("memory cycle", doc_types=["thesis"], tickers=["MU"], top_k=3)

    assert results[0]["doc_id"] == "doc-1"
    sql = fake.queries[0][0]
    assert "retrieval_chunks" in sql
    assert "retrieval_documents" in sql
    assert "<=>" in sql


def test_retrieval_sqlite_migrates_legacy_chunk_schema(monkeypatch, tmp_path):
    from api import retrieval

    db_path = tmp_path / "embeddings.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            doc_type TEXT NOT NULL,
            source_path TEXT,
            ticker TEXT,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            heading TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "thesis-META",
            "thesis",
            "investment_theses/META.md",
            "META",
            "META AI ad thesis",
            "2026-05-12T00:00:00+00:00",
            "2026-05-12T00:00:00+00:00",
        ),
    )
    conn.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        (
            "chunk-1",
            "thesis-META",
            0,
            "META AI ad ranking catalysts",
            retrieval._embedding_to_blob([1.0, 0.0]),
            "Key Catalysts",
        ),
    )
    conn.commit()
    conn.close()

    existing = retrieval._conn
    if existing is not None:
        existing.close()
    monkeypatch.setattr(retrieval, "_conn", None)
    monkeypatch.setattr(retrieval, "_DB_PATH", db_path)
    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setattr(retrieval, "_embed_single", lambda _query: [1.0, 0.0])

    results = retrieval.search("META AI catalysts", doc_types=["thesis"], tickers=["META"], top_k=1)

    assert results[0]["doc_id"] == "thesis-META"
    migrated = retrieval._get_conn()
    columns = {row["name"] for row in migrated.execute("PRAGMA table_info(chunks)").fetchall()}
    assert "object_uid" in columns
    assert "content_hash" in columns
    migrated.close()
    monkeypatch.setattr(retrieval, "_conn", None)


def test_retrieval_search_falls_back_to_lexical_when_embeddings_unavailable(monkeypatch, tmp_path):
    from api import retrieval

    existing = retrieval._conn
    if existing is not None:
        existing.close()
    monkeypatch.setattr(retrieval, "_conn", None)
    monkeypatch.setattr(retrieval, "_DB_PATH", tmp_path / "embeddings.db")
    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")

    conn = retrieval._get_conn()
    conn.execute(
        """
        INSERT INTO documents (doc_id, doc_type, source_path, ticker, content, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "thesis-META",
            "thesis",
            "investment_theses/META.md",
            "META",
            "META AI ad thesis",
            "2026-05-12T00:00:00+00:00",
            "2026-05-12T00:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO chunks (chunk_id, doc_id, chunk_index, content, embedding, heading)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "chunk-1",
            "thesis-META",
            0,
            "META AI ad ranking catalysts and Reels monetization",
            retrieval._embedding_to_blob([0.0, 1.0]),
            "Key Catalysts",
        ),
    )
    conn.commit()

    def fail_embed(_query):
        raise ImportError("sentence-transformers not installed")

    monkeypatch.setattr(retrieval, "_embed_single", fail_embed)

    results = retrieval.search("META ad catalysts", doc_types=["thesis"], tickers=["META"], top_k=1)

    assert results[0]["doc_id"] == "thesis-META"
    assert results[0]["retrieval_mode"] == "lexical"
    assert "sentence-transformers not installed" in results[0]["fallback_reason"]
    conn.close()
    monkeypatch.setattr(retrieval, "_conn", None)


def test_compat_layer_maps_legacy_table_names(monkeypatch):
    import api.postgres_compat as compat

    fake = _FakeConn()
    monkeypatch.setattr(compat, "open_connection", lambda register_pgvector=False: fake)

    conn = compat.PostgresCompatConnection(
        table_map={
            "items": "central_bank_items",
            "transcripts": "industry_transcripts",
            "nodes": "ontology_nodes",
        }
    )
    conn.execute("SELECT * FROM items WHERE guid=?", ("g1",))
    conn.execute("SELECT * FROM transcripts WHERE ticker=?", ("MU",))
    conn.execute("SELECT * FROM nodes ORDER BY id")

    queries = [sql for sql, _params in fake.queries]
    assert "central_bank_items" in queries[0]
    assert "industry_transcripts" in queries[1]
    assert "ontology_nodes" in queries[2]
    assert "%s" in queries[0]


def test_compat_cursor_is_iterable_like_sqlite_cursor():
    from api.postgres_compat import CompatCursor, CompatRow

    cursor = CompatCursor(
        [
            CompatRow({"cid": 0, "name": "guid"}, ["cid", "name"]),
            CompatRow({"cid": 1, "name": "content_url"}, ["cid", "name"]),
        ]
    )

    assert {row[1] for row in cursor} == {"guid", "content_url"}
    assert cursor.fetchone() is None


def test_core_provenance_updates_do_not_emit_untyped_null_predicates(monkeypatch):
    import api.postgres_compat as compat
    from portfolio import core_db

    fake = _FakeConn()
    monkeypatch.setattr(compat, "open_connection", lambda register_pgvector=False: fake)
    monkeypatch.setattr(core_db, "_conn", compat.PostgresCompatConnection())

    core_db.set_action_run_provenance_event(42, "pv:action_run:42")
    core_db.set_workflow_run_provenance_event("workflow-1", "pv:workflow:1")
    core_db.set_pending_approval_provenance(17, provenance_event_id="pv:approval:17")
    core_db.set_pending_approval_provenance(18, origin_provenance_event_id="pv:origin:18")

    sql_statements = [sql for sql, _params in fake.queries]
    assert all("IS NOT NULL" not in sql for sql in sql_statements)
    assert any("UPDATE action_runs" in sql and "lineage_completeness = 'complete'" in sql for sql in sql_statements)
    assert any("UPDATE workflow_runs" in sql and "lineage_completeness = 'complete'" in sql for sql in sql_statements)
