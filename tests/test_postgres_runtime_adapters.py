from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any


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


def test_portfolio_uses_postgres_in_production(monkeypatch):
    from portfolio import portfolio_db

    fake = _FakeConn()

    @contextmanager
    def fake_connect():
        yield fake

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(portfolio_db, "connect", fake_connect)

    portfolio_db.save_positions([{"ticker": "MU", "asset": "equity", "direction": "long"}])

    assert any("DELETE FROM positions WHERE role = %s" in sql for sql, _params in fake.queries)
    assert any("INSERT INTO positions" in sql for sql, _params in fake.queries)


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
