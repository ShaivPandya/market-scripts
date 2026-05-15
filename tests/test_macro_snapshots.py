from __future__ import annotations

import asyncio
from contextlib import contextmanager

from api.snapshot_store import SnapshotRecord


def _record(snapshot_key: str, payload: dict, *, status: str = "ok", error: str | None = None) -> SnapshotRecord:
    return SnapshotRecord(
        snapshot_key=snapshot_key,
        payload=payload,
        as_of_date="2026-05-01",
        fetched_at="2026-05-01T23:30:00",
        status=status,
        error=error,
        version=1,
        artifact_uri=None,
    )


def test_snapshot_backed_response_uses_fresh_snapshot(monkeypatch):
    from api import macro_snapshots as ms
    from api.cache import daily_cache

    snapshot = {"series": {}, "_meta": {"snapshot": {"key": "labor_market:current:v1", "stale": False}}}
    monkeypatch.setattr(ms, "get_snapshot_response", lambda _key: snapshot)

    result = ms.get_snapshot_backed_response(
        snapshot_key="labor_market:current:v1",
        cache=daily_cache,
        cache_key="test:fresh-snapshot",
        source="labor_market",
        load_payload=lambda: (_ for _ in ()).throw(AssertionError("loader should not run")),
    )

    assert result is snapshot


def test_snapshot_backed_response_writes_snapshot_on_cache_miss(monkeypatch):
    from api import macro_snapshots as ms
    from api.cache import daily_cache, delete_cached

    key = "test:macro-cache-miss"
    delete_cached(daily_cache, key)
    writes: list[tuple[str, dict]] = []

    monkeypatch.setattr(ms, "get_snapshot_response", lambda _key: None)
    monkeypatch.setattr(
        ms,
        "write_snapshot_success",
        lambda snapshot_key, payload, **_kwargs: (
            writes.append((snapshot_key, payload)) or _record(snapshot_key, payload)
        ),
    )

    result = ms.get_snapshot_backed_response(
        snapshot_key="housing:current:v1",
        cache=daily_cache,
        cache_key=key,
        source="housing",
        load_payload=lambda: {"latest": {"housing_starts": {"date": "2026-05-01", "value": 1.0}}},
    )

    assert writes[0][0] == "housing:current:v1"
    assert writes[0][1]["latest"] == {"housing_starts": {"date": "2026-05-01", "value": 1.0}}
    assert writes[0][1]["_meta"]["source_registry"]["source_id"] == "housing"
    assert result["_meta"]["snapshot"]["key"] == "housing:current:v1"
    assert result["_meta"]["source_registry"]["source_id"] == "housing"


def test_snapshot_backed_response_force_refresh_bypasses_cache(monkeypatch):
    from api import macro_snapshots as ms
    from api.cache import daily_cache, set_cached

    key = "test:macro-force-refresh"
    set_cached(daily_cache, key, {"cached": True})
    monkeypatch.setattr(
        ms,
        "get_snapshot_response",
        lambda _key: {"old": True, "_meta": {"snapshot": {"key": "housing:current:v1", "stale": False}}},
    )
    monkeypatch.setattr(
        ms, "write_snapshot_success", lambda snapshot_key, payload, **_kwargs: _record(snapshot_key, payload)
    )

    result = ms.get_snapshot_backed_response(
        snapshot_key="housing:current:v1",
        cache=daily_cache,
        cache_key=key,
        source="housing",
        load_payload=lambda: {"fresh": True},
        force_refresh=True,
    )

    assert result["fresh"] is True
    assert result["_meta"]["snapshot"]["key"] == "housing:current:v1"


def test_snapshot_backed_response_returns_stale_snapshot_on_loader_failure(monkeypatch):
    from api import macro_snapshots as ms
    from api.cache import daily_cache, delete_cached

    key = "test:macro-stale-fallback"
    delete_cached(daily_cache, key)
    stale = {"old": True, "_meta": {"snapshot": {"key": "housing:current:v1", "stale": True}}}
    monkeypatch.setattr(ms, "get_snapshot_response", lambda _key: stale)

    result = ms.get_snapshot_backed_response(
        snapshot_key="housing:current:v1",
        cache=daily_cache,
        cache_key=key,
        source="housing",
        load_payload=lambda: (_ for _ in ()).throw(RuntimeError("network down")),
    )

    assert result is stale


def test_refresh_macro_snapshots_records_independent_failures(monkeypatch):
    from api import macro_snapshots as ms
    from api.routers import economic_growth, housing, labor_market

    writes: list[tuple[str, dict]] = []
    failures: list[tuple[str, str]] = []

    monkeypatch.setattr(
        labor_market, "load_labor_market_payload", lambda: {"latest": {"claims": {"date": "2026-05-01"}}}
    )
    monkeypatch.setattr(housing, "load_housing_payload", lambda: (_ for _ in ()).throw(RuntimeError("FRED down")))
    monkeypatch.setattr(economic_growth, "load_economic_growth_payload", lambda: {"timestamp": "2026-05-01T00:00:00"})
    monkeypatch.setattr(
        ms,
        "write_snapshot_success",
        lambda snapshot_key, payload, **_kwargs: (
            writes.append((snapshot_key, payload)) or _record(snapshot_key, payload)
        ),
    )
    monkeypatch.setattr(
        ms,
        "write_snapshot_failure",
        lambda snapshot_key, error, **_kwargs: (
            failures.append((snapshot_key, error)) or _record(snapshot_key, {}, status="error", error=error)
        ),
    )

    result = ms.refresh_macro_snapshots()

    assert [row["status"] for row in result["snapshots"]] == ["ok", "error", "ok"]
    assert result["snapshots"][1]["source_registry"]["source_id"] == "housing"
    assert {key for key, _payload in writes} == {ms.SNAPSHOT_LABOR_MARKET, ms.SNAPSHOT_ECONOMIC_GROWTH}
    assert failures == [(ms.SNAPSHOT_HOUSING, "FRED down")]


def test_labor_route_accepts_force_refresh(auth_client, monkeypatch):
    from api.routers import labor_market

    seen: dict[str, object] = {}

    def fake_response(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(labor_market, "get_snapshot_backed_response", fake_response)

    resp = auth_client.get("/api/labor-market", params={"force_refresh": "true"})

    assert resp.status_code == 200
    assert seen["force_refresh"] is True
    assert seen["snapshot_key"] == labor_market.SNAPSHOT_LABOR_MARKET


def test_housing_snapshot_key_is_defined():
    from api.snapshot_keys import SNAPSHOT_HOUSING

    assert SNAPSHOT_HOUSING == "housing:current:v1"


def test_delete_snapshot_removes_sqlite_snapshot(monkeypatch, tmp_path):
    from api import snapshot_store

    monkeypatch.setattr(snapshot_store, "use_postgres_state", lambda: False)
    monkeypatch.setattr(snapshot_store, "_SQLITE_PATH", tmp_path / "computed_snapshots.sqlite3")

    snapshot_store.write_snapshot_success(
        "test:snapshot",
        {"ok": True},
        as_of_date="2026-05-01",
    )
    assert snapshot_store.read_snapshot("test:snapshot") is not None

    snapshot_store.delete_snapshot("test:snapshot")

    assert snapshot_store.read_snapshot("test:snapshot") is None


def test_delete_snapshot_expires_postgres_current_version(monkeypatch):
    from api import snapshot_store

    calls: list[tuple[str, tuple[object, ...]]] = []

    class _Conn:
        def execute(self, sql: str, params: tuple[object, ...]):
            calls.append((sql, params))

        def commit(self):
            return None

    class _Repo:
        @contextmanager
        def _connect(self):
            yield _Conn()

    monkeypatch.setattr(snapshot_store, "use_postgres_state", lambda: True)
    monkeypatch.setattr(snapshot_store, "TemporalOntologyRepository", _Repo)

    snapshot_store.delete_snapshot("economic_growth:current:v1")

    sql, params = calls[0]
    assert "UPDATE computed_snapshot_versions" in sql
    assert "tx_to IS NULL" in sql
    assert params[1] == "economic_growth:current:v1"


def test_economic_growth_crb_upload_invalidates_cache_and_snapshot(monkeypatch):
    from api.routers import economic_growth

    writes: list[tuple[bytes, dict]] = []
    cache_deletes: list[tuple[object, str]] = []
    snapshot_deletes: list[str] = []

    class _File:
        filename = "crb.xlsx"

    async def fake_read_upload_file_bytes(file, *, limit_bytes: int, limit_label: str) -> bytes:
        assert file.filename == "crb.xlsx"
        assert limit_bytes == economic_growth.MAX_CRB_UPLOAD_SIZE_BYTES
        assert limit_label == "10 MiB"
        return b"excel-bytes"

    metadata = {
        "filename": "crb.xlsx",
        "uploaded_at": "2026-05-06T12:00:00+00:00",
        "rows": 2,
        "latest_date": "2026-05-05",
        "latest_value": 123.4,
        "size_bytes": 11,
    }

    monkeypatch.setattr(economic_growth, "read_upload_file_bytes", fake_read_upload_file_bytes)
    monkeypatch.setattr(economic_growth, "_crb_metadata_from_upload", lambda payload, filename: metadata)
    monkeypatch.setattr(economic_growth, "_write_managed_crb", lambda payload, meta: writes.append((payload, meta)))
    monkeypatch.setattr(economic_growth, "delete_cached", lambda cache, key: cache_deletes.append((cache, key)))
    monkeypatch.setattr(economic_growth, "delete_snapshot", lambda snapshot_key: snapshot_deletes.append(snapshot_key))

    result = asyncio.run(economic_growth.upload_economic_growth_crb_file(_File()))

    assert result["status"] == "ok"
    assert result["crb"] == metadata
    assert writes == [(b"excel-bytes", metadata)]
    assert cache_deletes == [(economic_growth.daily_cache, economic_growth.ECONOMIC_GROWTH_CACHE_KEY)]
    assert snapshot_deletes == [economic_growth.SNAPSHOT_ECONOMIC_GROWTH]
