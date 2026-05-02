from __future__ import annotations

from datetime import datetime, timedelta

from api import snapshot_store as store


def test_snapshot_store_roundtrip_and_meta(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_SQLITE_PATH", tmp_path / "snapshots.sqlite3")
    record = store.write_snapshot_success(
        "unit:test",
        {"value": 42},
        as_of_date="2026-05-01",
        fetched_at=(datetime.now() - timedelta(hours=1)).isoformat(),
    )

    assert record.payload == {"value": 42}

    response = store.get_snapshot_response("unit:test")
    assert response is not None
    assert response["value"] == 42
    assert response["_meta"]["snapshot"]["key"] == "unit:test"
    assert response["_meta"]["snapshot"]["as_of"] == "2026-05-01"
    assert response["_meta"]["snapshot"]["stale"] is False


def test_snapshot_store_marks_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_SQLITE_PATH", tmp_path / "snapshots.sqlite3")
    store.write_snapshot_success(
        "unit:stale",
        {"value": 1},
        as_of_date="2026-04-30",
        fetched_at=(datetime.now() - timedelta(days=3)).isoformat(),
    )

    response = store.get_snapshot_response("unit:stale", max_age_seconds=60)
    assert response is not None
    assert response["_meta"]["snapshot"]["stale"] is True


def test_snapshot_failure_preserves_last_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_SQLITE_PATH", tmp_path / "snapshots.sqlite3")
    store.write_snapshot_success("unit:failure", {"value": "old"}, as_of_date="2026-05-01")

    record = store.write_snapshot_failure("unit:failure", "vendor timeout")
    assert record is not None
    assert record.payload == {"value": "old"}
    assert record.status == "error"

    response = store.get_snapshot_response("unit:failure")
    assert response is not None
    assert response["value"] == "old"
    assert response["_meta"]["snapshot"]["refresh_status"] == "error"
    assert response["_meta"]["snapshot"]["error"] == "vendor timeout"
