from __future__ import annotations

import sqlite3

import pytest

from api.local_write_guard import ProductionLocalWriteError
from paths import PROJECT_ROOT


def test_project_root_write_raises_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    target = PROJECT_ROOT / "tmp-production-write-guard.txt"

    with pytest.raises(ProductionLocalWriteError):
        target.write_text("should not be written", encoding="utf-8")

    assert not target.exists()


def test_non_project_temp_write_allowed_in_production(monkeypatch, tmp_path):
    monkeypatch.setenv("ENVIRONMENT", "production")
    target = tmp_path / "allowed.txt"

    target.write_text("ok", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "ok"


def test_project_root_sqlite_connect_raises_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    target = PROJECT_ROOT / "tmp-production-write-guard.sqlite3"

    with pytest.raises(ProductionLocalWriteError):
        sqlite3.connect(target)

    assert not target.exists()


def test_domain_table_writes_are_blocked_by_default(monkeypatch):
    from ontology.domain_write_service import assert_domain_table_write_allowed

    monkeypatch.delenv("ENVIRONMENT", raising=False)

    with pytest.raises(RuntimeError, match="Domain table write blocked"):
        assert_domain_table_write_allowed("test")
