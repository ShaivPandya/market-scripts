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


def test_legacy_domain_write_guard_is_enabled_in_ontology_primary_runtime(monkeypatch):
    from ontology.domain_write_service import legacy_write_guard_enabled

    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("ONTOLOGY_PRIMARY_WRITES", "true")

    assert legacy_write_guard_enabled() is True
