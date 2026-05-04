from __future__ import annotations

import portfolio.core_db as core_db
import portfolio.portfolio_db as portfolio_db
import portfolio.thesis_db as thesis_db


def test_operational_backfill_dry_run_inventories_legacy_rows(tmp_path, monkeypatch):
    monkeypatch.delenv("LEGACY_WRITE_GUARD", raising=False)
    for module, name in (
        (core_db, "core.db"),
        (portfolio_db, "portfolio.db"),
        (thesis_db, "thesis.db"),
    ):
        conn = getattr(module, "_conn", None)
        if conn:
            conn.close()
        monkeypatch.setattr(module, "DB_PATH", tmp_path / name)
        monkeypatch.setattr(module, "_conn", None)
    import paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)

    portfolio_db.save_positions([{"ticker": "MU", "asset": "equity", "direction": "long"}])
    thesis_db.upsert_thesis_meta("MU", status="active")
    core_db.create_action_item("Review MU", "review", ticker="MU")

    from ontology.backfill_operational import backfill_operational_legacy_state

    summary = backfill_operational_legacy_state(cutover_time="2026-05-04T00:00:00+00:00", dry_run=True)

    assert summary["positions:seen"] == 1
    assert summary["thesis_meta:seen"] == 1
    assert summary["action_items:seen"] == 1
    assert summary["dry_run"] == 1
