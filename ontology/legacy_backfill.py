"""Migration-only reader for legacy SQLite exports.

This module is intentionally not a runtime compatibility layer. It exists so a
maintenance-window cutover can read old SQLite exports, write audited ontology
objects, and then deploy the Postgres-only runtime.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ontology.command_service import OntologyCommandContext, OntologyCommandService
from ontology.policy import system_actor


class LegacyBackfillDisabled(RuntimeError):
    pass


def _enabled() -> bool:
    return (os.getenv("TALISMAN_ENABLE_LEGACY_BACKFILL") or "").strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    if not _enabled():
        raise LegacyBackfillDisabled(
            "Legacy SQLite reads are allowed only for maintenance-window backfill. "
            "Set TALISMAN_ENABLE_LEGACY_BACKFILL=true in the migration job."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def backfill_audit_minimum(
    *,
    portfolio_db_path: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill current portfolio positions through ontology command service.

    Additional audit-minimum domains should be fed through the same command
    boundary, not by importing legacy runtime modules.
    """

    positions = _read_positions(Path(portfolio_db_path))
    if dry_run:
        return {"dry_run": True, "positions": len(positions)}
    service = OntologyCommandService()
    context = OntologyCommandContext(
        actor=system_actor("legacy_backfill"),
        source_type="migration",
        source_id="legacy_backfill.audit_minimum",
    )
    approval = service.propose_action(
        "update_portfolio_positions",
        {"positions": positions},
        context,
        reason="Audit-minimum maintenance-window backfill from legacy portfolio export.",
    )
    applied = service.resolve_approval(approval["id"], "approved", "Approved migration backfill.", context)
    return {"dry_run": False, "positions": len(positions), "approval_id": applied["id"]}


def _read_positions(path: Path) -> list[dict[str, Any]]:
    with _connect(path) as conn:
        rows = conn.execute("SELECT * FROM positions ORDER BY ticker").fetchall()
    return [_jsonable(dict(row)) for row in rows]


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    out[key] = json.loads(stripped)
                    continue
                except json.JSONDecodeError:
                    pass
        out[key] = value
    return out
