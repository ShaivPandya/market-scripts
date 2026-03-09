"""
portfolio_db.py — SQLite-backed portfolio position store.

Single source of truth for portfolio positions. All backend modules should
import from here instead of reading portfolio.csv directly.

Public API:
  get_positions()     -> list[dict]       — all positions as plain dicts
  get_positions_df()  -> pd.DataFrame     — drop-in replacement for pd.read_csv(portfolio.csv)
  save_positions(positions: list[dict])   — full replacement (single transaction)
  DB_PATH: Path                           — path to the SQLite database file
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import pandas as pd

DB_PATH = Path(__file__).parent / "portfolio.db"
_CSV_PATH = Path(__file__).parent / "portfolio.csv"

_ASSET_CLASSES = {"equity", "commodity", "fx", "bond"}
_DIRECTIONS = {"long", "short"}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS positions (
    ticker      TEXT    PRIMARY KEY NOT NULL,
    asset       TEXT    NOT NULL
                        CHECK (asset IN ('equity','commodity','fx','bond')),
    direction   TEXT    NOT NULL
                        CHECK (direction IN ('long','short')),
    distressed  INTEGER NOT NULL DEFAULT 0,
    conviction  INTEGER NOT NULL DEFAULT 3
                        CHECK (conviction BETWEEN 1 AND 5),
    cost_basis  REAL,
    shares      REAL
)
"""

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                _conn.row_factory = sqlite3.Row
                _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE)
    conn.commit()
    # Migrate: add shares column if missing (added after initial schema)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(positions)").fetchall()}
    if "shares" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN shares REAL")
        conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    if count == 0 and _CSV_PATH.exists():
        _migrate_from_csv(conn)


def _migrate_from_csv(conn: sqlite3.Connection) -> None:
    df = pd.read_csv(_CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    rows: list[tuple[Any, ...]] = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        asset = str(row.get("asset", "equity")).strip().lower()
        if asset not in _ASSET_CLASSES:
            asset = "equity"
        direction = str(row.get("direction", "long")).strip().lower()
        if direction not in _DIRECTIONS:
            direction = "long"
        distressed_raw = row.get("distressed", False)
        if isinstance(distressed_raw, str):
            distressed = 1 if distressed_raw.strip().lower() in ("true", "1", "yes") else 0
        else:
            distressed = 1 if distressed_raw else 0
        try:
            conviction = int(row.get("conviction", 3))
            conviction = max(1, min(5, conviction))
        except (ValueError, TypeError):
            conviction = 3
        cost_basis_raw = row.get("cost_basis", None)
        try:
            cost_basis = (
                float(cost_basis_raw) if cost_basis_raw is not None and str(cost_basis_raw).strip() != "" else None
            )
        except (ValueError, TypeError):
            cost_basis = None
        rows.append((ticker, asset, direction, distressed, conviction, cost_basis))
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO positions (ticker, asset, direction, distressed, conviction, cost_basis) VALUES (?,?,?,?,?,?)",
            rows,
        )
        conn.commit()


def get_positions() -> list[dict]:
    """Return all positions as a list of plain dicts, ordered by insertion rowid."""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT ticker, asset, direction, distressed, conviction, cost_basis, shares FROM positions ORDER BY rowid"
        ).fetchall()
    return [dict(r) for r in rows]


def get_positions_df() -> pd.DataFrame:
    """Return positions as a DataFrame — drop-in replacement for pd.read_csv(portfolio.csv).

    Columns: ticker, asset, direction, distressed, conviction, cost_basis
    distressed is returned as bool for convenience.
    """
    positions = get_positions()
    if not positions:
        return pd.DataFrame(
            columns=["ticker", "asset", "direction", "distressed", "conviction", "cost_basis", "shares"]
        )
    df = pd.DataFrame(positions)
    df["distressed"] = df["distressed"].astype(bool)
    return df


def save_positions(positions: list[dict]) -> None:
    """Replace all positions in a single atomic transaction."""
    conn = _get_conn()
    rows = []
    for p in positions:
        ticker = str(p.get("ticker", "")).strip().upper()
        asset = str(p.get("asset", "equity")).strip().lower()
        direction = str(p.get("direction", "long")).strip().lower()
        distressed = 1 if p.get("distressed") else 0
        try:
            conviction = int(p.get("conviction", 3))
            conviction = max(1, min(5, conviction))
        except (ValueError, TypeError):
            conviction = 3
        cost_basis_raw = p.get("cost_basis")
        try:
            cost_basis = float(cost_basis_raw) if cost_basis_raw is not None else None
        except (ValueError, TypeError):
            cost_basis = None
        shares_raw = p.get("shares")
        try:
            shares = float(shares_raw) if shares_raw is not None else None
        except (ValueError, TypeError):
            shares = None
        rows.append((ticker, asset, direction, distressed, conviction, cost_basis, shares))
    with _lock:
        conn.execute("DELETE FROM positions")
        conn.executemany(
            "INSERT INTO positions (ticker, asset, direction, distressed, conviction, cost_basis, shares) VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
