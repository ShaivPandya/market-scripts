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

import pandas as pd

from api.postgres import connect, use_postgres_state
from api.postgres_compat import PostgresCompatConnection

DB_PATH = Path(__file__).parent / "portfolio.db"
CSV_PATH = Path(__file__).parent / "portfolio.csv"

_ASSET_CLASSES = {"equity", "commodity", "fx", "bond"}
_DIRECTIONS = {"long", "short"}
_POSITION_COLUMNS = ["ticker", "asset", "direction", "contrarian", "conviction", "cost_basis", "shares", "role"]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS positions (
    ticker      TEXT    PRIMARY KEY NOT NULL,
    asset       TEXT    NOT NULL
                        CHECK (asset IN ('equity','commodity','fx','bond')),
    direction   TEXT    NOT NULL
                        CHECK (direction IN ('long','short')),
    contrarian  INTEGER NOT NULL DEFAULT 0,
    conviction  INTEGER NOT NULL DEFAULT 3
                        CHECK (conviction BETWEEN 1 AND 5),
    cost_basis  REAL,
    shares      REAL,
    role        TEXT    NOT NULL DEFAULT 'position'
                        CHECK (role IN ('position','hedge'))
)
"""

_lock = threading.Lock()
_conn: sqlite3.Connection | PostgresCompatConnection | None = None


def _get_conn() -> sqlite3.Connection | PostgresCompatConnection:
    global _conn
    if _conn is not None:
        try:
            _conn.execute("SELECT 1")
        except Exception:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
    if _conn is None:
        with _lock:
            if _conn is None:
                if use_postgres_state():
                    _conn = PostgresCompatConnection()
                else:
                    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                    _conn.execute("PRAGMA journal_mode=WAL")
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
    if "role" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN role TEXT NOT NULL DEFAULT 'position'")
        conn.commit()
    # Migrate: rename distressed -> contrarian
    if "distressed" in cols and "contrarian" not in cols:
        conn.execute("ALTER TABLE positions RENAME COLUMN distressed TO contrarian")
        conn.commit()


def get_positions(include_hedges: bool = False) -> list[dict]:
    """Return positions as a list of plain dicts, ordered by insertion rowid.

    By default only ``role='position'`` rows are returned.  Pass
    ``include_hedges=True`` to also include ``role='hedge'`` rows.
    """
    if use_postgres_state():
        return _pg_get_positions(include_hedges=include_hedges)

    conn = _get_conn()
    with _lock:
        if include_hedges:
            rows = conn.execute(
                "SELECT ticker, asset, direction, contrarian, conviction, cost_basis, shares, role "
                "FROM positions ORDER BY rowid"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ticker, asset, direction, contrarian, conviction, cost_basis, shares, role "
                "FROM positions WHERE role = 'position' ORDER BY rowid"
            ).fetchall()
    return [dict(r) for r in rows]


def get_hedge_positions() -> list[dict]:
    """Return only hedge positions."""
    if use_postgres_state():
        return _pg_get_positions(include_hedges=True, role="hedge")

    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT ticker, asset, direction, contrarian, conviction, cost_basis, shares, role "
            "FROM positions WHERE role = 'hedge' ORDER BY rowid"
        ).fetchall()
    return [dict(r) for r in rows]


def load_positions_csv() -> pd.DataFrame:
    """Load ``portfolio.csv`` into the same shape as ``get_positions_df``."""
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=_POSITION_COLUMNS)

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return pd.DataFrame(columns=_POSITION_COLUMNS)

    df.columns = df.columns.str.strip()
    defaults = {
        "ticker": "",
        "asset": "equity",
        "direction": "long",
        "contrarian": False,
        "conviction": 3,
        "cost_basis": None,
        "shares": None,
        "role": "position",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    df = df[_POSITION_COLUMNS].copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["asset"] = df["asset"].astype(str).str.strip().str.lower()
    df["direction"] = df["direction"].astype(str).str.strip().str.lower()
    df["contrarian"] = df["contrarian"].astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
    df["conviction"] = pd.to_numeric(df["conviction"], errors="coerce").fillna(3).clip(1, 5).astype(int)
    df["cost_basis"] = pd.to_numeric(df["cost_basis"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["role"] = df["role"].astype(str).str.strip().str.lower().replace("", "position")
    return df


def get_positions_df(include_hedges: bool = False, fallback_to_csv: bool = False) -> pd.DataFrame:
    """Return positions as a DataFrame — drop-in replacement for pd.read_csv(portfolio.csv).

    Columns: ticker, asset, direction, contrarian, conviction, cost_basis, shares, role
    contrarian is returned as bool for convenience.
    """
    positions = get_positions(include_hedges=include_hedges)
    if not positions:
        if fallback_to_csv:
            csv_df = load_positions_csv()
            if not include_hedges and "role" in csv_df.columns:
                csv_df = csv_df[csv_df["role"].eq("position")]
            return csv_df.reset_index(drop=True)
        return pd.DataFrame(columns=_POSITION_COLUMNS)
    df = pd.DataFrame(positions)
    df["contrarian"] = df["contrarian"].astype(bool)
    return df


def save_positions(positions: list[dict], role: str = "position") -> None:
    """Replace all positions of the given *role* in a single atomic transaction.

    When ``role='position'`` (default) only regular position rows are deleted
    and re-inserted — hedge rows are left untouched, and vice-versa.
    """
    if role not in ("position", "hedge"):
        raise ValueError(f"Invalid role: {role!r}")
    rows = _normalize_position_rows(positions, role)
    if use_postgres_state():
        _pg_save_position_rows(rows, role=role)
        return

    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM positions WHERE role = ?", (role,))
        conn.executemany(
            "INSERT INTO positions (ticker, asset, direction, contrarian, conviction, cost_basis, shares, role) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()


def _normalize_position_rows(positions: list[dict], role: str) -> list[tuple]:
    rows = []
    for p in positions:
        ticker = str(p.get("ticker", "")).strip().upper()
        asset = str(p.get("asset", "equity")).strip().lower()
        direction = str(p.get("direction", "long")).strip().lower()
        contrarian = 1 if p.get("contrarian") else 0
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
        rows.append((ticker, asset, direction, contrarian, conviction, cost_basis, shares, role))
    return rows


def _pg_get_positions(*, include_hedges: bool = False, role: str | None = None) -> list[dict]:
    sql = "SELECT ticker, asset, direction, contrarian, conviction, cost_basis, shares, role FROM positions"
    params: tuple = ()
    if role:
        sql += " WHERE role = %s"
        params = (role,)
    elif not include_hedges:
        sql += " WHERE role = 'position'"
    sql += " ORDER BY ticker"
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    out = [dict(row) for row in rows]
    for row in out:
        row["contrarian"] = 1 if row.get("contrarian") else 0
    return out


def _pg_save_position_rows(rows: list[tuple], *, role: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM positions WHERE role = %s", (role,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO positions (ticker, asset, direction, contrarian, conviction, cost_basis, shares, role)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()
