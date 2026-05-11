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
from portfolio.instruments import (
    default_contract_multiplier,
    is_continuous_future_symbol,
    normalize_asset,
    normalize_instrument_type,
    normalize_quantity,
    normalize_spot_fx_symbol,
    normalize_symbol,
    spot_fx_currencies,
)
from portfolio.position_groups import (
    canonicalize_position_group_rows,
    normalize_position_group_fields,
    validate_position_groups,
)

DB_PATH = Path(__file__).parent / "portfolio.db"
CSV_PATH = Path(__file__).parent / "portfolio.csv"

_DIRECTIONS = {"long", "short"}
_POSITION_COLUMNS = [
    "ticker",
    "asset",
    "direction",
    "contrarian",
    "conviction",
    "cost_basis",
    "shares",
    "quantity",
    "instrument_type",
    "price_symbol",
    "contract_multiplier",
    "fx_base_currency",
    "fx_quote_currency",
    "currency",
    "country",
    "exchange",
    "base_currency",
    "fx_rate_to_base",
    "fx_rate_as_of",
    "cost_basis_base",
    "notional_base",
    "valuation_status",
    "group_name",
    "group_conviction",
    "role",
]
_SELECT_COLUMNS = ", ".join(_POSITION_COLUMNS)

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
    quantity    REAL,
    instrument_type TEXT NOT NULL DEFAULT 'security'
                        CHECK (instrument_type IN ('security','future','spot_fx')),
    price_symbol TEXT,
    contract_multiplier REAL NOT NULL DEFAULT 1.0
                        CHECK (contract_multiplier > 0),
    fx_base_currency TEXT,
    fx_quote_currency TEXT,
    currency    TEXT,
    country     TEXT,
    exchange    TEXT,
    base_currency TEXT NOT NULL DEFAULT 'USD',
    fx_rate_to_base REAL,
    fx_rate_as_of TEXT,
    cost_basis_base REAL,
    notional_base REAL,
    valuation_status TEXT NOT NULL DEFAULT 'missing_position_inputs',
    group_name  TEXT,
    group_conviction INTEGER CHECK (group_conviction BETWEEN 1 AND 5),
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
        cols.add("shares")
    if "quantity" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN quantity REAL")
        conn.commit()
        cols.add("quantity")
    if "instrument_type" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN instrument_type TEXT NOT NULL DEFAULT 'security'")
        conn.commit()
        cols.add("instrument_type")
    if "price_symbol" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN price_symbol TEXT")
        conn.commit()
        cols.add("price_symbol")
    if "contract_multiplier" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN contract_multiplier REAL NOT NULL DEFAULT 1.0")
        conn.commit()
        cols.add("contract_multiplier")
    if "fx_base_currency" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN fx_base_currency TEXT")
        conn.commit()
        cols.add("fx_base_currency")
    if "fx_quote_currency" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN fx_quote_currency TEXT")
        conn.commit()
        cols.add("fx_quote_currency")
    valuation_columns = {
        "currency": "TEXT",
        "country": "TEXT",
        "exchange": "TEXT",
        "base_currency": "TEXT NOT NULL DEFAULT 'USD'",
        "fx_rate_to_base": "REAL",
        "fx_rate_as_of": "TEXT",
        "cost_basis_base": "REAL",
        "notional_base": "REAL",
        "valuation_status": "TEXT NOT NULL DEFAULT 'missing_position_inputs'",
    }
    for column, definition in valuation_columns.items():
        if column not in cols:
            conn.execute(f"ALTER TABLE positions ADD COLUMN {column} {definition}")
            conn.commit()
            cols.add(column)
    if "role" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN role TEXT NOT NULL DEFAULT 'position'")
        conn.commit()
        cols.add("role")
    if "group_name" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN group_name TEXT")
        conn.commit()
        cols.add("group_name")
    if "group_conviction" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN group_conviction INTEGER")
        conn.commit()
        cols.add("group_conviction")
    # Migrate: rename distressed -> contrarian
    if "distressed" in cols and "contrarian" not in cols:
        conn.execute("ALTER TABLE positions RENAME COLUMN distressed TO contrarian")
        conn.commit()
    conn.execute("UPDATE positions SET quantity = shares WHERE quantity IS NULL AND shares IS NOT NULL")
    conn.execute(
        "UPDATE positions SET instrument_type = 'security' WHERE instrument_type IS NULL OR instrument_type = ''"
    )
    conn.execute("UPDATE positions SET price_symbol = ticker WHERE price_symbol IS NULL OR price_symbol = ''")
    conn.execute("UPDATE positions SET contract_multiplier = 1.0 WHERE contract_multiplier IS NULL")
    conn.execute("UPDATE positions SET base_currency = 'USD' WHERE base_currency IS NULL OR base_currency = ''")
    conn.execute(
        "UPDATE positions SET valuation_status = 'missing_position_inputs' "
        "WHERE valuation_status IS NULL OR valuation_status = ''"
    )
    conn.commit()
    _ensure_sqlite_spot_fx_schema(conn)


def _ensure_sqlite_spot_fx_schema(conn: sqlite3.Connection) -> None:
    create_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'positions'").fetchone()
    sql = str(create_sql[0] if create_sql else "")
    if "spot_fx" in sql:
        return
    temp = "positions_spot_fx_migration_old"
    conn.execute(f"ALTER TABLE positions RENAME TO {temp}")
    conn.execute(_CREATE_TABLE)
    old_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({temp})").fetchall()}
    common = [column for column in _POSITION_COLUMNS if column in old_cols]
    columns = ", ".join(common)
    conn.execute(f"INSERT INTO positions ({columns}) SELECT {columns} FROM {temp}")
    conn.execute(f"DROP TABLE {temp}")
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
            rows = conn.execute(f"SELECT {_SELECT_COLUMNS} FROM positions ORDER BY rowid").fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM positions WHERE role = 'position' ORDER BY rowid"
            ).fetchall()
    return [_position_dict(r) for r in rows]


def get_hedge_positions() -> list[dict]:
    """Return only hedge positions."""
    if use_postgres_state():
        return _pg_get_positions(include_hedges=True, role="hedge")

    conn = _get_conn()
    with _lock:
        rows = conn.execute(f"SELECT {_SELECT_COLUMNS} FROM positions WHERE role = 'hedge' ORDER BY rowid").fetchall()
    return [_position_dict(r) for r in rows]


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
        "asset": "",
        "direction": "long",
        "contrarian": False,
        "conviction": 3,
        "cost_basis": None,
        "shares": None,
        "quantity": None,
        "instrument_type": None,
        "price_symbol": None,
        "contract_multiplier": None,
        "fx_base_currency": None,
        "fx_quote_currency": None,
        "currency": None,
        "country": None,
        "exchange": None,
        "base_currency": "USD",
        "fx_rate_to_base": None,
        "fx_rate_as_of": None,
        "cost_basis_base": None,
        "notional_base": None,
        "valuation_status": None,
        "group_name": None,
        "group_conviction": None,
        "role": "position",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    df = df[_POSITION_COLUMNS].copy()
    df["instrument_type"] = df.apply(
        lambda row: normalize_instrument_type(
            row.get("instrument_type"), ticker=row["ticker"], price_symbol=row["price_symbol"]
        ),
        axis=1,
    )
    df["price_symbol"] = df.apply(
        lambda row: (
            normalize_spot_fx_symbol(row.get("price_symbol") or row.get("ticker"), field_name="price_symbol")
            if row["instrument_type"] == "spot_fx"
            else (
                normalize_symbol(row["price_symbol"], field_name="price_symbol")
                if str(row.get("price_symbol") or "").strip()
                else normalize_symbol(row["ticker"])
            )
        ),
        axis=1,
    )
    df["ticker"] = df.apply(
        lambda row: row["price_symbol"] if row["instrument_type"] == "spot_fx" else normalize_symbol(row["ticker"]),
        axis=1,
    )
    df["asset"] = df.apply(
        lambda row: normalize_asset(
            row.get("asset"), instrument_type=row["instrument_type"], symbol=row["price_symbol"]
        ),
        axis=1,
    )
    df["direction"] = df["direction"].astype(str).str.strip().str.lower()
    df["contrarian"] = df["contrarian"].astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
    df["conviction"] = pd.to_numeric(df["conviction"], errors="coerce").fillna(3).clip(1, 5).astype(int)
    df["group_name"] = df.apply(lambda row: normalize_position_group_fields(row)[0], axis=1)
    df["group_conviction"] = df.apply(lambda row: normalize_position_group_fields(row)[1], axis=1)
    df["cost_basis"] = pd.to_numeric(df["cost_basis"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(df["shares"])
    df["contract_multiplier"] = df.apply(
        lambda row: default_contract_multiplier(
            instrument_type=row["instrument_type"],
            symbol=row["price_symbol"],
            override=row.get("contract_multiplier"),
        ),
        axis=1,
    )
    for idx, row in df[df["instrument_type"].eq("spot_fx")].iterrows():
        fx_base, fx_quote = spot_fx_currencies(row["price_symbol"])
        df.at[idx, "fx_base_currency"] = fx_base
        df.at[idx, "fx_quote_currency"] = fx_quote
        df.at[idx, "currency"] = fx_quote
        df.at[idx, "asset"] = "fx"
        df.at[idx, "contract_multiplier"] = 1.0
    for column in (
        "fx_base_currency",
        "fx_quote_currency",
        "currency",
        "country",
        "exchange",
        "base_currency",
        "fx_rate_as_of",
        "valuation_status",
    ):
        df[column] = df[column].where(df[column].notna(), None)
    for column in ("fx_rate_to_base", "cost_basis_base", "notional_base"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["role"] = df["role"].astype(str).str.strip().str.lower().replace("", "position")
    return df


def get_positions_df(include_hedges: bool = False, fallback_to_csv: bool = False) -> pd.DataFrame:
    """Return positions as a DataFrame — drop-in replacement for pd.read_csv(portfolio.csv).

    Columns: ticker, asset, direction, contrarian, conviction, cost_basis,
    shares, quantity, instrument_type, price_symbol, contract_multiplier,
    valuation metadata, role
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


def save_positions(
    positions: list[dict],
    role: str = "position",
    *,
    preserve_existing_valuation: bool = False,
) -> None:
    """Replace all positions of the given *role* in a single atomic transaction.

    When ``role='position'`` (default) only regular position rows are deleted
    and re-inserted — hedge rows are left untouched, and vice-versa.
    """
    from ontology.domain_write_service import assert_legacy_domain_write_allowed

    assert_legacy_domain_write_allowed(f"portfolio_db.save_positions:{role}")
    if role not in ("position", "hedge"):
        raise ValueError(f"Invalid role: {role!r}")
    rows = _normalize_position_rows(positions, role, preserve_existing_valuation=preserve_existing_valuation)
    if role == "position":
        validate_position_groups(
            [
                {
                    "ticker": row[0],
                    "direction": row[2],
                    "group_name": row[22],
                    "group_conviction": row[23],
                }
                for row in rows
            ]
        )
    if use_postgres_state():
        _pg_save_position_rows(rows, role=role)
        return

    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM positions WHERE role = ?", (role,))
        conn.executemany(
            "INSERT INTO positions "
            "(ticker, asset, direction, contrarian, conviction, cost_basis, shares, quantity, "
            "instrument_type, price_symbol, contract_multiplier, fx_base_currency, fx_quote_currency, "
            "currency, country, exchange, base_currency, "
            "fx_rate_to_base, fx_rate_as_of, cost_basis_base, notional_base, valuation_status, "
            "group_name, group_conviction, role) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()


def _normalize_position_rows(
    positions: list[dict],
    role: str,
    *,
    preserve_existing_valuation: bool = False,
) -> list[tuple]:
    rows = []
    from portfolio.valuation import enrich_position_valuation

    input_positions = canonicalize_position_group_rows(positions) if role == "position" else positions
    for raw in input_positions:
        p = enrich_position_valuation(raw, preserve_existing=preserve_existing_valuation)
        raw_ticker = p.get("ticker", "")
        raw_price_symbol = p.get("price_symbol") or raw_ticker
        instrument_type = normalize_instrument_type(
            p.get("instrument_type"), ticker=str(raw_ticker), price_symbol=str(raw_price_symbol)
        )
        if instrument_type == "spot_fx":
            price_symbol = normalize_spot_fx_symbol(raw_price_symbol, field_name="price_symbol")
            ticker = price_symbol
        else:
            ticker = normalize_symbol(raw_ticker)
            price_symbol = normalize_symbol(raw_price_symbol or ticker, field_name="price_symbol")
        if instrument_type == "future" and not is_continuous_future_symbol(price_symbol):
            raise ValueError(f"Futures positions require a continuous '=F' price_symbol, got {price_symbol!r}.")
        asset = normalize_asset(p.get("asset"), instrument_type=instrument_type, symbol=price_symbol)
        direction = str(p.get("direction", "long")).strip().lower()
        if direction not in _DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction!r}")
        # Keep this as a bool: sqlite stores bools as 0/1, while psycopg sends
        # Python bools as native Postgres booleans.
        contrarian = bool(p.get("contrarian"))
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
        quantity = normalize_quantity(quantity=p.get("quantity"), shares=p.get("shares"), allow_negative=True)
        shares = quantity
        contract_multiplier = default_contract_multiplier(
            instrument_type=instrument_type,
            symbol=price_symbol,
            override=p.get("contract_multiplier"),
        )
        fx_base_currency = _optional_text(p.get("fx_base_currency"))
        fx_quote_currency = _optional_text(p.get("fx_quote_currency"))
        if instrument_type == "spot_fx":
            fx_base_currency, fx_quote_currency = spot_fx_currencies(price_symbol)
            asset = "fx"
            contract_multiplier = 1.0
        currency = _optional_text(p.get("currency"))
        if instrument_type == "spot_fx":
            currency = fx_quote_currency
        country = _optional_text(p.get("country"))
        exchange = _optional_text(p.get("exchange"))
        if instrument_type == "spot_fx" and not exchange:
            exchange = "FX"
        base_currency = _optional_text(p.get("base_currency")) or "USD"
        fx_rate_to_base = _optional_float(p.get("fx_rate_to_base"))
        fx_rate_as_of = _optional_text(p.get("fx_rate_as_of"))
        cost_basis_base = _optional_float(p.get("cost_basis_base"))
        notional_base = _optional_float(p.get("notional_base"))
        valuation_status = _optional_text(p.get("valuation_status")) or "missing_position_inputs"
        group_name, group_conviction = normalize_position_group_fields(p) if role == "position" else (None, None)
        rows.append(
            (
                ticker,
                asset,
                direction,
                contrarian,
                conviction,
                cost_basis,
                shares,
                quantity,
                instrument_type,
                price_symbol,
                contract_multiplier,
                fx_base_currency,
                fx_quote_currency,
                currency,
                country,
                exchange,
                base_currency,
                fx_rate_to_base,
                fx_rate_as_of,
                cost_basis_base,
                notional_base,
                valuation_status,
                group_name,
                group_conviction,
                role,
            )
        )
    return rows


def _pg_get_positions(*, include_hedges: bool = False, role: str | None = None) -> list[dict]:
    sql = f"SELECT {_SELECT_COLUMNS} FROM positions"
    params: tuple = ()
    if role:
        sql += " WHERE role = %s"
        params = (role,)
    elif not include_hedges:
        sql += " WHERE role = 'position'"
    sql += " ORDER BY ticker"
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    out = [_position_dict(row) for row in rows]
    for row in out:
        row["contrarian"] = 1 if row.get("contrarian") else 0
    return out


def _pg_save_position_rows(rows: list[tuple], *, role: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM positions WHERE role = %s", (role,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO positions (
                    ticker, asset, direction, contrarian, conviction, cost_basis, shares, quantity,
                    instrument_type, price_symbol, contract_multiplier, fx_base_currency, fx_quote_currency,
                    currency, country, exchange, base_currency,
                    fx_rate_to_base, fx_rate_as_of, cost_basis_base, notional_base, valuation_status,
                    group_name, group_conviction, role
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()


def _position_dict(row) -> dict:
    out = dict(row)
    quantity = out.get("quantity")
    if quantity is None:
        quantity = out.get("shares")
        out["quantity"] = quantity
    out["shares"] = quantity
    out["instrument_type"] = str(out.get("instrument_type") or "security")
    out["price_symbol"] = str(out.get("price_symbol") or out.get("ticker") or "").upper()
    out["contract_multiplier"] = float(out.get("contract_multiplier") or 1.0)
    if out["instrument_type"] == "spot_fx":
        out["price_symbol"] = normalize_spot_fx_symbol(out["price_symbol"])
        out["ticker"] = out["price_symbol"]
        out["asset"] = "fx"
        out["contract_multiplier"] = 1.0
        out["fx_base_currency"], out["fx_quote_currency"] = spot_fx_currencies(out["price_symbol"])
        out["currency"] = out["fx_quote_currency"]
        out["exchange"] = out.get("exchange") or "FX"
    out["base_currency"] = str(out.get("base_currency") or "USD").upper()
    out["valuation_status"] = str(out.get("valuation_status") or "missing_position_inputs")
    if str(out.get("role") or "position") == "position":
        out["group_name"], out["group_conviction"] = normalize_position_group_fields(out)
    else:
        out["group_name"] = None
        out["group_conviction"] = None
    return out


def _optional_text(value) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _optional_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out
