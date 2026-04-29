"""Small SQLite-style compatibility wrapper for legacy Postgres adapters."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from api.postgres import open_connection


class CompatRow(Mapping[str, Any]):
    def __init__(self, data: Mapping[str, Any], columns: Sequence[str]):
        self._data = dict(data)
        self._columns = list(columns)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._data[self._columns[key]]
        return self._data[key]

    def __iter__(self) -> Iterator[Any]:
        for column in self._columns:
            yield self._data.get(column)

    def __len__(self) -> int:
        return len(self._columns)

    def keys(self):
        return self._columns


class CompatCursor:
    def __init__(self, rows: list[CompatRow] | None = None, *, rowcount: int = -1, lastrowid: int | None = None):
        self._rows = rows or []
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchone(self) -> CompatRow | None:
        if not self._rows:
            return None
        return self._rows.pop(0)

    def fetchall(self) -> list[CompatRow]:
        rows = self._rows
        self._rows = []
        return rows


class PostgresCompatConnection:
    """A narrow sqlite3.Connection-compatible facade.

    It handles qmark placeholders, optional table renames, tuple-like rows, and
    identity ``lastrowid`` for legacy insert paths.
    """

    def __init__(
        self,
        *,
        table_map: dict[str, str] | None = None,
        identity_tables: set[str] | None = None,
        register_pgvector: bool = False,
    ):
        self._conn = open_connection(register_pgvector=register_pgvector)
        self._table_map = table_map or {}
        self._identity_tables = identity_tables or set()

    def __enter__(self) -> PostgresCompatConnection:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()

    def close(self) -> None:
        self._conn.close()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> CompatCursor:
        translated = self._translate_sql(sql)
        if self._is_schema_statement(translated):
            return CompatCursor(rowcount=0)

        translated, should_return_id = self._maybe_add_returning_id(translated)
        with self._conn.cursor() as cur:
            cur.execute(translated, tuple(params or ()))
            rows = self._wrap_rows(cur)
            lastrowid = None
            if should_return_id and rows:
                lastrowid = int(rows[0]["id"])
                rows = []
            return CompatCursor(rows, rowcount=cur.rowcount, lastrowid=lastrowid)

    def executemany(self, sql: str, params_seq: Sequence[Sequence[Any]]) -> CompatCursor:
        translated = self._translate_sql(sql)
        if self._is_schema_statement(translated):
            return CompatCursor(rowcount=0)
        with self._conn.cursor() as cur:
            cur.executemany(translated, [tuple(params) for params in params_seq])
            return CompatCursor(rowcount=cur.rowcount)

    def _wrap_rows(self, cur: Any) -> list[CompatRow]:
        if cur.description is None:
            return []
        columns = [col.name for col in cur.description]
        raw_rows = cur.fetchall()
        return [CompatRow(row, columns) for row in raw_rows]

    def _translate_sql(self, sql: str) -> str:
        out = sql
        out = re.sub(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", "INSERT INTO", out, flags=re.IGNORECASE)
        out = out.replace("?", "%s")
        out = re.sub(r"datetime\('now'\)", "CURRENT_TIMESTAMP::text", out, flags=re.IGNORECASE)
        out = re.sub(
            r"created_at\s+<\s+datetime\('now',\s*%s\)",
            "created_at::timestamptz < (CURRENT_TIMESTAMP + (%s)::interval)",
            out,
            flags=re.IGNORECASE,
        )
        for source, target in self._table_map.items():
            out = re.sub(rf"\b{re.escape(source)}\b", target, out)
        if re.search(r"\bINSERT\s+INTO\b", out, re.IGNORECASE) and "ON CONFLICT" not in out.upper():
            if re.search(r"\bVALUES\s*\(", out, re.IGNORECASE) and "INSERT OR IGNORE" in sql.upper():
                out = f"{out} ON CONFLICT DO NOTHING"
        return out

    @staticmethod
    def _is_schema_statement(sql: str) -> bool:
        stripped = sql.lstrip().upper()
        return stripped.startswith(("CREATE TABLE", "CREATE INDEX", "ALTER TABLE", "PRAGMA"))

    def _maybe_add_returning_id(self, sql: str) -> tuple[str, bool]:
        if " RETURNING " in sql.upper():
            return sql, False
        match = re.match(r"\s*INSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)\b", sql, flags=re.IGNORECASE)
        if not match:
            return sql, False
        table = match.group(1)
        if table not in self._identity_tables:
            return sql, False
        return f"{sql} RETURNING id", True
