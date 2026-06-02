"""First-party users, roles, and opaque server-side auth sessions."""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import bcrypt

from api.postgres import use_postgres_state
from api.postgres_state import PostgresStateConnection

DB_PATH = Path(__file__).parent / "auth.db"

_lock = threading.RLock()
_conn: sqlite3.Connection | PostgresStateConnection | None = None
_seeded = False

_CREATE_AUTH_USERS = """
CREATE TABLE IF NOT EXISTS auth_users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT,
    email         TEXT UNIQUE,
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
)
"""

_CREATE_AUTH_USER_ROLES = """
CREATE TABLE IF NOT EXISTS auth_user_roles (
    user_id TEXT NOT NULL,
    role    TEXT NOT NULL,
    PRIMARY KEY (user_id, role),
    FOREIGN KEY (user_id) REFERENCES auth_users(id) ON DELETE CASCADE
)
"""

_CREATE_AUTH_SESSIONS = """
CREATE TABLE IF NOT EXISTS auth_sessions (
    id               TEXT PRIMARY KEY,
    token_hash       TEXT NOT NULL UNIQUE,
    user_id          TEXT NOT NULL,
    csrf_token_hash  TEXT NOT NULL,
    expires_at       TEXT NOT NULL,
    revoked_at       TEXT,
    created_at       TEXT NOT NULL,
    user_agent       TEXT,
    ip_address       TEXT,
    FOREIGN KEY (user_id) REFERENCES auth_users(id) ON DELETE CASCADE
)
"""


@dataclass(frozen=True)
class AuthUser:
    id: str
    username: str
    roles: tuple[str, ...]
    email: str | None = None


@dataclass(frozen=True)
class AuthSession:
    session_id: str
    user: AuthUser
    session_token: str
    csrf_token: str
    expires_at: datetime


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _session_ttl_hours() -> int:
    raw = (os.environ.get("SESSION_TTL_HOURS") or os.environ.get("JWT_TTL_HOURS") or "12").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 12


def default_admin_username() -> str:
    return (os.environ.get("AUTH_DEFAULT_USERNAME") or "admin").strip() or "admin"


def _close_conn() -> None:
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


def reset_auth_store_for_tests() -> None:
    """Clear in-memory connection and re-seed on next access. Tests only."""
    global _conn, _seeded
    with _lock:
        _close_conn()
        _seeded = False
        if not use_postgres_state() and DB_PATH.exists():
            DB_PATH.unlink(missing_ok=True)


def _get_conn() -> sqlite3.Connection | PostgresStateConnection:
    global _conn
    if _conn is not None:
        try:
            _conn.execute("SELECT 1")
        except Exception:
            _close_conn()
    if _conn is None:
        with _lock:
            if _conn is None:
                if use_postgres_state():
                    _conn = PostgresStateConnection()
                else:
                    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                    _conn.row_factory = sqlite3.Row
                _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection | PostgresStateConnection) -> None:
    conn.execute(_CREATE_AUTH_USERS)
    conn.execute(_CREATE_AUTH_USER_ROLES)
    conn.execute(_CREATE_AUTH_SESSIONS)
    conn.commit()


def _row_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return {key: row[key] for key in row.keys()}


def _roles_for_user(conn: sqlite3.Connection | PostgresStateConnection, user_id: str) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT role FROM auth_user_roles WHERE user_id = ? ORDER BY role",
        (user_id,),
    ).fetchall()
    return tuple(str(r["role"] if isinstance(r, dict) else r[0]) for r in rows)


def _user_from_row(conn: sqlite3.Connection | PostgresStateConnection, row: Any) -> AuthUser | None:
    data = _row_dict(row)
    if not data:
        return None
    user_id = str(data["id"])
    return AuthUser(
        id=user_id,
        username=str(data["username"]),
        email=str(data["email"]) if data.get("email") else None,
        roles=_roles_for_user(conn, user_id),
    )


def _upsert_user(
    conn: sqlite3.Connection | PostgresStateConnection,
    *,
    username: str,
    password_hash: str | None,
    email: str | None,
    roles: tuple[str, ...],
) -> AuthUser:
    now = _now()
    existing = conn.execute(
        "SELECT id, username, email FROM auth_users WHERE username = ?",
        (username,),
    ).fetchone()
    if existing:
        user_id = str(_row_dict(existing)["id"])
        conn.execute(
            "UPDATE auth_users SET password_hash = ?, email = ?, active = 1, updated_at = ? WHERE id = ?",
            (password_hash, email, now, user_id),
        )
    else:
        user_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO auth_users (id, username, password_hash, email, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (user_id, username, password_hash, email, now, now),
        )
    conn.execute("DELETE FROM auth_user_roles WHERE user_id = ?", (user_id,))
    for role in roles:
        conn.execute(
            "INSERT INTO auth_user_roles (user_id, role) VALUES (?, ?)",
            (user_id, role),
        )
    conn.commit()
    row = conn.execute("SELECT * FROM auth_users WHERE id = ?", (user_id,)).fetchone()
    user = _user_from_row(conn, row)
    assert user is not None
    return user


def ensure_auth_users_seeded() -> None:
    """Seed admin and optional smoke users from env password hashes."""
    global _seeded
    with _lock:
        if _seeded:
            return
        conn = _get_conn()
        admin_hash = (os.environ.get("AUTH_PASSWORD_HASH") or "").strip()
        if admin_hash:
            _upsert_user(
                conn,
                username=default_admin_username(),
                password_hash=admin_hash,
                email=None,
                roles=("owner", "admin"),
            )
        smoke_hash = (os.environ.get("AUTH_SMOKE_PASSWORD_HASH") or "").strip()
        if smoke_hash:
            _upsert_user(
                conn,
                username="smoke",
                password_hash=smoke_hash,
                email=None,
                roles=("smoke", "viewer"),
            )
        _seeded = True


def get_user_by_username(username: str) -> AuthUser | None:
    ensure_auth_users_seeded()
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM auth_users WHERE username = ? AND active = 1",
        (username.strip(),),
    ).fetchone()
    return _user_from_row(conn, row)


def get_user_by_email(email: str) -> AuthUser | None:
    ensure_auth_users_seeded()
    conn = _get_conn()
    normalized = email.strip().lower()
    row = conn.execute(
        "SELECT * FROM auth_users WHERE lower(email) = ? AND active = 1",
        (normalized,),
    ).fetchone()
    return _user_from_row(conn, row)


def get_or_create_cloudflare_user(email: str) -> AuthUser:
    """Map Cloudflare Access identity to a first-party user."""
    ensure_auth_users_seeded()
    normalized_email = email.strip().lower()
    existing = get_user_by_email(normalized_email)
    if existing:
        return existing
    conn = _get_conn()
    username = normalized_email.split("@", 1)[0] or normalized_email
    base_username = username
    suffix = 0
    while get_user_by_username(username):
        suffix += 1
        username = f"{base_username}-{suffix}"
    default_roles = tuple(
        r.strip() for r in (os.environ.get("AUTH_CLOUDFLARE_DEFAULT_ROLES") or "admin,viewer").split(",") if r.strip()
    ) or ("admin", "viewer")
    return _upsert_user(
        conn,
        username=username,
        password_hash=None,
        email=normalized_email,
        roles=default_roles,
    )


def verify_password(user: AuthUser, password: str) -> bool:
    ensure_auth_users_seeded()
    conn = _get_conn()
    row = conn.execute("SELECT password_hash FROM auth_users WHERE id = ?", (user.id,)).fetchone()
    if not row:
        return False
    stored = _row_dict(row).get("password_hash")
    if not stored:
        return False
    return bcrypt.checkpw(password.encode(), str(stored).encode())


def create_session(
    user: AuthUser,
    *,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> AuthSession:
    ensure_auth_users_seeded()
    session_token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    session_id = str(uuid.uuid4())
    expires_at = datetime.now(UTC) + timedelta(hours=_session_ttl_hours())
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO auth_sessions (
            id, token_hash, user_id, csrf_token_hash, expires_at, revoked_at,
            created_at, user_agent, ip_address
        ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)
        """,
        (
            session_id,
            _hash_secret(session_token),
            user.id,
            _hash_secret(csrf_token),
            expires_at.isoformat(),
            _now(),
            user_agent,
            ip_address,
        ),
    )
    conn.commit()
    return AuthSession(
        session_id=session_id,
        user=user,
        session_token=session_token,
        csrf_token=csrf_token,
        expires_at=expires_at,
    )


def lookup_session(session_token: str) -> AuthSession | None:
    if not session_token:
        return None
    ensure_auth_users_seeded()
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT s.id, s.user_id, s.csrf_token_hash, s.expires_at, s.revoked_at,
               u.username, u.email
        FROM auth_sessions s
        JOIN auth_users u ON u.id = s.user_id
        WHERE s.token_hash = ? AND u.active = 1
        """,
        (_hash_secret(session_token),),
    ).fetchone()
    if not row:
        return None
    data = _row_dict(row)
    if data.get("revoked_at"):
        return None
    expires_at = _parse_iso(str(data["expires_at"]))
    if expires_at <= datetime.now(UTC):
        return None
    user_id = str(data["user_id"])
    user = AuthUser(
        id=user_id,
        username=str(data["username"]),
        email=str(data["email"]) if data.get("email") else None,
        roles=_roles_for_user(conn, user_id),
    )
    return AuthSession(
        session_id=str(data["id"]),
        user=user,
        session_token=session_token,
        csrf_token="",
        expires_at=expires_at,
    )


def verify_csrf(session_token: str, csrf_token: str | None) -> bool:
    if not session_token or not csrf_token:
        return False
    conn = _get_conn()
    row = conn.execute(
        "SELECT csrf_token_hash, revoked_at, expires_at FROM auth_sessions WHERE token_hash = ?",
        (_hash_secret(session_token),),
    ).fetchone()
    if not row:
        return False
    data = _row_dict(row)
    if data.get("revoked_at"):
        return False
    if _parse_iso(str(data["expires_at"])) <= datetime.now(UTC):
        return False
    return secrets.compare_digest(_hash_secret(csrf_token), str(data["csrf_token_hash"]))


def get_csrf_token_for_session(session_token: str) -> str | None:
    """Return a fresh CSRF token by rotating the stored hash (used on /me)."""
    session = lookup_session(session_token)
    if session is None:
        return None
    csrf_token = secrets.token_urlsafe(32)
    conn = _get_conn()
    conn.execute(
        "UPDATE auth_sessions SET csrf_token_hash = ? WHERE token_hash = ?",
        (_hash_secret(csrf_token), _hash_secret(session_token)),
    )
    conn.commit()
    return csrf_token


def revoke_session(session_token: str) -> None:
    if not session_token:
        return
    conn = _get_conn()
    conn.execute(
        "UPDATE auth_sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
        (_now(), _hash_secret(session_token)),
    )
    conn.commit()


def user_has_role(user: AuthUser, role: str) -> bool:
    return role in user.roles or "admin" in user.roles or "owner" in user.roles
