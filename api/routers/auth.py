"""
Authentication router — single master account, no registration.

Endpoints (all under /api prefix set in main.py):
    POST /api/auth/login   — verify password, set HTTP-only JWT cookie
    POST /api/auth/logout  — clear the cookie
    GET  /api/auth/me      — return {"username": "admin"} if cookie valid, else 401

Dependency:
    require_auth — inject into include_router() calls to protect all routes in a router
"""

import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Annotated, Any, cast

import bcrypt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.audit import emit_audit_event
from ontology.policy import Actor, admin_actor

_limiter = Limiter(key_func=get_remote_address)

# ── Config (read from .env via load_dotenv() in main.py) ─────────────────────
_LOGIN_RATE_LIMIT = (os.environ.get("AUTH_LOGIN_RATE_LIMIT") or "").strip() or "5/minute"
_DEFAULT_LOGIN_FAILURE_LIMIT = 5
_DEFAULT_LOGIN_FAILURE_WINDOW_SECONDS = 15 * 60
_DEFAULT_LOGIN_LOCKOUT_SECONDS = 15 * 60


@dataclass
class _LoginFailureState:
    failures: list[float]
    locked_until: float = 0.0


_login_failure_lock = threading.Lock()
_login_failures: dict[str, _LoginFailureState] = {}


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _login_failure_limit() -> int:
    return max(0, _int_env("AUTH_LOGIN_FAILURE_LIMIT", _DEFAULT_LOGIN_FAILURE_LIMIT))


def _login_failure_window_seconds() -> int:
    return max(1, _int_env("AUTH_LOGIN_FAILURE_WINDOW_SECONDS", _DEFAULT_LOGIN_FAILURE_WINDOW_SECONDS))


def _login_lockout_seconds() -> int:
    return max(1, _int_env("AUTH_LOGIN_LOCKOUT_SECONDS", _DEFAULT_LOGIN_LOCKOUT_SECONDS))


def _auth_mode() -> str:
    return (os.environ.get("AUTH_MODE") or "").strip().lower() or "password"


def _is_cloudflare_mode() -> bool:
    # Cloudflare Access gates the app; the API still remains protected by the proxy-secret middleware.
    return _auth_mode() == "cloudflare"


def _cloudflare_proxy_secret_configured() -> bool:
    return bool((os.environ.get("API_PROXY_SECRET") or "").strip())


def _is_production_runtime() -> bool:
    return os.environ.get("ENVIRONMENT", "development").strip().lower() == "production"


def _get_password_hash() -> bytes:
    value = os.environ.get("AUTH_PASSWORD_HASH")
    if not value:
        raise RuntimeError("AUTH_PASSWORD_HASH is not set")

    if _is_production_runtime():
        # Ensure it looks like a bcrypt hash in production
        if not value.startswith("$2b$") or len(value) < 50:
            raise RuntimeError("AUTH_PASSWORD_HASH must be a valid bcrypt hash in production.")

    return value.encode()


def _get_smoke_password_hash() -> bytes | None:
    """Return the optional smoke password bcrypt hash, or None if not configured."""
    value = (os.environ.get("AUTH_SMOKE_PASSWORD_HASH") or "").strip()
    return value.encode() if value else None


def _get_jwt_secret() -> str:
    value = os.environ.get("JWT_SECRET")
    if not value:
        raise RuntimeError("JWT_SECRET is not set")

    if _is_production_runtime():
        if value == "your_random_jwt_secret_here":
            raise RuntimeError("JWT_SECRET cannot be the default example value in production.")
        if len(value) < 32:
            raise RuntimeError("JWT_SECRET must be at least 32 characters in production.")

    return value


def _get_jwt_algorithm() -> str:
    return os.environ.get("JWT_ALGORITHM", "HS256")


def _get_jwt_ttl_hours() -> int:
    return int(os.environ.get("JWT_TTL_HOURS", "12"))


router = APIRouter(tags=["auth"])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _create_token(subject: str = "admin") -> str:
    ttl_hours = _get_jwt_ttl_hours()
    now = datetime.now(UTC)
    expire = now + timedelta(hours=ttl_hours)
    return cast(
        str,
        jwt.encode(
            {"sub": subject, "iat": now, "exp": expire},
            _get_jwt_secret(),
            algorithm=_get_jwt_algorithm(),
        ),
    )


def _login_attempt_key(request: Request) -> str:
    return f"ip:{get_remote_address(request) or 'unknown'}"


def _retry_after_seconds(until: float, now: float) -> int:
    return max(1, int(until - now + 0.999))


def _reset_login_attempt_state() -> None:
    """Clear in-memory login attempt state. Used by tests and local reloads."""
    with _login_failure_lock:
        _login_failures.clear()


def _current_lockout_retry_after(request: Request) -> int | None:
    limit = _login_failure_limit()
    if limit <= 0:
        return None

    key = _login_attempt_key(request)
    now = time.time()
    window_start = now - _login_failure_window_seconds()
    with _login_failure_lock:
        state = _login_failures.get(key)
        if state is None:
            return None
        state.failures = [ts for ts in state.failures if ts >= window_start]
        if state.locked_until > now:
            return _retry_after_seconds(state.locked_until, now)
        if not state.failures:
            _login_failures.pop(key, None)
    return None


def _record_failed_login(request: Request) -> int | None:
    limit = _login_failure_limit()
    if limit <= 0:
        return None

    key = _login_attempt_key(request)
    now = time.time()
    window_start = now - _login_failure_window_seconds()
    with _login_failure_lock:
        state = _login_failures.setdefault(key, _LoginFailureState(failures=[]))
        state.failures = [ts for ts in state.failures if ts >= window_start]
        state.failures.append(now)
        if len(state.failures) >= limit:
            state.locked_until = max(state.locked_until, now + _login_lockout_seconds())
            return _retry_after_seconds(state.locked_until, now)
    return None


def _clear_failed_logins(request: Request) -> None:
    key = _login_attempt_key(request)
    with _login_failure_lock:
        _login_failures.pop(key, None)


# ── Dependency — inject via dependencies=[Depends(require_auth)] ──────────────


def require_auth(access_token: str | None = Cookie(default=None, alias="__session")) -> str:
    """
    Returns the token subject ("admin") or raises HTTP 401.

    Usage in main.py:
        app.include_router(some_router, prefix="/api", dependencies=[Depends(require_auth)])
    """
    if _is_cloudflare_mode():
        if not _cloudflare_proxy_secret_configured():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API proxy secret is required in Cloudflare auth mode.",
            )
        return "admin"
    if access_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    try:
        payload = cast(
            dict[str, Any],
            jwt.decode(
                access_token,
                _get_jwt_secret(),
                algorithms=[_get_jwt_algorithm()],
                options={"require_iat": True, "require_exp": True},
            ),
        )
        sub = payload.get("sub")
        if not isinstance(sub, str) or not sub.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing or invalid subject",
            )
        return sub
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
        ) from exc


def require_actor(sub: str = Depends(require_auth)) -> Actor:
    """Return the typed actor context for the authenticated v1 user."""
    return admin_actor(sub, source="api")


ActorDep = Annotated[Actor, Depends(require_actor)]


# ── Schemas ───────────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=512)


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/auth/login")
@_limiter.limit(_LOGIN_RATE_LIMIT)
def login(request: Request, body: LoginRequest, response: Response):
    retry_after = _current_lockout_retry_after(request)
    if retry_after is not None:
        emit_audit_event(
            "auth.login",
            "permission",
            "denied",
            after_summary={"auth_mode": _auth_mode(), "reason": "failed_login_lockout"},
            metadata={"path": str(request.url.path), "retry_after_seconds": retry_after},
            error="Too many failed login attempts",
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    # Check smoke password first (if configured), then admin password
    smoke_hash = _get_smoke_password_hash()
    is_smoke = False
    if smoke_hash and bcrypt.checkpw(body.password.encode(), smoke_hash):
        is_smoke = True
    elif not bcrypt.checkpw(body.password.encode(), _get_password_hash()):
        retry_after = _record_failed_login(request)
        emit_audit_event(
            "auth.login",
            "permission",
            "denied" if retry_after is not None else "failed",
            after_summary={
                "auth_mode": _auth_mode(),
                "reason": "failed_login_lockout" if retry_after is not None else "incorrect_password",
            },
            metadata={
                "path": str(request.url.path),
                **({"retry_after_seconds": retry_after} if retry_after is not None else {}),
            },
            error="Too many failed login attempts" if retry_after is not None else "Incorrect password",
        )
        if retry_after is not None:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )
    _clear_failed_logins(request)
    subject = "smoke" if is_smoke else "admin"
    token = _create_token(subject)
    response.set_cookie(
        key="__session",
        value=token,
        httponly=True,
        samesite="strict",
        secure=os.environ.get("ENVIRONMENT", "development").strip().lower() == "production",
        path="/",
    )
    emit_audit_event(
        "auth.login",
        "permission",
        "succeeded",
        actor=admin_actor(subject, source="api"),
        after_summary={"auth_mode": _auth_mode(), "subject": subject},
        metadata={"path": str(request.url.path)},
    )
    return {"detail": "ok"}


@router.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(key="__session", path="/", samesite="strict")
    emit_audit_event(
        "auth.logout",
        "permission",
        "succeeded",
        after_summary={"auth_mode": _auth_mode()},
    )
    return {"detail": "ok"}


@router.get("/auth/me")
def me(actor: ActorDep):
    return {"username": actor.actor_id}
