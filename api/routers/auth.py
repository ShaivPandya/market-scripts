"""
Authentication router — first-party users and opaque server-side sessions.

Endpoints (all under /api prefix set in main.py):
    POST /api/auth/login   — verify credentials, set HTTP-only session cookie + CSRF token
    POST /api/auth/logout  — revoke session and clear cookie
    GET  /api/auth/me      — return user + roles + CSRF token when session valid

Dependencies:
    require_auth / require_actor — protect routes; role-aware Actor for policy checks
"""

import os
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Any

import bcrypt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.audit import emit_audit_event
from api.auth_store import (
    AuthUser,
    create_session,
    default_admin_username,
    ensure_auth_users_seeded,
    get_csrf_token_for_session,
    get_or_create_cloudflare_user,
    get_user_by_username,
    lookup_session,
    revoke_session,
    verify_password,
)
from ontology.policy import Actor, user_actor

_limiter = Limiter(key_func=get_remote_address)

_LOGIN_RATE_LIMIT = (os.environ.get("AUTH_LOGIN_RATE_LIMIT") or "").strip() or "5/minute"
_DEFAULT_LOGIN_FAILURE_LIMIT = 5
_DEFAULT_LOGIN_FAILURE_WINDOW_SECONDS = 15 * 60
_DEFAULT_LOGIN_LOCKOUT_SECONDS = 15 * 60

SESSION_COOKIE = "__session"
CSRF_HEADER = "x-csrf-token"


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
    return _auth_mode() == "cloudflare"


def _cloudflare_proxy_secret_configured() -> bool:
    return bool((os.environ.get("API_PROXY_SECRET") or "").strip())


def _is_production() -> bool:
    return os.environ.get("ENVIRONMENT", "development").strip().lower() == "production"


def _session_max_age_seconds() -> int:
    from api.auth_store import _session_ttl_hours

    return _session_ttl_hours() * 3600


def _set_session_cookie(response: Response, session_token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_token,
        httponly=True,
        samesite="strict",
        secure=_is_production(),
        path="/",
        max_age=_session_max_age_seconds(),
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/", samesite="strict")


def _cloudflare_identity_email(request: Request) -> str | None:
    for header in (
        "cf-access-authenticated-user-email",
        "Cf-Access-Authenticated-User-Email",
    ):
        value = (request.headers.get(header) or "").strip()
        if value:
            return value
    return None


router = APIRouter(tags=["auth"])


def _login_attempt_key(request: Request) -> str:
    return f"ip:{get_remote_address(request) or 'unknown'}"


def _retry_after_seconds(until: float, now: float) -> int:
    return max(1, int(until - now + 0.999))


def _reset_login_attempt_state() -> None:
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


def _resolve_session_user(
    request: Request,
    access_token: str | None,
) -> AuthUser:
    if _is_cloudflare_mode():
        if not _cloudflare_proxy_secret_configured():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API proxy secret is required in Cloudflare auth mode.",
            )
        email = _cloudflare_identity_email(request)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Cloudflare Access identity is required.",
            )
        return get_or_create_cloudflare_user(email)

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    session = lookup_session(access_token)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
        )
    return session.user


def require_auth(
    request: Request,
    access_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> str:
    """Returns the authenticated username or raises HTTP 401/403."""
    return _resolve_session_user(request, access_token).username


def require_actor(
    request: Request,
    access_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> Actor:
    user = _resolve_session_user(request, access_token)
    return user_actor(user.username, user.roles, source="api")


def require_admin(
    actor: Annotated[Actor, Depends(require_actor)],
) -> Actor:
    roles = {role.lower() for role in actor.roles}
    if actor.actor_type != "system" and "admin" not in roles and "owner" not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is required.",
        )
    return actor


ActorDep = Annotated[Actor, Depends(require_actor)]
AdminActorDep = Annotated[Actor, Depends(require_admin)]


class LoginRequest(BaseModel):
    username: str | None = Field(default=None, max_length=128)
    password: str = Field(..., min_length=1, max_length=512)


def _legacy_password_login(password: str) -> AuthUser | None:
    """Fallback when auth tables are empty: verify env bcrypt hashes directly."""
    admin_hash = (os.environ.get("AUTH_PASSWORD_HASH") or "").strip()
    smoke_hash = (os.environ.get("AUTH_SMOKE_PASSWORD_HASH") or "").strip()
    if smoke_hash and bcrypt.checkpw(password.encode(), smoke_hash.encode()):
        return AuthUser(id="legacy-smoke", username="smoke", roles=("smoke", "viewer"))
    if admin_hash and bcrypt.checkpw(password.encode(), admin_hash.encode()):
        return AuthUser(id="legacy-admin", username=default_admin_username(), roles=("owner", "admin"))
    return None


@router.post("/auth/login")
@_limiter.limit(_LOGIN_RATE_LIMIT)
def login(request: Request, body: LoginRequest, response: Response):
    if _is_cloudflare_mode():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password login is disabled in Cloudflare auth mode.",
        )

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

    username = (body.username or default_admin_username()).strip() or default_admin_username()
    ensure_auth_users_seeded()
    user = get_user_by_username(username)
    authenticated = user is not None and verify_password(user, body.password)
    if not authenticated:
        legacy = _legacy_password_login(body.password)
        if legacy is not None and (body.username is None or legacy.username == username):
            user = legacy
            authenticated = True

    if not authenticated:
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
                "username": username,
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

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )

    _clear_failed_logins(request)
    session = create_session(
        user,
        user_agent=request.headers.get("user-agent"),
        ip_address=get_remote_address(request),
    )
    _set_session_cookie(response, session.session_token)
    emit_audit_event(
        "auth.login",
        "permission",
        "succeeded",
        actor=user_actor(user.username, user.roles, source="api"),
        after_summary={"auth_mode": _auth_mode(), "subject": user.username},
        metadata={"path": str(request.url.path)},
    )
    return {
        "detail": "ok",
        "username": user.username,
        "roles": list(user.roles),
        "csrfToken": session.csrf_token,
    }


@router.post("/auth/logout")
def logout(
    response: Response,
    access_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    if access_token:
        revoke_session(access_token)
    _clear_session_cookie(response)
    emit_audit_event(
        "auth.logout",
        "permission",
        "succeeded",
        after_summary={"auth_mode": _auth_mode()},
    )
    return {"detail": "ok"}


@router.get("/auth/me")
def me(
    request: Request,
    access_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    user = _resolve_session_user(request, access_token)
    csrf_token: str | None = None
    if not _is_cloudflare_mode() and access_token:
        csrf_token = get_csrf_token_for_session(access_token)
    return {
        "username": user.username,
        "roles": list(user.roles),
        "csrfToken": csrf_token,
    }


def verify_request_csrf(request: Request) -> bool:
    """Validate CSRF for browser session cookies. Returns True if check passes or is skipped."""
    if _is_cloudflare_mode():
        return True
    session_token = request.cookies.get(SESSION_COOKIE)
    if not session_token:
        return True
    if request.method.upper() not in {"POST", "PUT", "PATCH", "DELETE"}:
        return True
    csrf = (request.headers.get(CSRF_HEADER) or request.headers.get("X-CSRF-Token") or "").strip()
    from api.auth_store import verify_csrf

    return verify_csrf(session_token, csrf or None)


def is_machine_authenticated_request(request: Request) -> bool:
    """True when scheduler or report-sync shared secrets authenticate the request."""
    import hmac

    scheduler_expected = (os.getenv("SCHEDULER_SECRET") or "").strip()
    if scheduler_expected:
        provided = request.headers.get("x-scheduler-secret")
        if provided and hmac.compare_digest(provided, scheduler_expected):
            return True
    report_expected = (os.getenv("REPORT_SYNC_SECRET") or "").strip()
    if report_expected:
        provided = request.headers.get("x-report-sync-secret")
        if provided and hmac.compare_digest(provided, report_expected):
            return True
    return False
