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


def _auth_mode() -> str:
    return (os.environ.get("AUTH_MODE") or "").strip().lower() or "password"


def _is_cloudflare_mode() -> bool:
    # Cloudflare Access gates the app; the API still remains protected by the proxy-secret middleware.
    return _auth_mode() == "cloudflare"


def _cloudflare_proxy_secret_configured() -> bool:
    return bool((os.environ.get("API_PROXY_SECRET") or "").strip())


def _get_password_hash() -> bytes:
    value = os.environ.get("AUTH_PASSWORD_HASH")
    if not value:
        raise RuntimeError("AUTH_PASSWORD_HASH is not set")
    return value.encode()


def _get_jwt_secret() -> str:
    value = os.environ.get("JWT_SECRET")
    if not value:
        raise RuntimeError("JWT_SECRET is not set")
    return value


def _get_jwt_algorithm() -> str:
    return os.environ.get("JWT_ALGORITHM", "HS256")


def _get_jwt_ttl_hours() -> int:
    return int(os.environ.get("JWT_TTL_HOURS", "12"))


router = APIRouter(tags=["auth"])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _create_token() -> str:
    ttl_hours = _get_jwt_ttl_hours()
    expire = datetime.now(UTC) + timedelta(hours=ttl_hours)
    return cast(
        str,
        jwt.encode(
            {"sub": "admin", "exp": expire},
            _get_jwt_secret(),
            algorithm=_get_jwt_algorithm(),
        ),
    )


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
            ),
        )
        sub = payload.get("sub")
        if isinstance(sub, str):
            return sub
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    except JWTError:
        raise HTTPException(  # noqa: B904
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


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
    if not bcrypt.checkpw(body.password.encode(), _get_password_hash()):
        emit_audit_event(
            "auth.login",
            "permission",
            "failed",
            after_summary={"auth_mode": _auth_mode(), "reason": "incorrect_password"},
            metadata={"path": str(request.url.path)},
            error="Incorrect password",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )
    token = _create_token()
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
        actor=admin_actor("admin", source="api"),
        after_summary={"auth_mode": _auth_mode()},
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
