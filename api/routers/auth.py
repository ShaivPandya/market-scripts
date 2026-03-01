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
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from jose import JWTError, jwt
from pydantic import BaseModel

# ── Config (read from .env via load_dotenv() in main.py) ─────────────────────
_AUTH_MODE = (os.environ.get("AUTH_MODE") or "").strip().lower() or "password"


def _is_cloudflare_mode() -> bool:
    # Cloudflare Access gates the app; the API still remains protected by the proxy-secret middleware.
    return _AUTH_MODE == "cloudflare"


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
    expire = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    return jwt.encode(
        {"sub": "admin", "exp": expire},
        _get_jwt_secret(),
        algorithm=_get_jwt_algorithm(),
    )


# ── Dependency — inject via dependencies=[Depends(require_auth)] ──────────────

def require_auth(access_token: str | None = Cookie(default=None)) -> str:
    """
    Returns the token subject ("admin") or raises HTTP 401.

    Usage in main.py:
        app.include_router(some_router, prefix="/api", dependencies=[Depends(require_auth)])
    """
    if _is_cloudflare_mode():
        return "admin"
    if access_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    try:
        payload = jwt.decode(
            access_token,
            _get_jwt_secret(),
            algorithms=[_get_jwt_algorithm()],
        )
        return payload["sub"]
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


# ── Schemas ───────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    password: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/auth/login")
def login(body: LoginRequest, response: Response):
    if not bcrypt.checkpw(body.password.encode(), _get_password_hash()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )
    token = _create_token()
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="strict",
        secure=bool(os.environ.get("RENDER")),  # True on Render (HTTPS), False locally (HTTP)
        path="/",
    )
    return {"detail": "ok"}


@router.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(key="access_token", path="/", samesite="strict")
    return {"detail": "ok"}


@router.get("/auth/me")
def me(sub: str = Depends(require_auth)):
    return {"username": sub}
