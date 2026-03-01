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
_PASSWORD_HASH = os.environ["AUTH_PASSWORD_HASH"].encode()
_JWT_SECRET    = os.environ["JWT_SECRET"]
_JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
_JWT_TTL_HOURS = int(os.environ.get("JWT_TTL_HOURS", "12"))

router = APIRouter(tags=["auth"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _create_token() -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=_JWT_TTL_HOURS)
    return jwt.encode(
        {"sub": "admin", "exp": expire},
        _JWT_SECRET,
        algorithm=_JWT_ALGORITHM,
    )


# ── Dependency — inject via dependencies=[Depends(require_auth)] ──────────────

def require_auth(access_token: str | None = Cookie(default=None)) -> str:
    """
    Returns the token subject ("admin") or raises HTTP 401.

    Usage in main.py:
        app.include_router(some_router, prefix="/api", dependencies=[Depends(require_auth)])
    """
    if access_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    try:
        payload = jwt.decode(access_token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
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
    if not bcrypt.checkpw(body.password.encode(), _PASSWORD_HASH):
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
        secure=False,  # set to True in production (HTTPS)
        max_age=_JWT_TTL_HOURS * 3600,
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
