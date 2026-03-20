"""
FastAPI backend for Market Analysis Dashboard.

Run from project root:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
import threading
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.gzip import GZipMiddleware

from api.exceptions import AppError
from api.logging_config import configure_logging, generate_request_id, request_id_var
from api.safe_import import get_degraded_modules, safe_import_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

configure_logging(json_format=IS_PRODUCTION)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Import routers AFTER path + env setup
# ---------------------------------------------------------------------------
# Core routers (must succeed)
from api.routers import (
    agent,
    memory,
    portfolio,
    portfolio_edit,
    thesis,
)
from api.routers import auth as auth_router
from api.routers.auth import require_auth

# Optional routers — gracefully degrade if dependencies fail
_optional_routers: dict[str, tuple] = {}

_OPTIONAL_MODULES = [
    ("api.routers.analyzer", "analyzer", "portfolio"),
    ("api.routers.hedging", "hedging", "portfolio"),
    ("api.routers.sizer", "sizer", "portfolio"),
    ("api.routers.momentum", "momentum", "portfolio"),
    ("api.routers.chart", "chart", "technical"),
    ("api.routers.quality", "quality", "equities"),
    ("api.routers.short_screen", "short_screen", "equities"),
    ("api.routers.fundamental_momentum", "fundamental_momentum", "equities"),
    ("api.routers.index_dashboard", "index_dashboard", "equities"),
    ("api.routers.fx_dashboard", "fx_dashboard", "fx"),
    ("api.routers.fx_model", "fx_model", "fx"),
    ("api.routers.commodities", "commodities", "commodities"),
    ("api.routers.commodities_curve", "commodities_curve", "commodities"),
    ("api.routers.commodity_research", "commodity_research", "commodities"),
    ("api.routers.market_technicals", "market_technicals", "equities"),
    ("api.routers.economic_growth", "economic_growth", "macro"),
    ("api.routers.labor_market", "labor_market", "macro"),
    ("api.routers.housing", "housing", "macro"),
    ("api.routers.liquidity", "liquidity", "macro"),
    ("api.routers.country_dashboard", "country_dashboard", "macro"),
    ("api.routers.positioning", "positioning", "macro"),
    ("api.routers.sentiment", "sentiment", "macro"),
    ("api.routers.breakout", "breakout", "macro"),
    ("api.routers.central_banks", "central_banks", "macro"),
    ("api.routers.sector_metrics", "sector_metrics", "equities"),
    ("api.routers.industry", "industry", "macro"),
    ("api.routers.yield_curve", "yield_curve", "fixed-income"),
    ("api.routers.bond_dashboard", "bond_dashboard", "fixed-income"),
    ("api.routers.financials", "financials", "equities"),
    ("api.routers.signal_aggregator", "signal_aggregator", "macro"),
    ("api.routers.portfolio_news", "portfolio_news", "portfolio"),
    ("api.routers.ontology", "ontology", "ontology"),
    ("api.routers.weekly_report", "weekly_report", "reports"),
]

for module_path, name, tag in _OPTIONAL_MODULES:
    router, healthy = safe_import_router(module_path)
    _optional_routers[name] = (router, tag, healthy)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Market Analysis API",
    description="REST API for portfolio analytics, market data, and macro indicators",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter


def _rate_limit_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, RateLimitExceeded):
        return _rate_limit_exceeded_handler(request, exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error", "type": "InternalError"})


app.add_exception_handler(RateLimitExceeded, _rate_limit_exception_handler)

app.add_middleware(GZipMiddleware, minimum_size=500)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(AppError)
async def _app_error_handler(request: Request, exc: AppError):
    logger.error("AppError [%s]: %s", exc.__class__.__name__, exc.message, exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "type": exc.__class__.__name__},
    )


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": "InternalError"},
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    """Assign a correlation ID to every request."""
    rid = request.headers.get("x-request-id") or generate_request_id()
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response


@app.middleware("http")
async def _request_timing_middleware(request: Request, call_next):
    """Log request duration for every API call."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    if request.url.path.startswith("/api/"):
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
    return response


_API_PROXY_SECRET = (os.environ.get("API_PROXY_SECRET") or "").strip() or None


@app.middleware("http")
async def _require_proxy_secret(request: Request, call_next):
    """
    When API_PROXY_SECRET is set (production), require every /api/* request except
    /api/health to include X-Api-Proxy-Secret (injected by the Cloudflare Pages proxy).
    """
    if _API_PROXY_SECRET and request.url.path.startswith("/api/"):
        if request.url.path != "/api/health":
            provided = request.headers.get("x-api-proxy-secret")
            if provided != _API_PROXY_SECRET:
                return JSONResponse({"detail": "Forbidden"}, status_code=403)
    return await call_next(request)


_CORS_ORIGINS = [
    o.strip()
    for o in (os.environ.get("CORS_ORIGINS") or "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Router registration — versioned at /api/v1
# ---------------------------------------------------------------------------
_auth_dep = [Depends(require_auth)]
_V1 = "/api/v1"

# Core routers (always available)
app.include_router(auth_router.router, prefix=_V1, tags=["auth"])
app.include_router(portfolio.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(portfolio_edit.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(thesis.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(memory.router, prefix=_V1, dependencies=_auth_dep, tags=["agent"])
app.include_router(agent.router, prefix=_V1, dependencies=_auth_dep, tags=["agent"])

# Investing OS routers (core_db entities + aggregates)
from api.routers import (
    action_items,
    approvals,
    dossier,
    process_entities,
    research_notes,
    triggers,
    workflow_runs,
    workspace,
)

app.include_router(workspace.router, prefix=_V1, dependencies=_auth_dep, tags=["workspace"])
app.include_router(dossier.router, prefix=_V1, dependencies=_auth_dep, tags=["workspace"])
app.include_router(approvals.router, prefix=_V1, dependencies=_auth_dep, tags=["approvals"])
app.include_router(action_items.router, prefix=_V1, dependencies=_auth_dep, tags=["actions"])
app.include_router(triggers.router, prefix=_V1, dependencies=_auth_dep, tags=["triggers"])
app.include_router(process_entities.router, prefix=_V1, dependencies=_auth_dep, tags=["process"])
app.include_router(research_notes.router, prefix=_V1, dependencies=_auth_dep, tags=["research"])
app.include_router(workflow_runs.router, prefix=_V1, dependencies=_auth_dep, tags=["workflows"])

# Optional routers (gracefully degraded if import failed)
for _name, (_router, _tag, _healthy) in _optional_routers.items():
    app.include_router(_router, prefix=_V1, dependencies=_auth_dep, tags=[_tag])


# ---------------------------------------------------------------------------
# Cache warming on startup
# ---------------------------------------------------------------------------
_WARM_TOOLS: list[tuple[str, dict[str, Any]]] = [
    ("get_portfolio", {}),
    ("get_market_breadth", {}),
    ("get_vix_term_structure", {}),
    ("get_liquidity", {}),
]


def _warm_caches() -> None:
    """Pre-fetch frequently used tool results into the in-memory TTL caches.

    Runs in a background thread so it does not delay server startup.
    """
    from api.agent_tools import execute_tool

    for tool_name, args in _WARM_TOOLS:
        try:
            execute_tool(tool_name, args)
            logger.info("cache_warm tool=%s status=ok", tool_name)
        except Exception:
            logger.warning("cache_warm tool=%s status=error", tool_name, exc_info=True)


@app.on_event("startup")
def _startup_warm_caches():
    thread = threading.Thread(target=_warm_caches, daemon=True, name="cache-warm")
    thread.start()
    logger.info("Cache warming started in background thread")


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------
@app.delete("/api/v1/cache", dependencies=_auth_dep, tags=["admin"])
def clear_cache():
    from api.cache import invalidate_all

    invalidate_all()
    return {"status": "cleared"}


@app.get("/api/health", tags=["admin"])
def health():
    checks: dict[str, str] = {}

    # Check portfolio DB connectivity
    try:
        from portfolio.portfolio_db import _get_conn as _portfolio_conn

        conn = _portfolio_conn()
        conn.execute("SELECT COUNT(*) FROM positions")
        checks["portfolio_db"] = "ok"
    except Exception as exc:
        checks["portfolio_db"] = f"error: {exc}"

    # Check thesis DB connectivity
    try:
        from portfolio.thesis_db import _get_conn as _thesis_conn

        conn = _thesis_conn()
        conn.execute("SELECT COUNT(*) FROM thesis_meta")
        checks["thesis_db"] = "ok"
    except Exception as exc:
        checks["thesis_db"] = f"error: {exc}"

    # Check core DB connectivity
    try:
        from portfolio.core_db import _get_conn as _core_conn

        conn = _core_conn()
        conn.execute("SELECT COUNT(*) FROM catalysts")
        checks["core_db"] = "ok"
    except Exception as exc:
        checks["core_db"] = f"error: {exc}"

    # Check FRED API reachability
    fred_key = os.environ.get("FRED_API_KEY")
    if fred_key:
        try:
            from fredapi import Fred

            fred = Fred(api_key=fred_key)
            fred.get_series("DGS10", limit=1)
            checks["fred_api"] = "ok"
        except Exception as exc:
            checks["fred_api"] = f"error: {exc}"
    else:
        checks["fred_api"] = "no_api_key"

    # Track degraded optional modules
    degraded = get_degraded_modules()
    if degraded:
        checks["degraded_modules"] = list(degraded.keys())

    db_ok = checks.get("portfolio_db") == "ok" and checks.get("thesis_db") == "ok" and checks.get("core_db") == "ok"
    all_ok = all(v == "ok" for v in checks.values() if isinstance(v, str))

    if all_ok and not degraded:
        status = "ok"
    elif db_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    status_code = 200 if status != "unhealthy" else 503
    return JSONResponse({"status": status, "checks": checks}, status_code=status_code)
