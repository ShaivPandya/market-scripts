"""
FastAPI backend for Market Analysis Dashboard.

Run from project root:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.gzip import GZipMiddleware

from api.audit import emit_audit_event
from api.exceptions import AppError, DataFetchError
from api.logging_config import configure_logging, generate_request_id, request_id_var
from api.request_limits import MULTIPART_FORM_DATA_OVERHEAD_BYTES, BodySizeLimitMiddleware
from api.request_schema import (
    collect_api_request_schema_definitions,
    validate_and_upgrade_request_schema,
)
from api.safe_import import get_degraded_modules, safe_import_router
from ontology.schema_definitions import (
    domain_action_schema_definitions,
    ontology_schema_definitions,
    seed_schema_definitions,
)

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
    document_generation,
    management_quality,
    memory,
    overview,
    portfolio,
    portfolio_edit,
    settings,
    thesis,
)
from api.routers import auth as auth_router
from api.routers.auth import require_actor
from ontology.policy import PolicyDenied

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
    ("api.routers.long_screen", "long_screen", "equities"),
    ("api.routers.fundamental_momentum", "fundamental_momentum", "equities"),
    ("api.routers.price_momentum", "price_momentum", "equities"),
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
    ("api.routers.central_banks", "central_banks", "macro"),
    ("api.routers.sector_metrics", "sector_metrics", "equities"),
    ("api.routers.industry", "industry", "macro"),
    ("api.routers.yield_curve", "yield_curve", "fixed-income"),
    ("api.routers.bond_dashboard", "bond_dashboard", "fixed-income"),
    ("api.routers.financials", "financials", "equities"),
    ("api.routers.valuation", "valuation", "equities"),
    ("api.routers.signal_aggregator", "signal_aggregator", "macro"),
    ("api.routers.portfolio_news", "portfolio_news", "portfolio"),
    ("api.routers.ontology", "ontology", "ontology"),
    ("api.routers.source_ingestion", "source_ingestion", "ontology"),
    ("api.routers.risk", "risk", "risk"),
    ("api.routers.dcf", "dcf", "equities"),
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
    docs_url=None if IS_PRODUCTION else "/api/docs",
    redoc_url=None if IS_PRODUCTION else "/api/redoc",
    openapi_url=None if IS_PRODUCTION else "/api/openapi.json",
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


def _multipart_request_body_limit(file_limit_bytes: int) -> int:
    return file_limit_bytes + MULTIPART_FORM_DATA_OVERHEAD_BYTES


_ENDPOINT_BODY_LIMITS = {
    "/api/v1/thesis/generate": _multipart_request_body_limit(30 * 1024 * 1024),
    "/api/v1/overview/generate": _multipart_request_body_limit(30 * 1024 * 1024),
    "/api/v1/management-quality/generate": _multipart_request_body_limit(30 * 1024 * 1024),
    "/api/v1/economic-growth/crb-file": _multipart_request_body_limit(10 * 1024 * 1024),
    "/api/v1/portfolio-news": _multipart_request_body_limit(10 * 1024 * 1024),
}

app.add_middleware(BodySizeLimitMiddleware, path_limits=_ENDPOINT_BODY_LIMITS)
app.add_middleware(GZipMiddleware, minimum_size=500)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(AppError)
async def _app_error_handler(request: Request, exc: AppError):
    logger.error("AppError [%s]: %s", exc.__class__.__name__, exc.message, exc_info=True)
    content = {"error": exc.message, "type": exc.__class__.__name__}
    if isinstance(exc, DataFetchError):
        content["source"] = exc.source
        if exc.detail:
            content["detail"] = exc.detail
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": "InternalError"},
    )


@app.exception_handler(PolicyDenied)
async def _policy_denied_handler(request: Request, exc: PolicyDenied):
    logger.warning("PolicyDenied on %s %s: %s", request.method, request.url.path, exc.reason)
    emit_audit_event(
        "permission.denied",
        "permission",
        "denied",
        metadata={"method": request.method, "path": request.url.path},
        error=exc.reason,
    )
    return JSONResponse(status_code=403, content={"detail": exc.reason, "type": "PolicyDenied"})


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
_DOCS_PATHS = {"/api/docs", "/api/redoc", "/api/openapi.json"}


def _is_production_runtime() -> bool:
    return (os.environ.get("ENVIRONMENT") or ENVIRONMENT).strip().lower() == "production"


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    """Assign a correlation ID to every request."""
    rid = request.headers.get("x-request-id") or generate_request_id()
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response


@app.middleware("http")
async def _security_headers_middleware(request: Request, call_next):
    """Apply conservative security headers to every response."""
    response = await call_next(request)
    if "x-content-type-options" not in response.headers:
        response.headers["X-Content-Type-Options"] = "nosniff"
    if "referrer-policy" not in response.headers:
        response.headers["Referrer-Policy"] = "no-referrer"
    csp = response.headers.get("Content-Security-Policy")
    if csp:
        if "frame-ancestors" not in csp.lower():
            response.headers["Content-Security-Policy"] = f"{csp}; frame-ancestors 'none'"
    else:
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    if _is_production_runtime() and "strict-transport-security" not in response.headers:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
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


@app.middleware("http")
async def _production_schema_middleware(request: Request, call_next):
    """Disable interactive docs and schema routes in production."""
    if _is_production_runtime() and request.url.path in _DOCS_PATHS:
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return await call_next(request)


@app.middleware("http")
async def _request_schema_version_middleware(request: Request, call_next):
    """Validate and upgrade versioned API request bodies before route validation."""
    return await validate_and_upgrade_request_schema(app, request, call_next)


def _api_proxy_secret() -> str | None:
    return (os.environ.get("API_PROXY_SECRET") or "").strip() or None


def _auth_mode() -> str:
    return (os.environ.get("AUTH_MODE") or "").strip().lower() or "password"


def _proxy_secret_required() -> bool:
    if _auth_mode() == "cloudflare":
        return True
    return (os.environ.get("REQUIRE_API_PROXY_SECRET") or "").strip().lower() in ("1", "true", "yes")


_WRITE_FREEZE = (os.environ.get("WRITE_FREEZE") or "").strip().lower() in ("1", "true", "yes")


@app.middleware("http")
async def _require_proxy_secret(request: Request, call_next):
    """
    Require X-Api-Proxy-Secret for deployments that have an edge proxy capable
    of injecting it. Firebase Hosting rewrites cannot add this header, so merely
    configuring API_PROXY_SECRET must not block password-mode browser traffic.
    """
    if request.url.path.startswith("/api/") and request.url.path != "/api/health":
        if _proxy_secret_required():
            proxy_secret = _api_proxy_secret()
            if not proxy_secret:
                emit_audit_event(
                    "permission.proxy_secret_denied",
                    "permission",
                    "denied",
                    metadata={"method": request.method, "path": request.url.path, "reason": "proxy_secret_missing"},
                    error="API proxy secret is required for this auth mode.",
                )
                return JSONResponse({"detail": "API proxy secret is required for this auth mode."}, status_code=403)
            provided = request.headers.get("x-api-proxy-secret")
            if provided != proxy_secret:
                emit_audit_event(
                    "permission.proxy_secret_denied",
                    "permission",
                    "denied",
                    metadata={"method": request.method, "path": request.url.path, "reason": "invalid_proxy_secret"},
                    error="Forbidden",
                )
                return JSONResponse({"detail": "Forbidden"}, status_code=403)
    return await call_next(request)


@app.middleware("http")
async def _write_freeze_middleware(request: Request, call_next):
    """Reject mutating API calls during cutover freeze."""
    if _WRITE_FREEZE and request.url.path.startswith("/api/") and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        if request.url.path not in {"/api/v1/auth/login", "/api/health", "/api/v1/admin/quiescence"}:
            emit_audit_event(
                "permission.write_freeze_denied",
                "permission",
                "denied",
                metadata={"method": request.method, "path": request.url.path},
                error="Writes are frozen for migration cutover.",
            )
            return JSONResponse({"detail": "Writes are frozen for migration cutover."}, status_code=423)
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
_auth_dep = [Depends(require_actor)]
_V1 = "/api/v1"

# Core routers (always available)
app.include_router(auth_router.router, prefix=_V1, tags=["auth"])
app.include_router(portfolio.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(portfolio_edit.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(thesis.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(overview.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(management_quality.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(document_generation.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(memory.router, prefix=_V1, dependencies=_auth_dep, tags=["agent"])
app.include_router(agent.router, prefix=_V1, dependencies=_auth_dep, tags=["agent"])
app.include_router(settings.router, prefix=_V1, dependencies=_auth_dep, tags=["settings"])

# Investing OS routers (core_db entities + aggregates)
from api.routers import (
    action_items,
    admin_jobs,
    approvals,
    domain_actions,
    dossier,
    ideas,
    optimization,
    policy_gate,
    process_entities,
    provenance,
    recommendations,
    report_sync,
    triggers,
    workflow_runs,
    workspace,
)

app.include_router(workspace.router, prefix=_V1, dependencies=_auth_dep, tags=["workspace"])
app.include_router(dossier.router, prefix=_V1, dependencies=_auth_dep, tags=["workspace"])
app.include_router(ideas.router, prefix=_V1, dependencies=_auth_dep, tags=["ideas"])
app.include_router(approvals.router, prefix=_V1, dependencies=_auth_dep, tags=["approvals"])
app.include_router(domain_actions.router, prefix=_V1, dependencies=_auth_dep, tags=["domain-actions"])
app.include_router(action_items.router, prefix=_V1, dependencies=_auth_dep, tags=["actions"])
app.include_router(triggers.router, prefix=_V1, dependencies=_auth_dep, tags=["triggers"])
app.include_router(process_entities.router, prefix=_V1, dependencies=_auth_dep, tags=["process"])
app.include_router(provenance.router, prefix=_V1, dependencies=_auth_dep, tags=["provenance"])
app.include_router(recommendations.router, prefix=_V1, dependencies=_auth_dep, tags=["recommendations"])
app.include_router(optimization.router, prefix=_V1, dependencies=_auth_dep, tags=["optimization"])
app.include_router(policy_gate.router, prefix=_V1, dependencies=_auth_dep, tags=["policy-gate"])
app.include_router(workflow_runs.router, prefix=_V1, dependencies=_auth_dep, tags=["workflows"])
app.include_router(admin_jobs.router, prefix=_V1, tags=["admin"])
app.include_router(report_sync.router, prefix=_V1, tags=["reports"])

# Optional routers (gracefully degraded if import failed)
for _name, (_router, _tag, _healthy) in _optional_routers.items():
    app.include_router(_router, prefix=_V1, dependencies=_auth_dep, tags=[_tag])


def _seed_runtime_schema_registry() -> None:
    try:
        from ontology.repository import OntologyRepository

        repo = OntologyRepository()
        with repo._connect() as conn:
            seed_schema_definitions(
                conn,
                [
                    *ontology_schema_definitions(),
                    *domain_action_schema_definitions(),
                    *collect_api_request_schema_definitions(app.router.routes),
                ],
            )
    except Exception:
        logger.warning("schema registry seed failed", exc_info=True)


_seed_runtime_schema_registry()


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
    return {"status": "ok"}


def _detailed_health_response() -> JSONResponse:
    checks: dict[str, str | list[str]] = {}

    # Check ontology/Postgres connectivity
    try:
        from api.postgres import connect

        with connect() as conn:
            conn.execute("SELECT 1")
        checks["postgres"] = "ok"
    except Exception:
        logger.warning("postgres health check failed", exc_info=True)
        checks["postgres"] = "error"

    # Check FRED API reachability
    fred_key = os.environ.get("FRED_API_KEY")
    if fred_key:
        try:
            from fredapi import Fred

            fred = Fred(api_key=fred_key)
            fred.get_series("DGS10", limit=1)
            checks["fred_api"] = "ok"
        except Exception:
            logger.warning("fred_api health check failed", exc_info=True)
            checks["fred_api"] = "error"
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


@app.get("/api/v1/admin/health", dependencies=_auth_dep, tags=["admin"])
def admin_health():
    return _detailed_health_response()


@app.get("/api/v1/admin/quiescence", dependencies=_auth_dep, tags=["admin"])
def quiescence():
    active_jobs = 0
    try:
        from api.job_queue import count_active_jobs

        active_jobs = count_active_jobs()
    except Exception:
        active_jobs = 0
    return {
        "write_freeze": _WRITE_FREEZE,
        "active_jobs": active_jobs,
        "pending_writes": 0,
    }
