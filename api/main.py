"""
FastAPI backend for Market Analysis Dashboard.

Run from project root:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
import time

from paths import setup_paths

setup_paths()  # must happen before any project module imports

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
from api.routers import (
    agent,
    analyzer,
    breakout,
    central_banks,
    chart,
    commodities,
    commodities_curve,
    country_dashboard,
    economic_growth,
    financials,
    fundamental_momentum,
    fx_dashboard,
    fx_model,
    hedging,
    index_dashboard,
    industry,
    labor_market,
    liquidity,
    market_technicals,
    momentum,
    ontology,
    portfolio,
    portfolio_news,
    positioning,
    quality,
    sector_metrics,
    sentiment,
    short_screen,
    sizer,
    weekly_report,
    yield_curve,
)
from api.routers import auth as auth_router
from api.routers.auth import require_auth

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

app.include_router(auth_router.router, prefix=_V1, tags=["auth"])
app.include_router(portfolio.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(analyzer.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(hedging.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(sizer.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(momentum.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(chart.router, prefix=_V1, dependencies=_auth_dep, tags=["technical"])
app.include_router(quality.router, prefix=_V1, dependencies=_auth_dep, tags=["equities"])
app.include_router(short_screen.router, prefix=_V1, dependencies=_auth_dep, tags=["equities"])
app.include_router(fundamental_momentum.router, prefix=_V1, dependencies=_auth_dep, tags=["equities"])
app.include_router(index_dashboard.router, prefix=_V1, dependencies=_auth_dep, tags=["equities"])
app.include_router(fx_dashboard.router, prefix=_V1, dependencies=_auth_dep, tags=["fx"])
app.include_router(commodities.router, prefix=_V1, dependencies=_auth_dep, tags=["commodities"])
app.include_router(market_technicals.router, prefix=_V1, dependencies=_auth_dep, tags=["equities"])
app.include_router(economic_growth.router, prefix=_V1, dependencies=_auth_dep, tags=["macro"])
app.include_router(labor_market.router, prefix=_V1, dependencies=_auth_dep, tags=["macro"])
app.include_router(liquidity.router, prefix=_V1, dependencies=_auth_dep, tags=["macro"])
app.include_router(country_dashboard.router, prefix=_V1, dependencies=_auth_dep, tags=["macro"])
app.include_router(positioning.router, prefix=_V1, dependencies=_auth_dep, tags=["macro"])
app.include_router(sentiment.router, prefix=_V1, dependencies=_auth_dep, tags=["macro"])
app.include_router(breakout.router, prefix=_V1, dependencies=_auth_dep, tags=["macro"])
app.include_router(fx_model.router, prefix=_V1, dependencies=_auth_dep, tags=["fx"])
app.include_router(central_banks.router, prefix=_V1, dependencies=_auth_dep, tags=["macro"])
app.include_router(sector_metrics.router, prefix=_V1, dependencies=_auth_dep, tags=["equities"])
app.include_router(industry.router, prefix=_V1, dependencies=_auth_dep, tags=["macro"])
app.include_router(yield_curve.router, prefix=_V1, dependencies=_auth_dep, tags=["fixed-income"])
app.include_router(financials.router, prefix=_V1, dependencies=_auth_dep, tags=["equities"])
app.include_router(portfolio_news.router, prefix=_V1, dependencies=_auth_dep, tags=["portfolio"])
app.include_router(ontology.router, prefix=_V1, dependencies=_auth_dep, tags=["ontology"])
app.include_router(weekly_report.router, prefix=_V1, dependencies=_auth_dep, tags=["reports"])
app.include_router(commodities_curve.router, prefix=_V1, dependencies=_auth_dep, tags=["commodities"])
app.include_router(agent.router, prefix=_V1, dependencies=_auth_dep, tags=["agent"])


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
    checks = {
        "fred_api_key": bool(os.environ.get("FRED_API_KEY")),
    }
    status = "ok" if all(v for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}
