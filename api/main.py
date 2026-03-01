"""
FastAPI backend for Market Analysis Dashboard.

Run from project root:
    uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path

# Replicate sys.path setup from gui/app.py lines 19-42
# Must happen before any project module imports
PROJECT_ROOT = Path(__file__).parent.parent
_PATHS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "equities" / "market_technicals",
    PROJECT_ROOT / "macro" / "economic_growth",
    PROJECT_ROOT / "macro" / "liquidity",
    PROJECT_ROOT / "macro" / "breakout",
    PROJECT_ROOT / "macro" / "positioning",
    PROJECT_ROOT / "equities" / "portfolio",
    PROJECT_ROOT / "portfolio" / "momentum" / "price_momentum",
    PROJECT_ROOT / "fx" / "model",
    PROJECT_ROOT / "fx" / "fx_dashboard",
    PROJECT_ROOT / "commodities",
    PROJECT_ROOT / "equities" / "index_dashboard",
    PROJECT_ROOT / "portfolio",
    PROJECT_ROOT / "macro" / "central_banks",
    PROJECT_ROOT / "macro" / "industry",
    PROJECT_ROOT / "portfolio" / "technical_analysis",
    PROJECT_ROOT / "equities" / "quality",
    PROJECT_ROOT / "equities",
    PROJECT_ROOT / "macro" / "country_dashboard",
    PROJECT_ROOT / "equities" / "short_screen",
    PROJECT_ROOT / "equities" / "sector_metrics",
    PROJECT_ROOT / "portfolio" / "momentum" / "fundamental_momentum",
]
for _p in reversed(_PATHS):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers AFTER sys.path is configured
from api.routers import (
    portfolio,
    optimizer,
    momentum,
    chart,
    quality,
    short_screen,
    fundamental_momentum,
    index_dashboard,
    fx_dashboard,
    commodities,
    market_technicals,
    economic_growth,
    liquidity,
    country_dashboard,
    positioning,
    breakout,
    fx_model,
    central_banks,
    sector_metrics,
    industry,
)

app = FastAPI(
    title="Market Analysis API",
    description="REST API wrapping the market analysis data modules",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolio.router, prefix="/api")
app.include_router(optimizer.router, prefix="/api")
app.include_router(momentum.router, prefix="/api")
app.include_router(chart.router, prefix="/api")
app.include_router(quality.router, prefix="/api")
app.include_router(short_screen.router, prefix="/api")
app.include_router(fundamental_momentum.router, prefix="/api")
app.include_router(index_dashboard.router, prefix="/api")
app.include_router(fx_dashboard.router, prefix="/api")
app.include_router(commodities.router, prefix="/api")
app.include_router(market_technicals.router, prefix="/api")
app.include_router(economic_growth.router, prefix="/api")
app.include_router(liquidity.router, prefix="/api")
app.include_router(country_dashboard.router, prefix="/api")
app.include_router(positioning.router, prefix="/api")
app.include_router(breakout.router, prefix="/api")
app.include_router(fx_model.router, prefix="/api")
app.include_router(central_banks.router, prefix="/api")
app.include_router(sector_metrics.router, prefix="/api")
app.include_router(industry.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
