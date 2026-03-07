"""
Centralized sys.path configuration for market-scripts.

All entry points (api/main.py, auto_report scripts, standalone modules)
should call ``setup_paths()`` once before importing any project modules.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

_MODULE_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "equities" / "market_technicals",
    PROJECT_ROOT / "macro" / "economic_growth",
    PROJECT_ROOT / "macro" / "labor_market",
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
    PROJECT_ROOT / "portfolio" / "portfolio_optimizer",
    PROJECT_ROOT / "government_bonds",
    PROJECT_ROOT / "macro" / "sentiment",
]

_setup_done = False


def setup_paths() -> None:
    """Add all project module directories to sys.path (idempotent)."""
    global _setup_done
    if _setup_done:
        return
    for p in reversed(_MODULE_DIRS):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    _setup_done = True
