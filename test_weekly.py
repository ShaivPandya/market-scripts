import sys
import os
from pathlib import Path

# Setup paths so imports work like main
PROJECT_ROOT = Path(".").resolve()
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

from dotenv import load_dotenv
load_dotenv()

from api.routers.weekly_report import get_weekly_report

print("Running weekly report generation...")
try:
    res = get_weekly_report()
    print("Success:", res.keys())
except Exception as e:
    import traceback
    traceback.print_exc()

