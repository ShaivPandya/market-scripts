"""Background maintenance jobs enqueued by Cloud Scheduler."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("api.maintenance_jobs")

WARM_TOOLS: list[tuple[str, dict[str, Any]]] = [
    ("get_portfolio", {}),
    ("get_market_breadth", {}),
    ("get_vix_term_structure", {}),
    ("get_liquidity", {}),
]


def warm_caches(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    from api.agent_tools import execute_tool

    results: list[dict[str, str]] = []
    for tool_name, args in WARM_TOOLS:
        try:
            execute_tool(tool_name, args)
            logger.info("cache_warm tool=%s status=ok", tool_name)
            results.append({"tool": tool_name, "status": "ok"})
        except Exception as exc:
            logger.warning("cache_warm tool=%s status=error", tool_name, exc_info=True)
            results.append({"tool": tool_name, "status": "error", "error": str(exc)})
    return {"tools": results}


def sweep_async_jobs(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    from api.job_queue import sweep_expired_jobs

    deleted = sweep_expired_jobs()
    logger.info("async_job_sweep deleted=%d", deleted)
    return {"deleted": deleted}
