"""Background maintenance jobs enqueued by Cloud Scheduler.

Only the async-job sweep is scheduled by default. Cache warming is kept as an
admin/manual job because the generic Cloud Run Job does not share the API
service's in-memory TTL cache or local filesystem cache.
"""

from __future__ import annotations

import logging
import os
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
    from ontology.policy import system_actor

    actor = system_actor("maintenance.cache_warm")
    results: list[dict[str, str]] = []
    for tool_name, args in WARM_TOOLS:
        try:
            execute_tool(tool_name, args, actor=actor)
            logger.info("cache_warm tool=%s status=ok", tool_name)
            results.append({"tool": tool_name, "status": "ok"})
        except Exception as exc:
            logger.warning("cache_warm tool=%s status=error", tool_name, exc_info=True)
            results.append({"tool": tool_name, "status": "error", "error": str(exc)})
    return {"tools": results}


def refresh_workspace_sources(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Refresh the persisted inputs that drive the Workspace source-health panel."""
    from api.macro_snapshots import refresh_macro_snapshots
    from api.market_snapshots import refresh_market_snapshots
    from api.position_risk import refresh_portfolio_risk

    steps: list[dict[str, Any]] = []
    for name, fn in (
        ("market_snapshot_refresh", refresh_market_snapshots),
        ("macro_snapshot_refresh", refresh_macro_snapshots),
        ("portfolio_risk_refresh", refresh_portfolio_risk),
    ):
        try:
            result = fn()
            logger.info("workspace_source_refresh step=%s status=ok", name)
            steps.append({"step": name, "status": "ok", "result": result})
        except Exception as exc:
            logger.warning("workspace_source_refresh step=%s status=error", name, exc_info=True)
            steps.append({"step": name, "status": "error", "error": str(exc) or exc.__class__.__name__})

    errors = [step for step in steps if step["status"] == "error"]
    if errors:
        summary = "; ".join(f"{step['step']}: {step['error']}" for step in errors)
        raise RuntimeError(f"Workspace source refresh failed: {summary}")

    return {"steps": steps}


def sweep_async_jobs(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    from api.async_job_runner import fail_stale_active_jobs
    from api.job_queue import sweep_expired_jobs

    stale_failed = fail_stale_active_jobs()
    deleted = sweep_expired_jobs()
    analyzer_input_snapshots_deleted = 0
    try:
        from portfolio.portfolio_optimizer.portfolio_analyzer import cleanup_analyzer_input_snapshots

        analyzer_input_snapshots_deleted = cleanup_analyzer_input_snapshots()
    except Exception:
        logger.warning("async_job_sweep analyzer input snapshot cleanup failed", exc_info=True)
    logger.info(
        "async_job_sweep stale_failed=%d deleted=%d analyzer_input_snapshots_deleted=%d",
        stale_failed,
        deleted,
        analyzer_input_snapshots_deleted,
    )
    return {
        "stale_failed": stale_failed,
        "deleted": deleted,
        "analyzer_input_snapshots_deleted": analyzer_input_snapshots_deleted,
    }


def _env_int(name: str, default: int) -> int:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def drain_governance_outbox(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    logger.info("governance_outbox_drain skipped: outbox removed from ontology-primary runtime")
    return {
        "result": {"claimed": 0, "completed": 0, "failed": 0, "dead_lettered": 0},
        "metrics": {"pending": 0, "failed": 0, "dead_letter": 0},
        "lineage_state": "ontology",
    }
