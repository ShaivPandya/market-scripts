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


def sweep_async_jobs(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    from api.async_job_runner import fail_stale_active_jobs
    from api.job_queue import sweep_expired_jobs

    stale_failed = fail_stale_active_jobs()
    deleted = sweep_expired_jobs()
    logger.info("async_job_sweep stale_failed=%d deleted=%d", stale_failed, deleted)
    return {"stale_failed": stale_failed, "deleted": deleted}


def _env_int(name: str, default: int) -> int:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def drain_governance_outbox(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    from portfolio.core_db import drain_governance_outbox as drain_outbox
    from portfolio.core_db import get_governance_outbox_metrics

    result = drain_outbox(
        limit=_env_int("GOVERNANCE_OUTBOX_BATCH_SIZE", 50),
        lease_seconds=_env_int("GOVERNANCE_OUTBOX_LEASE_SECONDS", 300),
        max_attempts=_env_int("GOVERNANCE_OUTBOX_MAX_ATTEMPTS", 8),
        retry_base_seconds=_env_int("GOVERNANCE_OUTBOX_RETRY_BASE_SECONDS", 30),
        retry_max_seconds=_env_int("GOVERNANCE_OUTBOX_RETRY_MAX_SECONDS", 3600),
        retry_jitter_seconds=_env_int("GOVERNANCE_OUTBOX_RETRY_JITTER_SECONDS", 30),
    )
    metrics = get_governance_outbox_metrics()
    logger.info(
        "governance_outbox_drain claimed=%s completed=%s failed=%s dead_lettered=%s "
        "pending=%s failed_count=%s dead_letter=%s oldest_pending_age_seconds=%s",
        result.get("claimed"),
        result.get("completed"),
        result.get("failed"),
        result.get("dead_lettered"),
        metrics.get("pending"),
        metrics.get("failed"),
        metrics.get("dead_letter"),
        metrics.get("oldest_pending_age_seconds"),
    )
    return {"result": result, "metrics": metrics}
